# Documentazione del progetto

Rilevamento automatico di lesioni cerebrali in bambini con **paralisi cerebrale**, con nnU-Net v2.
I dati originali non vengono duplicati: sono referenziati da `data/raw/` tramite symlink.

## Da dove iniziare

Leggi in quest'ordine. Sono cinque documenti, nessuno ripete gli altri.

1. **[Obiettivo e piano](obiettivo-e-piano.md)** — a cosa serve il progetto, i quattro dataset e il ruolo
   di ciascuno, cosa va fatto adesso e in che ordine. **Se leggi un solo file, leggi questo.**
2. **[Primer: MRI e segmentazione](primer-mri-e-segmentazione.md)** — che cos'è una risonanza, voxel,
   spacing, geometrie, metriche, come funziona nnU-Net. Non presuppone background in neuroimaging.
3. **[Dataset e decisioni](dataset-e-decisioni.md)** — cosa contiene ciascun dataset, i problemi trovati
   misurandoli, e le decisioni prese con la loro motivazione. Include cosa **non** facciamo e perché.
4. **[Cosa è stato fatto](cosa-e-stato-fatto.md)** — pratico: quale script, quale comando, cosa ha
   prodotto, cosa ne è uscito.
5. **[Training](training.md)** — dai dataset preparati al modello addestrato e valutato, con il tracking
   su W&B.

[Panoramica del progetto](panoramica-progetto.md) è il documento di **comunicazione**: una sintesi
compatta di dati, scelte, training e risultati, da mostrare a chi sta fuori dal progetto. Non è
documentazione tecnica e non va mantenuta in parallelo agli altri: si aggiorna quando serve mostrarla.

Per aprire una **nuova sessione di lavoro con un assistente**, usa il prompt in
[prompt-nuova-chat.md](prompt-nuova-chat.md): fa partire la sessione da questi documenti invece di far
riesplorare i dati da zero.

## Stato al 29 agosto 2026

**In una riga:** le baseline e le due ablazioni chiuse sono misurate; l'esclusione dei dati a 1.5 T è
stata provata e scartata; SynthStrip è validato e i tre dataset successivi sono pronti ma **nessuno dei
tre training è ancora partito**.

### Risultati misurati

| Esperimento | Dice fold 0 | Cosa dice |
| --- | --- | --- |
| `501` WMH, FLAIR | **0.774** | baseline |
| `502` ATLAS, T1 | **0.610** | baseline; i casi falliti sono lesioni piccole non cavitate |
| `503` DBB, 6 tessuti | **0.971** | baseline; validation su 23 casi *sani* |
| `511` WMH, solo T1 | **0.626** | −0.148 contro la FLAIR, peggiore in **16 casi su 16** (p=0.00003) |
| `521` ATLAS solo 3 T | **0.616** | +0.0093 appaiato su 103 casi, **p=0.67**: escludere i 1.5 T non serve |

### Pronto, in attesa di GPU

| Dataset | Domanda | Confronto |
| --- | --- | --- |
| `512` WMH FLAIR+T1 | il T1 aggiunge qualcosa alla FLAIR? | contro `501` |
| `522` WMH skull-stripped | lo skull stripping aiuta? | contro `501` |
| `523` ATLAS skull-stripped | lo skull stripping aiuta? | contro `502` |

Tutti e tre costruiti, verificati e preprocessati in `3d_fullres`. Circa 7 ore di training ciascuno.

### Da fare dopo

| Voce | Stato |
| --- | --- |
| **Variante D** — modello unico sui tre dataset a 1 mm, 7 classi | **obiettivo prioritario**, da costruire |
| Varianti A (WMH+ATLAS), B (A+AInCP), C (tutti con `ignore`) | alternative se D non regge |
| Transfer DBB → ATLAS | pianificato |
| Confronto `2d` contro `3d_fullres` in cross-validation | pianificato |
| Prova sui 2 pazienti AInCP a 3 T | in attesa dei dati |
| Dataset privato AInCP | non ancora consegnato |
| Stadio 2 — inpainting e gemello sano | dopo la segmentazione |

### Fasi concluse

Esplorazione dei dati, preparazione e split di WMH e ATLAS, sblocco GPU (`torch 2.6.0+cu124` su A100),
download di DBB (3576 file), misura del bias di campo (**N4 non giustificato**), analisi di DBB e
confronto a tre, conversione di DBB, tracking W&B verificato end-to-end, planning e preprocessing di
501/502/503, esame dei casi falliti di ATLAS, verifica delle maschere DBB (**le maschere di lesione non
esistono**), figure di confronto predizione/ground truth, validazione di SynthStrip su 244 casi DBB.
Il dettaglio di ciascuna, con comandi e misure, è in [cosa-e-stato-fatto.md](cosa-e-stato-fatto.md).

## Vincoli operativi da ricordare

- `/data01` è **in sola lettura**: i dati sorgente non si modificano. Tutto l'output sta su `/home`, che è
  oltre il 90% di occupazione, con circa 250 GB liberi che variano con l’uso degli altri utenti (`df -h /home`).
- Le 4 A100 80GB sono **condivise con altri utenti**: quale sia libera cambia di giorno in giorno, va
  controllato con `nvidia-smi` prima di ogni lancio.
- `.env` contiene `WANDB_API_KEY` (e le credenziali brainlife, se usate): è escluso da Git e deve avere
  permessi stretti. Non va mai citato in notebook, JSON o report.
- `splits_final.json` **non si rigenera** fra un esperimento e l'altro: se cambia, i confronti fra run
  precedenti e successive perdono validità.
