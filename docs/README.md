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

[Componenti](components.md) è la mappa di cartelle e file, da consultare al bisogno.

[Recap per il relatore](recap-per-relatore.md) non è documentazione tecnica ma un documento di
**comunicazione**: riassume per chi sta fuori dal progetto cosa è stato analizzato, cosa deciso e perché.
Va riscritto quando serve, non mantenuto in parallelo agli altri.

Per aprire una **nuova sessione di lavoro con un assistente**, usa il prompt in
[prompt-nuova-chat.md](prompt-nuova-chat.md): fa partire la sessione da questi documenti invece di far
riesplorare i dati da zero.

## Stato

| Fase | Stato |
| --- | --- |
| Esplorazione dei dati | completata (`reports/dataset_exploration.md`) |
| Preparazione e split di WMH e ATLAS | completata (`reports/dataset_preparation.md`) |
| Sblocco GPU | completato: `torch 2.6.0+cu124`, verificato su A100 |
| Download di DBB | completato: 3576/3576 file, 4.4 GB |
| Fase 0 — misura del bias di campo | completata: **N4 non giustificato**, la Fase 1 decade |
| Analisi di DBB e confronto a tre | completata (`reports/dbb_analysis.md`, `reports/dataset_comparison_three.md`) |
| Conversione di DBB | completata: `Dataset503_DBB`, integrità verificata |
| Variante T1-only di WMH | completata: `Dataset511_WMH_T1` |
| Tracking W&B | **verificato** end-to-end sullo smoke test |
| Planning e preprocessing di 501/502/503 | completato: 24 GB, spacing verificate (`notebooks/preprocessing_501_502_503.ipynb`) |
| Baseline di training (fold 0) | 501 e 502 **fatte** (Dice 0.774 e 0.610, `reports/baseline_validation.md`); 503 in corso |
| **Ablazione FLAIR (511 contro 501)** | **da fare — è il passo successivo** |
| Transfer DBB → ATLAS | pianificato |
| Dataset privato AInCP | non ancora consegnato |
| Stadio 2 — inpainting e gemello sano | da fare, dopo la segmentazione |

## Vincoli operativi da ricordare

- `/data01` è **in sola lettura**: i dati sorgente non si modificano. Tutto l'output sta su `/home`, che è
  oltre il 90% di occupazione, con circa 250 GB liberi che variano con l’uso degli altri utenti (`df -h /home`).
- Le 4 A100 80GB sono **condivise con altri utenti**: quale sia libera cambia di giorno in giorno, va
  controllato con `nvidia-smi` prima di ogni lancio.
- `.env` contiene `WANDB_API_KEY` (e le credenziali brainlife, se usate): è escluso da Git e deve avere
  permessi stretti. Non va mai citato in notebook, JSON o report.
- `splits_final.json` **non si rigenera** fra un esperimento e l'altro: se cambia, i confronti fra run
  precedenti e successive perdono validità.
