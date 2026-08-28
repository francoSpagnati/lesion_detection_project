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
| Baseline di training (fold 0) | **completate**: 501 `0.774`, 502 `0.610`, 503 `0.971` (`reports/baseline_validation.md`) |
| Ablazione FLAIR (511 contro 501) | **completata**: T1 `0.626` contro FLAIR `0.774`, −0.148, peggiore in **16 casi su 16** |
| Esame dei casi falliti di ATLAS | completato: limite del T1, non un bug (`reports/failure_analysis_502.md`) |
| Verifica maschere DBB e campo magnetico | completata il 27/08, vedi [dataset-e-decisioni](dataset-e-decisioni.md) |
| **Ablazione a due canali (512: T1+FLAIR)** | **da fare — chiude il confronto sui canali** |
| Immagini di confronto predizione/ground truth | da fare, richiesta dai relatori |
| Studio delle maschere DBB per lesioni | **verificato: non esistono**, vedi decisioni |
| Esclusione dei dati 1.5 T | **misurata e scartata**: +0.0093 appaiato su 103 casi, p=0.67 (`reports/test_esclusione_1_5T.md`) |
| Dataset skull-stripped | **costruiti**: `Dataset522_WMH_SS` e `Dataset523_ATLAS_SS`, geometria e fold invariati |
| Dataset unificato a 1 mm sui tre dataset | pianificato |
| Skull stripping con SynthStrip | **validato**: 0.957 sui patologici DBB; taglia il CSF negli idrocefali, ma ritagliando WMH/ATLAS si perde lo **0.00%/0.12%** delle lesioni (`reports/synthstrip_dbb.md`, `reports/lesioni_fuori_maschera.md`) |
| Prova su 2 pazienti AInCP a 3 T | in attesa dei dati |
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
