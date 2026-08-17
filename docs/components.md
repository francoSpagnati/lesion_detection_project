# Componenti

Mappa di cartelle e file. Il *perché* delle scelte è in
[dataset-e-decisioni.md](dataset-e-decisioni.md); il *cosa è stato eseguito* in
[cosa-e-stato-fatto.md](cosa-e-stato-fatto.md).

## Configurazione

| | |
| --- | --- |
| `configs/datasets.json` | Percorsi canonici dei dataset sorgente. |
| `configs/experiments.json` | Dataset ID, configurazioni nnU-Net, trainer e fold da confrontare. |
| `configs/wandb.json` | Progetto W&B, tag e modalità. **Non contiene segreti.** |
| `.env` | Segreti locali (`WANDB_API_KEY`, credenziali brainlife). Escluso da Git, permessi stretti. |
| `scripts/env.sh` | Variabili nnU-Net, trainer esterni, scelta GPU. Uso: `source scripts/env.sh`. |

## Dati

| | |
| --- | --- |
| `data/raw/` | Symlink ai sorgenti su `/data01`, in sola lettura. Eccezione: `data/raw/DBB/` sono file veri (4.4 GB), con `manifest.csv`. |
| `data/processed/manifests/` | Un CSV per dataset, una riga per caso: spacing, shape, orientamento, volumi. Rigenerabile, usato come cache. |
| `data/processed/splits/` | Split riproducibili: holdout di test e 5 fold stratificati, con il seed. |
| `data/processed/qc/` | Figure di QC: spacing, anisotropia, dimensione delle lesioni, overlay. |
| `data/processed/nnUNet_raw/` | Dataset convertiti in formato nnU-Net. Immagini come symlink. |
| `nnUNet/nnUNet_raw` | Symlink ai dataset sopra, per la variabile `nnUNet_raw`. |
| `nnUNet/nnUNet_preprocessed` | Fingerprint, piani e dati preprocessati. Contiene anche `splits_final.json`, scritto da noi. |
| `nnUNet/nnUNet_results` | Checkpoint, predizioni e metriche. |

## Script

| | |
| --- | --- |
| `scripts/explore_datasets.py` | Inventario read-only di file e header NIfTI → `reports/dataset_exploration.*`. |
| `scripts/download_dbb.py` | Scarica il progetto brainlife DBB. Ripartibile; credenziali da ambiente o `.env`. |
| `scripts/measure_bias_field.py` | Fase 0: misura il bias di campo con N4 senza modificare i dati → `reports/bias_field_assessment.*`. |
| `scripts/analyze_dbb.py` | Analisi read-only di DBB → `manifests/DBB_cases.csv` e `reports/dbb_analysis.*`. |
| `scripts/compare_datasets.py` | Confronto a tre dai manifest → `reports/dataset_comparison_three.*`. |
| `scripts/convert_dbb_to_nnunet.py` | Converte DBB in `Dataset503_DBB`; fold 0 = split ufficiale del benchmark. |
| `scripts/make_wmh_t1_dataset.py` | Costruisce `Dataset511_WMH_T1`: ablazione della FLAIR a parità di casi, label e fold. |
| `scripts/make_smoke_dataset.py` | Dataset minuscolo (`Dataset599_*`) per lo smoke test della pipeline. |
| `scripts/nnunet_trainers/nnUNetTrainerWandb.py` | Accende il logger W&B integrato di nnU-Net e rende il run identificabile. Varianti a 5 e 250 epoche. |
| `scripts/wandb_setup.py` | Autenticazione e test di connessione W&B. |

I trainer stanno **fuori** da `site-packages`: nnU-Net li trova tramite `nnUNet_extTrainer`.

## Notebook

| | |
| --- | --- |
| `notebooks/dataset_preparation.ipynb` | Preparazione completa di WMH e ATLAS: QC, risoluzione, split, conversione → `reports/dataset_preparation.*`. |
| `notebooks/dataset_comparison_exploration.ipynb` | Confronto esplorativo read-only WMH vs ATLAS. |
| `notebooks/wmh_validation.ipynb` | Verifica rapida del dataset WMH e dei percorsi nnU-Net. |

## Report

`reports/` contiene solo file **rigenerabili** da uno script o da un notebook: `dataset_exploration`,
`dataset_preparation`, `bias_field_assessment`, `dbb_analysis`, `dataset_comparison_three`. **Da non
modificare a mano.**

## Dove scrivono i dati

I dati originali stanno su `/data01`, in sola lettura. Tutto ciò che viene prodotto sta sotto `/home`,
quindi lo spazio disco di `/home` è il vincolo reale.

I dataset nnU-Net vivono fisicamente in `data/processed/nnUNet_raw/` e sono esposti tramite symlink da
`nnUNet/nnUNet_raw/`: un solo contenuto, raggiungibile sia dalla convenzione del progetto sia dalla
variabile d'ambiente, senza duplicazione. Le immagini a loro volta sono symlink ai file originali, quindi i
dataset preparati occupano decine di MB invece di ~14 GB.
