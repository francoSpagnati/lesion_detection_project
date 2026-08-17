# Cosa è stato fatto, in pratica

Una riga per passo: **quale script**, **quale comando**, **cosa ha prodotto**, **cosa ne è uscito**. Il
*perché* di ogni decisione è in [dataset-e-decisioni.md](dataset-e-decisioni.md); il *cosa manca* è in
[obiettivo-e-piano.md](obiettivo-e-piano.md#piano-di-lavoro-in-ordine).

Tutti i comandi si eseguono da `/home/ccantini/lesion_detection_project`, dopo `source scripts/env.sh`.
Tutti i report in `reports/` sono **rigenerabili**: non modificarli a mano.

---

## 1. Inventario grezzo dei file

```bash
python scripts/explore_datasets.py --config configs/datasets.json
```

→ `reports/dataset_exploration.{md,json}`. Attraversa le cartelle sorgente in sola lettura, conta i NIfTI,
ne classifica il ruolo dal nome e campiona gli header.

**Esito:** 594 file per WMH, 2266 per ATLAS. Serve soprattutto a scoprire che il numero di file è molto
maggiore del numero di casi utilizzabili, perché le cartelle contengono sottoprodotti.

## 2. Confronto esplorativo e verifica

`notebooks/dataset_comparison_exploration.ipynb` e `notebooks/wmh_validation.ipynb`, read-only.

**Esito:** emergono i problemi reali dei due dataset. I tre che contano: solo
`_T1w_BiasCorr_2FLAIR_Aligned` condivide la griglia della FLAIR; le due maschere ATLAS sono la stessa
segmentazione; `sub-r039s002` è codificato `0/0.01` e va binarizzato con `> 0`.

## 3. Preparazione, split e conversione di WMH e ATLAS

`notebooks/dataset_preparation.ipynb` — il passo centrale. Nell'ordine: manifest per caso appaiato → analisi
della risoluzione → figure di QC → split (holdout + 5 fold stratificati, seed `42`) → conversione nnU-Net.

→ `reports/dataset_preparation.{md,json}`, `data/processed/manifests/{WMH,ATLAS}_cases.csv`,
`data/processed/splits/`, `data/processed/qc/`, `Dataset501_WMH`, `Dataset502_ATLAS`.

| | Casi appaiati | Sviluppo (5 fold) | Test interno | Target spacing |
| --- | --- | --- | --- | --- |
| WMH | 99, su 4 siti | 79 | 20 | `1.2 × 0.9766 × 3.0` mm |
| ATLAS | 655, su 33 coorti | 556 | 99 | `1.0 × 1.0 × 1.0` mm |

I manifest sono in cache: per ricalcolarli da zero impostare `FORCE_RECOMPUTE = True`.

## 4. Sblocco della GPU

Nessuno script, un comando. `torch.cuda.is_available()` era `False`: il pacchetto installato era
`torch 2.13.0+cu130`, compilato per CUDA 13.0, mentre il driver della macchina (550.144.03) espone **CUDA
12.4**. CUDA garantisce compatibilità solo dentro la stessa major version.

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0+cu124 torchvision==0.21.0+cu124
```

L'indice `cu124` da solo non basta: non ospita le dipendenze `nvidia-*`, quindi serve
`--extra-index-url` (**non** `--index-url`) perché pip possa prenderle da PyPI.

**Esito** verificato con calcolo reale, non solo con l'import: matmul 4096², più `Conv3d` con autocast fp16
e backward. `torch 2.6.0+cu124`, A100 80GB (sm_80).

## 5. Download del terzo dataset (DBB)

```bash
BRAINLIFE_USER=... BRAINLIFE_PASSWORD=... python scripts/download_dbb.py
```

Credenziali dall'ambiente o da `.env`, **mai** a riga di comando (finirebbero nella history). Il download è
ripartibile: rilanciarlo completa un trasferimento interrotto senza riscaricare nulla.

→ `data/raw/DBB/`, 3576 dataset, 4.4 GB, con `manifest.csv` per risalire da ogni file a soggetto, coorte e
split. A differenza degli altri due **non è un symlink**: i file sono scaricati davvero.

**Esito:** 3576/3576, 0 errori. Un bug non ovvio incontrato e risolto: l'API brainlife paginata **senza
chiave di ordinamento** restituisce record duplicati e ne omette altri in silenzio (3576 record, 3383
unici). La correzione — `sort: _id` nella query — è dentro lo script.

## 6. Fase 0 — misura del bias di campo

```bash
python scripts/measure_bias_field.py --n 10 --seed 0
```

→ `reports/bias_field_assessment.{md,json}`. Stima il campo con N4 (ANTs via `antspyx`, già installato:
non serve né FSL né ANTs da riga di comando) e ne misura l'ampiezza dentro la testa, erodendo di margini
fisici crescenti. **Non modifica alcun dato.**

**Esito: N4 non è giustificato su nessuno dei tre dataset**, la Fase 1 decade. Numeri, cautele
metodologiche e limite dichiarato in
[dataset-e-decisioni.md](dataset-e-decisioni.md#niente-n4-la-misura-lo-esclude).

## 7. Analisi di DBB e confronto a tre

```bash
python scripts/analyze_dbb.py          # -> reports/dbb_analysis.md + manifest per caso
python scripts/compare_datasets.py     # -> reports/dataset_comparison_three.md
```

Il primo porta DBB allo stesso livello di descrizione degli altri due, scrivendo `DBB_cases.csv` con lo
**stesso schema** di `WMH_cases.csv` e `ATLAS_cases.csv` — è questo che rende possibile il confronto. Il
secondo legge i tre manifest e produce le tabelle comparate. Entrambi read-only.

**Esito:** due scoperte che hanno cambiato la conversione — l'eterogeneità dello spazio in DBB (31 casi
nativi, 12 orientati RAS) e il carico di foreground che separa DBB dagli altri di tre ordini di grandezza.
Entrambe in [dataset-e-decisioni.md](dataset-e-decisioni.md#dbb--dataset503_dbb-t1-6-tessuti) e
[obiettivo-e-piano.md](obiettivo-e-piano.md#non-sono-tre-versioni-dello-stesso-problema).

## 8. Conversione di DBB e variante T1 di WMH

```bash
python scripts/convert_dbb_to_nnunet.py    # -> Dataset503_DBB
python scripts/make_wmh_t1_dataset.py      # -> Dataset511_WMH_T1
```

`Dataset503_DBB`: 207 casi di sviluppo (sani) + 37 di test (patologici), 6 classi, canale T1. Immagini **e**
label sono symlink. Verificato con `nnUNetv2_extract_fingerprint -d 503 --verify_dataset_integrity`: nessun
errore. Alcuni casi emettono un avviso di affine non identica fra immagine e label, ma la differenza è alla
settima cifra decimale — precisione in virgola mobile, non disallineamento.

`Dataset511_WMH_T1`: identico a `Dataset501_WMH` in tutto — stessi 99 casi, stesse label, **stessi fold**
copiati e non rigenerati — tranne il canale. Cambia **una sola variabile**, quindi la differenza di Dice fra
501 e 511 misura esattamente il contributo della FLAIR.

## 9. Tracking su W&B e smoke test

```bash
source scripts/env.sh
python scripts/make_smoke_dataset.py --sorgente Dataset502_ATLAS --n 16
nnUNetv2_plan_and_preprocess -d 599 -c 3d_fullres --verify_dataset_integrity
nnUNetv2_train 599 3d_fullres 0 -tr nnUNetTrainerWandb_5epochs
```

**Il fatto che ha semplificato le cose:** nnU-Net 2.8 **ha già** un logger W&B integrato (`MetaLogger` →
`WandbLogger`), acceso da `nnUNet_wandb_enabled=1`. Non serve riscriverlo.
`scripts/nnunet_trainers/nnUNetTrainerWandb.py` lo accende e ne copre i tre buchi: il run nasce senza nome,
il config non contiene la **target spacing effettiva del piano** (senza la quale due run non sono
confrontabili), e non registra canali di input né mappa delle label — cioè proprio le variabili che qui
cambiano fra un esperimento e l'altro.

Nome e tag vanno passati **per ambiente** (`WANDB_NAME`, `WANDB_TAGS`) prima che `wandb.init()` venga
chiamato: assegnarli dopo non ha effetto sul run già creato. Il trainer vive **nel repository**, non in
`site-packages`: nnU-Net ≥ 2.8 lo trova tramite `nnUNet_extTrainer`, impostata da `scripts/env.sh`.

Lo smoke dataset prende 16 casi da ATLAS **a passo costante** — non i primi 16, che verrebbero da una sola
coorte.

**Esito verificato** end-to-end: 5 epoche su GPU, ~23 s/epoca, validation completata, e su W&B sono arrivati
config (`target_spacing [1,1,1]`, `patch_size [128,128,128]`, `channel_names {'0': 'T1'}`, label,
iperparametri, GPU), le metriche per epoca e le metriche finali in summary (`final_val/foreground_dice`).
Il Dice dello smoke **non va letto**: 16 casi, 5 epoche.

---

## Dataset nnU-Net attualmente disponibili

| Dataset | Canale | Classi | Sviluppo | Test | Prodotto da |
| --- | --- | --- | --- | --- | --- |
| `Dataset501_WMH` | FLAIR | lesione | 79 | 20 | `notebooks/dataset_preparation.ipynb` |
| `Dataset502_ATLAS` | T1 | lesione | 556 | 99 | `notebooks/dataset_preparation.ipynb` |
| `Dataset503_DBB` | T1 | 6 tessuti | 207 | 37 | `scripts/convert_dbb_to_nnunet.py` |
| `Dataset511_WMH_T1` | T1 | lesione | 79 | 20 | `scripts/make_wmh_t1_dataset.py` |
| `Dataset599_SMOKE_ATLAS` | T1 | lesione | 16 | — | `scripts/make_smoke_dataset.py` (usa e getta) |

Nessuno di questi è ancora stato preprocessato tranne il 599. Il passo successivo è
`nnUNetv2_plan_and_preprocess` su 501, 502 e 503 — vedi [training.md](training.md).
