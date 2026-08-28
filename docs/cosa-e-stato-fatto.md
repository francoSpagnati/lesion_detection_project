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

## 10. Planning e preprocessing di 501, 502 e 503

```bash
source scripts/env.sh
nnUNetv2_plan_and_preprocess -d 501 -npfp 16 -np 16 --verify_dataset_integrity   # e cosi' per 502 e 503
```

Eseguito da `notebooks/preprocessing_501_502_503.ipynb`, che aggiunge attorno al comando i controlli che
lo rendono verificabile: spazio disco con guardia a 60 GB fra un dataset e l'altro, hash degli
`splits_final.json` prima e dopo, e la verifica del piano. Log completi in `reports/logs/`.

→ `nnUNetPlans.json`, `dataset_fingerprint.json` e i dati preprocessati nelle tre cartelle di
`nnUNet_preprocessed/`. **24 GB in tutto** (WMH 1.1, ATLAS 15.9, DBB 6.8 — meno dei ~35 GB stimati),
**3 minuti** di parete con 16 processi.

| | Target spacing del piano | Coincide con l'attesa | Patch `3d_fullres` | Batch |
| --- | --- | --- | --- | --- |
| 501 WMH | `3.0 × 0.9766 × 1.2` | sì | `64 × 224 × 128` | 2 |
| 502 ATLAS | `1.0 × 1.0 × 1.0` | sì | `128 × 128 × 128` | 2 |
| 503 DBB | `1.0 × 1.0 × 1.0` | sì | `128 × 160 × 112` | 2 |

I valori del piano sono nell'ordine assi di nnU-Net, che mette per prima la direzione a risoluzione
peggiore: sono gli stessi numeri di `1.2 × 0.9766 × 3.0` permutati, non una spacing diversa.

**Tre esiti che cambiano qualcosa:**

- **`3d_lowres` scartata su tutti e tre** dal planner («image size difference too small»): resta il
  confronto a due, `2d` contro `3d_fullres`, e la cascade non è disponibile.
- **Su WMH la patch copre 64 delle 79 fette (81% del volume).** Era l'argomento principale a favore
  della `2d` — «con fette da 3 mm il contesto attraverso il piano è scarso» — e il piano lo indebolisce:
  la rete 3D vede quasi tutto il volume in una volta. La `2d` va comunque misurata, ma l'aspettativa a
  priori non è più a suo favore.
- **Batch size 2 ovunque in `3d_fullres`**, cioè il minimo: le patch sono grandi quanto il budget di
  memoria consente. Gli split sono rimasti invariati (hash identici) e le dimensioni dei fold sono
  quelle attese.

## 11. Baseline: fold 0 di 3d_fullres su 501, 502, 503

```bash
setsid nohup bash scripts/run_baselines.sh 501 502 503 > reports/logs/baselines.log 2>&1 < /dev/null &
python scripts/summarize_validation.py     # -> reports/baseline_validation.md
```

1000 epoche, ~22.9 s/epoca su A100, circa 6 ore per dataset in sequenza sulla stessa GPU. Il tracking W&B
funziona; `scripts/summarize_validation.py` legge i `summary.json` della validation e ne riassume la
distribuzione, che è il punto: il Dice medio da solo su questi compiti non descrive il comportamento.

| | casi val | Dice medio | mediana | Q1–Q3 |
| --- | --- | --- | --- | --- |
| 501 WMH | 16 | 0.774 | 0.837 | 0.705 – 0.880 |
| 502 ATLAS | 112 | 0.610 | 0.721 | 0.465 – 0.839 |
| 503 DBB | 23 | 0.971 | — | per classe: 0.928 (CSF) – 0.989 (sostanza bianca) |

**Cosa dicono i numeri, oltre alla media:**

- la **mediana è più alta della media** in entrambi (+6 e +11 punti): pochi casi difficili tirano giù la
  media, il modello lavora bene sulla maggioranza;
- il **volume del bersaglio domina**. Metà dei casi con lesioni più piccole contro metà con le più grandi:
  0.667 / 0.881 su WMH, 0.509 / 0.712 su ATLAS. Su ATLAS i casi sotto 100 voxel stanno a 0.28;
- su ATLAS la **coorte `r009`** (19 casi su 112) ha lesioni mediane da 1398 voxel contro 7053 delle altre e
  Dice 0.337 contro 0.666. Buona parte del divario fra i due dataset è composizione del campione, non
  qualità del modello;
- **8 casi su 112 hanno Dice 0**, e non sono solo i minuscoli: uno ha 9586 voxel di riferimento e 217
  predetti. Sono errori di **localizzazione**, categoria diversa dai bordi imprecisi, da guardare a parte;
- **WMH per sito**: Singapore 0.818, Amsterdam_GE3T 0.790, Amsterdam_GE1T5 0.750 (1 caso), Utrecht 0.715.
  Con 16 casi su un fold non è ancora interpretabile;
- **DBB sta a 0.971**, ma il numero non è confrontabile con gli altri due: il compito è la partizione in
  tessuti, il foreground è un voxel su cinque e il Dice è alto per costruzione. La classe più difficile è
  il **CSF** (0.928), che ha molta superficie per poco volume; le classi piccole tengono bene (grigia
  profonda 0.972, tronco encefalico 0.973). Soprattutto: **questa validation è sui 23 casi sani**, mentre
  il senso di DBB sono i 37 malformati che stanno nel test e non sono ancora stati toccati.

**Due problemi risolti durante l'esecuzione, entrambi documentati in [training.md](training.md):** i
training lanciati con `nohup` muoiono con la sessione che li ha avviati (serve `setsid`), e la ripresa con
`--c` di un run tracciato su W&B falliva per un bug del logger di nnU-Net 2.8, corretto nel nostro trainer.

## 12. Esame dei casi falliti di ATLAS

```bash
python scripts/inspect_failures.py 502 --soglia 0.05   # -> reports/failure_analysis_502.md
```

Per ogni caso sotto soglia: volumi, componenti connesse, centroidi in millimetri, distanza fra i
centroidi, **test dello specchio** (predizione controlaterale?), probabilità assegnate ai voxel di
lesione (le legge dai `.npz` salvati con `--npz`), e rapporto di intensità lesione/cervello confrontato
con i casi riusciti.

**Esito: i 10 fallimenti non sono un difetto della pipeline ma un limite del T1.** Il modello non vede
quelle lesioni perché non hanno contrasto (rapporto 1.07 contro 0.62 nei casi riusciti), e le probabilità
predette al loro interno sono nulle. Dettaglio e conseguenze in
[dataset-e-decisioni.md](dataset-e-decisioni.md#atlas-una-parte-dei-casi-non-è-risolvibile-con-il-solo-t1).

## 13. Ablazione FLAIR: 511 contro 501

```bash
nnUNetv2_plan_and_preprocess -d 511 -npfp 16 -np 16 --verify_dataset_integrity
setsid nohup bash scripts/run_baselines.sh 511 > reports/logs/baselines.log 2>&1 < /dev/null &
```

Prima di lanciare è stato verificato che il **piano di 511 fosse identico a quello di 501** — stessa
spacing, stessa patch `64×224×128`, stessi kernel e strides — e che gli `splits_final.json` fossero
bit-identici. Solo così la differenza è attribuibile al canale e non all'architettura. L'unica voce
diversa è `median_image_size_in_voxels` (`79×242×134` contro `74×237×134`), statistica del fingerprint
che non entra nella rete: nasce dal ritaglio, perché il T1 allineato alla griglia FLAIR ha qualche fetta
azzerata in più ai bordi.

| | Dice medio | mediana |
| --- | --- | --- |
| 501, **FLAIR** | **0.7739** | 0.8369 |
| 511, **T1** | **0.6257** | 0.7138 |

**Esito: la FLAIR vale 14.8 punti di Dice, ed è migliore in 16 casi su 16** (Wilcoxon appaiato sugli
stessi casi, p = 0.00003). Il confronto è appaiato perché i due dataset condividono casi e fold.

Il dettaglio che conta più della media: **la perdita dipende dalla dimensione della lesione**.

| | perdita passando da FLAIR a T1 |
| --- | --- |
| lesioni sotto la mediana (< 2266 voxel) | **−0.242** |
| lesioni sopra la mediana | **−0.055** |

Sulle lesioni grandi il T1 se la cava; su quelle piccole crolla. Converge con quanto trovato in modo
indipendente sui fallimenti di ATLAS (passo 12): il T1 non porta l'informazione necessaria per le lesioni
piccole e non cavitate.

## 14. Verifiche richieste dall'incontro del 27 agosto

Due domande poste dai relatori, entrambe risolte leggendo i dati invece di ragionarci sopra.

**DBB ha maschere di lesione o malformazione?** No. I cinque tipi di dato pubblicati su brainlife sono
tutti già scaricati: la `mask` è una maschera **cerebrale binaria**, e le parcellazioni dei casi
patologici contengono **gli stessi valori `1..6`** di quelle dei sani, senza classi aggiuntive. La
patologia in quel benchmark è implicita nell'anatomia, non annotata.

**Dove sono i dati a 1.5 T?** Censiti per la prima volta, dai metadati di ATLAS
(`20211112_ATLAS_2.0_SupplementaryInfo.csv`) e dal nome del sito su WMH: 10 casi su 99 in WMH, 52 sui 655
annotati di ATLAS (coorti `R029`, `R039`, `R040`). Nel fold 0 di ATLAS i casi a 1.5 T hanno Dice **0.653**
contro **0.607** dei 3 T, quindi la misura preliminare **non** supporta l'esclusione.

Entrambe in [dataset-e-decisioni.md](dataset-e-decisioni.md) con i dettagli e le cautele.

---

## Dataset nnU-Net attualmente disponibili

| Dataset | Canale | Classi | Sviluppo | Test | Prodotto da |
| --- | --- | --- | --- | --- | --- |
| `Dataset501_WMH` | FLAIR | lesione | 79 | 20 | `notebooks/dataset_preparation.ipynb` |
| `Dataset502_ATLAS` | T1 | lesione | 556 | 99 | `notebooks/dataset_preparation.ipynb` |
| `Dataset503_DBB` | T1 | 6 tessuti | 207 | 37 | `scripts/convert_dbb_to_nnunet.py` |
| `Dataset511_WMH_T1` | T1 | lesione | 79 | 20 | `scripts/make_wmh_t1_dataset.py` |
| `Dataset599_SMOKE_ATLAS` | T1 | lesione | 16 | — | `scripts/make_smoke_dataset.py` (usa e getta) |

501, 502 e 503 sono preprocessati (passo 10) in `2d` e `3d_fullres`. Il 511 no: va fatto quando si
lancia l'ablazione FLAIR. Il passo successivo è il training delle baseline — vedi
[training.md](training.md#step-5---strategia-di-training).
