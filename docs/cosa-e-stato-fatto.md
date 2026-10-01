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

1000 epoche, circa 23 s/epoca su A100: 5h42 per 501, 6h43 per 502, 8h03 per 503, in sequenza sulla stessa
GPU fra il 17 e il 18 agosto. Il tracking W&B funziona.

Esito: **501 = 0.774, 502 = 0.610, 503 = 0.971**. Architetture, piani e lettura completa dei risultati
sono in [modelli-ed-esperimenti.md](modelli-ed-esperimenti.md); la distribuzione caso per caso, che su
questi compiti conta piu' della media, e' in `reports/baseline_validation.md`, prodotto da
`scripts/summarize_validation.py`.

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

Prima di lanciare e' stato verificato che il **piano di 511 fosse identico a quello di 501** — stessa
spacing, stessa patch `64x224x128`, stessi kernel e strides — e che gli `splits_final.json` fossero
bit-identici. Solo cosi' la differenza e' attribuibile al canale e non all'architettura. L'unica voce
diversa e' `median_image_size_in_voxels` (`79x242x134` contro `74x237x134`), statistica del fingerprint
che non entra nella rete: nasce dal ritaglio, perche' il T1 allineato alla griglia FLAIR ha qualche fetta
azzerata in piu' ai bordi.

Esito: **0.6257 contro 0.7739**, la FLAIR migliore in 16 casi su 16 (p = 0.00003), con la perdita
concentrata sulle lesioni piccole. Numeri e interpretazione in
[modelli-ed-esperimenti.md](modelli-ed-esperimenti.md#511--wmh-sul-solo-t1-quanto-vale-la-flair).

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

## 15. Test 1 — escludere i dati a 1.5 T non serve

`scripts/make_atlas_3t_dataset.py` costruisce `Dataset521_ATLAS_3T` togliendo le tre coorti a 1.5 T
(`r029`, `r039`, `r040`): **511 casi di sviluppo invece di 556** e 92 di test invece di 99. I fold sono
**filtrati da quelli di 502, mai rigenerati**, quindi la validation del fold 0 e' esattamente il
sottoinsieme 3 T di quella di 502, 103 casi su 112. Training identico, 1000 epoche, `3d_fullres`, fold 0.

Il confronto e' stato fatto con `scripts/confronto_appaiato.py`, sui **casi in comune** e non fra le due
medie complessive. Poi, per rispondere al rovescio della domanda — un modello addestrato solo su 3 T sa
ancora segmentare una 1.5 T? — `scripts/inferenza_incrociata_15T.py` ha applicato 521 ai 45 casi a 1.5 T
mai visti, senza toccare il test set. La verifica che nessuno dei 45 comparisse nel training di 521 e'
parte dello script: se ne fosse passato uno, l'intera misura sarebbe stata priva di senso.

Esito: **+0.0093 con p = 0.67** sui casi appaiati, e **0.6721** in inferenza incrociata, cioe' nessun
crollo fuori dominio. Numeri completi, limiti della sottoanalisi sui 9 casi puliti e conclusione in
[modelli-ed-esperimenti.md](modelli-ed-esperimenti.md#521--atlas-senza-i-dati-a-15-t-escluderli-non-serve).
Report: `reports/test_esclusione_1_5T.md` e `reports/inferenza_incrociata_15T.md`.

**Conclusione: l'esclusione non e' supportata.** L'ipotesi dei relatori era che i dati a 1.5 T fossero
troppo diversi per essere utili; la misura dice che togliere **45 casi** dal training (52 contando anche
il test) costa dati e non compra accuratezza. Restano dentro.

---

## 16. Skull stripping: validazione e costruzione dei dataset

Tre passaggi, in ordine, il 28 agosto.

**a. SynthStrip regge sulle anatomie alterate?** `scripts/run_synthstrip.py` lo applica a tutti i 244
casi DBB e confronta con le maschere di riferimento. Dice medio **0.9657**. Per gruppo: sani 0.9673,
ACC 0.9675, MCD 0.9605, PFM 0.9604, idrocefalo **0.9267** (minimo assoluto 0.8791). Sui cervelli
malformati si perde **0.010** rispetto ai sani: e' la risposta che serviva, ed e' il contrario di cio'
che farebbe BET. Nei casi con ventricoli dilatati il volume escluso e' pero' quasi tutto CSF (in
`sub-0955`, 400 ml su 423). Report: `reports/synthstrip_dbb.md`, 244 figure di controllo in
`reports/figures/synthstrip/` col Dice come prefisso del nome.

**b. Quante lesioni si perderebbero ritagliando?** `scripts/lesioni_fuori_maschera.py`. Su WMH la
perdita e' **zero esatto** in tutti e 79 i casi; su ATLAS (tutti i 556) la media e' **0.0012** con un
massimo di **0.0649**: 18 casi oltre l'1% e tre oltre il 5%. Una prima misura su 100 casi campionati
dava 0.0007 e nessun caso oltre il 5% — la coda si vede solo sul dataset intero, quindi la misura va
sempre fatta su tutto. I casi peggiori sono infarti massivi che arrivano alla superficie corticale, e i voxel persi
stanno a 1-2 mm mediani dal bordo della maschera: rifilatura di superficie, non asportazione. Dilatare
la maschera di 2 mm dimezzerebbe la perdita ma rimetterebbe dentro ~160 ml di guscio non cerebrale per
caso, quindi **non si dilata**. Report: `reports/lesioni_fuori_maschera.md`.

**c. Costruzione dei dataset.** `scripts/make_skullstripped_datasets.py` produce `Dataset522_WMH_SS` e
`Dataset523_ATLAS_SS`: i voxel fuori dalla maschera sono azzerati, **geometria, affine e label restano
identiche**, e `splits_final.json` e' copiato dal dataset di partenza. Verificato su otto casi a
campione: geometria invariata e voxel di lesione conservati (sette casi al 100%, uno al 98.4%).
Frazione di volume tenuta: **0.161** su WMH, **0.240** su ATLAS — la differenza e' il campo di vista,
non il cervello: le immagini WMH includono collo e spalle.

DBB **non** e' incluso di proposito: li' il compito e' segmentare i tessuti e l'unione delle sei classi
coincide con la maschera cerebrale (Dice 0.9999), quindi azzerare fuori dalla maschera regalerebbe alla
rete l'estensione del foreground.

Nessun resampling: la spacing di destinazione si sceglie al planning con
`nnUNetv2_plan_experiment -overwrite_target_spacing`, quindi "tutto a 1 mm" e' un'opzione di planning e
non un secondo dataset da 3 GB.

**Cosa ci si aspetta.** nnU-Net accende `use_mask_for_norm` da solo quando il ritaglio al non-zero
scende sotto i 3/4 del volume. Oggi e' `False` su 501, 502 e 503 perche' sono teste intere; su 522 e 523
si accendera', e lo z-score verra' calcolato **solo sui voxel cerebrali** invece che su una
distribuzione che include grasso sottocutaneo, diploe e aria. E' l'ipotesi che il confronto 522/501 e
523/502 mette alla prova.

**Note operative costate tempo.** SynthStrip su CPU passa da 6 s a oltre 100 s per caso quando la
macchina e' contesa; su GPU costa 10 s di cui 9 di solo avvio del processo, quindi conviene tenerne
6 in parallelo (0.27 casi/s). E il progresso di un training va letto in
`nnUNet_results/<dataset>/*/fold_0/training_log_*.txt`, non nello stdout rediretto in `reports/logs/`,
che Python bufferizza a blocchi e puo' restare indietro di mezz'ora.

---

## 17. Test 3 — aggiungere il T1 alla FLAIR peggiora

`scripts/make_wmh_two_channel_dataset.py` costruisce `Dataset512_WMH_T1_FLAIR` mettendo la FLAIR di 501
come canale 0 e il T1 di 511 come canale 1. Lo script non si fida della co-registrazione dichiarata dal
dataset: verifica caso per caso che forma e affine coincidano, e si ferma se non lo fanno. Training fold 0
`3d_fullres` avviato il 29 agosto alle 18:06, concluso il 30 alle 00:58.

Esito: **0.7535 contro 0.7739**, peggiore in 11 casi su 16, Wilcoxon **p = 0.025**. Il confronto e' pulito
perche' 512 eredita da 501 la stessa patch e lo stesso batch, quindi l'unica variabile e' il canale in
piu'. Lettura completa e conseguenze per il modello unificato in
[modelli-ed-esperimenti.md](modelli-ed-esperimenti.md#512--wmh-con-flair-e-t1-insieme-no-non-aiuta).
Report: `reports/confronto_512_due_canali.md`.

## 18. Variante D: costruzione del dataset unificato

```bash
python scripts/make_union_dataset.py --verifica    # controlli, non scrive
python scripts/make_union_dataset.py               # -> Dataset531_UNION_D
nnUNetv2_extract_fingerprint -d 531 --verify_dataset_integrity -np 12
nnUNetv2_plan_experiment -d 531
nnUNetv2_preprocess -d 531 -c 3d_fullres -np 12
python scripts/bilancio_classi_union.py            # -> reports/bilancio_classi_531.md
```

`Dataset531_UNION_D` unisce WMH (col T1), ATLAS e DBB: **842 casi di sviluppo e 156 di test**, un canale,
sette classi. Immagini e label di lesione sono symlink ai sorgenti; solo le 244 label di DBB sono
riscritte, rimappate da `1..6` a `2..7`. I nomi sono prefissati `wmh_`, `atlas_`, `dbb_`, e i fold sono
l'unione dei fold k dei tre sorgenti: fold 0 = 691 train, 151 validation.

Verifiche fatte prima di preprocessare: rimappatura DBB confrontata voxel per voxel con l'atteso su 4
casi campionati, label di WMH e ATLAS ancora `0/1`, geometria immagine/label coerente, unicita' dei nomi,
nessun caso presente sia in train sia in val, e `--verify_dataset_integrity` sull'intero dataset.

**Il planner ha scelto da solo `1x1x1` mm e patch `128x128x128` batch 2, identici a 502**: non e' servito
forzare la risoluzione, e il confronto fra D e la baseline ATLAS resta pulito. 19 GB di preprocessing.

Lo sbilanciamento fra le classi e' stato misurato invece che stimato: un caso DBB porta **266 volte** il
foreground di un caso lesionale, e la lesione e' il **5.87%** del foreground totale. Conseguenze sulla
lettura del risultato in [modelli-ed-esperimenti.md](modelli-ed-esperimenti.md).

Training non ancora avviato: la GPU 0 e' occupata da 522 e 523.

## 19. Test 2 — lo skull stripping non cambia il risultato

```bash
setsid nohup bash scripts/run_baselines.sh 522 523 > reports/logs/coda_522_523.log 2>&1 < /dev/null &
python scripts/confronto_appaiato.py Dataset501_WMH  Dataset522_WMH_SS   --out reports/confronto_522_skullstrip.md
python scripts/confronto_appaiato.py Dataset502_ATLAS Dataset523_ATLAS_SS --out reports/confronto_523_skullstrip.md
```

522 dal 30 agosto 12:34 alle 19:00, 523 dalle 19:00 alle 02:44 del 31, in sequenza sulla GPU 0.

Esito: **+0.0045 con p = 0.86 su WMH** e **-0.0037 con p = 0.58 su ATLAS**. Nessun effetto in nessuna
delle due direzioni. Su ATLAS piu' casi migliorano che peggiorano (59 contro 46) ma pochi casi crollano e
decidono il segno; verificato che **non e' colpa della maschera**, perche' quei casi perdono fra lo 0% e
l'1.7% dei voxel di lesione e i tre casi che ne perdono davvero non sono fra i peggiorati.

Lettura completa e conseguenze in
[modelli-ed-esperimenti.md](modelli-ed-esperimenti.md#522-e-523--skull-stripping-nessun-effetto).

---

## 20. Variante D: training avviato

```bash
CUDA_VISIBLE_DEVICES=0 setsid nohup bash scripts/run_baselines.sh 531 > reports/logs/coda_531.log 2>&1 < /dev/null &
python scripts/analisi_union_d.py       # -> reports/risultato_531.md
python scripts/plot_predictions.py 531  # -> reports/figures/predizioni_531_fold_0.png
```

Avviato il 31 agosto alle 16:34, concluso alle 23:40. Verificato all'avvio che nnU-Net legga il
**nostro** `splits_final.json` e non ne generi uno proprio: il log dice "This split has 691 training and
151 validation cases", che sono i numeri attesi.

`scripts/analisi_union_d.py` esiste perche' il `summary.json` di D non e' leggibile come quello degli
altri: riporta una media per classe calcolata su tutti i 151 casi, ma le classi non esistono ovunque —
la lesione solo nei casi WMH e ATLAS, i tessuti solo in quelli DBB. Lo script misura ogni classe **sui
soli casi che possono contenerla** e la confronta con la propria baseline sugli stessi casi.

Esito: la lesione tiene (**-0.0072** su ATLAS con p=0.073, **-0.0225** su WMH con p=0.10, nessuna delle
due significativa), ma D **perde su tutte e sette le classi**, e sui tessuti la perdita e' piccola e
perfettamente sistematica (23 casi su 23 in cinque classi su sei). Lettura completa in
[modelli-ed-esperimenti.md](modelli-ed-esperimenti.md#531--variante-d-tutti-e-tre-i-dataset-in-un-modello-solo).

## 21. Verifiche richieste dai chiarimenti del 5 settembre

```bash
python scripts/analisi_curve_training.py    # -> reports/curve_training.md + figures/curve_training.png
python scripts/metriche_per_lesione.py 502 501 511 512   # -> reports/metriche_per_lesione.md
setsid nohup bash scripts/run_val_best.sh > reports/logs/coda_val_best.log 2>&1 < /dev/null &
```

**a. nnU-Net fa early stopping?** No, verificato nel sorgente: nel `nnUNetTrainer` non compaiono `early`,
`patience` ne' `stop`. Salva pero' `checkpoint_best.pth` sulla pseudo-dice EMA, mentre la validation
finale usa `checkpoint_final.pth`: il meccanismo per fermarsi al punto migliore esiste, non lo usiamo.

**b. Curve di training.** `analisi_curve_training.py` estrae dai log le tre serie per epoca e calcola
epoca del minimo, risalita dopo il minimo e divario finale fra dati visti e non visti. Il log va letto
concatenando **tutti** i file in ordine di data: un training ripreso con `--c` ne apre uno nuovo, e su
501 leggendo solo l'ultimo mancherebbero le prime 150 epoche.

Esito: overfitting modesto sui dataset grandi (su 502, 503 e 531 il minimo cade oltre il 90% delle
epoche), evidente su 501 (minimo all'epoca 347 su 1039), **massiccio sui 10 casi di 541** con divario
+0.534, che scende monotonicamente a +0.398, +0.275, +0.179 al ridursi del passo di apprendimento.

**c. Metriche di rilevamento per lesione.** Erano nel piano da settimane (`docs/training.md`, passo 7) e
non erano mai state implementate. Etichettano le componenti connesse di riferimento e predizione con
connettivita' a 26 vicini; una lesione conta come rilevata se almeno un voxel della predizione ci cade
dentro. Risultato che cambia la lettura di tutto il progetto:

| ingresso | minuscole (<=10 mm³) | piccole (<=50) | medie (<=200) | grandi | totale |
| --- | --- | --- | --- | --- | --- |
| WMH, FLAIR | 45.2% | 75.4% | 95.4% | 98.7% | 72.8% |
| WMH, FLAIR+T1 | 43.4% | 72.0% | 97.7% | 100% | 71.1% |
| WMH, solo T1 | **11.9%** | 31.2% | 73.3% | 93.4% | **38.2%** |
| ATLAS, solo T1 | **8.7%** | 5.9% | 44.4% | 82.1% | 64.3% |

Le fasce sono in **mm³** dal 13 settembre (prima erano in voxel: su ATLAS coincidono, su WMH un voxel
vale 3.5 mm³ e le fasce vecchie erano tre volte e mezzo più larghe). I totali non cambiano; le righe per
fascia sì, e la conclusione qualitativa resta identica.

**Corregge una nostra conclusione precedente**: avevamo riportato che il modello a due canali peggiora,
sulla base del Dice (0.753 contro 0.774). Sul *rilevamento* i due canali sono equivalenti alla sola FLAIR
(71.1% contro 72.8%). Non perde lesioni, perde qualche punto di contorno.

**d. Rivalutazione con `checkpoint_best`** (conclusa l'8 settembre, letta solo il 13). `--val_best`
riscrive la cartella `validation`, quindi i risultati originali stanno in `validation_final`. Da zero
0.4507 → 0.4491; fine-tuning a passo `1e-2` 0.5642 → **0.6763**; `1e-3` 0.6419 → 0.6624; `1e-4`
0.6851 → 0.6873. Corregge la lettura del 7 settembre, fatta sul solo braccio da zero: il criterio
di nnU-Net **vede** l'overfitting del fine-tuning e recupera undici punti dove il passo e' sbagliato.
Dettaglio in [modelli-ed-esperimenti.md](modelli-ed-esperimenti.md).

## 22. Primo lotto AInCP e preparazione del training (11-13 settembre)

```bash
python scripts/costruisci_notebook_aincp.py && jupyter nbconvert --execute --inplace notebooks/aincp_primo_lotto.ipynb
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=24 python scripts/preprocess_aincp.py     # ~2 min/paziente
python scripts/figura_registrazione_aincp.py                                  # -> reports/aincp_scala_mni.json
python scripts/make_aincp_dataset.py && nnUNetv2_extract_fingerprint -d 504
nnUNetv2_move_plans_between_datasets -s 502 -t 504 -sp nnUNetPlans -tp nnUNetPlansDa502   # + 3 correzioni a mano
nnUNetv2_preprocess -d 504 -plans_name nnUNetPlansDa502 -c 3d_fullres
python scripts/make_aincp_dataset.py --solo-split
```

**a. I dati.** Dieci pazienti, T1 e FLAIR originali non allineate, ROI sulla FLAIR; `COREGISTERED`
e' un nome fuorviante. Bersaglio = ROI_1 ∪ ROI_2 ∪ ROI_4, ventricolo (ROI_3) escluso. Le misure
stanno nel notebook e in [obiettivo-e-piano.md](obiettivo-e-piano.md): mediana 7.2 ml, 65 componenti
connesse (26 vicini) di cui 42 sotto i 10 mm³ per lo 0.03% del volume, legenda confermata dai
rapporti d'intensita'.

**b. Preprocessing** (`preprocess_aincp.py`, ANTsPy): N4 → FLAIR→T1 rigida → T1→MNI SyN → maschera
binaria. Due misure: FLAIR→T1 frammenta le lesioni sottili (fette 0.8 → 1.0 mm), MNI le **ingrandisce**
(+51% mediano) perche' il template e' adulto — il determinante dell'affine lo prevede entro 4 punti.
Decisione: si addestra in spazio T1 nativo.

**c. Dataset504 e piano.** Il piano di 502 trasferito su 504 richiede tre correzioni a mano per il
secondo canale (documentate in `make_aincp_dataset.py`). Il ricampionamento a 1 mm conserva il
volume ma spezza le componenti (≥10 mm³: da 29 a 48): le metriche si calcolano sulla griglia originale.

**d. Le componenti piccole non sono rumore** (14 settembre, `scripts/figura_lesioni_piccole.py`).
Due correzioni in una. Primo, il conteggio: il notebook usava la connettivita' a **6 vicini** (il
default di scipy) e `metriche_per_lesione.py` quella a **26**; le stesse maschere davano 158
componenti contro 65. Il volume e' identico, cambia quanti pezzi si conta, e con 6 vicini una
lesione sottile e obliqua si spezza in frammenti che sembrano lesioni minuscole distinte. Ora si usa
26 ovunque. Secondo, la sostanza: misurando l'intensita' FLAIR di ogni componente rispetto al tessuto
sano, quelle di gliosi **anche da un solo voxel** stanno fra 2.3 e 4.4 volte, cioe' hanno il segnale
di una lesione vera. Solo 3 componenti su 65 cadono dove l'immagine ha il livello dell'aria, fuori
dal cervello. **Corregge quanto scritto il 13 settembre** («scorie del disegno a mano»): quella
lettura veniva dal conteggio, non dall'immagine. Decisione: **nessuna soglia**, tutto cio' che e'
annotato conta come lesione, in attesa del parere del clinico sulle due figure.

**e. Tre strumenti per il training vero** (13 settembre):
- `metriche_per_lesione.py`: fasce in **mm³**, `--soglia-mm3` simmetrica, `--fold all`, `--trainer`,
  JSON per caso in `reports/lesioni_per_caso/`. Il report per 501/502/511/512 e' stato rigenerato:
  cambiano le righe per fascia su WMH, non i totali.
- `nnUNetTrainerWandb.pazienza_val`: arresto sulla perdita di validazione, `checkpoint_val_best`,
  `checkpoint_final` = epoca migliore, ultima epoca in `checkpoint_ultima_epoca`. Tre classi con
  pazienza 50: `nnUNetTrainerWandb_250ep_arresto50` (T0) e le due `EspansioneCanali_*_250ep_lr1e4_arresto50`.
  Provato a CPU con epoche minuscole.
- `confronto_fold.py`: unisce i fold in una predizione fuori-fold per paziente e confronta due
  trainer appaiati in Dice e rilevamento. Provato su 541 (`lr1e3` contro `lr1e4`).

## 23. Primo training sui pazienti (14-15 settembre)

```bash
MAX_GPU=2 SOGLIA_MEM=60000 SOGLIA_UTIL=101 setsid nohup bash scripts/attendi_gpu_e_lancia.sh > reports/logs/attesa_gpu.log 2>&1 < /dev/null &
python scripts/metriche_per_lesione.py 504 --trainer <nome>__ --fold all --soglia-mm3 0
python scripts/confronto_fold.py 504 nnUNetTrainerWandb_250ep_arresto50 nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50 --soglia-mm3 0
python scripts/figura_predizioni_504.py
```

Lanciato il 13 sera e ritentato a vuoto per 17 ore (`nnUNetv2_train` non era nel PATH dello script
staccato: corretto in `env.sh`, vedi commit e2e84e8). Rilanciato il 14 alle 11:44 su due GPU
condivise con il job di un altro utente, per decisione dell'utente. Quindici training in 20 ore;
l'arresto sulla perdita di validazione si e' attivato in tutti, fra l'epoca 26 e la 199.

**Esito**: Dice medio fuori-fold 0.581 da zero, 0.555 Proposta B `zero`, 0.596 `duplica`; nessuna
differenza appaiata. Diagnosi sui pesi: con passo `1e-4` il canale FLAIR dell'espansione `zero` e'
rimasto a norma 0.05 di quella del T1 — il modello e' ancora il 502 a solo T1. Il passo misurato su
541 non trasferisce a un canale e un dominio nuovi. Dettaglio e prossimi esperimenti in
[modelli-ed-esperimenti.md](modelli-ed-esperimenti.md), sezione 504.

---

## 24. Il passo differenziato, e un difetto nel freno (16 e 22 settembre)

```bash
# i cinque fold sono girati da soli, accodati da attendi_gpu_e_lancia.sh fra il 15 sera e il 16
python scripts/confronto_fold.py 504 nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50 \
    nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_arresto50 --soglia-mm3 0
```

**Esito**: Dice medio fuori-fold **0.621**, il migliore delle quattro varianti (0.581 da zero,
0.555 `zero` a passo uniforme, 0.596 `duplica`), con 7 coppie su 10 in miglioramento rispetto a
`zero`. La norma del filtro FLAIR passa da 0.04–0.08 volte quella del T1 a 0.65–2.7, cioe' nello
stesso regime di un modello addestrato da zero: **la diagnosi del primo giro era giusta**, il
canale nuovo non imparava perche' il passo era troppo piccolo, non perche' mancassero i dati.

**Difetto trovato il 22 settembre.** Il freno sui canali pre-addestrati scalava il gradiente
dentro un hook, ma `torch.optim.SGD` aggiunge il decadimento dei pesi *dopo* l'hook e lo moltiplica
per il passo del gruppo: il canale T1 ha ricevuto un decadimento cento volte piu' forte del suo
passo. La norma del suo filtro e' scesa da 8.36 a 3.50–7.49 in proporzione alla durata del fold,
con coseno 0.996–0.9998 rispetto alla partenza — direzione intatta, solo scala ridotta: decadimento
puro, non apprendimento. I run del 16 settembre restano validi come misura, ma **non sono a una
variabile sola**: il T1 e' stato insieme frenato e rimpicciolito. Corretto mettendo il peso del
primo strato in un gruppo con `weight_decay=0` e applicando il decadimento dentro il freno.

---

## 25. Il terzo giro e l'analisi per fenotipo (22 settembre)

```bash
MAX_GPU=2 setsid nohup bash scripts/attendi_gpu_e_lancia.sh > reports/logs/attesa_gpu.log 2>&1 < /dev/null &
python scripts/analisi_rilevamento_504.py        # ~6 min a CPU
```

In coda su due GPU libere: il passo differenziato con il freno corretto e il controllo a passo
`1e-3` uniforme, cinque fold ciascuno. Verificato al primo minuto che il run reale registra il
decadimento dentro il freno e produce l'epoca 1.

**Proposta respinta**: un modello solo-FLAIR, suggerito perché su WMH la FLAIR da sola batte
FLAIR+T1. Il progetto usa i due canali perché vedono fenotipi diversi, e il valore di ciascuno si
misura per fenotipo — cosa che l'analisi di oggi fa senza addestrare nulla.

**Esito dell'analisi**: il pre-addestramento su ATLAS porta l'encefalomalacia da 0.40 a 0.78–0.81 di
copertura (4 pazienti su 4); la FLAIR attiva migliora la gliosi (7 su 10), dove però il modello da
zero resta il migliore; i falsi positivi calano con il pre-addestramento (8 pazienti su 10,
p = 0.033); Dice e numero di lesioni rilevate non distinguono le varianti. 27 lesioni su 71
spariscono dal bersaglio quando nnU-Net lo ricampiona a 1 mm. Stadio 2 su WMH promosso a prossimo
esperimento. Trovato in `/data01/fsm_mri_data/` un undicesimo paziente (FSM_028) preparato da un
altro utente: non usato (escluso definitivamente il 24/9, provenienza ignota). Dettaglio in
[obiettivo-e-piano.md](obiettivo-e-piano.md) e [modelli-ed-esperimenti.md](modelli-ed-esperimenti.md).

---

## 26. Il terzo giro letto per fenotipo (23 settembre)

```bash
python scripts/analisi_rilevamento_504.py        # ora sulle sei varianti, ~8 min
```

Dieci addestramenti conclusi nella notte, codice 0. Il freno corretto tiene il filtro T1 fermo
(norma 8.37–8.43) e non cambia il risultato (Dice 0.615 contro 0.621): il guadagno del 16
settembre veniva dalla FLAIR. Passo differenziato e passo 1e-3 uniforme sono indistinguibili nel Dice
e nella copertura per paziente; il differenziato è l'unico con evidenza significativa che la FLAIR
porti la gliosi (9 su 10, p = 0.006) e ha meno falsi positivi (10 su 10, p = 0.002): è la ricetta
scelta. Corretta l'affermazione del 22 settembre sulla gliosi (media pesata, non paziente per
paziente) e ridimensionato il rapporto fra le norme dei filtri come misura dell'uso della FLAIR.

---

## 27. Pulizia del disco: 110 GB di dati ricreabili (23 settembre)

`/home` era al 99% (64 GB liberi); dopo la pulizia 167 GB. Tolto **solo ciò che si ricrea da ciò
che resta**, dopo aver verificato che la sorgente esistesse:

| cosa | dove | GB circa | come si ricrea |
| --- | --- | --- | --- |
| mappe di probabilità `.npz` e i loro `.pkl` (845 casi), prodotte da `--npz` | `validation*/` di tutti i dataset **tranne 504** | 45 | prima si rigenerano i volumi preprocessati (riga sotto), poi `nnUNetv2_train <ID> 3d_fullres <fold> -tr <trainer> -p <piano> --val --npz`: rifà la sola validazione, stesse predizioni |
| volumi preprocessati (`nnUNetPlans_3d_fullres/`, `_2d/`, `gt_segmentations/`) | `nnUNet_preprocessed/` di 502, 503, 511, 521, 522, 523, 531, 541, 599 | 58 | `nnUNetv2_preprocess -d <ID> -plans_name nnUNetPlans -c 3d_fullres` (per 541 anche `nnUNetPlansFT`): usa i piani salvati, quindi dà lo stesso risultato |
| `checkpoint_val_best.pth` (30 file) | `Dataset504_AINCP/*/fold_*/` | 7.5 | identici a `checkpoint_final.pth` nei pesi, verificato su tutti e 30: `cp checkpoint_final.pth checkpoint_val_best.pth` |

**Restano intatti**: tutti i `.json` delle cartelle preprocessate, e in particolare
`splits_final.json`, che non si rigenera mai; tutti i checkpoint dei modelli; le predizioni
`.nii.gz` e i `summary.json` di validazione, che sono ciò che leggono gli script di analisi; i
preprocessati di 501, 504 e 512 (piccoli, e 504 e 512 servono allo stadio 2); i `.npz` di 504;
`data/` per intero, compreso l'archivio `dataset_FSM_v1.7z`, unica copia sulla macchina dei dati
AInCP originali oltre alla cartella estratta.

Le sorgenti dei dati grezzi: 501, 502, 511, 512, 521, 531, 541 e 599 sono collegamenti a `/data01`
(sola lettura); 503 punta a `data/raw/DBB`; 522 e 523 stanno in `data/processed/nnUNet_raw`.
Unico script che non gira più senza rigenerare: `inspect_failures.py`, che legge i `.npz` di 502
(il suo report `reports/failure_analysis_502.md` resta). Nota per i training futuri: il trainer
salva ancora `checkpoint_val_best.pth`, che a fine addestramento è una copia di `checkpoint_final`.

---

## 28. Dopo l'incontro del 23 settembre: MNI, soglia, stadio 2

```bash
python scripts/confronto_template_aincp.py         # template adulto contro pediatrico, ~20 min
python scripts/elenco_residui_annotazione.py       # componenti <= 1 mm3, per il clinico
python scripts/make_wmh_mni_dataset.py --processi 12 && nnUNetv2_extract_fingerprint -d 542 -np 16
python scripts/make_wmh_mni_dataset.py --piano && nnUNetv2_preprocess -d 542 -plans_name nnUNetPlansDa502 -c 3d_fullres -np 16
CUDA_VISIBLE_DEVICES=1 setsid nohup bash scripts/run_stadio2_542.sh <trainer> > reports/logs/stadio2_<breve>.log 2>&1 < /dev/null &
```

Decisioni dell'incontro: standardizzazione in MNI con ritorno al nativo alla fine; componenti fino a
1 mm³ tolte; 17 pazienti a 1.5 T in arrivo e nessun altro; split 70/30 stratificato per centro;
Proposta B confermata. Misurato che la proporzione di cervello lesionato si conserva in MNI (mediana
−10%) e che il template pediatrico MNIPediatricAsym allinea meglio (Dice 0.95–0.98 contro 0.82–0.93)
senza togliere l'ingrandimento. Le maschere del cervello (SynthStrip) sono salvate come
`data/processed/AINCP/<soggetto>/cervello_inT1.nii.gz` e `cervello_FLAIR_originale.nii.gz`, le
trasformazioni verso il template pediatrico in `transforms_ped/`, i template in `data/templates/`.
Aggiunto il modality dropout al trainer di base. Costruito `Dataset542_WMH_MNI` e lanciato lo
stadio 2 in due varianti. Metriche di 504 ricalcolate con la soglia: 47 lesioni.

## 29. Oblio dello stadio 2, training congiunto, pazienti nel template (23 settembre, sera)

```bash
python scripts/oblio_atlas_stadio2.py --gpu 1               # 542 su ATLAS con FLAIR vuota, ~30 min
python scripts/make_union_atlas_wmh_mni.py && nnUNetv2_extract_fingerprint -d 543 -np 12
python scripts/make_union_atlas_wmh_mni.py --piano && nnUNetv2_preprocess -d 543 -plans_name nnUNetPlansDa502 -c 3d_fullres -np 12
DS=543 CUDA_VISIBLE_DEVICES=1 setsid nohup bash scripts/run_stadio2_542.sh nnUNetTrainerEspansioneCanali_zero_500ep_lr1e3_dropmod30_arresto50 > reports/logs/stadio2_543_lr1e3.log 2>&1 < /dev/null &
python scripts/make_aincp_mni_dataset.py && nnUNetv2_extract_fingerprint -d 505 -np 8
python scripts/make_aincp_mni_dataset.py --piano && nnUNetv2_preprocess -d 505 -plans_name nnUNetPlansDa502 -c 3d_fullres -np 8
CUDA_VISIBLE_DEVICES=2 setsid nohup bash scripts/run_aincp_505.sh A > reports/logs/coda_505_A.log 2>&1 < /dev/null &
python scripts/porta_505_nativo.py                          # a fold finiti: predizioni nel nativo
```

Lo stadio 2 in sequenza impara WMH (0.735) ma dimentica ATLAS (0.610 → 0.217 con la FLAIR vuota),
in entrambe le varianti: sostituito da un training congiunto ATLAS + WMH (543). I dieci pazienti sono
stati portati nel template pediatrico (505) riusando le trasformazioni di `transforms_ped/`; i
centri e le età sono stati letti dal foglio Excel in `docs/`. Aggiunti i trainer `..._da542`.


## 30. Pronti per i pazienti nuovi: pipeline, postprocessing, valutazione (24 settembre)

Scritti `scripts/pipeline_aincp.py` (dai file ricevuti al dataset, due formati di ingresso, FLAIR
facoltativa, centro e campo dal foglio Excel, split 70/30 per strato), `scripts/postprocessing_aincp.py`
(scelta di soglia, volume minimo e maschera del cervello sui fuori-fold; valutazione nel nativo con
volume e percentuale di cervello lesionato), `scripts/run_stadio3.sh` e `scripts/predici_test.sh`.
Collaudati sui dieci pazienti del primo lotto: le immagini nel template coincidono con quelle di 505, le etichette a meno
di 1–5 voxel di arrotondamento. FSM_028, usato per provare il secondo formato, è poi stato escluso
(provenienza ignota): la pipeline legge solo i lotti AInCP in `data/raw/AINCP/`. D su 505 finito; due varianti di 543
lanciate. La procedura è in `obiettivo-e-piano.md`.

## 31. Il secondo lotto AInCP e il dataset finale (1 ottobre)

```bash
curl -L "<link FileSender>/download.php?token=...&files_ids=1113201" -o data/raw/AINCP/dataset_FSM_v2.zip
cd data/raw/AINCP && unzip -q dataset_FSM_v2.zip && cd -
python scripts/pipeline_aincp.py prepara --casi FSM_001 FSM_028 FSM_030 FSM_033 FSM_035 FSM_036 FSM_049
python scripts/elenco_residui_annotazione.py                    # ora su entrambi i lotti
python scripts/pipeline_aincp.py dataset --id 506 && nnUNetv2_extract_fingerprint -d 506 -np 8
python scripts/pipeline_aincp.py piano --id 506 && nnUNetv2_preprocess -d 506 -plans_name nnUNetPlansDa502 -c 3d_fullres -np 8
# 15 run (A, D, Z x 5 fold) sulle 4 GPU condivise, SOGLIA_MEM=60000: vedi reports/logs/s3_506_gpu*.log
```

Arrivati 7 pazienti (totale 17, non 27), tutti a 1.5 T, nello stesso formato del primo lotto. Foglio
aggiornato `docs/AINCP_SERIES_TYPE_dataset_FSM_v2.xlsx`, ora letto dalla pipeline. Controllo a vista
dei 7 nel template: allineamento corretto. In FSM_030 e FSM_033 la lesione cresce nel template più
del cervello (2.7 contro 1.9 e 1.5 volte): la registrazione non lineare deforma di più la zona
periventricolare; le misure si fanno nel nativo. Split 12/5 per strato con quote a resti maggiori
(arrotondare per strato dava 6 di test). Aggiunto il riferimento da zero con dropout
(`nnUNetTrainerWandb_250ep_dropmod30_arresto50`). Elenco dei residui per il clinico: 46 componenti,
12 fuori dal cervello.

---

## Dataset nnU-Net attualmente disponibili

| Dataset | Canale | Classi | Sviluppo | Test | Prodotto da |
| --- | --- | --- | --- | --- | --- |
| `Dataset501_WMH` | FLAIR | lesione | 79 | 20 | `notebooks/dataset_preparation.ipynb` |
| `Dataset502_ATLAS` | T1 | lesione | 556 | 99 | `notebooks/dataset_preparation.ipynb` |
| `Dataset503_DBB` | T1 | 6 tessuti | 207 | 37 | `scripts/convert_dbb_to_nnunet.py` |
| `Dataset511_WMH_T1` | T1 | lesione | 79 | 20 | `scripts/make_wmh_t1_dataset.py` |
| `Dataset512_WMH_T1_FLAIR` | FLAIR + T1 | lesione | 79 | 20 | `scripts/make_wmh_two_channel_dataset.py` |
| `Dataset521_ATLAS_3T` | T1 | lesione | 511 | 92 | `scripts/make_atlas_3t_dataset.py` |
| `Dataset522_WMH_SS` | FLAIR skull-stripped | lesione | 79 | 20 | `scripts/make_skullstripped_datasets.py` |
| `Dataset523_ATLAS_SS` | T1 skull-stripped | lesione | 556 | 99 | `scripts/make_skullstripped_datasets.py` |
| `Dataset531_UNION_D` | T1 | lesione + 6 tessuti | 842 | 156 | `scripts/make_union_dataset.py` (abbandonato il 7/09) |
| `Dataset541_SIM_AINCP` | T1 | lesione | 45 (10 train / 35 val) | — | `scripts/make_finetuning_sim_dataset.py` |
| `Dataset504_AINCP` | FLAIR + T1, spazio T1 nativo | lesione | 10 (5 fold) | — (i lotti futuri) | `scripts/make_aincp_dataset.py` |
| `Dataset505_AINCP_MNI` | FLAIR + T1, template pediatrico | lesione | 10 (split di 504) | — | `scripts/make_aincp_mni_dataset.py` |
| `Dataset542_WMH_MNI` | FLAIR + T1, MNI adulto | lesione | 79 (split di 512) | — | `scripts/make_wmh_mni_dataset.py` |
| `Dataset506_AINCP_FINALE` | FLAIR + T1, template pediatrico | lesione | 12 (5 fold) | **5** (`imagesTs`, mai guardati) | `scripts/pipeline_aincp.py` |
| `Dataset543_ATLAS_WMH_MNI` | FLAIR (vuota per ATLAS) + T1, MNI adulto | lesione | 635 (unione dei fold) | — | `scripts/make_union_atlas_wmh_mni.py` |
| `Dataset599_SMOKE_ATLAS` | T1 | lesione | 16 | — | `scripts/make_smoke_dataset.py` (usa e getta) |

501, 502 e 503 sono preprocessati in `2d` e `3d_fullres` (passo 10); 512, 522, 523 e 531 in `3d_fullres`.

Un dettaglio che cambia come vanno letti i confronti: **512 eredita da 501 lo stesso patch
[64, 224, 128] e lo stesso batch 2**, quindi 512 contro 501 isola davvero la sola aggiunta del canale
T1. Su 522 e 523 invece nnU-Net ha ricalcolato il piano, perche' dopo il ritaglio i volumi sono piu'
piccoli (su WMH il patch passa a [48, 192, 128] con batch 3, e arriva a coprire il 100% delle fette
invece dell'81%). Quei due confronti misurano quindi "la pipeline nnU-Net con e senza skull stripping",
che e' la domanda pratica giusta, ma non una variabile sola.

Stato dei training al 30 agosto: 501, 502, 503, 511, 521 e 512 conclusi; **522 in corso** (avviato alle
12:34 sulla GPU 0) e **523 in coda** dietro di lui. Come si lanciano e dove si legge il progresso e'
in [training.md](training.md).
