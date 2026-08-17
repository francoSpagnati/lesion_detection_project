# Training

Prerequisito: i dataset devono essere già preparati. Stato atteso sotto `nnUNet/nnUNet_raw/`:

| Dataset | Canale | Sviluppo | Test | Prodotto da |
| --- | --- | --- | --- | --- |
| `Dataset501_WMH` | FLAIR | 79 | 20 | `notebooks/dataset_preparation.ipynb` |
| `Dataset502_ATLAS` | T1 | 556 | 99 | `notebooks/dataset_preparation.ipynb` |
| `Dataset503_DBB` | T1 | 207 | 37 | `scripts/convert_dbb_to_nnunet.py` |
| `Dataset511_WMH_T1` | T1 | 79 | 20 | `scripts/make_wmh_t1_dataset.py` |

Quali di questi conviene addestrare, in che ordine e quali possono condividere pesi è in
[obiettivo-e-piano.md](obiettivo-e-piano.md).

## Step 0 - Scegliere la GPU

La GPU è sbloccata: `torch 2.6.0+cu124`, allineato al driver della macchina, verificato su A100. La
macchina ha 4× A100 80GB ed è **condivisa con altri utenti**: quale scheda sia libera cambia di giorno
in giorno, quindi va controllato prima di ogni lancio.

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=0   # la scheda risultata libera
```

## Step 1 - Variabili d'ambiente

```bash
source scripts/env.sh
```

Imposta `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results` e `nnUNet_extTrainer`. Quest'ultima è
quella che permette di tenere i trainer del progetto in `scripts/nnunet_trainers/` invece che dentro
`site-packages`: senza, `-tr nnUNetTrainerWandb` non viene trovato.

## Step 2 - Planning e preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity   # WMH (FLAIR)
nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity   # ATLAS (T1)
nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity   # DBB (T1, 6 tessuti)
```

`--verify_dataset_integrity` va usato la prima volta: controlla appaiamento immagini/label, geometrie e
coerenza con `dataset.json`. Se fallisce, il problema è nella preparazione, non nel training.

Attenzione allo spazio disco: `/home` ha circa 250 GB liberi (verificare con `df -h /home`: la macchina è condivisa), e `/data01` non è scrivibile.
Il preprocessing genera stime dell'ordine di ~25 GB per ATLAS per configurazione (i dati vengono
scompattati in `.npy` durante il training), pochi GB per WMH e ~8 GB per DBB. `2d` + `3d_fullres` sui tre
dataset stanno nel budget, ma vanno monitorati:

```bash
du -sh "$nnUNet_preprocessed"; df -h /home
```

## Step 3 - Verificare il piano prima di addestrare

Questo step è quello che viene saltato più spesso ed è quello che rende i risultati interpretabili.

```bash
python - <<'PY'
import json, os
for d in ('Dataset501_WMH', 'Dataset502_ATLAS', 'Dataset503_DBB'):
    p = json.load(open(os.path.join(os.environ['nnUNet_preprocessed'], d, 'nnUNetPlans.json')))
    for cfg, c in p['configurations'].items():
        print(d, cfg, 'spacing', c.get('spacing'), 'patch', c.get('patch_size'),
              'batch', c.get('batch_size'))
PY
```

Confronti da fare:

- **Target spacing** contro quella prevista (`reports/dataset_preparation.md`, `reports/dbb_analysis.md`):
  `1.2 x 0.9766 x 3.0` mm per WMH, `1.0 x 1.0 x 1.0` mm per ATLAS e per DBB. Una discrepanza significa
  che il fingerprint ha visto dati diversi da quelli attesi.
- **Patch size lungo l'asse a 3 mm** per WMH: con circa 83 fette, se la patch copre pochi slice il
  contesto attraverso il piano è limitato, ed è un argomento a favore della configurazione `2d`.
- **Batch size**: se è 2 (il minimo), il modello sta già saturando la memoria pianificata.

## Step 4 - Verificare che i nostri fold siano rispettati

`splits_final.json` è già stato scritto da noi e nnU-Net ne genera uno proprio **solo se il file non
esiste**. All'avvio del training nnU-Net stampa la dimensione di train e validation: devono corrispondere
a `63/16` (WMH e WMH_T1, fold 0-3), `64/15` (fold 4), `444/112` o `445/111` (ATLAS), `184/23` (DBB fold 0,
che è lo split ufficiale del benchmark) e `166/41` (DBB fold 1-4).
Se i numeri non corrispondono, il file è stato sovrascritto e il confronto fra esperimenti non è più
valido.

## Step 5 - Strategia di training

Il budget completo sarebbe 3 dataset × 2 configurazioni × 5 fold = **30 training**. Con il default di 1000
epoche sono settimane di GPU, quindi non si fa così. L'ordine consigliato:

1. **Un solo fold, per misurare.** `fold 0` di `3d_fullres` su WMH (il dataset piccolo) e leggere il tempo
   per epoca dal log: serve a stimare il costo reale prima di impegnare la GPU.
2. **Primo giro rapido**, per confrontare le configurazioni senza pagare il prezzo pieno:
   `-tr nnUNetTrainerWandb_250epochs` su `fold 0`. Serve a scegliere fra `2d` e `3d_fullres`, non a
   produrre i numeri finali.
3. **Giro definitivo** sui 5 fold, solo sulle configurazioni sopravvissute.

```bash
# baseline, un fold alla volta
nnUNetv2_train 501 3d_fullres 0 --npz -tr nnUNetTrainerWandb
nnUNetv2_train 502 3d_fullres 0 --npz -tr nnUNetTrainerWandb
nnUNetv2_train 503 3d_fullres 0 --npz -tr nnUNetTrainerWandb

# tutti i fold, quando la configurazione è decisa
for f in 0 1 2 3 4; do nnUNetv2_train 501 3d_fullres $f --npz -tr nnUNetTrainerWandb; done
```

La priorità fra dataset ed esperimenti (baseline → ablazione FLAIR → transfer) è in
[obiettivo-e-piano.md](obiettivo-e-piano.md#piano-di-lavoro-in-ordine).

`--npz` salva le probabilità di validation: senza, `nnUNetv2_find_best_configuration` non può valutare
gli ensemble.

Su WMH la configurazione `2d` merita attenzione reale e non è una formalità: con fette da 3 mm il
contesto attraverso il piano è scarso, ed è il caso tipico in cui la 2d è competitiva con la 3d.
`3d_lowres` e la cascade vanno usate solo se è il planner a proporle.

## Step 6 - Scegliere configurazione ed ensemble

```bash
nnUNetv2_find_best_configuration 501 -c 2d 3d_fullres
```

Decide su **cross-validation**, non sul test. Stampa anche i comandi di inferenza e post-processing da
usare dopo.

## Step 7 - Valutare sul test interno

Il test è stato tenuto fuori dal fingerprint e dal planning: va usato **una volta sola**, alla fine,
sulla configurazione già scelta.

```bash
D=Dataset501_WMH
nnUNetv2_predict -i "$nnUNet_raw/$D/imagesTs" -o /tmp/pred_501 -d 501 -c 3d_fullres -f 0 1 2 3 4
nnUNetv2_evaluate_folder "$nnUNet_raw/$D/labelsTs" /tmp/pred_501 \
    -djfile "$nnUNet_raw/$D/dataset.json" -pfile "$nnUNet_preprocessed/$D/nnUNetPlans.json"
```

Per ATLAS, `imagesTs_official/` (300 casi senza label) serve **solo** per un'eventuale submission: non
produce alcuna metrica locale.

### Metriche

Il Dice globale da solo è fuorviante su questi compiti e va affiancato da metriche per lesione:

- **WMH**: Dice, Hausdorff 95, differenza di volume assoluta, e recall/F1 **per lesione** — con lesioni
  piccole e numerose il Dice pesa quasi solo le lesioni grandi.
- **ATLAS**: Dice, F1 per lesione, differenza di volume. 19 casi su 655 hanno meno di 100 voxel di
  lesione alla risoluzione target: su questi il Dice è instabile per costruzione, e il Dice medio va
  sempre letto insieme alla distribuzione, non da solo.
- **WMH, per sito**: riportare il Dice separatamente per `Amsterdam_GE1T5`, `Amsterdam_GE3T`,
  `Singapore`, `Utrecht`. La generalizzazione fra scanner è la domanda interessante del dataset, e una
  media unica la nasconde.
- **DBB**: qui il compito è diverso (6 tessuti, foreground denso) e il Dice globale è alto per
  costruzione. Va riportato **per classe** — tronco encefalico e grigia profonda sono piccoli e sono le
  classi dove si vede davvero la qualità — e **per patologia** sui 37 casi di test (PFM, MCD, ACC, HD),
  perché il senso del benchmark è proprio la tenuta su anatomie mai viste.

## Step 8 - Tracking

Il tracking è **già configurato e verificato**: si ottiene selezionando il trainer del progetto.

```bash
nnUNetv2_train 502 3d_fullres 0 --npz -tr nnUNetTrainerWandb
```

`nnUNetTrainerWandb` (in `scripts/nnunet_trainers/`) accende il logger W&B **integrato in nnU-Net 2.8**
e vi aggiunge nome del run, tag, e le voci di config che il logger di serie non registra: target
spacing del piano, patch size, canali di input, mappa delle label. Il resto del comportamento è quello
del `nnUNetTrainer` di default, quindi un run con questo trainer e uno con il default restano
confrontabili. Varianti disponibili: `nnUNetTrainerWandb_250epochs` per il primo giro rapido e
`nnUNetTrainerWandb_5epochs` per gli smoke test.

Se W&B non è raggiungibile il training **prosegue senza tracking**: un problema di logging non deve far
perdere ore di GPU. Per un run deliberatamente non tracciato, usare il `nnUNetTrainer` di default.

Per verificare la catena senza impegnare la GPU per ore:

```bash
python scripts/make_smoke_dataset.py --sorgente Dataset502_ATLAS --n 16
nnUNetv2_plan_and_preprocess -d 599 -c 3d_fullres --verify_dataset_integrity
nnUNetv2_train 599 3d_fullres 0 -tr nnUNetTrainerWandb_5epochs
```

## Regole che rendono valido il confronto

- Non guardare il test set finché la configurazione non è decisa sulla cross-validation.
- Non rigenerare `splits_final.json` fra un esperimento e l'altro.
- Cambiare **una** variabile per volta: canali (FLAIR contro FLAIR+T1), politica della label 2,
  configurazione, target spacing. Cambiarne due insieme rende il risultato non attribuibile.
- Per cambiare risoluzione usare un piano alternativo
  (`nnUNetv2_plan_experiment -overwrite_target_spacing ...`) invece di sovrascrivere il default, così i
  due esperimenti restano entrambi riproducibili.
