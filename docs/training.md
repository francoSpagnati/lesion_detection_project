# Training

Questo documento e' la **procedura**: come si prepara, si lancia e si valuta un training. I modelli gia'
addestrati, con architetture, numeri e interpretazione, stanno invece nel registro
[modelli-ed-esperimenti.md](modelli-ed-esperimenti.md).

Prerequisito: i dataset devono essere già preparati. Stato atteso sotto `nnUNet/nnUNet_raw/`:

| Dataset | Canale | Sviluppo | Test | Prodotto da |
| --- | --- | --- | --- | --- |
| `Dataset501_WMH` | FLAIR | 79 | 20 | `notebooks/dataset_preparation.ipynb` |
| `Dataset502_ATLAS` | T1 | 556 | 99 | `notebooks/dataset_preparation.ipynb` |
| `Dataset503_DBB` | T1 | 207 | 37 | `scripts/convert_dbb_to_nnunet.py` |
| `Dataset511_WMH_T1` | T1 | 79 | 20 | `scripts/make_wmh_t1_dataset.py` |

Quali di questi conviene addestrare, in che ordine e quali possono condividere pesi è in
[obiettivo-e-piano.md](obiettivo-e-piano.md).

> **I modelli per i pazienti AInCP seguono una procedura diversa**, descritta qui sotto solo nei passi
> comuni (GPU, ambiente, lettura del progresso). Dati, split, pesi di partenza e lancio:
> `scripts/pipeline_aincp.py` → `scripts/run_stadio3.sh <ID> <A|D|Z> [fold]`; postprocessing e
> valutazione: `scripts/postprocessing_aincp.py`; test: `scripts/predici_test.sh`. I comandi esatti
> sono in [obiettivo-e-piano.md](obiettivo-e-piano.md) («Il giorno in cui arrivano i pazienti»), le
> ricette nel registro. Le GPU si condividono con `SOGLIA_MEM=60000` solo se l'utente lo ha deciso.

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

**Fatto** per 501, 502, 503, 511, 512, 521, 522 e 523 — esito e numeri reali in
[cosa-e-stato-fatto.md](cosa-e-stato-fatto.md#10-planning-e-preprocessing-di-501-502-e-503), i piani
risultanti nel registro [modelli-ed-esperimenti.md](modelli-ed-esperimenti.md).

```bash
nnUNetv2_plan_and_preprocess -d 501 -npfp 16 -np 16 --verify_dataset_integrity   # WMH (FLAIR)
nnUNetv2_plan_and_preprocess -d 502 -npfp 16 -np 16 --verify_dataset_integrity   # ATLAS (T1)
nnUNetv2_plan_and_preprocess -d 503 -npfp 16 -np 16 --verify_dataset_integrity   # DBB (T1, 6 tessuti)
```

`-np`/`-npfp` cambiano solo il tempo, non il risultato: con 16 processi i tre dataset richiedono circa
3 minuti in tutto. La macchina ha 96 core ma è condivisa, quindi non vanno presi tutti.

`--verify_dataset_integrity` va usato la prima volta: controlla appaiamento immagini/label, geometrie e
coerenza con `dataset.json`. Se fallisce, il problema è nella preparazione, non nel training.

Attenzione allo spazio disco: `/home` ha circa 250 GB liberi (verificare con `df -h /home`: la macchina è condivisa), e `/data01` non è scrivibile.
**Misurato:** `2d` + `3d_fullres` sui tre dataset occupano **24 GB** (ATLAS 15.9, DBB 6.8, WMH 1.1), meno
delle stime iniziali. Il costo vero arriva dopo: durante il training i casi vengono scompattati in `.npy`
non compressi, quindi lo spazio va ricontrollato al primo training su ATLAS:

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

**Come lanciare, perché il modo conta.** Un training dura ore e **non deve morire con il terminale o
con la sessione che lo ha avviato**. `nohup` da solo non basta: protegge dal segnale di riaggancio del
terminale, non dalla terminazione dell'intero gruppo di processi quando il padre viene chiuso. Serve
`setsid`, che mette il comando in una sessione propria. `scripts/run_baselines.sh` incapsula questo più
le guardie su spazio disco e GPU occupata, riprende da `checkpoint_latest.pth` se lo trova e si ferma
invece di proseguire quando un training fallisce:

```bash
setsid nohup bash scripts/run_baselines.sh 501 502 503 > reports/logs/baselines.log 2>&1 < /dev/null &
```

**Dove leggere il progresso.** Non in `reports/logs/train_<d>_*.log`: quello è lo stdout rediretto, che
Python bufferizza a blocchi e puo' restare indietro di mezz'ora, facendo sembrare fermo un training che
sta lavorando. Il log vero, scritto con flush esplicito da nnU-Net, è:

```bash
tail -f nnUNet/nnUNet_results/Dataset<NNN>_*/*/fold_0/training_log_*.txt
```

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

Su WMH la configurazione `2d` va misurata, ma **l'argomento a priori a suo favore si è indebolito**: il
piano assegna alla `3d_fullres` una patch di 64 fette su 79, cioè l'81% del volume, quindi il contesto
attraverso il piano non è così scarso come si temeva con fette da 3 mm. Resta un confronto da fare, non
più un candidato favorito.

`3d_lowres` **non è disponibile**: il planner l'ha scartata su tutti e tre i dataset perché la riduzione
di dimensione sarebbe trascurabile. Di conseguenza nemmeno la cascade è utilizzabile, e il confronto di
configurazione è a due: `2d` contro `3d_fullres`.

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

**Ripresa di un training interrotto** (`nnUNetv2_train ... --c`, riparte da `checkpoint_latest.pth`, che
nnU-Net salva ogni 50 epoche). Funziona, ma solo grazie a una correzione nel nostro trainer: il
`WandbLogger` di nnU-Net 2.8 riaggancia il run W&B esistente e ne riscrive la config chiamando
`run.config.update()` **senza** `allow_val_change=True`; wandb rifiuta di sovrascrivere una chiave già
presente e il training muore in inizializzazione. I due valori sono lo stesso piano, diverso solo
all'ultima cifra della serializzazione dei float. Senza quella correzione nessun run tracciato è
ripartibile. Alla ripresa W&B ignora le epoche già registrate («Tried to log to step N») e riprende
dalla prima nuova: è atteso, non un errore.

Per verificare la catena senza impegnare la GPU per ore:

```bash
python scripts/make_smoke_dataset.py --sorgente Dataset502_ATLAS --n 16
nnUNetv2_plan_and_preprocess -d 599 -c 3d_fullres --verify_dataset_integrity
nnUNetv2_train 599 3d_fullres 0 -tr nnUNetTrainerWandb_5epochs
```

## Regole che rendono valido il confronto

- Non guardare il test set finché la configurazione non è decisa sulla cross-validation.
- Non rigenerare `splits_final.json` fra un esperimento e l'altro.
- **Confrontare i casi, non le medie.** Se due dataset non hanno la stessa validation, una differenza fra
  le due medie complessive può venire dalla composizione invece che dal cambiamento.
  `scripts/confronto_appaiato.py` interseca i casi e applica un Wilcoxon appaiato — senza assumere
  normalità, che sul Dice di lesioni piccole non vale.
- **Controllare se nnU-Net ha ricalcolato il piano.** Un dataset derivato può ricevere patch e batch
  diversi (è successo su `522`, dove il ritaglio ha ridotto i volumi): in quel caso il confronto misura la
  pipeline intera, non la singola variabile, e va detto.
- Cambiare **una** variabile per volta: canali (FLAIR contro FLAIR+T1), politica della label 2,
  configurazione, target spacing. Cambiarne due insieme rende il risultato non attribuibile.
- Per cambiare risoluzione usare un piano alternativo
  (`nnUNetv2_plan_experiment -overwrite_target_spacing ...`) invece di sovrascrivere il default, così i
  due esperimenti restano entrambi riproducibili.
