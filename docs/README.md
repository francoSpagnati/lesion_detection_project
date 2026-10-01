# Documentazione del progetto

Rilevamento automatico di lesioni cerebrali in bambini con **paralisi cerebrale**, con nnU-Net v2.
I dati originali non vengono duplicati: sono referenziati da `data/raw/` tramite symlink.

## Da dove iniziare

Leggi in quest'ordine. Sono sei documenti, nessuno ripete gli altri. Per il percorso in breve, senza dettagli tecnici, parti dalla [panoramica](panoramica-progetto.md).

1. **[Obiettivo e piano](obiettivo-e-piano.md)** — a cosa serve il progetto, i quattro dataset e il ruolo
   di ciascuno, cosa va fatto adesso e in che ordine. **Se leggi un solo file, leggi questo.**
2. **[Primer: MRI e segmentazione](primer-mri-e-segmentazione.md)** — che cos'è una risonanza, voxel,
   spacing, geometrie, metriche, come funziona nnU-Net. Non presuppone background in neuroimaging.
3. **[Dataset e decisioni](dataset-e-decisioni.md)** — cosa contiene ciascun dataset, i problemi trovati
   misurandoli, e le decisioni prese con la loro motivazione. Include cosa **non** facciamo e perché.
4. **[Modelli ed esperimenti](modelli-ed-esperimenti.md)** — il registro dei modelli: per ciascuno il
   dataset visto, l'architettura, la ricetta di training, il risultato e cosa ci ha detto. **È il file da
   aprire per "che modelli abbiamo e cosa hanno dimostrato".**
5. **[Cosa è stato fatto](cosa-e-stato-fatto.md)** — cronologia pratica: quale script, quale comando,
   cosa ha prodotto.
6. **[Training](training.md)** — la procedura: come si lancia un training, dove si legge il progresso,
   come si valuta, il tracking su W&B.

[Panoramica del progetto](panoramica-progetto.md) è il documento di **comunicazione**: una sintesi
compatta di dati, scelte, training e risultati, da mostrare a chi sta fuori dal progetto. Non è
documentazione tecnica e non va mantenuta in parallelo agli altri: si aggiorna quando serve mostrarla.

Per aprire una **nuova sessione di lavoro con un assistente**, usa il prompt in
[prompt-nuova-chat.md](prompt-nuova-chat.md): fa partire la sessione da questi documenti invece di far
riesplorare i dati da zero.

## Stato al 1 ottobre 2026

**In una riga:** sono arrivati tutti i pazienti (**17**, non 27) e sono in corso i training del
**modello finale** (`Dataset506`): una rete nnU-Net a due canali (FLAIR + T1) nel template pediatrico,
in tre ricette — da ATLAS con il primo strato espanso (A), dal pretraining congiunto ATLAS + WMH (D), da
zero (Z) — su 12 pazienti, con 5 pazienti di test messi da parte. Domani: scelta della ricetta e del
postprocessing sui 12, poi il test una volta sola.

Il percorso, spiegato per chi non conosce il progetto, è in [panoramica-progetto.md](panoramica-progetto.md).

### Risultati misurati

| Esperimento | Dice fold 0 | Cosa dice |
| --- | --- | --- |
| `501` WMH, FLAIR | **0.774** | baseline |
| `502` ATLAS, T1 | **0.610** | baseline; i casi falliti sono lesioni piccole non cavitate |
| `503` DBB, 6 tessuti | **0.971** | baseline; validation su 23 casi *sani* |
| `511` WMH, solo T1 | **0.626** | −0.148 contro la FLAIR, peggiore in **16 casi su 16** (p=0.00003) |
| `521` ATLAS solo 3 T | **0.616** | +0.0093 appaiato su 103 casi, **p=0.67**: escludere i 1.5 T non serve |
| `512` WMH, FLAIR+T1 | **0.753** | −0.0205 appaiato su 16 casi, peggiore in 11 su 16 (p=0.025): il T1 **toglie** |
| `521` applicato ai casi 1.5 T mai visti | **0.672** | nessun crollo fuori dominio; sui 9 casi puliti −0.072 con **p=0.25**, e due terzi vengono da un caso solo |
| `522` WMH skull-stripped | **0.779** | +0.0045 appaiato, **p=0.86**: nessun effetto |
| `523` ATLAS skull-stripped | **0.607** | −0.0037 appaiato, **p=0.58**: nessun effetto |
| `531` **variante D**, tutti e tre insieme | **0.603** lesione | −0.007 su ATLAS (p=0.073) e −0.022 su WMH (p=0.10): non crolla; ma perde su tutte e 7 le classi |
| `541` **test del pretraining** | **0.685** contro **0.451** da zero | il pretraining vale **+0.23**, ma solo con passo `1e-4`: col default (0.564) *danneggia* il modello, che senza adattamento farebbe 0.677 |
| `504` **AInCP**, 10 pazienti, 5 fold | da zero **0.581**, Proposta B `zero` **0.555**, `duplica` **0.596**, passo differenziato **0.621** | nessuna differenza di Dice distinguibile dal rumore (miglior variante: IC 95% [−0.036, +0.110]); i pre-addestrati fanno un terzo dei falsi positivi (p = 0.033). Con il passo differenziato la FLAIR impara. Per fenotipo: ATLAS porta le cavità, la FLAIR la gliosi (`reports/analisi_rilevamento_504.md`) |

Sui pazienti AInCP (cinque fold, dieci pazienti; dettaglio nel registro):

| Esperimento | Dice | Cosa dice |
| --- | --- | --- |
| `504` nativo, sei ricette | **0.555–0.621** | Dice indistinguibile fra ricette; il pretraining porta le cavità e taglia i falsi positivi (10 pazienti su 10, p = 0.002); scelta la ricetta a passo differenziato |
| `542` WMH dopo ATLAS, in sequenza | 0.735 su WMH | **dimentica ATLAS**: 0.610 → 0.217 con la FLAIR vuota. Abbandonato |
| `543` ATLAS + WMH insieme, 1000 epoche | 0.606 ATLAS, 0.723 WMH | non dimentica e impara la FLAIR: i pesi della ricetta D |
| `505` i dieci nel template pediatrico | 0.563–0.600 | il template non costa nulla di misurabile (p = 0.49 contro il nativo) |
| postprocessing su `505` | F1 per lesione 0.61 → 0.68 | soglia e volume minimo: falsi positivi da 22 a 6 senza perdere lesioni |

### In corso

| Cosa | Stato |
| --- | --- |
| **Modello finale** `Dataset506_AINCP_FINALE` | 15 run (A, D, Z × 5 fold) dal 1/10 sulle 4 GPU condivise con un altro utente; fine prevista il 2/10 |
| Split | 12 di addestramento, **5 di test** (FSM_030, 046, 049, 052, 054), per strato di acquisizione: `reports/split_Dataset506_AINCP_FINALE.md` |
| Domande aperte al clinico | 46 componenti residue da correggere, 12 fuori dal cervello (`reports/residui_annotazione.md`); encefalomalacia a intensità marginale (FSM_040, FSM_045) |

### Dopo il modello finale

| Voce | Stato |
| --- | --- |
| Scelta di ricetta e postprocessing sui fuori-fold dei 12, poi il test | procedura pronta (`postprocessing_aincp.py`, `predici_test.sh`) |
| Ablazione dei canali in inferenza sul modello scelto | da fare: prova definitiva che usa entrambe le sequenze |
| Stadio 2 della tesi — inpainting e gemello sano | dopo la segmentazione |

**Abbandonati o superati**, con il motivo nel registro: variante D a sette classi (531), catena DBB →
ATLAS, stadio 2 in sequenza (542), lavoro nello spazio nativo (sostituito dal template, incontro del 23/9),
Proposta C (lesioni sintetiche su DBB: proposta ai relatori, non avviata; con i pazienti arrivati non è più sul percorso critico), skull stripping
nelle baseline (nessun effetto).

### Verifiche introdotte dopo i chiarimenti del relatore (7 settembre)

| verifica | script | cosa ha detto |
| --- | --- | --- |
| nnU-Net fa early stopping? | — | **No**. Il suo `checkpoint_best` (pseudo-Dice) recupera però **+0.112** sul fine-tuning a passo default di `541` e nulla sul braccio da zero. Dal 13/9 il trainer del progetto ha l'arresto sulla perdita di validazione (`pazienza_val`) |
| curve di training e overfitting | `analisi_curve_training.py` | modesto sui dataset grandi, **massiccio sui 10 casi** (+0.534 di divario), e si riduce al ridursi del passo |
| **rilevamento per singola lesione** | `metriche_per_lesione.py` | su ATLAS/T1 il modello trova l'**8.7%** delle lesioni sotto 10 mm³; su WMH con FLAIR il **45.2%**, con solo T1 l'**11.9%** (fasce in mm³ dal 13/9; totali invariati: 72.8% e 38.2%) |

### Materiale per gli incontri

Le figure stanno in `reports/figures/` e si rigenerano tutte da script.

| figura | cosa mostra | script |
| --- | --- | --- |
| `risultati.png` | tutti i confronti in due pannelli | `figura_risultati.py` |
| `flair_contro_t1.png` | **stesso paziente, stessa fetta: FLAIR contro T1** | `figura_flair_contro_t1.py` |
| `modello_unico_7_classi.png` | le due modalità di `531`: tessuti sui pediatrici, lesione sugli adulti | `figura_modello_unico.py` |
| `skullstrip_prima_dopo.png` | cosa toglie SynthStrip su WMH e ATLAS | `figura_skullstrip_prima_dopo.py` |
| `predizioni_<NNN>_fold_0.png` | predizione contro riferimento, per ognuno dei nove modelli | `plot_predictions.py` |
| `synthstrip/` | 244 figure di validazione su DBB, il nome inizia con il Dice | `run_synthstrip.py` |

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
