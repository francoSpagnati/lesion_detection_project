# Rilevamento automatico di lesioni cerebrali nella paralisi cerebrale infantile

Progetto di tesi magistrale. Una rete **nnU-Net v2** a due canali (FLAIR + T1) segmenta le lesioni
cerebrali su risonanza magnetica di bambini con paralisi cerebrale. Il bersaglio è un dataset privato di
**17 pazienti** dell'AInCP (Stella Maris e altri centri), 3–15 anni; tre dataset pubblici (WMH, ATLAS 2.0, DBB)
servono a costruire il metodo e i pesi di partenza.

> **Questo repository non contiene dati di pazienti.** Ci sono il codice, i report con le misure e la
> documentazione. Immagini, maschere, modelli addestrati e dati preprocessati (circa 150 GB) restano fuori
> da git; i dati AInCP sono privati. Senza di essi gli script di addestramento non si possono
> rieseguire; i numeri si leggono nei report.

## Il metodo in breve

1. **Preprocessing** (`scripts/pipeline_aincp.py`): correzione del bias di campo N4, registrazione
   non lineare della T1 sul template pediatrico MNIPediatricAsym (cohort-1, 1 mm), FLAIR portata nello
   stesso spazio con una sola interpolazione.
2. **Addestramento**: nnU-Net 3D a piena risoluzione, cross-validation a 5 fold su 12 pazienti, 5 pazienti
   riservati al test. Il modello della tesi (**Z**) parte da zero, con modality dropout 0.3 per poter
   usare anche una sola sequenza, e arresto anticipato sulla perdita di validazione.
3. **Postprocessing** (`scripts/postprocessing_aincp.py`): le probabilità tornano nello spazio nativo
   del paziente, soglia 0.3, componenti sotto 50 mm³ tolte. Parametri scelti sulla cross-validation e fissati
   prima di guardare il test.
4. **Valutazione** nello spazio nativo, oltre il Dice: lesioni trovate per fascia di volume, falsi
   positivi, CC-Dice, percentuale di cervello lesionato, intervalli bootstrap sui pazienti.

## Risultati principali

| | Dice medio | lesioni trovate | falsi positivi | F1 per lesione |
| --- | --- | --- | --- | --- |
| Cross-validation (12 pazienti, fuori-fold) | 0.584 | 27/68 | 1 | 0.560 |
| Test (5 pazienti, usato una sola volta) | 0.622 | 21/40 | 4 | 0.605 |

Le lesioni grandi sono segmentate bene (Dice 0.83 sopra 10 000 mm³); **quelle sotto 50 mm³ sono in gran
parte mancate**, ed è il limite principale. Sono stati provati senza successo pretraining da ATLAS e WMH
(ricette A e D), addestramento misto con i dati pubblici, blob loss per dare peso alle lesioni piccole,
ventricolo come classe aggiuntiva: con 12 pazienti nessuna ricetta è distinguibile dalle altre.
Una rete a tre classi (gliosi, lesione cistica, encefalomalacia) riconosce il tipo di lesione sul 86% dei
voxel trovati, a prezzo di più falsi positivi. Dettaglio e limiti in
[`docs/modelli-ed-esperimenti.md`](docs/modelli-ed-esperimenti.md).

## Dove guardare

| Cosa | Dove |
| --- | --- |
| Il percorso spiegato a chi non conosce il progetto | [`docs/panoramica-progetto.md`](docs/panoramica-progetto.md) |
| Obiettivo, dataset e piano | [`docs/obiettivo-e-piano.md`](docs/obiettivo-e-piano.md) |
| Registro di ogni modello ed esperimento, con motivo e risultato | [`docs/modelli-ed-esperimenti.md`](docs/modelli-ed-esperimenti.md) |
| Dati, problemi trovati misurando, decisioni | [`docs/dataset-e-decisioni.md`](docs/dataset-e-decisioni.md) |
| Che cos'è una risonanza, come funziona nnU-Net (senza background) | [`docs/primer-mri-e-segmentazione.md`](docs/primer-mri-e-segmentazione.md) |
| Come si lancia un training | [`docs/training.md`](docs/training.md) |
| Indice completo della documentazione | [`docs/README.md`](docs/README.md) |
| Tabelle e analisi prodotte dagli script | `reports/` |
| Codice: pipeline, trainer, valutazione | `scripts/`, trainer personalizzati in `scripts/nnunet_trainers/` |

## Struttura

`configs` · `data` (template, manifesti dei pazienti, alcune predizioni su ATLAS) · `docs` · `notebooks` (esplorazione dei dataset pubblici) ·
`reports` · `scripts`. Gli script hanno nell'intestazione il perché e il come; i report sono generati dagli
script e non vanno modificati a mano.

## Ambiente

nnU-Net v2.8, Python 3.10, ANTsPy (registrazione), SynthStrip (maschera del cervello), SynthSeg (ventricoli).
Le variabili d'ambiente e il percorso dei trainer del progetto si impostano con `source scripts/env.sh`.
Il tracking opzionale su Weights & Biases legge la chiave da un file `.env` locale, mai versionato.

## Dataset pubblici usati

WMH Segmentation Challenge 2017, ATLAS v2.0 e DBB (Distorted Brain Benchmark): vanno scaricati dalle fonti
originali, rispettando le loro licenze. Il template pediatrico è MNIPediatricAsym (Fonov et al., 2011).
