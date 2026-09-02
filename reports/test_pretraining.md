# Test del pretraining: serve, con dieci casi di target?

Prodotto da `scripts/analisi_test_pretraining.py`, **non modificare a mano**.

Stesso dataset, stessa partizione (10 casi di training, 35 di validation), stessa
architettura, stesse 250 epoche. Cambia **solo** l'inizializzazione dei pesi.

| braccio | Dice medio sui 35 casi |
| --- | --- |
| addestrato da zero | 0.4507 |
| fine-tuning da `Dataset521_ATLAS_3T` | **0.5642** |
| **differenza** | **+0.1135** |

Migliora in **24** casi su 35, peggiora in **5**, pari in 6.
Wilcoxon appaiato: **p = 0.000124**.

## Il terzo braccio: e se non si facesse fine-tuning affatto?

Il modello pre-addestrato applicato **tale e quale** ai casi di valutazione, senza
vedere i 10 casi di training. Stessi casi, quindi il confronto e' appaiato.

| | Dice sui 35 casi |
| --- | --- |
| addestrato da zero sui 10 | 0.4507 |
| fine-tuning da `521` sui 10 | 0.5642 |
| **`521` senza alcun fine-tuning** | **0.6772** |
| differenza fine-tuning meno zero-shot | **-0.1130** |

Il fine-tuning batte lo zero-shot in **13** casi su 35. Wilcoxon appaiato: **p = 0.01288**.

**Il fine-tuning con la ricetta predefinita peggiora un modello gia' buono.** Il
pretraining serve, e molto, ma la messa a punto su dieci casi cosi' com'e' distrugge
piu' di quanto aggiunga: nnU-Net riparte con passo di apprendimento 0.01, che e' quello
di un addestramento completo, e con dieci casi cancella le feature pre-addestrate.

| termine di paragone | Dice |
| --- | --- |
| `502`, addestrato su 556 casi | 0.6103 |
| `521` zero-shot su questi casi | 0.6772 |

## Come si distribuisce il guadagno

Mediana del volume di lesione nei 35 casi: 13.2 ml.

| gruppo | casi | da zero | fine-tuning | differenza |
| --- | --- | --- | --- | --- |
| lesioni sotto la mediana | 18 | 0.3433 | 0.4303 | **+0.0870** |
| lesioni sopra la mediana | 17 | 0.5644 | 0.7059 | **+0.1415** |

## I cinque casi in cui il pretraining cambia di piu'

| caso | lesione (ml) | da zero | fine-tuning | delta |
| --- | --- | --- | --- | --- |
| sub-r040s002 | 1.0 | 0.0000 | 0.7813 | **+0.7813** |
| sub-r040s008 | 9.3 | 0.0030 | 0.5799 | **+0.5769** |
| sub-r029s004 | 22.7 | 0.2883 | 0.7751 | **+0.4867** |
| sub-r040s039 | 29.3 | 0.3471 | 0.7695 | **+0.4224** |
| sub-r040s032 | 130.7 | 0.1029 | 0.5132 | **+0.4103** |

## Limiti da dichiarare

- Una sola estrazione dei 10 casi di training: con dieci casi la varianza e' alta.
- Una sola sorgente di pretraining. Quella fra compiti diversi (DBB) richiede prima di
  spostare il piano di 502 su DBB e riaddestrarlo, perche' le architetture non combaciano.
- 39 dei 45 casi vengono dalla coorte `r040`: e' in larga parte un solo scanner.
- 250 epoche per entrambi i bracci. Un budget maggiore favorirebbe forse il braccio da
  zero, che parte da piu' lontano; il confronto vale a parita' di calcolo.

