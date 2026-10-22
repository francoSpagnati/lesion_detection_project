# Blob loss (B, 506) contro il modello della tesi (Z, 506), fuori-fold

Prodotto da `scripts/confronto_misto_532.py B`, **non modificare a mano**. Criterio fissato prima dei risultati: misura principale le lesioni sotto 50 mm³ trovate (McNemar appaiato), vincolo i falsi positivi e l'F1.

## Misura principale

| | Z | B | trovate solo da B / solo da Z | p (McNemar) |
| --- | --- | --- | --- | --- |
| lesioni < 50 mm³ trovate | 10/46 | 9/46 | 1 / 2 | 1.000 |
| tutte le lesioni trovate | 27/68 | 27/68 | 2 / 2 | 1.000 |

## Vincolo e misure secondarie

| | postprocessing | F1 lesioni | falsi positivi | Dice medio | mediana | globale |
| --- | --- | --- | --- | --- | --- | --- |
| Z | soglia 0.3, volume minimo 50 mm³ | 0.560 | 1 | 0.584 | 0.565 | 0.859 |
| B | soglia 0.3, volume minimo 5 mm³ | 0.532 | 8 | 0.600 | 0.582 | 0.862 |

Dice per paziente, B meno Z: +0.016 in media, B meglio in 7 pazienti e peggio in 4, p = 0.465 (Wilcoxon).

## Lesioni trovate per fascia di volume

| fascia | lesioni | Z | B |
| --- | --- | --- | --- |
| 1–10 mm³ | 34 | 7 | 6 |
| 10–50 mm³ | 12 | 3 | 3 |
| 50–200 mm³ | 9 | 6 | 6 |
| 200–1 000 mm³ | 3 | 1 | 2 |
| 1 000–10 000 mm³ | 4 | 4 | 4 |
| > 10 000 mm³ | 6 | 6 | 6 |

## Copertura per fenotipo

| fenotipo | voxel | Z | B |
| --- | --- | --- | --- |
| gliosi | 238509 | 75% | 75% |
| cistica | 954382 | 91% | 92% |
| encefalomalacia | 378574 | 77% | 80% |

## Dice per paziente

| paziente | Z | B |
| --- | --- | --- |
| FSM_001 | 0.458 | 0.384 |
| FSM_028 | 0.892 | 0.882 |
| FSM_032 | 0.859 | 0.876 |
| FSM_033 | 0.000 | 0.000 |
| FSM_035 | 0.854 | 0.835 |
| FSM_036 | 0.870 | 0.879 |
| FSM_039 | 0.517 | 0.582 |
| FSM_040 | 0.891 | 0.903 |
| FSM_041 | 0.160 | 0.225 |
| FSM_045 | 0.439 | 0.581 |
| FSM_053 | 0.460 | 0.479 |
| FSM_056 | 0.613 | 0.579 |
