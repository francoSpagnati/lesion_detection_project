# Training misto (M, 532) contro il modello della tesi (Z, 506), fuori-fold

Prodotto da `scripts/confronto_misto_532.py`, **non modificare a mano**. Criterio fissato prima dei risultati: misura principale le lesioni sotto 50 mm³ trovate (McNemar appaiato), vincolo i falsi positivi e l'F1.

## Misura principale

| | Z | M | trovate solo da M / solo da Z | p (McNemar) |
| --- | --- | --- | --- | --- |
| lesioni < 50 mm³ trovate | 10/46 | 11/46 | 3 / 2 | 1.000 |
| tutte le lesioni trovate | 27/68 | 28/68 | 3 / 2 | 1.000 |

## Vincolo e misure secondarie

| | postprocessing | F1 lesioni | falsi positivi | Dice medio | mediana | globale |
| --- | --- | --- | --- | --- | --- | --- |
| Z | soglia 0.3, volume minimo 50 mm³ | 0.560 | 1 | 0.584 | 0.565 | 0.859 |
| M | soglia 0.4, volume minimo 50 mm³ | 0.542 | 6 | 0.596 | 0.604 | 0.853 |

Dice per paziente, M meno Z: +0.011 in media, M meglio in 6 pazienti e peggio in 5, p = 0.465 (Wilcoxon).

## Lesioni trovate per fascia di volume

| fascia | lesioni | Z | M |
| --- | --- | --- | --- |
| 1–10 mm³ | 34 | 7 | 8 |
| 10–50 mm³ | 12 | 3 | 3 |
| 50–200 mm³ | 9 | 6 | 6 |
| 200–1 000 mm³ | 3 | 1 | 1 |
| 1 000–10 000 mm³ | 4 | 4 | 4 |
| > 10 000 mm³ | 6 | 6 | 6 |

## Copertura per fenotipo

| fenotipo | voxel | Z | M |
| --- | --- | --- | --- |
| gliosi | 238509 | 75% | 74% |
| cistica | 954382 | 91% | 89% |
| encefalomalacia | 378574 | 77% | 77% |

## Dice per paziente

| paziente | Z | M |
| --- | --- | --- |
| FSM_001 | 0.458 | 0.386 |
| FSM_028 | 0.892 | 0.886 |
| FSM_032 | 0.859 | 0.849 |
| FSM_033 | 0.000 | 0.000 |
| FSM_035 | 0.854 | 0.782 |
| FSM_036 | 0.870 | 0.888 |
| FSM_039 | 0.517 | 0.588 |
| FSM_040 | 0.891 | 0.904 |
| FSM_041 | 0.160 | 0.194 |
| FSM_045 | 0.439 | 0.464 |
| FSM_053 | 0.460 | 0.598 |
| FSM_056 | 0.613 | 0.609 |
