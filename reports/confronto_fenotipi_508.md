# Fenotipi come classi (F, 508) contro il modello della tesi (Z, 506), fuori-fold

Prodotto da `scripts/confronto_misto_532.py F`, **non modificare a mano**. Criterio fissato prima dei risultati: il rilevamento non deve peggiorare (F1, lesioni trovate, Dice); il riconoscimento del tipo di lesione è in reports/fenotipi_508.md.

## Misura principale

| | Z | F | trovate solo da F / solo da Z | p (McNemar) |
| --- | --- | --- | --- | --- |
| lesioni < 50 mm³ trovate | 10/46 | 12/46 | 3 / 1 | 0.625 |
| tutte le lesioni trovate | 27/68 | 29/68 | 3 / 1 | 0.625 |

## Vincolo e misure secondarie

| | postprocessing | F1 lesioni | falsi positivi | Dice medio | mediana | globale |
| --- | --- | --- | --- | --- | --- | --- |
| Z | soglia 0.3, volume minimo 50 mm³ | 0.560 | 1 | 0.584 | 0.565 | 0.859 |
| F | soglia 0.5, volume minimo 20 mm³, cervello | 0.511 | 12 | 0.582 | 0.654 | 0.808 |

Dice per paziente, F meno Z: -0.002 in media, F meglio in 3 pazienti e peggio in 8, p = 0.577 (Wilcoxon).

## Lesioni trovate per fascia di volume

| fascia | lesioni | Z | F |
| --- | --- | --- | --- |
| 1–10 mm³ | 34 | 7 | 9 |
| 10–50 mm³ | 12 | 3 | 3 |
| 50–200 mm³ | 9 | 6 | 6 |
| 200–1 000 mm³ | 3 | 1 | 1 |
| 1 000–10 000 mm³ | 4 | 4 | 4 |
| > 10 000 mm³ | 6 | 6 | 6 |

## Copertura per fenotipo

| fenotipo | voxel | Z | F |
| --- | --- | --- | --- |
| gliosi | 238509 | 75% | 71% |
| cistica | 954382 | 91% | 77% |
| encefalomalacia | 378574 | 77% | 60% |

## Dice per paziente

| paziente | Z | F |
| --- | --- | --- |
| FSM_001 | 0.458 | 0.314 |
| FSM_028 | 0.892 | 0.851 |
| FSM_032 | 0.859 | 0.732 |
| FSM_033 | 0.000 | 0.000 |
| FSM_035 | 0.854 | 0.843 |
| FSM_036 | 0.870 | 0.844 |
| FSM_039 | 0.517 | 0.624 |
| FSM_040 | 0.891 | 0.866 |
| FSM_041 | 0.160 | 0.150 |
| FSM_045 | 0.439 | 0.685 |
| FSM_053 | 0.460 | 0.521 |
| FSM_056 | 0.613 | 0.561 |
