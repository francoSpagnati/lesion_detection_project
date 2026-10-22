# Lesione + ventricolo dilatato (C, 507) contro il modello della tesi (Z, 506), fuori-fold

Prodotto da `scripts/confronto_misto_532.py C`, **non modificare a mano**. Criterio fissato prima dei risultati: misura principale i falsi positivi e l'F1 per lesione, vincolo le lesioni trovate (McNemar, non devono scendere).

## Misura principale

| | Z | C | trovate solo da C / solo da Z | p (McNemar) |
| --- | --- | --- | --- | --- |
| lesioni < 50 mm³ trovate | 10/46 | 9/46 | 2 / 3 | 1.000 |
| tutte le lesioni trovate | 27/68 | 26/68 | 3 / 4 | 1.000 |

## Vincolo e misure secondarie

| | postprocessing | F1 lesioni | falsi positivi | Dice medio | mediana | globale |
| --- | --- | --- | --- | --- | --- | --- |
| Z | soglia 0.3, volume minimo 50 mm³ | 0.560 | 1 | 0.584 | 0.565 | 0.859 |
| C | soglia 0.3, volume minimo 20 mm³, cervello | 0.517 | 5 | 0.584 | 0.614 | 0.857 |

Dice per paziente, C meno Z: +0.000 in media, C meglio in 5 pazienti e peggio in 6, p = 0.520 (Wilcoxon).

## Lesioni trovate per fascia di volume

| fascia | lesioni | Z | C |
| --- | --- | --- | --- |
| 1–10 mm³ | 34 | 7 | 6 |
| 10–50 mm³ | 12 | 3 | 3 |
| 50–200 mm³ | 9 | 6 | 5 |
| 200–1 000 mm³ | 3 | 1 | 2 |
| 1 000–10 000 mm³ | 4 | 4 | 4 |
| > 10 000 mm³ | 6 | 6 | 6 |

## Copertura per fenotipo

| fenotipo | voxel | Z | C |
| --- | --- | --- | --- |
| gliosi | 238509 | 75% | 76% |
| cistica | 954382 | 91% | 88% |
| encefalomalacia | 378574 | 77% | 78% |

## Dice per paziente

| paziente | Z | C |
| --- | --- | --- |
| FSM_001 | 0.458 | 0.436 |
| FSM_028 | 0.892 | 0.926 |
| FSM_032 | 0.859 | 0.819 |
| FSM_033 | 0.000 | 0.000 |
| FSM_035 | 0.854 | 0.809 |
| FSM_036 | 0.870 | 0.880 |
| FSM_039 | 0.517 | 0.633 |
| FSM_040 | 0.891 | 0.906 |
| FSM_041 | 0.160 | 0.162 |
| FSM_045 | 0.439 | 0.400 |
| FSM_053 | 0.460 | 0.448 |
| FSM_056 | 0.613 | 0.595 |
