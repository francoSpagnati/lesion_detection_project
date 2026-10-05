# Ablazione dei canali in inferenza: ricetta Z di Dataset506 (fuori-fold)

Prodotto da `scripts/ablazione_canali_506.py`, **non modificare a mano**. 12 pazienti di addestramento, ognuno predetto dal modello del fold che non l'ha visto; la sequenza tolta e' azzerata in ingresso. Postprocessing congelato di Z (soglia 0.3, volume minimo 50 mm³), metriche nel nativo. Controllo delle ROI: l'unione dei fenotipi ricostruiti coincide con l'etichetta pulita con Dice minimo 0.987.

## Complessivo

| ingresso | Dice medio | mediana | globale | Δ Dice vs entrambe | meglio/peggio | p (Wilcoxon) | lesioni trovate | solo qui / solo con entrambe | p (McNemar) | falsi positivi | F1 lesioni |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FLAIR + T1 | 0.584 | 0.565 | 0.859 | — | — | — | 27/68 | — | — | 1 | 0.560 |
| solo T1 (FLAIR azzerata) | 0.418 | 0.372 | 0.820 | -0.167 | 0/11 | < 0.001 | 14/68 | 0 / 13 | < 0.001 | 2 | 0.331 |
| solo FLAIR (T1 azzerata) | 0.507 | 0.496 | 0.734 | -0.077 | 2/9 | 0.042 | 28/68 | 1 / 0 | 1.000 | 3 | 0.558 |

## Lesioni trovate per fascia di volume

| fascia | lesioni | FLAIR + T1 | solo T1 (FLAIR azzerata) | solo FLAIR (T1 azzerata) |
| --- | --- | --- | --- | --- |
| 1–10 mm³ | 34 | 7/34 | 2/34 | 7/34 |
| 10–50 mm³ | 12 | 3/12 | 2/12 | 3/12 |
| 50–200 mm³ | 9 | 6/9 | 2/9 | 6/9 |
| 200–1 000 mm³ | 3 | 1/3 | 1/3 | 2/3 |
| 1 000–10 000 mm³ | 4 | 4/4 | 1/4 | 4/4 |
| > 10 000 mm³ | 6 | 6/6 | 6/6 | 6/6 |

## Per fenotipo

Copertura: voxel del fenotipo coperti dalla predizione su voxel totali, sommati sui pazienti (pesa chi ha piu' lesione, come il Dice). Lesioni trovate: ogni lesione annotata ha il fenotipo con piu' voxel.

| fenotipo | voxel | copertura FLAIR + T1 | copertura solo T1 (FLAIR azzerata) | copertura solo FLAIR (T1 azzerata) | lesioni | trovate FLAIR + T1 | trovate solo T1 (FLAIR azzerata) | trovate solo FLAIR (T1 azzerata) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gliosi | 238509 | 75% | 44% | 62% | 60 | 20/60 | 7/60 | 21/60 |
| cistica | 954382 | 91% | 95% | 88% | 6 | 5/6 | 5/6 | 5/6 |
| encefalomalacia | 378574 | 77% | 81% | 58% | 2 | 2/2 | 2/2 | 2/2 |

## Dice per paziente

| paziente | FLAIR + T1 | solo T1 (FLAIR azzerata) | solo FLAIR (T1 azzerata) |
| --- | --- | --- | --- |
| FSM_001 | 0.458 | 0.000 | 0.449 |
| FSM_028 | 0.892 | 0.872 | 0.830 |
| FSM_032 | 0.859 | 0.835 | 0.841 |
| FSM_033 | 0.000 | 0.000 | 0.000 |
| FSM_035 | 0.854 | 0.761 | 0.844 |
| FSM_036 | 0.870 | 0.819 | 0.138 |
| FSM_039 | 0.517 | 0.000 | 0.544 |
| FSM_040 | 0.891 | 0.846 | 0.882 |
| FSM_041 | 0.160 | 0.000 | 0.169 |
| FSM_045 | 0.439 | 0.176 | 0.381 |
| FSM_053 | 0.460 | 0.136 | 0.437 |
| FSM_056 | 0.613 | 0.568 | 0.570 |
