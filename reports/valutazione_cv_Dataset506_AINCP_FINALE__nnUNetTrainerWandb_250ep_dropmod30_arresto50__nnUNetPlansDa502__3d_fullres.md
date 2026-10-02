# Valutazione di `Dataset506_AINCP_FINALE__nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py valuta`, **non modificare a mano**. Predizioni fuori-fold (cross-validation), 12 pazienti, nella griglia T1 nativa. Postprocessing: soglia 0.3, volume minimo 50 mm³, maschera del cervello no.

| | Dice (IC 95%) | lesioni rilevate | falsi positivi / paziente | F1 lesioni |
| --- | --- | --- | --- | --- |
| senza postprocessing | 0.580 (0.418–0.736) | 27/68 | 1.7 | 0.491 |
| con postprocessing | 0.584 (0.420–0.742) | 27/68 | 0.1 | 0.560 |

## Per fascia di volume (con postprocessing)

| fascia | lesioni | rilevate |
| --- | --- | --- |
| 1–10 mm³ | 34 | 7 (21%) |
| 10–50 | 12 | 3 (25%) |
| 50–200 | 9 | 6 (67%) |
| > 200 | 13 | 11 (85%) |

## Percentuale di cervello lesionato: predetta contro vera

Correlazione di Pearson 0.991, per ranghi (Spearman) 0.923; differenza media (predetta − vera) -0.11 punti percentuali, limiti di accordo ±1.21.

| paziente | strato | % vera | % predetta | Dice | lesioni rilevate | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- |
| FSM_001 | altri centri | 0.01 | 0.01 | 0.458 | 1/2 | 0 |
| FSM_028 | Stella Maris 1.5 T | 12.12 | 11.19 | 0.892 | 2/9 | 1 |
| FSM_032 | Stella Maris 3 T | 9.17 | 9.21 | 0.859 | 1/3 | 0 |
| FSM_033 | Stella Maris 1.5 T | 0.05 | 0.00 | 0.000 | 0/1 | 0 |
| FSM_035 | Stella Maris 1.5 T | 3.97 | 4.89 | 0.854 | 4/7 | 0 |
| FSM_036 | Stella Maris 1.5 T | 9.38 | 8.00 | 0.870 | 4/21 | 0 |
| FSM_039 | Stella Maris 3 T | 0.09 | 0.05 | 0.517 | 1/2 | 0 |
| FSM_040 | Stella Maris 3 T | 4.04 | 4.62 | 0.891 | 3/6 | 0 |
| FSM_041 | Stella Maris 3 T | 0.14 | 0.01 | 0.160 | 2/5 | 0 |
| FSM_045 | Stella Maris 3 T | 0.86 | 0.30 | 0.439 | 1/1 | 0 |
| FSM_053 | Stella Maris 3 T | 0.04 | 0.09 | 0.460 | 6/6 | 0 |
| FSM_056 | Stella Maris 3 T | 0.88 | 1.06 | 0.613 | 2/5 | 0 |

## Per strato di acquisizione

| strato | pazienti | Dice | lesioni rilevate | FP / paziente |
| --- | --- | --- | --- | --- |
| Stella Maris 1.5 T | 4 | 0.654 | 10/38 | 0.2 |
| Stella Maris 3 T | 7 | 0.563 | 16/28 | 0.0 |
| altri centri | 1 | 0.458 | 1/2 | 0.0 |
