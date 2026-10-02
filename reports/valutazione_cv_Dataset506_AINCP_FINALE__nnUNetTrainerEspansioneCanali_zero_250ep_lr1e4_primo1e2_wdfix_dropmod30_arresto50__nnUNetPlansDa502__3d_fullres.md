# Valutazione di `Dataset506_AINCP_FINALE__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py valuta`, **non modificare a mano**. Predizioni fuori-fold (cross-validation), 12 pazienti, nella griglia T1 nativa. Postprocessing: soglia 0.6, volume minimo 5 mm³, maschera del cervello no.

| | Dice (IC 95%) | lesioni rilevate | falsi positivi / paziente | F1 lesioni |
| --- | --- | --- | --- | --- |
| senza postprocessing | 0.562 (0.390–0.729) | 26/68 | 0.8 | 0.502 |
| con postprocessing | 0.563 (0.391–0.732) | 26/68 | 0.4 | 0.520 |

## Per fascia di volume (con postprocessing)

| fascia | lesioni | rilevate |
| --- | --- | --- |
| 1–10 mm³ | 34 | 5 (15%) |
| 10–50 | 12 | 3 (25%) |
| 50–200 | 9 | 6 (67%) |
| > 200 | 13 | 12 (92%) |

## Percentuale di cervello lesionato: predetta contro vera

Correlazione di Pearson 0.979, per ranghi (Spearman) 0.902; differenza media (predetta − vera) -0.08 punti percentuali, limiti di accordo ±1.78.

| paziente | strato | % vera | % predetta | Dice | lesioni rilevate | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- |
| FSM_001 | altri centri | 0.01 | 0.01 | 0.332 | 1/2 | 2 |
| FSM_028 | Stella Maris 1.5 T | 12.12 | 9.64 | 0.865 | 1/9 | 1 |
| FSM_032 | Stella Maris 3 T | 9.17 | 10.09 | 0.886 | 1/3 | 0 |
| FSM_033 | Stella Maris 1.5 T | 0.05 | 0.00 | 0.000 | 0/1 | 0 |
| FSM_035 | Stella Maris 1.5 T | 3.97 | 4.49 | 0.849 | 4/7 | 0 |
| FSM_036 | Stella Maris 1.5 T | 9.38 | 9.01 | 0.889 | 3/21 | 0 |
| FSM_039 | Stella Maris 3 T | 0.09 | 0.07 | 0.558 | 1/2 | 0 |
| FSM_040 | Stella Maris 3 T | 4.04 | 5.15 | 0.869 | 3/6 | 0 |
| FSM_041 | Stella Maris 3 T | 0.14 | 0.01 | 0.063 | 2/5 | 1 |
| FSM_045 | Stella Maris 3 T | 0.86 | 0.17 | 0.299 | 1/1 | 0 |
| FSM_053 | Stella Maris 3 T | 0.04 | 0.06 | 0.551 | 6/6 | 1 |
| FSM_056 | Stella Maris 3 T | 0.88 | 1.05 | 0.599 | 3/5 | 0 |

## Per strato di acquisizione

| strato | pazienti | Dice | lesioni rilevate | FP / paziente |
| --- | --- | --- | --- | --- |
| Stella Maris 1.5 T | 4 | 0.651 | 8/38 | 0.2 |
| Stella Maris 3 T | 7 | 0.546 | 17/28 | 0.3 |
| altri centri | 1 | 0.332 | 1/2 | 2.0 |
