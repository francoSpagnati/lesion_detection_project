# Valutazione di `Dataset506_AINCP_FINALE__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_dropmod30_arresto50_dapre2c__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py valuta`, **non modificare a mano**. Predizioni fuori-fold (cross-validation), 12 pazienti, nella griglia T1 nativa. Postprocessing: soglia 0.4, volume minimo 20 mm³, maschera del cervello no.

| | Dice (IC 95%) | lesioni rilevate | falsi positivi / paziente | F1 lesioni |
| --- | --- | --- | --- | --- |
| senza postprocessing | 0.577 (0.408–0.737) | 26/68 | 1.3 | 0.482 |
| con postprocessing | 0.577 (0.407–0.740) | 25/68 | 0.2 | 0.522 |

## Per fascia di volume (con postprocessing)

| fascia | lesioni | rilevate |
| --- | --- | --- |
| 1–10 mm³ | 34 | 5 (15%) |
| 10–50 | 12 | 3 (25%) |
| 50–200 | 9 | 6 (67%) |
| > 200 | 13 | 11 (85%) |

## Percentuale di cervello lesionato: predetta contro vera

Correlazione di Pearson 0.993, per ranghi (Spearman) 0.909; differenza media (predetta − vera) -0.19 punti percentuali, limiti di accordo ±1.25.

| paziente | strato | % vera | % predetta | Dice | lesioni rilevate | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- |
| FSM_001 | altri centri | 0.01 | 0.02 | 0.338 | 1/2 | 2 |
| FSM_028 | Stella Maris 1.5 T | 12.12 | 10.53 | 0.898 | 1/9 | 0 |
| FSM_032 | Stella Maris 3 T | 9.17 | 8.31 | 0.866 | 1/3 | 0 |
| FSM_033 | Stella Maris 1.5 T | 0.05 | 0.00 | 0.000 | 0/1 | 0 |
| FSM_035 | Stella Maris 1.5 T | 3.97 | 4.44 | 0.857 | 5/7 | 0 |
| FSM_036 | Stella Maris 1.5 T | 9.38 | 8.81 | 0.893 | 2/21 | 0 |
| FSM_039 | Stella Maris 3 T | 0.09 | 0.07 | 0.621 | 1/2 | 0 |
| FSM_040 | Stella Maris 3 T | 4.04 | 4.87 | 0.888 | 3/6 | 0 |
| FSM_041 | Stella Maris 3 T | 0.14 | 0.01 | 0.139 | 2/5 | 0 |
| FSM_045 | Stella Maris 3 T | 0.86 | 0.24 | 0.376 | 1/1 | 0 |
| FSM_053 | Stella Maris 3 T | 0.04 | 0.10 | 0.470 | 6/6 | 0 |
| FSM_056 | Stella Maris 3 T | 0.88 | 1.08 | 0.578 | 2/5 | 0 |

## Per strato di acquisizione

| strato | pazienti | Dice | lesioni rilevate | FP / paziente |
| --- | --- | --- | --- | --- |
| Stella Maris 1.5 T | 4 | 0.662 | 8/38 | 0.0 |
| Stella Maris 3 T | 7 | 0.563 | 16/28 | 0.0 |
| altri centri | 1 | 0.338 | 1/2 | 2.0 |
