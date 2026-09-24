# Valutazione di `Dataset505_AINCP_MNI__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50_da543__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py valuta`, **non modificare a mano**. Predizioni fuori-fold (cross-validation), 10 pazienti, nella griglia T1 nativa. Postprocessing: soglia 0.4, volume minimo 20 mm³, maschera del cervello no.

| | Dice (IC 95%) | lesioni rilevate | falsi positivi / paziente | F1 lesioni |
| --- | --- | --- | --- | --- |
| senza postprocessing | 0.595 (0.443–0.741) | 27/47 | 2.2 | 0.610 |
| con postprocessing | 0.593 (0.440–0.741) | 28/47 | 0.6 | 0.680 |

## Per fascia di volume (con postprocessing)

| fascia | lesioni | rilevate |
| --- | --- | --- |
| 1–10 mm³ | 21 | 8 (38%) |
| 10–50 | 7 | 2 (29%) |
| 50–200 | 7 | 6 (86%) |
| > 200 | 12 | 12 (100%) |

## Percentuale di cervello lesionato: predetta contro vera

Correlazione di Pearson 0.989, per ranghi (Spearman) 0.903; differenza media (predetta − vera) -0.12 punti percentuali, limiti di accordo ±0.92.

| paziente | strato | % vera | % predetta | Dice | lesioni rilevate | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- |
| FSM_032 | — | 9.17 | 8.10 | 0.883 | 1/3 | 0 |
| FSM_039 | — | 0.09 | 0.11 | 0.567 | 1/2 | 0 |
| FSM_040 | — | 4.04 | 3.85 | 0.891 | 2/6 | 0 |
| FSM_041 | — | 0.14 | 0.02 | 0.098 | 2/5 | 2 |
| FSM_045 | — | 0.86 | 0.22 | 0.347 | 1/1 | 0 |
| FSM_046 | — | 0.03 | 0.08 | 0.499 | 3/3 | 3 |
| FSM_052 | — | 0.37 | 0.26 | 0.731 | 4/8 | 0 |
| FSM_053 | — | 0.04 | 0.12 | 0.468 | 6/6 | 0 |
| FSM_054 | — | 2.52 | 3.25 | 0.830 | 5/8 | 1 |
| FSM_056 | — | 0.88 | 0.94 | 0.611 | 3/5 | 0 |
