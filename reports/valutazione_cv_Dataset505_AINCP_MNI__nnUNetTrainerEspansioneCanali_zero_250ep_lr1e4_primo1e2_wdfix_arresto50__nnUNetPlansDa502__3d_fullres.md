# Valutazione di `Dataset505_AINCP_MNI__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py valuta`, **non modificare a mano**. Predizioni fuori-fold (cross-validation), 10 pazienti, nella griglia T1 nativa. Postprocessing: soglia 0.3, volume minimo 50 mm³, maschera del cervello no.

| | Dice (IC 95%) | lesioni rilevate | falsi positivi / paziente | F1 lesioni |
| --- | --- | --- | --- | --- |
| senza postprocessing | 0.569 (0.421–0.723) | 25/47 | 1.2 | 0.607 |
| con postprocessing | 0.568 (0.415–0.727) | 24/47 | 0.2 | 0.649 |

## Per fascia di volume (con postprocessing)

| fascia | lesioni | rilevate |
| --- | --- | --- |
| 1–10 mm³ | 21 | 4 (19%) |
| 10–50 | 7 | 2 (29%) |
| 50–200 | 7 | 6 (86%) |
| > 200 | 12 | 12 (100%) |

## Percentuale di cervello lesionato: predetta contro vera

Correlazione di Pearson 0.993, per ranghi (Spearman) 0.879; differenza media (predetta − vera) +0.03 punti percentuali, limiti di accordo ±0.70.

| paziente | strato | % vera | % predetta | Dice | lesioni rilevate | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- |
| FSM_032 | — | 9.17 | 9.08 | 0.900 | 1/3 | 0 |
| FSM_039 | — | 0.09 | 0.12 | 0.479 | 1/2 | 0 |
| FSM_040 | — | 4.04 | 4.02 | 0.907 | 1/6 | 0 |
| FSM_041 | — | 0.14 | 0.01 | 0.149 | 2/5 | 0 |
| FSM_045 | — | 0.86 | 0.26 | 0.388 | 1/1 | 0 |
| FSM_046 | — | 0.03 | 0.15 | 0.362 | 3/3 | 2 |
| FSM_052 | — | 0.37 | 0.25 | 0.715 | 2/8 | 0 |
| FSM_053 | — | 0.04 | 0.12 | 0.359 | 6/6 | 0 |
| FSM_054 | — | 2.52 | 3.38 | 0.822 | 4/8 | 0 |
| FSM_056 | — | 0.88 | 1.03 | 0.597 | 3/5 | 0 |
