# Valutazione di `Dataset506_AINCP_FINALE__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py valuta`, **non modificare a mano**. Predizioni **test** (/home/ccantini/lesion_detection_project/nnUNet/nnUNet_results/Dataset506_AINCP_FINALE/nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres/test_insieme), 5 pazienti, nella griglia T1 nativa. Postprocessing: soglia 0.6, volume minimo 5 mm³, maschera del cervello no.

| | Dice (IC 95%) | lesioni rilevate | falsi positivi / paziente | F1 lesioni |
| --- | --- | --- | --- | --- |
| senza postprocessing | 0.667 (0.572–0.766) | 21/40 | 2.2 | 0.542 |
| con postprocessing | 0.655 (0.552–0.761) | 18/40 | 1.2 | 0.523 |

## Per fascia di volume (con postprocessing)

| fascia | lesioni | rilevate |
| --- | --- | --- |
| 1–10 mm³ | 23 | 9 (39%) |
| 10–50 | 3 | 1 (33%) |
| 50–200 | 6 | 2 (33%) |
| > 200 | 8 | 6 (75%) |

## Percentuale di cervello lesionato: predetta contro vera

Correlazione di Pearson 0.983, per ranghi (Spearman) 0.900; differenza media (predetta − vera) -1.30 punti percentuali, limiti di accordo ±5.97.

| paziente | strato | % vera | % predetta | Dice | lesioni rilevate | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- |
| FSM_030 | Stella Maris 1.5 T | 0.45 | 0.21 | 0.478 | 1/11 | 0 |
| FSM_046 | Stella Maris 3 T | 0.03 | 0.04 | 0.629 | 3/3 | 2 |
| FSM_049 | altri centri | 16.21 | 9.49 | 0.644 | 8/10 | 4 |
| FSM_052 | Stella Maris 3 T | 0.37 | 0.21 | 0.684 | 2/8 | 0 |
| FSM_054 | Stella Maris 3 T | 2.52 | 3.13 | 0.840 | 4/8 | 0 |

## Per strato di acquisizione

| strato | pazienti | Dice | lesioni rilevate | FP / paziente |
| --- | --- | --- | --- | --- |
| Stella Maris 1.5 T | 1 | 0.478 | 1/11 | 0.0 |
| Stella Maris 3 T | 3 | 0.718 | 9/19 | 0.7 |
| altri centri | 1 | 0.644 | 8/10 | 4.0 |
