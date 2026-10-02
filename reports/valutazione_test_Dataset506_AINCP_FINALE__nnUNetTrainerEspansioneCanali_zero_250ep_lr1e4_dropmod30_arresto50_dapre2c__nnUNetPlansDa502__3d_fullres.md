# Valutazione di `Dataset506_AINCP_FINALE__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_dropmod30_arresto50_dapre2c__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py valuta`, **non modificare a mano**. Predizioni **test** (/home/ccantini/lesion_detection_project/nnUNet/nnUNet_results/Dataset506_AINCP_FINALE/nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_dropmod30_arresto50_dapre2c__nnUNetPlansDa502__3d_fullres/test_insieme), 5 pazienti, nella griglia T1 nativa. Postprocessing: soglia 0.4, volume minimo 20 mm³, maschera del cervello no.

| | Dice (IC 95%) | lesioni rilevate | falsi positivi / paziente | F1 lesioni |
| --- | --- | --- | --- | --- |
| senza postprocessing | 0.671 (0.591–0.762) | 22/40 | 4.2 | 0.547 |
| con postprocessing | 0.672 (0.590–0.761) | 24/40 | 1.0 | 0.683 |

## Per fascia di volume (con postprocessing)

| fascia | lesioni | rilevate |
| --- | --- | --- |
| 1–10 mm³ | 23 | 10 (43%) |
| 10–50 | 3 | 2 (67%) |
| 50–200 | 6 | 4 (67%) |
| > 200 | 8 | 8 (100%) |

## Percentuale di cervello lesionato: predetta contro vera

Correlazione di Pearson 0.987, per ranghi (Spearman) 0.900; differenza media (predetta − vera) -1.24 punti percentuali, limiti di accordo ±6.01.

| paziente | strato | % vera | % predetta | Dice | lesioni rilevate | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- |
| FSM_030 | Stella Maris 1.5 T | 0.45 | 0.36 | 0.590 | 7/11 | 0 |
| FSM_046 | Stella Maris 3 T | 0.03 | 0.07 | 0.553 | 3/3 | 4 |
| FSM_049 | altri centri | 16.21 | 9.50 | 0.661 | 6/10 | 0 |
| FSM_052 | Stella Maris 3 T | 0.37 | 0.42 | 0.723 | 5/8 | 0 |
| FSM_054 | Stella Maris 3 T | 2.52 | 3.03 | 0.832 | 3/8 | 1 |

## Per strato di acquisizione

| strato | pazienti | Dice | lesioni rilevate | FP / paziente |
| --- | --- | --- | --- | --- |
| Stella Maris 1.5 T | 1 | 0.590 | 7/11 | 0.0 |
| Stella Maris 3 T | 3 | 0.703 | 11/19 | 1.7 |
| altri centri | 1 | 0.661 | 6/10 | 0.0 |
