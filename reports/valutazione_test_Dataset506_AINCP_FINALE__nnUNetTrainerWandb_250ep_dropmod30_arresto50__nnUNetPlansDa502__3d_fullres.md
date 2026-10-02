# Valutazione di `Dataset506_AINCP_FINALE__nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py valuta`, **non modificare a mano**. Predizioni **test** (/home/ccantini/lesion_detection_project/nnUNet/nnUNet_results/Dataset506_AINCP_FINALE/nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres/test_insieme), 5 pazienti, nella griglia T1 nativa. Postprocessing: soglia 0.3, volume minimo 50 mm³, maschera del cervello no.

| | Dice (IC 95%) | lesioni rilevate | falsi positivi / paziente | F1 lesioni |
| --- | --- | --- | --- | --- |
| senza postprocessing | 0.615 (0.347–0.815) | 24/40 | 8.2 | 0.394 |
| con postprocessing | 0.622 (0.341–0.825) | 21/40 | 0.8 | 0.605 |

## Per fascia di volume (con postprocessing)

| fascia | lesioni | rilevate |
| --- | --- | --- |
| 1–10 mm³ | 23 | 12 (52%) |
| 10–50 | 3 | 1 (33%) |
| 50–200 | 6 | 2 (33%) |
| > 200 | 8 | 6 (75%) |

## Percentuale di cervello lesionato: predetta contro vera

Correlazione di Pearson 0.996, per ranghi (Spearman) 0.600; differenza media (predetta − vera) -0.02 punti percentuali, limiti di accordo ±1.81.

| paziente | strato | % vera | % predetta | Dice | lesioni rilevate | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- |
| FSM_030 | Stella Maris 1.5 T | 0.45 | 0.28 | 0.573 | 1/11 | 0 |
| FSM_046 | Stella Maris 3 T | 0.03 | 0.70 | 0.090 | 3/3 | 2 |
| FSM_049 | altri centri | 16.21 | 14.77 | 0.862 | 8/10 | 2 |
| FSM_052 | Stella Maris 3 T | 0.37 | 0.30 | 0.770 | 3/8 | 0 |
| FSM_054 | Stella Maris 3 T | 2.52 | 3.45 | 0.817 | 6/8 | 0 |

## Per strato di acquisizione

| strato | pazienti | Dice | lesioni rilevate | FP / paziente |
| --- | --- | --- | --- | --- |
| Stella Maris 1.5 T | 1 | 0.573 | 1/11 | 0.0 |
| Stella Maris 3 T | 3 | 0.559 | 12/19 | 0.7 |
| altri centri | 1 | 0.862 | 8/10 | 2.0 |
