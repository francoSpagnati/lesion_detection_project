# Postprocessing scelto per `Dataset505_AINCP_MNI__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50_da543__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py scegli`, **non modificare a mano**. Predizioni fuori-fold di 10 pazienti, nel nativo.

| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita' | precisione | Dice | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default nnU-Net | 0.5 | 0 | no | 0.610 | 0.574 | 0.651 | 0.595 | 22 |
| **scelto** | 0.4 | 20 | no | 0.680 | 0.596 | 0.793 | 0.593 | 6 |

Le dieci combinazioni migliori per F1:

| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 50 | no | 0.683 | 0.574 | 0.842 | 0.582 | 3 |
| 0.3 | 50 | si | 0.683 | 0.574 | 0.842 | 0.582 | 3 |
| 0.3 | 10 | no | 0.683 | 0.617 | 0.765 | 0.588 | 8 |
| 0.3 | 10 | si | 0.683 | 0.617 | 0.765 | 0.588 | 8 |
| 0.3 | 20 | no | 0.681 | 0.617 | 0.759 | 0.587 | 7 |
| 0.3 | 20 | si | 0.681 | 0.617 | 0.759 | 0.587 | 7 |
| 0.4 | 20 | no | 0.680 | 0.596 | 0.793 | 0.593 | 6 |
| 0.4 | 20 | si | 0.680 | 0.596 | 0.793 | 0.593 | 6 |
| 0.4 | 10 | no | 0.680 | 0.617 | 0.757 | 0.592 | 9 |
| 0.4 | 10 | si | 0.680 | 0.617 | 0.757 | 0.592 | 9 |
