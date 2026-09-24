# Postprocessing scelto per `Dataset505_AINCP_MNI__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py scegli`, **non modificare a mano**. Predizioni fuori-fold di 10 pazienti, nel nativo.

| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita' | precisione | Dice | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default nnU-Net | 0.5 | 0 | no | 0.607 | 0.532 | 0.707 | 0.569 | 12 |
| **scelto** | 0.3 | 50 | no | 0.649 | 0.511 | 0.889 | 0.568 | 2 |

Le dieci combinazioni migliori per F1:

| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 50 | no | 0.649 | 0.511 | 0.889 | 0.568 | 2 |
| 0.3 | 50 | si | 0.649 | 0.511 | 0.889 | 0.568 | 2 |
| 0.7 | 20 | no | 0.645 | 0.511 | 0.875 | 0.560 | 3 |
| 0.7 | 20 | si | 0.645 | 0.511 | 0.875 | 0.560 | 3 |
| 0.6 | 0 | no | 0.641 | 0.532 | 0.805 | 0.567 | 8 |
| 0.6 | 0 | si | 0.641 | 0.532 | 0.805 | 0.567 | 8 |
| 0.7 | 10 | no | 0.632 | 0.511 | 0.828 | 0.560 | 5 |
| 0.7 | 10 | si | 0.632 | 0.511 | 0.828 | 0.560 | 5 |
| 0.5 | 20 | no | 0.631 | 0.511 | 0.826 | 0.569 | 4 |
| 0.5 | 20 | si | 0.631 | 0.511 | 0.826 | 0.569 | 4 |
