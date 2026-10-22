# Postprocessing scelto per `Dataset506_AINCP_FINALE__nnUNetTrainerBlob_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py scegli`, **non modificare a mano**. Predizioni fuori-fold di 12 pazienti, nel nativo.

| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita' | precisione | Dice | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default nnU-Net | 0.5 | 0 | no | 0.516 | 0.382 | 0.793 | 0.593 | 12 |
| **scelto** | 0.3 | 5 | no | 0.532 | 0.397 | 0.805 | 0.600 | 8 |

Le dieci combinazioni migliori per F1:

| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 5 | no | 0.532 | 0.397 | 0.805 | 0.600 | 8 |
| 0.3 | 5 | si | 0.532 | 0.397 | 0.805 | 0.600 | 8 |
| 0.3 | 20 | no | 0.531 | 0.368 | 0.955 | 0.598 | 1 |
| 0.3 | 20 | si | 0.531 | 0.368 | 0.955 | 0.598 | 1 |
| 0.4 | 5 | no | 0.527 | 0.382 | 0.850 | 0.598 | 6 |
| 0.4 | 5 | si | 0.527 | 0.382 | 0.850 | 0.598 | 6 |
| 0.3 | 10 | no | 0.519 | 0.382 | 0.806 | 0.599 | 6 |
| 0.3 | 10 | si | 0.519 | 0.382 | 0.806 | 0.599 | 6 |
| 0.5 | 0 | no | 0.516 | 0.382 | 0.793 | 0.593 | 12 |
| 0.5 | 0 | si | 0.516 | 0.382 | 0.793 | 0.593 | 12 |
