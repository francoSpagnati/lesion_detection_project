# Postprocessing scelto per `Dataset506_AINCP_FINALE__nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py scegli`, **non modificare a mano**. Predizioni fuori-fold di 12 pazienti, nel nativo.

| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita' | precisione | Dice | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default nnU-Net | 0.5 | 0 | no | 0.491 | 0.397 | 0.643 | 0.580 | 20 |
| **scelto** | 0.3 | 50 | no | 0.560 | 0.397 | 0.947 | 0.584 | 1 |

Le dieci combinazioni migliori per F1:

| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 50 | no | 0.560 | 0.397 | 0.947 | 0.584 | 1 |
| 0.3 | 50 | si | 0.560 | 0.397 | 0.947 | 0.584 | 1 |
| 0.3 | 10 | no | 0.542 | 0.412 | 0.794 | 0.584 | 7 |
| 0.3 | 10 | si | 0.542 | 0.412 | 0.794 | 0.584 | 7 |
| 0.4 | 20 | no | 0.532 | 0.382 | 0.875 | 0.586 | 3 |
| 0.4 | 20 | si | 0.532 | 0.382 | 0.875 | 0.586 | 3 |
| 0.3 | 5 | no | 0.532 | 0.412 | 0.750 | 0.585 | 10 |
| 0.3 | 5 | si | 0.532 | 0.412 | 0.750 | 0.585 | 10 |
| 0.4 | 50 | no | 0.528 | 0.368 | 0.938 | 0.580 | 1 |
| 0.4 | 50 | si | 0.528 | 0.368 | 0.938 | 0.580 | 1 |
