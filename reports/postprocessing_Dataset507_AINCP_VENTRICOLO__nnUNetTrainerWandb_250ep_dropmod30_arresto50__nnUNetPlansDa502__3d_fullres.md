# Postprocessing scelto per `Dataset507_AINCP_VENTRICOLO__nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py scegli`, **non modificare a mano**. Predizioni fuori-fold di 12 pazienti, nel nativo.

| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita' | precisione | Dice | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default nnU-Net | 0.5 | 0 | no | 0.437 | 0.368 | 0.538 | 0.569 | 30 |
| **scelto** | 0.3 | 20 | si | 0.517 | 0.382 | 0.800 | 0.584 | 5 |

Le dieci combinazioni migliori per F1:

| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 20 | si | 0.523 | 0.368 | 0.905 | 0.576 | 2 |
| 0.3 | 20 | si | 0.517 | 0.382 | 0.800 | 0.584 | 5 |
| 0.4 | 20 | no | 0.516 | 0.368 | 0.864 | 0.575 | 3 |
| 0.5 | 10 | si | 0.511 | 0.368 | 0.840 | 0.569 | 4 |
| 0.3 | 20 | no | 0.511 | 0.382 | 0.769 | 0.583 | 6 |
| 0.5 | 20 | si | 0.507 | 0.353 | 0.900 | 0.568 | 2 |
| 0.5 | 10 | no | 0.505 | 0.368 | 0.808 | 0.569 | 5 |
| 0.5 | 20 | no | 0.500 | 0.353 | 0.857 | 0.567 | 3 |
| 0.4 | 10 | si | 0.499 | 0.368 | 0.778 | 0.578 | 6 |
| 0.6 | 10 | si | 0.497 | 0.353 | 0.840 | 0.557 | 4 |
