# Postprocessing scelto per `Dataset532_MISTO_AINCP__nnUNetTrainerMisto_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py scegli`, **non modificare a mano**. Predizioni fuori-fold di 12 pazienti, nel nativo.

| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita' | precisione | Dice | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default nnU-Net | 0.5 | 0 | no | 0.460 | 0.426 | 0.500 | 0.589 | 38 |
| **scelto** | 0.4 | 50 | no | 0.542 | 0.412 | 0.793 | 0.596 | 6 |

Le dieci combinazioni migliori per F1:

| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 50 | si | 0.549 | 0.412 | 0.821 | 0.596 | 5 |
| 0.7 | 20 | si | 0.544 | 0.412 | 0.800 | 0.563 | 7 |
| 0.4 | 50 | no | 0.542 | 0.412 | 0.793 | 0.596 | 6 |
| 0.5 | 50 | no | 0.539 | 0.397 | 0.840 | 0.585 | 4 |
| 0.5 | 50 | si | 0.539 | 0.397 | 0.840 | 0.585 | 4 |
| 0.7 | 20 | no | 0.538 | 0.412 | 0.778 | 0.563 | 8 |
| 0.6 | 20 | no | 0.537 | 0.412 | 0.771 | 0.578 | 8 |
| 0.6 | 20 | si | 0.537 | 0.412 | 0.771 | 0.578 | 8 |
| 0.3 | 50 | si | 0.537 | 0.426 | 0.724 | 0.597 | 8 |
| 0.6 | 10 | no | 0.535 | 0.426 | 0.717 | 0.578 | 13 |
