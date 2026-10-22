# Postprocessing scelto per `Dataset508_AINCP_FENOTIPI__nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py scegli`, **non modificare a mano**. Predizioni fuori-fold di 12 pazienti, nel nativo.

| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita' | precisione | Dice | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default nnU-Net | 0.5 | 0 | no | 0.387 | 0.456 | 0.336 | 0.578 | 81 |
| **scelto** | 0.5 | 20 | si | 0.511 | 0.426 | 0.636 | 0.582 | 12 |

Le dieci combinazioni migliori per F1:

| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 50 | si | 0.520 | 0.426 | 0.667 | 0.573 | 9 |
| 0.5 | 20 | si | 0.511 | 0.426 | 0.636 | 0.582 | 12 |
| 0.7 | 10 | si | 0.505 | 0.412 | 0.652 | 0.576 | 16 |
| 0.6 | 20 | si | 0.504 | 0.426 | 0.615 | 0.580 | 15 |
| 0.4 | 50 | si | 0.502 | 0.397 | 0.682 | 0.575 | 7 |
| 0.4 | 20 | si | 0.501 | 0.426 | 0.606 | 0.580 | 13 |
| 0.6 | 10 | si | 0.499 | 0.426 | 0.600 | 0.582 | 20 |
| 0.7 | 20 | si | 0.495 | 0.382 | 0.700 | 0.575 | 9 |
| 0.5 | 10 | si | 0.492 | 0.426 | 0.581 | 0.584 | 18 |
| 0.5 | 100 | si | 0.487 | 0.368 | 0.722 | 0.573 | 5 |
