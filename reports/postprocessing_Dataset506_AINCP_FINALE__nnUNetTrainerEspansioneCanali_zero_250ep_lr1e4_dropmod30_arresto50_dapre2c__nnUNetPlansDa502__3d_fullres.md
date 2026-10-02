# Postprocessing scelto per `Dataset506_AINCP_FINALE__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_dropmod30_arresto50_dapre2c__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py scegli`, **non modificare a mano**. Predizioni fuori-fold di 12 pazienti, nel nativo.

| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita' | precisione | Dice | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default nnU-Net | 0.5 | 0 | no | 0.482 | 0.382 | 0.652 | 0.577 | 16 |
| **scelto** | 0.4 | 20 | no | 0.522 | 0.368 | 0.900 | 0.577 | 2 |

Le dieci combinazioni migliori per F1:

| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 50 | no | 0.528 | 0.368 | 0.938 | 0.571 | 1 |
| 0.3 | 50 | si | 0.528 | 0.368 | 0.938 | 0.571 | 1 |
| 0.4 | 20 | no | 0.522 | 0.368 | 0.900 | 0.577 | 2 |
| 0.4 | 20 | si | 0.522 | 0.368 | 0.900 | 0.577 | 2 |
| 0.3 | 0 | no | 0.516 | 0.397 | 0.738 | 0.577 | 11 |
| 0.3 | 0 | si | 0.516 | 0.397 | 0.738 | 0.577 | 11 |
| 0.3 | 20 | no | 0.515 | 0.368 | 0.857 | 0.574 | 3 |
| 0.3 | 20 | si | 0.515 | 0.368 | 0.857 | 0.574 | 3 |
| 0.6 | 20 | no | 0.514 | 0.353 | 0.947 | 0.576 | 1 |
| 0.6 | 20 | si | 0.514 | 0.353 | 0.947 | 0.576 | 1 |
