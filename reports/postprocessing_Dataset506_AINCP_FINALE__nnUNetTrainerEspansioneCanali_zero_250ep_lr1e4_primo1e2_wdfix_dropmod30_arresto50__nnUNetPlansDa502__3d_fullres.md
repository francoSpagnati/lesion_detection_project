# Postprocessing scelto per `Dataset506_AINCP_FINALE__nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres`

Prodotto da `scripts/postprocessing_aincp.py scegli`, **non modificare a mano**. Predizioni fuori-fold di 12 pazienti, nel nativo.

| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita' | precisione | Dice | falsi positivi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default nnU-Net | 0.5 | 0 | no | 0.502 | 0.382 | 0.730 | 0.562 | 10 |
| **scelto** | 0.6 | 5 | no | 0.520 | 0.382 | 0.815 | 0.563 | 5 |

Le dieci combinazioni migliori per F1:

| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 5 | no | 0.522 | 0.382 | 0.821 | 0.562 | 5 |
| 0.5 | 5 | si | 0.522 | 0.382 | 0.821 | 0.562 | 5 |
| 0.6 | 5 | no | 0.520 | 0.382 | 0.815 | 0.563 | 5 |
| 0.6 | 5 | si | 0.520 | 0.382 | 0.815 | 0.563 | 5 |
| 0.3 | 10 | no | 0.516 | 0.382 | 0.792 | 0.553 | 5 |
| 0.3 | 10 | si | 0.516 | 0.382 | 0.792 | 0.553 | 5 |
| 0.3 | 20 | no | 0.513 | 0.368 | 0.850 | 0.552 | 3 |
| 0.3 | 20 | si | 0.513 | 0.368 | 0.850 | 0.552 | 3 |
| 0.4 | 5 | no | 0.512 | 0.382 | 0.774 | 0.558 | 7 |
| 0.4 | 5 | si | 0.512 | 0.382 | 0.774 | 0.558 | 7 |
