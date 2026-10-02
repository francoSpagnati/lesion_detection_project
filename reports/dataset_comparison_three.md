# Confronto fra i tre dataset

Generato da `scripts/compare_datasets.py` dai manifest per caso. Read-only.

## Che problema pone ciascuno

| | WMH | ATLAS | DBB |
| --- | --- | --- | --- |
| dataset nnU-Net | `Dataset501_WMH` | `Dataset502_ATLAS` | `Dataset503_DBB` |
| compito | lesioni della sostanza bianca (iperintensita) | lesioni da ictus (post-stroke) | segmentazione di 6 tessuti (non lesioni) |
| classi | binaria (wmh); label 2 "altra patologia" esclusa | binaria (lesion) | 6 tessuti: CSF, GM, WM, GM profonda, tronco, cervelletto |
| canale usato | FLAIR (T1 co-registrato disponibile) | T1 | T1 |
| popolazione | adulti, 4 siti / scanner diversi | adulti, 33 coorti | pediatrica; training su sani, test su malformazioni |
| casi con immagine + label | 99 | 655 | 244 |
| immagini senza label | 0 | 300 | 0 |

## Geometria

| | WMH | ATLAS | DBB |
| --- | --- | --- | --- |
| spazio | nativo dello scanner | MNI152NLin2009aSym | MNI152 (213 casi) + nativo (31 casi) |
| orientamento | `LPS:99` | `RAS:655` | `LAS:232, RAS:12` |
| spacing mediana (mm) | 1.2 x 0.9766 x 3 | 1 x 1 x 1 | 1 x 1 x 1 |
| spacing distinti | 7 | 1 | 7 |
| shape mediana | 132x256x83 | 197x233x189 | 182x218x182 |
| anisotropia | 3.07 | 1.0 | 1.0 |
| volume voxel (mm3) | 3.516 | 1.0 | 1.0 |
| campo di vista (mm³) | 9 861 000 | 8 675 000 | 7 221 000 |

## Carico di foreground

Per WMH e ATLAS e' il volume di lesione; per DBB e' il volume complessivo dei 6 tessuti,
cioe' quasi tutto il cervello. La differenza di **tre ordini di grandezza** nella frazione
di foreground e' la ragione per cui i tre compiti non sono lo stesso problema.

| | WMH | ATLAS | DBB |
| --- | --- | --- | --- |
| foreground mediano (mm³) | 10 930 | 5 290 | 1 478 810 |
| p10 - p90 (mm³) | 1 610 - 40 120 | 380 - 106 090 | 1 295 000 - 1 689 000 |
| frazione del campo di vista | 0.0010 | 0.0006 | 0.2034 |
