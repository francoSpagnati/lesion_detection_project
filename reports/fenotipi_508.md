# Ricetta F: il tipo di lesione predetto contro quello del clinico (fuori-fold, 12 pazienti)

Prodotto da `scripts/fenotipi_508.py`, **non modificare a mano**. Griglia T1 nativa; maschera dopo il postprocessing di F; tipo predetto = classe più probabile; tipo vero dalle ROI con la precedenza di 508.

## Voxel di lesione trovati: matrice di confusione (mm³; % della riga)

| vero \ predetto | gliosi | cistica | encefalomalacia |
| --- | --- | --- | --- |
| gliosi | 51 661 mm³ (79%) | 9.5 mm³ (0%) | 13 664 mm³ (21%) |
| cistica | 109 mm³ (0%) | 210 051 mm³ (95%) | 11 166 mm³ (5%) |
| encefalomalacia | 8 482 mm³ (9%) | 18 887 mm³ (20%) | 68 663 mm³ (71%) |

Voxel trovati con il tipo giusto: **86%**. Voxel trovati senza tipo vero (bordo dell'interpolazione delle ROI): 3 260 mm³.

## Dice per classe (voxel di tutti i pazienti con quella classe)

| classe | pazienti con la classe | volume vero | Dice |
| --- | --- | --- | --- |
| gliosi | 12 | 93 596 mm³ | 0.596 |
| cistica | 6 | 281 899 mm³ | 0.794 |
| encefalomalacia | 6 | 131 594 mm³ | 0.556 |

## Lesione per lesione (lesioni annotate trovate)

Tipo prevalente giusto in **28 lesioni su 29**; la regola banale «sempre gliosi» ne indovina 22.

| vero \ predetto | gliosi | cistica | encefalomalacia |
| --- | --- | --- | --- |
| gliosi | 22 | 0 | 0 |
| cistica | 1 | 4 | 0 |
| encefalomalacia | 0 | 0 | 2 |
