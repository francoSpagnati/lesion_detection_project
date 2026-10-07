# Rapporto d'intensita' lesione / cervello per fenotipo (17 pazienti AInCP)

Prodotto da `scripts/contrasto_fenotipi_aincp.py`, **non modificare a mano**. Media dentro la ROI divisa per la media nel cervello fuori dalla lesione, griglia T1 nativa dopo N4. 1 = stessa intensita' del tessuto attorno; sotto 1 piu' scura, sopra 1 piu' chiara. Su ATLAS (T1): 0.62 nei casi risolti, 1.07 nei falliti (`failure_analysis_502.md`).

## Mediana (min–max) sui pazienti che hanno il fenotipo

| fenotipo | pazienti | T1 | FLAIR |
| --- | --- | --- | --- |
| gliosis | 17 | 0.96 (0.73–1.19) | 1.50 (1.22–1.79) |
| cystic lesion | 8 | 0.35 (0.22–0.40) | 0.16 (0.08–0.41) |
| encephalomalacia | 7 | 0.48 (0.41–0.56) | 0.72 (0.31–0.97) |

## Per paziente

| paziente | fenotipo | volume | T1 | FLAIR |
| --- | --- | --- | --- | --- |
| FSM_001 | gliosis | 166 mm³ | 0.79 | 1.48 |
| FSM_028 | gliosis | 25 498 mm³ | 0.82 | 1.25 |
| FSM_028 | cystic lesion | 120 427 mm³ | 0.30 | 0.19 |
| FSM_028 | encephalomalacia | 8 505 mm³ | 0.50 | 0.97 |
| FSM_030 | gliosis | 4 879 mm³ | 0.94 | 1.48 |
| FSM_032 | gliosis | 9 685 mm³ | 0.97 | 1.39 |
| FSM_032 | cystic lesion | 88 645 mm³ | 0.40 | 0.08 |
| FSM_032 | encephalomalacia | 29 807 mm³ | 0.47 | 0.31 |
| FSM_033 | gliosis | 667 mm³ | 1.12 | 1.22 |
| FSM_035 | gliosis | 8 178 mm³ | 0.98 | 1.47 |
| FSM_035 | cystic lesion | 24 060 mm³ | 0.22 | 0.14 |
| FSM_035 | encephalomalacia | 7 491 mm³ | 0.48 | 0.72 |
| FSM_036 | gliosis | 39 182 mm³ | 0.73 | 1.25 |
| FSM_036 | cystic lesion | 1 190 mm³ | 0.32 | 0.41 |
| FSM_036 | encephalomalacia | 71 631 mm³ | 0.44 | 0.94 |
| FSM_039 | gliosis | 1 394 mm³ | 0.94 | 1.52 |
| FSM_040 | gliosis | 3 594 mm³ | 0.99 | 1.74 |
| FSM_040 | cystic lesion | 40 311 mm³ | 0.39 | 0.10 |
| FSM_040 | encephalomalacia | 12 677 mm³ | 0.56 | 0.49 |
| FSM_041 | gliosis | 1 902 mm³ | 1.19 | 1.23 |
| FSM_045 | gliosis | 2 606 mm³ | 0.75 | 1.54 |
| FSM_045 | encephalomalacia | 6 913 mm³ | 0.51 | 0.72 |
| FSM_046 | gliosis | 454 mm³ | 1.11 | 1.79 |
| FSM_049 | gliosis | 4 822 mm³ | 0.96 | 1.69 |
| FSM_049 | cystic lesion | 191 972 mm³ | 0.38 | 0.36 |
| FSM_052 | gliosis | 4 780 mm³ | 1.04 | 1.60 |
| FSM_053 | gliosis | 582 mm³ | 1.12 | 1.50 |
| FSM_054 | gliosis | 6 684 mm³ | 0.89 | 1.79 |
| FSM_054 | cystic lesion | 26 476 mm³ | 0.33 | 0.08 |
| FSM_054 | encephalomalacia | 484 mm³ | 0.41 | 0.33 |
| FSM_056 | gliosis | 5 475 mm³ | 0.87 | 1.73 |
| FSM_056 | cystic lesion | 6 967 mm³ | 0.37 | 0.18 |
