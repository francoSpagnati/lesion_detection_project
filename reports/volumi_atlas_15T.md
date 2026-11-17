# Volume delle lesioni ATLAS: 1.5 T contro 3 T (insiemi del modello 521)

Prodotto da `scripts/volumi_atlas_15T.py`, **non modificare a mano**. Etichette di Dataset502 (`labelsTr`, voxel da 1 mm³); connettività 26. Il Dice di 521 sui due insiemi è in `reports/inferenza_incrociata_15T.md`.

| insieme | casi | volume per caso: Q1 / mediana / Q3 (mm³) | minimo – massimo (mm³) | componenti | volume per componente: Q1 / mediana / Q3 (mm³) |
| --- | --- | --- | --- | --- | --- |
| 1.5 T, mai visti da 521 | 45 | 2 291 / 13 172 / 80 554 | 162 – 373 955 | 80 | 260 / 1 340 / 26 223 |
| 3 T, validazione di 521 | 103 | 1 158 / 5 232 / 27 210 | 23 – 230 745 | 230 | 114 / 569 / 2 732 |
