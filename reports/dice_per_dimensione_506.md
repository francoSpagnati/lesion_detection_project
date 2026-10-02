# Dice per singola lesione, per fascia di volume (Dataset506)

Prodotto da `scripts/dice_per_dimensione_506.py`, **non modificare a mano**. CC-Dice: Dice di ogni lesione annotata dentro la sua regione di Voronoi, 0 se mancata; ogni ricetta con il suo postprocessing congelato. "Dice delle rilevate" esclude le mancate: dice quanto bene e' segmentata una lesione *quando* viene trovata.

## Cross-validation, fuori-fold (12 pazienti)

| fascia | lesioni | Z: Dice medio (mediana) · rilevate · Dice delle rilevate | A: Dice medio (mediana) · rilevate · Dice delle rilevate | D: Dice medio (mediana) · rilevate · Dice delle rilevate |
| --- | --- | --- | --- | --- |
| 1–10 mm³ | 34 | 0.04 (0.00) · 7/34 · 0.18 | 0.03 (0.00) · 5/34 · 0.24 | 0.03 (0.00) · 5/34 · 0.21 |
| 10–50 mm³ | 12 | 0.09 (0.00) · 3/12 · 0.38 | 0.11 (0.00) · 3/12 · 0.46 | 0.09 (0.00) · 3/12 · 0.36 |
| 50–200 mm³ | 9 | 0.32 (0.27) · 6/9 · 0.47 | 0.23 (0.17) · 6/9 · 0.34 | 0.28 (0.33) · 6/9 · 0.42 |
| 200–1 000 mm³ | 3 | 0.21 (0.00) · 1/3 · 0.62 | 0.37 (0.44) · 2/3 · 0.56 | 0.20 (0.00) · 1/3 · 0.61 |
| 1 000–10 000 mm³ | 4 | 0.44 (0.48) · 4/4 · 0.44 | 0.38 (0.43) · 4/4 · 0.38 | 0.40 (0.44) · 4/4 · 0.40 |
| > 10 000 mm³ | 6 | 0.83 (0.87) · 6/6 · 0.83 | 0.83 (0.87) · 6/6 · 0.83 | 0.83 (0.88) · 6/6 · 0.83 |

## Test (5 pazienti)

| fascia | lesioni | Z: Dice medio (mediana) · rilevate · Dice delle rilevate | A: Dice medio (mediana) · rilevate · Dice delle rilevate | D: Dice medio (mediana) · rilevate · Dice delle rilevate |
| --- | --- | --- | --- | --- |
| 1–10 mm³ | 23 | 0.09 (0.07) · 12/23 · 0.16 | 0.08 (0.00) · 9/23 · 0.20 | 0.08 (0.00) · 10/23 · 0.18 |
| 10–50 mm³ | 3 | 0.08 (0.00) · 1/3 · 0.24 | 0.12 (0.00) · 1/3 · 0.36 | 0.38 (0.47) · 2/3 · 0.57 |
| 50–200 mm³ | 6 | 0.02 (0.00) · 2/6 · 0.07 | 0.15 (0.00) · 2/6 · 0.44 | 0.36 (0.44) · 4/6 · 0.54 |
| 200–1 000 mm³ | 4 | 0.16 (0.08) · 2/4 · 0.31 | 0.24 (0.11) · 2/4 · 0.47 | 0.51 (0.54) · 4/4 · 0.51 |
| 1 000–10 000 mm³ | 2 | 0.75 (0.75) · 2/2 · 0.75 | 0.66 (0.66) · 2/2 · 0.66 | 0.69 (0.69) · 2/2 · 0.69 |
| > 10 000 mm³ | 2 | 0.84 (0.84) · 2/2 · 0.84 | 0.74 (0.74) · 2/2 · 0.74 | 0.75 (0.75) · 2/2 · 0.75 |
