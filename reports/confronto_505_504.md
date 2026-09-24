# Stadio 3: template (505) contro nativo (504)

Prodotto da `scripts/confronto_505_504.py`, **non modificare a mano**.

Tutto nella griglia T1 nativa, contro gli stessi riferimenti, predizioni fuori-fold, soglia 1.001 mm³ su lesioni e falsi positivi. Differenze appaiate contro «504 ricetta (nativo)», Wilcoxon (con dieci pazienti p < 0.05 richiede un effetto quasi sempre nello stesso verso).

| variante | Dice medio | Δ Dice (p) | lesioni rilevate | falsi positivi | Δ FP (p) |
| --- | --- | --- | --- | --- | --- |
| 504 da zero | 0.581 | -0.034 (0.56) | 25/47 | 39 (3.9/paz.) | +3.0 (0.02) |
| 504 ricetta (nativo) | 0.615 | — | 21/47 | 9 (0.9/paz.) | — |
| 505 A: ricetta nel template | 0.563 | -0.052 (0.49) | 23/47 | 15 (1.5/paz.) | +0.6 (0.22) |
| 505 B: da 542 (sequenza) | 0.600 | -0.015 (0.77) | 25/47 | 32 (3.2/paz.) | +2.3 (0.03) |

## Dice per paziente

| paziente | 504 da zero | 504 ricetta (nativo) | 505 A: ricetta nel template | 505 B: da 542 (sequenza) |
| --- | --- | --- | --- | --- |
| FSM_032 | 0.807 | 0.886 | 0.893 | 0.851 |
| FSM_039 | 0.360 | 0.390 | 0.480 | 0.572 |
| FSM_040 | 0.825 | 0.909 | 0.892 | 0.871 |
| FSM_041 | 0.044 | 0.089 | 0.122 | 0.142 |
| FSM_045 | 0.353 | 0.613 | 0.351 | 0.432 |
| FSM_046 | 0.448 | 0.639 | 0.437 | 0.479 |
| FSM_052 | 0.767 | 0.605 | 0.645 | 0.693 |
| FSM_053 | 0.643 | 0.563 | 0.391 | 0.514 |
| FSM_054 | 0.880 | 0.829 | 0.834 | 0.834 |
| FSM_056 | 0.686 | 0.626 | 0.587 | 0.615 |
