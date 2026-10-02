# Ensemble delle ricette di Dataset506 (fuori-fold)

Prodotto da `scripts/ensemble_506.py`, **non modificare a mano**. 12 pazienti di addestramento, nel nativo; ogni combinazione col suo postprocessing scelto con la regola di `postprocessing_aincp.py`.

| | postprocessing | F1 lesioni | Dice medio | mediana | globale | rilevate | FP | Δ Dice vs Z | meglio/peggio | p (Wilcoxon) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Z | t 0.3, vmin 50 | 0.560 | 0.584 | 0.565 | 0.859 | 27/68 | 1 | — | — | — |
| A+D+Z | t 0.3, vmin 10 | 0.566 | 0.584 | 0.621 | 0.874 | 29/68 | 4 | -0.000 | 7/4 | 0.70 |
| A+D | t 0.4, vmin 5 | 0.547 | 0.576 | 0.609 | 0.870 | 27/68 | 3 | -0.008 | 6/5 | 0.83 |
| D+Z | t 0.3, vmin 20 | 0.565 | 0.588 | 0.621 | 0.871 | 28/68 | 2 | +0.004 | 8/3 | 0.37 |
