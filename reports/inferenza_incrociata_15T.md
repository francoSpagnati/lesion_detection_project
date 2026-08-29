# Il modello solo-3 T applicato ai casi a 1.5 T

Risponde a "quanto **costa** escludere i 1.5 T", non a "quanto si guadagna":
la seconda domanda ha gia' risposto `reports/test_esclusione_1_5T.md` (+0.0093, p=0.67).

## Confronto appaiato sui 9 casi mai visti da nessuno dei due modelli

| | Dice medio | mediana |
| --- | --- | --- |
| `502`, addestrato **con** i 1.5 T | 0.6528 | 0.7930 |
| `521`, addestrato **senza** | 0.5805 | 0.7308 |
| **differenza** | **-0.0723** | |

Migliora in 3, peggiora in 5. Wilcoxon appaiato: **p = 0.25**.

### Tre limiti da tenere presenti leggendo questo numero

1. **Potere statistico basso**: con 9 casi la misura serve a escludere un crollo, non a stimare una differenza fine.
2. **Una sola coorte**: i casi puliti vengono tutti da `['r040']`. Quindi si sta misurando la generalizzazione a **uno scanner**, non ai 1.5 T in generale.
3. **Un solo caso pesa per due terzi**: togliendo `sub-r040s059` la differenza media passa da -0.0723 a -0.0270. Non e' un effetto diffuso.

## `521` da solo su tutti i 45 casi a 1.5 T mai visti

Dice medio **0.6721**, mediana 0.7668, minimo 0.0000, massimo 0.9074. Composizione per coorte: {'r040': 39, 'r029': 4, 'r039': 2}.
Sui 36 casi che 502 aveva in training il solo `521` vale 0.6950; il numero di 502 su quei casi sarebbe ottimistico e non va confrontato.

Per riferimento, `521` sui suoi 103 casi di validation a **3 T** vale **0.6160**.

## Dettaglio dei casi puliti

| caso | 502 (con 1.5 T) | 521 (senza) | delta |
| --- | --- | --- | --- |
| sub-r040s030 | 0.8914 | 0.8213 | -0.0701 |
| sub-r040s031 | 0.9084 | 0.7668 | -0.1416 |
| sub-r040s045 | 0.8471 | 0.8435 | -0.0037 |
| sub-r040s046 | 0.7930 | 0.8010 | +0.0080 |
| sub-r040s049 | 0.0000 | 0.0000 | +0.0000 |
| sub-r040s054 | 0.6983 | 0.7306 | +0.0324 |
| sub-r040s059 | 0.8375 | 0.4027 | -0.4348 |
| sub-r040s063 | 0.1092 | 0.1280 | +0.0189 |
| sub-r040s075 | 0.7906 | 0.7308 | -0.0598 |
