# Variante D: il modello unico su tutti e tre i dataset

Prodotto da `scripts/analisi_union_d.py`, **non modificare a mano**.

Fold 0, 151 casi di validation: 112 da ATLAS, 23 da DBB, 16 da WMH. Il Dice medio che
nnU-Net stampa a fine training (**0.8966**) e' la media sulle sette classi ed e' dominato
dai sei tessuti, che sono densi e facili: non descrive il compito che ci interessa.

## La classe che conta: la lesione

| sorgente | casi | Dice della lesione in D |
| --- | --- | --- |
| ATLAS | 112 | **0.6031** |
| WMH | 16 | **0.6032** |

## Confronto appaiato con le baseline, sugli stessi casi

| confronto | classe | casi | baseline | D | differenza | migliora / peggiora | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D contro `Dataset502_ATLAS` | lesione | 112 | 0.6103 | 0.6031 | **-0.0072** | 41 / 65 | 0.07316 |
| D contro `Dataset511_WMH_T1` | lesione | 16 | 0.6257 | 0.6032 | **-0.0225** | 4 / 12 | 0.1046 |
| D contro `Dataset503_DBB` | csf | 23 | 0.9278 | 0.9034 | **-0.0245** | 0 / 23 | 2.384e-07 |
| D contro `Dataset503_DBB` | grigia | 23 | 0.9845 | 0.9741 | **-0.0104** | 0 / 23 | 2.384e-07 |
| D contro `Dataset503_DBB` | bianca | 23 | 0.9892 | 0.9811 | **-0.0080** | 0 / 23 | 2.384e-07 |
| D contro `Dataset503_DBB` | grigia profonda | 23 | 0.9718 | 0.9514 | **-0.0205** | 0 / 23 | 2.384e-07 |
| D contro `Dataset503_DBB` | tronco | 23 | 0.9732 | 0.9639 | **-0.0093** | 1 / 22 | 7.868e-06 |
| D contro `Dataset503_DBB` | cervelletto | 23 | 0.9806 | 0.9753 | **-0.0054** | 0 / 23 | 2.384e-07 |

