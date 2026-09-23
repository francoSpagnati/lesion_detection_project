# Oblio delle cavita' di ATLAS dopo lo stadio 2

Prodotto da `scripts/oblio_atlas_stadio2.py`, **non modificare a mano**.

Rete dello stadio 2 (Dataset542, fold 0, `checkpoint_final`) sui 112 casi di validazione del fold 0 di 502, con la FLAIR vuota. Riferimento: la rete di 502 sugli stessi casi. Confronto appaiato, Wilcoxon.

| rete | Dice medio | differenza appaiata da 502 | p |
| --- | --- | --- | --- |
| 502 (stadio 1) | 0.610 | — | — |
| `zero_500ep_lr1e3_dropmod30_arresto50` | 0.217 | -0.394 | 9.8e-17 |
| `zero_500ep_lr1e4_primo1e2_dropmod30_arresto50` | 0.220 | -0.390 | 1.3e-16 |

Per volume della lesione (terzili sui casi di validazione):

| fascia | casi | 502 | lr1e3_dropmod30_arresto50 | lr1e4_primo1e2_dropmod30_arresto50 |
| --- | --- | --- | --- | --- |
| < 1545 mm3 | 37 | 0.489 | 0.040 | 0.047 |
| 1545–15656 | 37 | 0.584 | 0.225 | 0.253 |
| >= 15656 | 38 | 0.753 | 0.380 | 0.357 |
