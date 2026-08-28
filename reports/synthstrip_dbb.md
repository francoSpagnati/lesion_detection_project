# SynthStrip su DBB: qualita' contro le maschere di riferimento

Generato da `scripts/run_synthstrip.py`, **non modificare a mano**.

Modello: `synthstrip.1.pt` (principale, con CSF). Casi: **244**.

Il riferimento e' la maschera cerebrale fornita da DBB. La domanda a cui risponde questa
misura: **lo skull stripping regge sulle anatomie alterate?** — il caso in cui BET fallisce.

## Dice per gruppo

| gruppo | n | Dice medio | mediana | minimo |
| --- | --- | --- | --- | --- |
| ACC | 10 | 0.9675 | 0.9681 | 0.9621 |
| HD | 6 | 0.9267 | 0.9366 | 0.8791 |
| Healthy | 207 | 0.9673 | 0.9678 | 0.9476 |
| MCDs | 10 | 0.9605 | 0.9644 | 0.9284 |
| PFM | 11 | 0.9604 | 0.9652 | 0.9365 |

**Sani: 0.9673   |   patologici: 0.9569** (differenza -0.0104)

## I dieci casi peggiori

| soggetto | gruppo | Dice | ml SynthStrip | ml riferimento | solo SynthStrip | solo riferimento |
| --- | --- | --- | --- | --- | --- | --- |
| sub-0955 | HD | 0.8791 | 1684 | 2076 | 31.5 | 423.0 |
| sub-0924 | HD | 0.9089 | 1860 | 2197 | 16.2 | 353.3 |
| sub-0922 | MCDs | 0.9284 | 1490 | 1630 | 41.6 | 181.8 |
| sub-0958 | HD | 0.9290 | 1989 | 2194 | 46.0 | 251.1 |
| sub-0932 | PFM | 0.9365 | 1005 | 889 | 118.0 | 2.3 |
| sub-0929 | MCDs | 0.9421 | 1003 | 899 | 107.1 | 3.1 |
| sub-0956 | HD | 0.9442 | 1430 | 1499 | 47.0 | 116.3 |
| sub-0942 | PFM | 0.9459 | 778 | 735 | 62.5 | 19.3 |
| sub-0417 | Healthy | 0.9476 | 1634 | 1480 | 158.6 | 4.6 |
| sub-0926 | PFM | 0.9484 | 702 | 647 | 62.1 | 7.5 |

Immagini di controllo in `reports/figures/synthstrip/`, **una per caso, con il Dice come
prefisso del nome**: ordinando la cartella per nome si guardano prima i casi peggiori.
Maschere in `data/processed/synthstrip/DBB/`.

