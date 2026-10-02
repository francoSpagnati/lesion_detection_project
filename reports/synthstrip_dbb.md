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

| soggetto | gruppo | Dice | mm³ SynthStrip | mm³ riferimento | mm³ solo SynthStrip | mm³ solo riferimento |
| --- | --- | --- | --- | --- | --- | --- |
| sub-0955 | HD | 0.8791 | 1 684 000 | 2 076 000 | 31 500 | 423 000 |
| sub-0924 | HD | 0.9089 | 1 860 000 | 2 197 000 | 16 200 | 353 300 |
| sub-0922 | MCDs | 0.9284 | 1 490 000 | 1 630 000 | 41 600 | 181 800 |
| sub-0958 | HD | 0.9290 | 1 989 000 | 2 194 000 | 46 000 | 251 100 |
| sub-0932 | PFM | 0.9365 | 1 005 000 | 889 000 | 118 000 | 2 300 |
| sub-0929 | MCDs | 0.9421 | 1 003 000 | 899 000 | 107 100 | 3 100 |
| sub-0956 | HD | 0.9442 | 1 430 000 | 1 499 000 | 47 000 | 116 300 |
| sub-0942 | PFM | 0.9459 | 778 000 | 735 000 | 62 500 | 19 300 |
| sub-0417 | Healthy | 0.9476 | 1 634 000 | 1 480 000 | 158 600 | 4 600 |
| sub-0926 | PFM | 0.9484 | 702 000 | 647 000 | 62 100 | 7 500 |

Immagini di controllo in `reports/figures/synthstrip/`, **una per caso, con il Dice come
prefisso del nome**: ordinando la cartella per nome si guardano prima i casi peggiori.
Maschere in `data/processed/synthstrip/DBB/`.

