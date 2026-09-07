# Curve di training e verifica dell'overfitting

Prodotto da `scripts/analisi_curve_training.py`, **non modificare a mano**.

## nnU-Net fa early stopping?

No. Nel sorgente del trainer non compaiono ne' `early`, ne' `patience`, ne' `stop`:
il numero di epoche e' fissato in partenza e vengono eseguite tutte.

nnU-Net salva pero' `checkpoint_best.pth` sul massimo della pseudo-dice mediata
esponenzialmente, e la validation finale usa `checkpoint_final.pth`, cioe' **l'ultima
epoca**. Se le curve mostrassero overfitting, valutare il checkpoint migliore invece
dell'ultimo sarebbe un intervento immediato e gratuito.

## Gli indicatori, modello per modello

| modello | epoche | epoca del minimo di val_loss | val_loss al minimo | val_loss finale | risalita | train_loss finale | divario train/val |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 501 WMH FLAIR | 1039 | 347 (33%) | -0.7974 | -0.7560 | **+0.0414** | -0.8911 | +0.1351 |
| 502 ATLAS T1 | 1003 | 905 (90%) | -0.6671 | -0.5734 | **+0.0937** | -0.7235 | +0.1501 |
| 503 DBB tessuti | 1000 | 982 (98%) | -0.9266 | -0.9178 | **+0.0088** | -0.8969 | -0.0209 |
| 531 unione D | 1000 | 980 (98%) | -0.2355 | -0.1644 | **+0.0711** | -0.2479 | +0.0835 |
| 541 da zero, 10 casi | 250 | 128 (51%) | -0.4588 | -0.3447 | **+0.1141** | -0.8783 | +0.5336 |
| 541 fine-tuning lr 1e-2 | 250 | 90 (36%) | -0.5980 | -0.4955 | **+0.1025** | -0.8930 | +0.3975 |
| 541 fine-tuning lr 1e-3 | 250 | 87 (35%) | -0.6789 | -0.5852 | **+0.0937** | -0.8596 | +0.2745 |
| 541 fine-tuning lr 1e-4 | 250 | 69 (28%) | -0.7210 | -0.6349 | **+0.0861** | -0.8143 | +0.1793 |

La perdita di nnU-Net e' Dice + cross-entropy ed e' **negativa**: piu' bassa e'
migliore. La risalita e' quindi la quantita' da guardare: positiva e grande
significa che il modello, dopo un certo punto, ha peggiorato sui dati non visti
continuando a migliorare su quelli visti.

Figura: `reports/figures/curve_training.png`.

