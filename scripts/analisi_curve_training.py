#!/usr/bin/env python
"""Curve di training di tutti i modelli, e misura quantitativa dell'overfitting.

    python scripts/analisi_curve_training.py   # -> reports/curve_training.md + figura

Nasce da una domanda dei relatori: nnU-Net fa early stopping, e rischiamo overfitting?

La risposta alla prima e' no, e si legge nel sorgente: nel `nnUNetTrainer` non compaiono ne'
`early`, ne' `patience`, ne' `stop`. Il numero di epoche e' fissato in partenza e il training le
esegue tutte. nnU-Net salva pero' `checkpoint_best.pth` sul massimo della pseudo-dice mediata
esponenzialmente, quindi il meccanismo per scegliere un'epoca diversa dall'ultima **esiste**: e'
solo che la validation finale usa `checkpoint_final.pth`, cioe' l'ultima epoca.

Alla seconda domanda si risponde con i numeri, non a occhio. Per ogni modello si estraggono dal
log le tre curve che nnU-Net registra a ogni epoca -- perdita di training, perdita di validation,
pseudo-dice -- e si misurano tre indicatori:

    - **epoca del minimo di val_loss**: se cade molto prima della fine, le epoche successive non
      stanno piu' imparando;
    - **risalita dopo il minimo**: di quanto la perdita di validation e' risalita alla fine
      rispetto al suo minimo. E' la definizione operativa di overfitting;
    - **divario finale train/val**: quanto il modello fa meglio su cio' che ha visto.

Attenzione a come si legge la pseudo-dice: e' calcolata su ritagli durante il training, non sui
volumi interi, e non coincide con il Dice riportato altrove.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
RES = PROGETTO / 'nnUNet' / 'nnUNet_results'

# Nome leggibile -> (dataset, cartella del run)
MODELLI = {
    '501 WMH FLAIR':        ('Dataset501_WMH', 'nnUNetTrainerWandb__nnUNetPlans__3d_fullres'),
    '502 ATLAS T1':         ('Dataset502_ATLAS', 'nnUNetTrainerWandb__nnUNetPlans__3d_fullres'),
    '503 DBB tessuti':      ('Dataset503_DBB', 'nnUNetTrainerWandb__nnUNetPlans__3d_fullres'),
    '531 unione D':         ('Dataset531_UNION_D', 'nnUNetTrainerWandb__nnUNetPlans__3d_fullres'),
    '541 da zero, 10 casi': ('Dataset541_SIM_AINCP',
                             'nnUNetTrainerWandb_250epochs__nnUNetPlans__3d_fullres'),
    '541 fine-tuning lr 1e-2': ('Dataset541_SIM_AINCP',
                                'nnUNetTrainerWandb_250epochs__nnUNetPlansFT__3d_fullres'),
    '541 fine-tuning lr 1e-3': ('Dataset541_SIM_AINCP',
                                'nnUNetTrainerWandb_250epochs_lr1e3__nnUNetPlans__3d_fullres'),
    '541 fine-tuning lr 1e-4': ('Dataset541_SIM_AINCP',
                                'nnUNetTrainerWandb_250epochs_lr1e4__nnUNetPlans__3d_fullres'),
}

BLU, ARANCIO, VERDE, MUTO = '#2a78d6', '#eb6834', '#0f6f4d', '#8b8f8f'


def curve(dataset: str, run: str):
    """Le tre serie per epoca, lette dai log che nnU-Net scrive con flush esplicito.

    I log vanno **concatenati in ordine di data**: un training ripreso con `--c` ne apre uno
    nuovo, e leggere solo l'ultimo taglierebbe via le epoche iniziali. Su 501, che e' stato
    ripreso, mancherebbero le prime 150 -- proprio quelle in cui la curva scende.
    """
    file = sorted((RES / dataset / run / 'fold_0').glob('training_log_*.txt'))
    if not file:
        raise FileNotFoundError(f'nessun log in {dataset}/{run}')
    testo = '\n'.join(f.read_text() for f in file)
    tr = [float(x) for x in re.findall(r'train_loss (-?[\d.]+)', testo)]
    va = [float(x) for x in re.findall(r'val_loss (-?[\d.]+)', testo)]
    ps = [float(x) for x in re.findall(r'Pseudo dice \[np\.float32\(([\d.]+)\)', testo)]
    n = min(len(tr), len(va))
    return np.array(tr[:n]), np.array(va[:n]), np.array(ps[:n] if len(ps) >= n else ps)


def indicatori(tr, va):
    """Coda = ultimo 10% delle epoche: un solo valore finale sarebbe rumore."""
    coda = max(1, len(va) // 10)
    i_min = int(np.argmin(va))
    val_fin = float(va[-coda:].mean())
    risalita = val_fin - float(va[i_min])
    return i_min, float(va[i_min]), val_fin, risalita, float(tr[-coda:].mean())


def main():
    dati = {}
    for nome, (d, r) in MODELLI.items():
        try:
            dati[nome] = curve(d, r)
        except (StopIteration, IndexError, FileNotFoundError):
            print(f'saltato (log assente): {nome}')

    righe = ['# Curve di training e verifica dell\'overfitting', '',
             'Prodotto da `scripts/analisi_curve_training.py`, **non modificare a mano**.', '',
             '## nnU-Net fa early stopping?', '',
             'No. Nel sorgente del trainer non compaiono ne\' `early`, ne\' `patience`, ne\' `stop`:',
             'il numero di epoche e\' fissato in partenza e vengono eseguite tutte.',
             '',
             'nnU-Net salva pero\' `checkpoint_best.pth` sul massimo della pseudo-dice mediata',
             'esponenzialmente, e la validation finale usa `checkpoint_final.pth`, cioe\' **l\'ultima',
             'epoca**. Se le curve mostrassero overfitting, valutare il checkpoint migliore invece',
             'dell\'ultimo sarebbe un intervento immediato e gratuito.', '',
             '## Gli indicatori, modello per modello', '',
             '| modello | epoche | epoca del minimo di val_loss | val_loss al minimo | val_loss finale | risalita | train_loss finale | divario train/val |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for nome, (tr, va, ps) in dati.items():
        i_min, v_min, v_fin, ris, t_fin = indicatori(tr, va)
        righe.append(f'| {nome} | {len(va)} | {i_min} ({i_min / len(va):.0%}) | {v_min:.4f} | '
                     f'{v_fin:.4f} | **{ris:+.4f}** | {t_fin:.4f} | {v_fin - t_fin:+.4f} |')
    righe += ['',
              'La perdita di nnU-Net e\' Dice + cross-entropy ed e\' **negativa**: piu\' bassa e\'',
              'migliore. La risalita e\' quindi la quantita\' da guardare: positiva e grande',
              'significa che il modello, dopo un certo punto, ha peggiorato sui dati non visti',
              'continuando a migliorare su quelli visti.', '']

    # --- figura ------------------------------------------------------------
    n = len(dati)
    righe_fig = (n + 1) // 2
    fig, assi = plt.subplots(righe_fig, 2, figsize=(13, 3.3 * righe_fig), facecolor='white')
    assi = np.atleast_1d(assi).ravel()
    for ax, (nome, (tr, va, ps)) in zip(assi, dati.items()):
        e = np.arange(len(tr))
        ax.plot(e, tr, color=BLU, lw=1.4, label='perdita, training')
        ax.plot(e, va, color=ARANCIO, lw=1.4, label='perdita, validation')
        i_min = int(np.argmin(va))
        ax.axvline(i_min, color=MUTO, lw=1, ls=':')
        ax.annotate(f'minimo\nepoca {i_min}', xy=(i_min, va[i_min]), xytext=(6, 14),
                    textcoords='offset points', fontsize=8, color=MUTO)
        if len(ps):
            a2 = ax.twinx()
            a2.plot(np.arange(len(ps)), ps, color=VERDE, lw=1.1, alpha=.75)
            a2.set_ylabel('pseudo-dice', color=VERDE, fontsize=8.5)
            a2.tick_params(axis='y', labelsize=8, colors=VERDE)
            a2.set_ylim(0, 1)
        ax.set_title(nome, fontsize=10.5)
        ax.set_xlabel('epoca', fontsize=8.5)
        ax.set_ylabel('perdita', fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=.25, lw=.7)
    for ax in assi[len(dati):]:
        ax.axis('off')
    assi[0].legend(fontsize=8.5, loc='upper right', frameon=False)
    fig.suptitle('Curve di training: perdita su dati visti (blu) e non visti (arancio),\n'
                 'con la pseudo-dice di validation in verde sull\'asse destro', fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.6)
    out_fig = PROGETTO / 'reports' / 'figures' / 'curve_training.png'
    fig.savefig(out_fig, dpi=140, facecolor='white')

    righe += [f'Figura: `reports/figures/curve_training.png`.', '']
    (PROGETTO / 'reports' / 'curve_training.md').write_text('\n'.join(righe) + '\n')
    print('\n'.join(righe))
    print(f'figura -> {out_fig}')


if __name__ == '__main__':
    main()
