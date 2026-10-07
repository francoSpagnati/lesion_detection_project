#!/usr/bin/env python
"""Curve di addestramento dei 15 run di Dataset506: c'e' overfitting, e l'arresto lo ferma?

    python scripts/figura_curve_506.py    # -> reports/figures/aincp/curve_506.png + reports/curve_506.md

PERCHE'. Con 12 pazienti e 31 milioni di parametri la rete puo' imparare a memoria. Il segno e' la
forbice fra perdita di addestramento (continua a scendere) e perdita di validazione (si ferma e
risale). L'arresto (pazienza 50, nnUNetTrainerWandb.on_epoch_end) tiene i pesi dell'epoca col
minimo della validazione: la figura mostra dove cade quel minimo e quanto risale dopo.

DA DOVE. `checkpoint_ultima_epoca.pth` di ogni fold, che contiene il log completo di nnU-Net
(train_losses, val_losses, mean_fg_dice per epoca) fino all'ultima epoca eseguita. La perdita e'
quella di nnU-Net, Dice morbido + entropia incrociata: piu' bassa e' meglio, -1 e' il limite.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROGETTO = Path(__file__).resolve().parent.parent
R = PROGETTO / 'nnUNet/nnUNet_results/Dataset506_AINCP_FINALE'
RICETTE = [('Z, from scratch (the thesis model)', 'nnUNetTrainerWandb_250ep_dropmod30_arresto50'),
           ('A, from ATLAS with expanded first layer', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_dropmod30_arresto50'),
           ('D, from joint ATLAS + WMH pretraining', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_dropmod30_arresto50_dapre2c')]
BLU, ARANCIO = '#2a78d6', '#eb6834'          # stessa coppia validata delle altre figure AInCP
INK, INK2, GRIGIO, FONDO = '#0b0b0b', '#52514e', '#c9c8c3', '#fcfcfb'
OUT = PROGETTO / 'reports/figures/aincp/curve_506.png'


def log(tr, f):
    ck = torch.load(R / f'{tr}__nnUNetPlansDa502__3d_fullres' / f'fold_{f}' / 'checkpoint_ultima_epoca.pth',
                    map_location='cpu', weights_only=False)
    L = ck['logging']
    t0, t1 = np.array(L['epoch_start_timestamps']), np.array(L['epoch_end_timestamps'])
    return {'train': np.array(L['train_losses']), 'val': np.array(L['val_losses']),
            'dice': np.array(L['mean_fg_dice']), 'lr0': L['lrs'][0], 'sec': float(np.median(t1 - t0))}


def main():
    fig, ax = plt.subplots(3, 5, figsize=(17, 10.5), sharey=True, facecolor=FONDO)
    righe = ['| ricetta | fold | epoche eseguite | epoca migliore | train al migliore | val al migliore | '
             'val all\'ultima | risalita val | passo iniziale | s/epoca |', '| --- ' * 10 + '|']
    for r, (nome, tr) in enumerate(RICETTE):
        for f in range(5):
            d = log(tr, f)
            b = int(np.argmin(d['val']))
            x = ax[r, f]
            x.set_facecolor(FONDO)
            ep = np.arange(len(d['train']))
            # le 50 epoche di pazienza dopo il minimo: quelle che l'arresto butta via
            x.axvspan(b, len(ep) - 1, color='#efeeea', lw=0)
            x.plot(ep, d['train'], color=BLU, lw=1.6)
            x.plot(ep, d['val'], color=ARANCIO, lw=1.6, ls='--')
            x.plot([b], [d['val'][b]], 'o', ms=7, color=ARANCIO, mec=FONDO, mew=2)
            x.axvline(b, color=ARANCIO, lw=0.8, alpha=0.6)
            x.text(b + 3, 0.0, f'best: epoch {b}', ha='left', va='top', fontsize=8.5, color=INK2)
            x.set_xlim(0, 200)
            x.set_ylim(-0.9, 0.05)
            x.grid(axis='y', color=GRIGIO, lw=0.6, alpha=0.6)
            for s in ('top', 'right'):
                x.spines[s].set_visible(False)
            for s in ('left', 'bottom'):
                x.spines[s].set_color(GRIGIO)
            x.tick_params(colors=INK2, labelsize=8.5)
            if r == 0:
                x.set_title(f'fold {f}', fontsize=11, color=INK, loc='left')
            if f == 0:
                x.set_ylabel(nome.split(',')[0] + '\nloss (Dice + CE)', fontsize=10, color=INK)
            if r == 2:
                x.set_xlabel('epoch', fontsize=9.5, color=INK2)
            righe.append(f"| {nome.split(',')[0]} | {f} | {len(ep)} | {b} | {d['train'][b]:.3f} | {d['val'][b]:.3f} | "
                         f"{d['val'][-1]:.3f} | {d['val'][-1] - d['val'][b]:+.3f} | {d['lr0']:.0e} | {d['sec']:.0f} |")
    # etichette dirette nel primo pannello, oltre alla legenda
    ax[0, 0].text(30, -0.80, 'training', color=BLU, fontsize=9, fontweight='bold')
    ax[0, 0].text(30, -0.36, 'validation', color=ARANCIO, fontsize=9, fontweight='bold')
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([], [], color=BLU, lw=2, label='training loss'),
                        Line2D([], [], color=ARANCIO, lw=2, ls='--', label='validation loss (the 2–3 patients of the fold)'),
                        Patch(color='#efeeea', label='50 patience epochs after the minimum: discarded')],
               loc='upper left', bbox_to_anchor=(0.005, 0.935), ncol=3, frameon=False, fontsize=10.5, labelcolor=INK)
    fig.suptitle('Dataset506: training curves of the 15 runs — early stopping halts overfitting',
                 x=0.008, ha='left', fontsize=15, color=INK, fontweight='bold', y=0.995)
    fig.text(0.008, 0.962, 'Rows: the three recipes; columns: the 5 folds. The dot marks the epoch with the lowest validation loss: '
             'its weights are the fold model. Afterwards the training loss keeps falling and the validation loss rises.',
             ha='left', va='top', fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), h_pad=1.2, w_pad=0.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100, facecolor=FONDO)
    md = PROGETTO / 'reports/curve_506.md'
    md.write_text('# Curve di addestramento di Dataset506\n\nProdotto da `scripts/figura_curve_506.py`, **non modificare '
                  'a mano**. Perdita di nnU-Net (Dice morbido + entropia incrociata, piu\' bassa e\' meglio).\n\n'
                  + '\n'.join(righe) + '\n')
    print('scritto', OUT, md)
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
