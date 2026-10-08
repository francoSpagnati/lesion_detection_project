#!/usr/bin/env python
"""Confronto visivo fra il modello sulla FLAIR e quello sul T1, sugli stessi casi WMH.

    python scripts/figura_flair_contro_t1.py

La misura dice che la FLAIR vale 14.8 punti di Dice ed e' migliore in 16 casi su 16, e che la
perdita si concentra sulle lesioni piccole. Questa figura mostra **la stessa cosa a occhio**, che
e' cio' che serve per verificare un risultato di segmentazione invece di doverlo credere sulla
parola: stessi pazienti, stessa fetta, stessa rete, cambia solo la sequenza in ingresso.

Colonna 1: FLAIR con riferimento e predizione del modello 501.
Colonna 2: T1 dello stesso paziente, allineato, con riferimento e predizione del modello 511.

I casi non sono scelti a piacere: si prendono quelli in cui il divario fra i due modelli e'
maggiore, cioe' quelli che mostrano il fenomeno, e uno in cui i due modelli sono simili, per non
dare l'impressione che il T1 fallisca sempre.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D

PROGETTO = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get('nnUNet_results', PROGETTO / 'nnUNet' / 'nnUNet_results'))
PREP = Path(os.environ.get('nnUNet_preprocessed', PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'))
RAW = PROGETTO / 'data/processed/nnUNet_raw'
TR = 'nnUNetTrainerWandb__nnUNetPlans__3d_fullres'

RIF, PRED = '#2a78d6', '#eb6834'     # stessa coppia validata delle altre figure della tesi
N_CASI = 4


def dadi(dataset):
    s = RES / dataset / TR / 'fold_0' / 'validation' / 'summary.json'
    j = json.loads(s.read_text())
    return {Path(c['prediction_file']).name.replace('.nii.gz', ''): c['metrics']['1']['Dice']
            for c in j['metric_per_case']}


def carica(p):
    # orientamento canonico RAS, per mostrare le fette in convenzione radiologica
    return np.asanyarray(nib.as_closest_canonical(nib.load(str(p))).dataobj).astype(np.float32)


def spaziatura(p):
    return nib.as_closest_canonical(nib.load(str(p))).header.get_zooms()[:3]


def finestra(a):
    lo, hi = np.percentile(a[a > 0], (1, 99)) if (a > 0).any() else (0, 1)
    return lo, max(hi, lo + 1e-6)


def main():
    d501, d511 = dadi('Dataset501_WMH'), dadi('Dataset511_WMH_T1')
    comuni = sorted(set(d501) & set(d511), key=lambda c: d511[c] - d501[c])
    # I tre con il divario maggiore, piu' quello in cui i due modelli sono piu' vicini.
    scelti = comuni[:N_CASI - 1] + [comuni[-1]]

    # formato della tesi: FLAIR sopra e T1 sotto, un caso per colonna; titolo e numero di fetta
    # vanno nella didascalia o non servono
    H = 3.75
    fig, assi = plt.subplots(2, len(scelti), figsize=(6.3, H), facecolor='white')
    for c, caso in enumerate(scelti):
        gt = carica(PREP / 'Dataset501_WMH' / 'gt_segmentations' / f'{caso}.nii.gz') > 0
        # la fetta con piu' lesione: al centro del volume spesso non c'e' nulla da vedere
        z = int(np.argmax(gt.sum(axis=(0, 1))))
        for r, (dataset, dice, canale) in enumerate((
                ('Dataset501_WMH', d501[caso], 'FLAIR'),
                ('Dataset511_WMH_T1', d511[caso], 'T1'))):
            f_img = RAW / dataset / 'imagesTr' / f'{caso}_0000.nii.gz'
            vol = carica(f_img)
            sp = spaziatura(f_img)
            img = vol[:, :, z]
            pred = carica(RES / dataset / TR / 'fold_0' / 'validation' / f'{caso}.nii.gz')[:, :, z] > 0
            lo, hi = finestra(img)
            testa = img > lo + 0.08 * (hi - lo)
            xs, ys = np.where(testa)
            box = (slice(max(xs.min() - 3, 0), xs.max() + 4), slice(max(ys.min() - 3, 0), ys.max() + 4))
            # RAS -> anteriore in alto, destra del paziente a sinistra
            orienta = lambda a: np.fliplr(np.flipud(a[box].T))
            est = (0, (box[0].stop - box[0].start) * sp[0], 0, (box[1].stop - box[1].start) * sp[1])
            ax = assi[r, c]
            ax.imshow(orienta(img), cmap='gray', vmin=lo, vmax=hi, extent=est, interpolation='lanczos')
            for m, col, ls in ((gt[:, :, z], RIF, 'dashed'), (pred, PRED, 'solid')):
                o = orienta(m.astype(float))
                if o.any():
                    ax.contour(o, levels=[0.5], colors=[col], linewidths=0.8, linestyles=ls,
                               extent=est, origin='upper')
            ax.set_title((f'{caso}\n' if r == 0 else '') + f'{canale} · Dice {dice:.2f}', fontsize=7,
                         loc='left', pad=2, linespacing=1.2)
            ax.axis('off')
            if c == 0:
                ax.text(0.03, 0.97, 'R', transform=ax.transAxes, color='white', fontsize=6, va='top')
                ax.text(0.97, 0.97, 'L', transform=ax.transAxes, color='white', fontsize=6, va='top', ha='right')

    fig.legend(handles=[Line2D([], [], color=RIF, ls='--', lw=1.2, label='manual reference'),
                        Line2D([], [], color=PRED, lw=1.2, label='model prediction')],
               loc='lower center', ncol=2, frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.35 / H, bottom=0.22 / H, wspace=0.04, hspace=0.14)
    out = PROGETTO / 'reports' / 'figures' / 'flair_contro_t1.png'
    fig.savefig(out, dpi=300, facecolor='white')
    print(f'scritta {out}  —  casi: {", ".join(scelti)}')


if __name__ == '__main__':
    main()
