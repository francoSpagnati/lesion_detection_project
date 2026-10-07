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

VERDE, ROSSO = '#00d67d', '#ff4d4d'
N_CASI = 4


def dadi(dataset):
    s = RES / dataset / TR / 'fold_0' / 'validation' / 'summary.json'
    j = json.loads(s.read_text())
    return {Path(c['prediction_file']).name.replace('.nii.gz', ''): c['metrics']['1']['Dice']
            for c in j['metric_per_case']}


def carica(p):
    return np.asanyarray(nib.load(str(p)).dataobj).astype(np.float32)


def finestra(a):
    lo, hi = np.percentile(a[a > 0], (1, 99)) if (a > 0).any() else (0, 1)
    return lo, max(hi, lo + 1e-6)


def main():
    d501, d511 = dadi('Dataset501_WMH'), dadi('Dataset511_WMH_T1')
    comuni = sorted(set(d501) & set(d511), key=lambda c: d511[c] - d501[c])
    # I tre con il divario maggiore, piu' quello in cui i due modelli sono piu' vicini.
    scelti = comuni[:N_CASI - 1] + [comuni[-1]]

    fig, assi = plt.subplots(len(scelti), 2, figsize=(8.6, 4.1 * len(scelti)), facecolor='white')
    for r, caso in enumerate(scelti):
        gt = carica(PREP / 'Dataset501_WMH' / 'gt_segmentations' / f'{caso}.nii.gz') > 0
        # La fetta con piu' lesione: al centro del volume spesso non c'e' nulla da vedere.
        z = int(np.argmax(gt.sum(axis=(0, 1))))
        for c, (dataset, dice, canale) in enumerate((
                ('Dataset501_WMH', d501[caso], 'FLAIR'),
                ('Dataset511_WMH_T1', d511[caso], 'T1'))):
            img = carica(RAW / dataset / 'imagesTr' / f'{caso}_0000.nii.gz')[:, :, z]
            pred = carica(RES / dataset / TR / 'fold_0' / 'validation' / f'{caso}.nii.gz')[:, :, z] > 0
            ax = assi[r, c]
            lo, hi = finestra(img)
            ax.imshow(np.rot90(img), cmap='gray', vmin=lo, vmax=hi)
            g = np.rot90(gt[:, :, z])
            if g.any():
                ax.contour(g, levels=[0.5], colors=[VERDE], linewidths=1.3, linestyles='dashed')
            if pred.any():
                ax.contour(np.rot90(pred), levels=[0.5], colors=[ROSSO], linewidths=1.3)
            ax.set_title(f'{canale}  —  Dice {dice:.3f}', fontsize=11,
                         fontweight='bold' if c == 0 else 'normal')
            ax.axis('off')
        assi[r, 0].annotate(f'{caso}\nslice {z}', xy=(0, 0.5), xycoords='axes fraction',
                            xytext=(-12, 0), textcoords='offset points', rotation=90,
                            va='center', ha='center', fontsize=9, color='#444')

    fig.legend(handles=[Line2D([], [], color=VERDE, ls='--', lw=1.6, label='manual reference'),
                        Line2D([], [], color=ROSSO, lw=1.6, label='model prediction')],
               loc='lower center', ncol=2, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle('Same patient, same slice, same network: only the input sequence changes\n'
                 'the three cases with the largest gap, and at the bottom one where T1 holds up',
                 fontsize=12.5)
    fig.tight_layout(rect=(0.012, 0.028, 1, 0.955), h_pad=1.4)
    out = PROGETTO / 'reports' / 'figures' / 'flair_contro_t1.png'
    fig.savefig(out, dpi=140, facecolor='white')
    print(f'scritta {out}  —  casi: {", ".join(scelti)}')


if __name__ == '__main__':
    main()
