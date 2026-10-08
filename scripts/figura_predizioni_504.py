#!/usr/bin/env python
"""Le tre varianti di 504 a confronto, paziente per paziente, sulla predizione fuori-fold.

    python scripts/figura_predizioni_504.py     # -> reports/figures/aincp/predizioni_504.png

Una riga per paziente, una colonna per variante. Ogni riquadro e' la fetta della FLAIR (canale 0
del dataset, nello spazio T1 del paziente) con piu' lesione annotata; il riferimento e' in verde
e la predizione in rosso, con le convenzioni delle figure di predizione dei dataset pubblici
(scripts/plot_predictions.py). Le predizioni vengono dal fold in cui il paziente era in
validazione, quindi nessuna e' stata vista in addestramento.

Serve a leggere i numeri del confronto appaiato (reports/confronto_504_*.md): un Dice di 0.04 su
FSM_041 e uno di 0.92 su FSM_040 sono la stessa rete su due pazienti diversi, e la differenza
si capisce solo guardando quanto e' grande e com'e' fatta la lesione in ciascuno.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D

PROGETTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROGETTO / 'scripts'))
from plot_predictions import RIF, PRED, finestra, scegli_fette  # noqa: E402

RES = Path(os.environ.get('nnUNet_results', PROGETTO / 'nnUNet' / 'nnUNet_results')) / 'Dataset504_AINCP'
PREP = Path(os.environ.get('nnUNet_preprocessed', PROGETTO / 'nnUNet' / 'nnUNet_preprocessed')) / 'Dataset504_AINCP'
RAW = Path(os.environ.get('nnUNet_raw', PROGETTO / 'nnUNet' / 'nnUNet_raw')) / 'Dataset504_AINCP'
FIG = PROGETTO / 'reports/figures/aincp'
INK, INK2 = '#0b0b0b', '#52514e'

VARIANTI = [
    ('from scratch', 'nnUNetTrainerWandb_250ep_arresto50'),
    ('ATLAS, FLAIR\nzero-initialised', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50'),
    ('ATLAS, T1 filter\nduplicated', 'nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50'),
]


def fuori_fold(trainer):
    """caso -> (cartella validation, Dice) dal fold in cui il caso era in validazione."""
    out = {}
    for s in sorted((RES / f'{trainer}__nnUNetPlansDa502__3d_fullres').glob('fold_*/validation/summary.json')):
        for m in json.load(open(s))['metric_per_case']:
            out[Path(m['reference_file']).name[:-7]] = (s.parent, m['metrics']['1']['Dice'])
    return out


def ritaglio(sl):
    pieno = sl > np.percentile(sl[sl > 0], 5) if (sl > 0).any() else sl > 0
    r = np.where(pieno.any(axis=1))[0]; c = np.where(pieno.any(axis=0))[0]
    if len(r) == 0:
        return slice(None), slice(None)
    return slice(max(r[0] - 8, 0), r[-1] + 8), slice(max(c[0] - 8, 0), c[-1] + 8)


def main():
    # formato della tesi: 16 cm a 300 dpi, due pazienti per riga (le tre ricette di ciascuno),
    # niente titolo ne' fold ne' numero di fetta: stanno nella didascalia o non servono
    plt.rcParams.update({'font.size': 7})
    pred = {tr: fuori_fold(tr) for _, tr in VARIANTI}
    casi = sorted(pred[VARIANTI[0][1]])
    nv = len(VARIANTI)
    nr, nc = (len(casi) + 1) // 2, 2 * nv
    fig, axes = plt.subplots(nr, nc, figsize=(6.3, 4.95), squeeze=False)
    fig.subplots_adjust(left=.005, right=.995, top=.925, bottom=.06, hspace=.13, wspace=.03)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, caso in enumerate(casi):
        r, c0 = i // 2, (i % 2) * nv
        gt = np.asarray(nib.load(str(PREP / 'gt_segmentations' / f'{caso}.nii.gz')).dataobj) > 0
        img = np.asarray(nib.load(str(RAW / 'imagesTr' / f'{caso}_0000.nii.gz')).dataobj, dtype=np.float32)
        k = scegli_fette(gt, 1, asse=0)[0]          # asse 0: la FLAIR in spazio T1 e' sagittale
        fondo = np.rot90(img[k]); box = ritaglio(fondo); fondo = fondo[box]
        lo, hi = finestra(fondo)
        g = np.rot90(gt[k])[box]
        for j, (_, tr) in enumerate(VARIANTI):
            ax = axes[r][c0 + j]
            valdir, dice = pred[tr][caso]
            pm = np.rot90(np.asarray(nib.load(str(valdir / f'{caso}.nii.gz')).dataobj)[k] > 0)[box]
            ax.imshow(fondo, cmap='gray', vmin=lo, vmax=hi, interpolation='lanczos')
            if g.any():
                ax.contour(g.astype(float), levels=[.5], colors=RIF, linewidths=.7, linestyles='--')
            if pm.any():
                ax.contour(pm.astype(float), levels=[.5], colors=PRED, linewidths=.7)
            ax.set_title((f'{caso} · ' if j == 0 else '') + f'Dice {dice:.2f}', fontsize=6.5, loc='left',
                         color=INK, pad=2)
    for c in range(nc):
        x = (c + .5) / nc
        fig.text(x, .995, VARIANTI[c % nv][0], ha='center', va='top', fontsize=7, color=INK,
                 fontweight='bold', linespacing=1.2)
    fig.legend(handles=[Line2D([], [], color=RIF, lw=1.5, ls='--', label='reference (clinician)'),
                        Line2D([], [], color=PRED, lw=1.5, label='prediction')],
               frameon=False, ncol=2, loc='lower center', fontsize=7.5, bbox_to_anchor=(.5, .0))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'predizioni_504.png', dpi=300, facecolor='white')
    print('scritto', FIG / 'predizioni_504.png')


if __name__ == '__main__':
    main()
