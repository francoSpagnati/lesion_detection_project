#!/usr/bin/env python
"""Due figure per la tesi sui dati pubblici, in inglese.

    python scripts/figura_dati_pubblici.py
    # -> reports/figures/esempi_dataset_pubblici.png   una fetta per WMH, ATLAS, DBB con il riferimento
    # -> reports/figures/atlas_mancato_trovato.png     un caso ATLAS mancato da 502 e uno trovato

PERCHE'. Il capitolo sui dati presenta i tre dataset pubblici: serve vedere che cosa contiene il
riferimento di ciascuno (lesione in FLAIR, lesione in T1, sei tessuti). La seconda figura e' il
risultato di failure_analysis_502.md in immagine: il caso mancato ha una lesione con l'intensita'
del tessuto attorno (rapporto lesione / cervello 0.91), quello trovato e' una cavita' scura (0.64).

COME. Immagini e etichette di nnUNet_raw (labelsTr: le gt_segmentations sono state cancellate per
spazio e ne sono copie), predizioni della validazione del fold 0 di 502. Riorientamento canonico
(RAS) con nibabel, fetta assiale con piu' etichetta, anteriore in alto e convenzione radiologica
(destra del paziente a sinistra), estensione in mm: stessa scala in tutti i pannelli.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROGETTO = Path(__file__).resolve().parent.parent
RAW = PROGETTO / 'nnUNet/nnUNet_raw'
VAL502 = PROGETTO / 'nnUNet/nnUNet_results/Dataset502_ATLAS/nnUNetTrainerWandb__nnUNetPlans__3d_fullres/fold_0/validation'
FIG = PROGETTO / 'reports/figures'
INK, INK2 = '#1f2328', '#59636e'
BLU, ARANCIO = '#2a78d6', '#eb6834'          # riferimento e predizione, come nelle figure di AInCP
TESSUTI = [('CSF', '#2a78d6'), ('grey matter', '#1baf7a'), ('white matter', '#eda100'),
           ('deep grey matter', '#e87ba4'), ('brainstem', '#4a3aa7'), ('cerebellum', '#eb6834')]


def carica(f):
    img = nib.as_closest_canonical(nib.load(str(f)))
    return np.asanyarray(img.dataobj), img.header.get_zooms()[:3]


def fetta(v, k):
    # RAS: colonne verso la destra del paziente, righe verso l'anteriore -> anteriore in alto,
    # destra del paziente a sinistra
    return np.fliplr(np.flipud(v[:, :, k].T))


def finestra(a):
    lo, hi = np.percentile(a[a > 0], [0.5, 99.5]) if (a > 0).any() else (0, 1)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def pannello(ax, img, sp, k, titolo):
    s = fetta(img, k)
    est = (0, s.shape[1] * sp[0], 0, s.shape[0] * sp[1])
    ax.imshow(finestra(s), cmap='gray', extent=est, interpolation='lanczos')
    ax.set_title(titolo, loc='left', fontsize=10.5, color=INK, linespacing=1.3)
    ax.set_axis_off()
    ax.text(0.02, 0.98, 'R', transform=ax.transAxes, color='white', fontsize=9, va='top', fontweight='bold')
    ax.text(0.98, 0.98, 'L', transform=ax.transAxes, color='white', fontsize=9, va='top', ha='right',
            fontweight='bold')
    return est


def contorno(ax, m, k, est, colore, ls='-'):
    s = fetta(m.astype(float), k)
    if s.any():
        ax.contour(s, levels=[0.5], colors=[colore], linewidths=1.4, linestyles=ls,
                   extent=est, origin='upper')


def esempi():
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.3), dpi=170)
    # WMH: FLAIR (canale 0) e lesione
    img, sp = carica(RAW / 'Dataset501_WMH/imagesTr/Amsterdam_GE3T_100_0000.nii.gz')
    lab, _ = carica(RAW / 'Dataset501_WMH/labelsTr/Amsterdam_GE3T_100.nii.gz')
    k = int(np.argmax((lab == 1).sum((0, 1))))
    est = pannello(ax[0], img, sp, k, 'WMH 2017 · FLAIR\nwhite matter hyperintensities (adult)')
    contorno(ax[0], lab == 1, k, est, ARANCIO)
    # ATLAS: T1 e lesione
    img, sp = carica(RAW / 'Dataset502_ATLAS/imagesTr/sub-r004s009_0000.nii.gz')
    lab, _ = carica(RAW / 'Dataset502_ATLAS/labelsTr/sub-r004s009.nii.gz')
    k = int(np.argmax((lab == 1).sum((0, 1))))
    est = pannello(ax[1], img, sp, k, 'ATLAS v2.0 · T1-weighted\nchronic stroke lesion (adult)')
    contorno(ax[1], lab == 1, k, est, ARANCIO)
    # DBB: T1 e sei tessuti
    img, sp = carica(RAW / 'Dataset503_DBB/imagesTr/sub-0419_0000.nii.gz')
    lab, _ = carica(RAW / 'Dataset503_DBB/labelsTr/sub-0419.nii.gz')
    k = int(np.argmax((lab == 4).sum((0, 1))))                     # fetta con i nuclei profondi
    est = pannello(ax[2], img, sp, k, 'DBB · T1-weighted\nsix tissue classes (paediatric, no lesions)')
    s = fetta(lab.astype(float), k)
    cmap = ListedColormap([c for _, c in TESSUTI])
    ax[2].imshow(np.ma.masked_where(s == 0, s), cmap=cmap, norm=BoundaryNorm(np.arange(0.5, 7.5), 6),
                 alpha=0.5, extent=est, interpolation='nearest')
    h = [Line2D([], [], color=ARANCIO, lw=2, label='annotated lesion')] + \
        [Patch(facecolor=c, alpha=0.7, label=n) for n, c in TESSUTI]
    fig.legend(handles=h, loc='lower center', ncol=7, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08, wspace=0.05)
    out = FIG / 'esempi_dataset_pubblici.png'
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    return out


def mancato_trovato():
    import json
    dice = {Path(m['reference_file']).name[:-7]: m['metrics']['1']['Dice']
            for m in json.loads((VAL502 / 'summary.json').read_text())['metric_per_case']}
    casi = [('sub-r040s049', 'missed', 0.91), ('sub-r004s009', 'detected', 0.64)]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 6.2), dpi=170)
    for a, (c, esito, rap) in zip(ax, casi):
        img, sp = carica(RAW / f'Dataset502_ATLAS/imagesTr/{c}_0000.nii.gz')
        ref, _ = carica(RAW / f'Dataset502_ATLAS/labelsTr/{c}.nii.gz')
        pr, _ = carica(VAL502 / f'{c}.nii.gz')
        k = int(np.argmax((ref == 1).sum((0, 1))))
        mm3 = float((ref == 1).sum() * np.prod(sp))
        est = pannello(a, img, sp, k, f'{c} · lesion {esito}\n{mm3:,.0f} mm³ · lesion / brain intensity {rap:.2f} · '
                                      f'Dice {dice[c]:.2f}')
        contorno(a, ref == 1, k, est, BLU)
        contorno(a, pr == 1, k, est, ARANCIO, ls='--')
    fig.legend(handles=[Line2D([], [], color=BLU, lw=2, label='reference'),
                        Line2D([], [], color=ARANCIO, lw=2, ls='--', label='prediction of model 502 (T1 only)')],
               loc='lower center', ncol=2, frameon=False, fontsize=10, labelcolor=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.07, wspace=0.04)
    out = FIG / 'atlas_mancato_trovato.png'
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    return out


if __name__ == '__main__':
    print(esempi())
    print(mancato_trovato())
