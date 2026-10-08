#!/usr/bin/env python
"""Figure di confronto fra predizione e ground truth, sui casi di validation.

**Perche' sulla validation e non sul test.** Guardare le immagini del test set
significa usarlo: nel momento in cui si vede come va e in base a quello si cambia
qualcosa, quei dati diventano un secondo validation e il numero finale non e' piu'
onesto. I casi di validation della cross-validation danno la stessa informazione
diagnostica - sono casi che il modello non ha visto in training - senza bruciare
niente. Le predizioni sono gia' su disco: non serve rilanciare inferenza.

**Come sceglie i casi.** Non a caso: li ordina per Dice e ne campiona lungo tutto
lo spettro (i migliori, i mediani, i peggiori). Una griglia di soli casi riusciti
non dice niente di utile, e una di soli fallimenti nemmeno.

**Come sceglie le fette.** Quelle con piu' lesione nella ground truth, distanziate
fra loro, cosi' si guarda dove c'e' qualcosa da vedere invece che al centro del
volume per convenzione.

Uso:
    python scripts/plot_predictions.py 501
    python scripts/plot_predictions.py 502 --n-casi 12 --fette 4
"""
import argparse
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
RESULTS = Path(os.environ.get('nnUNet_results', PROGETTO / 'nnUNet' / 'nnUNet_results'))
PREP = Path(os.environ.get('nnUNet_preprocessed', PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'))
RAW = Path(os.environ.get('nnUNet_raw', PROGETTO / 'nnUNet' / 'nnUNet_raw'))

# riferimento blu tratteggiato, predizione arancio: la coppia validata delle figure della tesi
# (il verde e rosso di prima non si distingue con la deuteranopia)
RIF = '#2a78d6'
PRED = '#eb6834'


def finestra(img):
    """Contrasto robusto: taglia le code, che nelle MRI sono grasso e artefatti."""
    lo, hi = np.percentile(img[img > 0], (1, 99)) if (img > 0).any() else (0, 1)
    return lo, max(hi, lo + 1e-6)


def scegli_fette(gt, n, asse=2):
    """Le fette con piu' foreground, tenute distanti fra loro per non mostrare
    tre volte la stessa lesione."""
    carico = gt.sum(axis=tuple(i for i in range(3) if i != asse))
    ordine = np.argsort(carico)[::-1]
    scelte, minimo = [], max(3, gt.shape[asse] // 20)
    for k in ordine:
        if carico[k] == 0:
            break
        if all(abs(k - s) >= minimo for s in scelte):
            scelte.append(int(k))
        if len(scelte) == n:
            break
    while len(scelte) < n:                     # ground truth vuota o quasi
        scelte.append(gt.shape[asse] // 2 + 4 * len(scelte))
    return sorted(scelte)


def campiona(casi, n):
    """Migliori, mediani e peggiori: tre gruppi da n/3, senza ripetizioni."""
    if len(casi) <= n:
        return casi
    k = max(1, n // 3)
    meta = len(casi) // 2
    idx = list(range(k)) + list(range(meta - k // 2, meta - k // 2 + k)) + list(range(len(casi) - k, len(casi)))
    visti, out = set(), []
    for i in idx:
        if 0 <= i < len(casi) and i not in visti:
            visti.add(i)
            out.append(casi[i])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--n-casi', type=int, default=3)
    ap.add_argument('--fette', type=int, default=3)
    ap.add_argument('--asse', type=int, default=2, help='asse delle fette (default 2, assiale)')
    ap.add_argument('--trainer', default='nnUNetTrainerWandb__nnUNetPlans__3d_fullres')
    ap.add_argument('--fold', default='fold_0')
    args = ap.parse_args()

    nome = next(d.name for d in RAW.iterdir() if d.name.startswith(f'Dataset{args.dataset}'))
    valdir = RESULTS / nome / args.trainer / args.fold / 'validation'
    gtdir = PREP / nome / 'gt_segmentations'
    if not any(gtdir.glob('*.nii.gz')):          # cancellate per spazio: sono copie di labelsTr
        gtdir = PROGETTO / 'nnUNet/nnUNet_raw' / nome / 'labelsTr'
    imgdir = RAW / nome / 'imagesTr'
    dj = json.loads((RAW / nome / 'dataset.json').read_text())
    canale = list(dj['channel_names'].values())[0]
    multiclasse = len(dj['labels']) > 2

    s = json.loads((valdir / 'summary.json').read_text())
    classe = sorted(s['mean'].keys(), key=int)[0]
    casi = sorted(((Path(c['reference_file']).name[:-7], c['metrics'][classe]['Dice'])
                   for c in s['metric_per_case']), key=lambda t: -t[1])
    scelti = campiona(casi, args.n_casi)

    # formato della tesi: 16 cm di larghezza a 300 dpi, niente titolo (sta nella didascalia), nessun
    # numero di fetta (inutile a chi guarda), orientamento radiologico e proporzioni in mm
    nr, nc = len(scelti), args.fette
    plt.rcParams.update({'font.size': 8})
    fig, axes = plt.subplots(nr, nc, figsize=(6.3, 1.72 * nr + 0.25), squeeze=False)
    fig.patch.set_facecolor('white')

    for r, (caso, dice) in enumerate(scelti):
        carica = lambda f: nib.as_closest_canonical(nib.load(str(f)))
        ig = carica(imgdir / f'{caso}_0000.nii.gz')
        sp = ig.header.get_zooms()[:3]
        img = np.asarray(ig.dataobj, dtype=np.float32)
        gt = np.asarray(carica(gtdir / f'{caso}.nii.gz').dataobj)
        pr = np.asarray(carica(valdir / f'{caso}.nii.gz').dataobj)
        gtb, prb = gt > 0, pr > 0
        lo, hi = finestra(img)
        # riquadro comune alle fette del caso: la testa, cosi' il cervello riempie il pannello
        testa = img > lo + 0.08 * (hi - lo)
        xs, ys = np.where(testa.any(axis=2))
        x0, x1, y0, y1 = max(xs.min() - 3, 0), xs.max() + 4, max(ys.min() - 3, 0), ys.max() + 4
        # RAS -> anteriore in alto, destra del paziente a sinistra (convenzione radiologica)
        orienta = lambda a: np.fliplr(np.flipud(a[x0:x1, y0:y1].T))
        est = (0, (x1 - x0) * sp[0], 0, (y1 - y0) * sp[1])
        for c, k in enumerate(scegli_fette(gtb, nc, 2)):
            ax = axes[r][c]
            ax.imshow(orienta(img[:, :, k]), cmap='gray', vmin=lo, vmax=hi, extent=est, interpolation='lanczos')
            g, p = orienta(gtb[:, :, k]).astype(float), orienta(prb[:, :, k]).astype(float)
            if multiclasse:
                mc = orienta(pr[:, :, k]).astype(float)
                ax.imshow(np.ma.masked_where(mc == 0, mc), cmap='tab10', alpha=.45,
                          vmin=0, vmax=10, interpolation='nearest', extent=est)
            else:
                if p.any():
                    ax.contour(p, levels=[.5], colors=PRED, linewidths=0.9, extent=est, origin='upper')
                if g.any():
                    ax.contour(g, levels=[.5], colors=RIF, linewidths=0.8, linestyles='--',
                               extent=est, origin='upper')
            ax.set_xticks([]); ax.set_yticks([])
            for spn in ax.spines.values():
                spn.set_visible(False)
            if c == 0:
                ax.set_ylabel(f'{caso}\nDice {dice:.2f}', fontsize=8, color='#333')
                ax.text(0.03, 0.97, 'R', transform=ax.transAxes, color='white', fontsize=6.5, va='top')
                ax.text(0.97, 0.97, 'L', transform=ax.transAxes, color='white', fontsize=6.5, va='top', ha='right')

    legenda = ([Line2D([], [], color=RIF, ls='--', label='reference'),
                Line2D([], [], color=PRED, label='prediction')] if not multiclasse
               else [Line2D([], [], color='#888', label='prediction (colour = class)')])
    fig.legend(handles=legenda, loc='lower center', ncol=2, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.995, bottom=0.25 / (1.72 * nr + 0.25) + 0.01,
                        wspace=0.02, hspace=0.04)

    out = PROGETTO / 'reports' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    f = out / f'predizioni_{args.dataset}_{args.fold}.png'
    fig.savefig(f, dpi=300, facecolor='white')
    print(f'{len(scelti)} casi -> {f.relative_to(PROGETTO)}')


if __name__ == '__main__':
    main()
