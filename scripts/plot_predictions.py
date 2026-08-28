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

VERDE = '#00d67d'   # ground truth
ROSSO = '#ff4d4d'   # predizione


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
    ap.add_argument('--n-casi', type=int, default=9)
    ap.add_argument('--fette', type=int, default=3)
    ap.add_argument('--asse', type=int, default=2, help='asse delle fette (default 2, assiale)')
    ap.add_argument('--trainer', default='nnUNetTrainerWandb__nnUNetPlans__3d_fullres')
    ap.add_argument('--fold', default='fold_0')
    args = ap.parse_args()

    nome = next(d.name for d in RAW.iterdir() if d.name.startswith(f'Dataset{args.dataset}'))
    valdir = RESULTS / nome / args.trainer / args.fold / 'validation'
    gtdir = PREP / nome / 'gt_segmentations'
    imgdir = RAW / nome / 'imagesTr'
    dj = json.loads((RAW / nome / 'dataset.json').read_text())
    canale = list(dj['channel_names'].values())[0]
    multiclasse = len(dj['labels']) > 2

    s = json.loads((valdir / 'summary.json').read_text())
    classe = sorted(s['mean'].keys(), key=int)[0]
    casi = sorted(((Path(c['reference_file']).name[:-7], c['metrics'][classe]['Dice'])
                   for c in s['metric_per_case']), key=lambda t: -t[1])
    scelti = campiona(casi, args.n_casi)

    nr, nc = len(scelti), args.fette
    fig, axes = plt.subplots(nr, nc, figsize=(3.1 * nc, 3.2 * nr), squeeze=False)
    fig.patch.set_facecolor('white')

    for r, (caso, dice) in enumerate(scelti):
        gt = np.asarray(nib.load(str(gtdir / f'{caso}.nii.gz')).dataobj)
        pr = np.asarray(nib.load(str(valdir / f'{caso}.nii.gz')).dataobj)
        img = np.asarray(nib.load(str(imgdir / f'{caso}_0000.nii.gz')).dataobj, dtype=np.float32)
        gtb, prb = gt > 0, pr > 0
        lo, hi = finestra(img)
        for c, k in enumerate(scegli_fette(gtb, nc, args.asse)):
            ax = axes[r][c]
            sl = [slice(None)] * 3
            sl[args.asse] = k
            fondo = np.rot90(img[tuple(sl)])
            ax.imshow(fondo, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
            g, p = np.rot90(gtb[tuple(sl)]), np.rot90(prb[tuple(sl)])
            if multiclasse:
                mc = np.rot90(pr[tuple(sl)]).astype(float)
                ax.imshow(np.ma.masked_where(mc == 0, mc), cmap='tab10', alpha=.45,
                          vmin=0, vmax=10, interpolation='nearest')
            else:
                if p.any():
                    ax.contour(p, levels=[.5], colors=ROSSO, linewidths=1.4)
                if g.any():
                    ax.contour(g, levels=[.5], colors=VERDE, linewidths=1.0, linestyles='--')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(f'{caso}\nDice {dice:.3f}', fontsize=8.5,
                              color=('#b00020' if dice < 0.3 else '#333'))
            ax.set_title(f'fetta {k}', fontsize=7.5, color='#666')

    legenda = ([Line2D([], [], color=VERDE, ls='--', label='ground truth'),
                Line2D([], [], color=ROSSO, label='predizione')] if not multiclasse
               else [Line2D([], [], color='#888', label='predizione (colore = classe)')])
    fig.legend(handles=legenda, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f'{nome} — {args.fold}, canale {canale} — casi campionati per Dice '
                 f'(migliori, mediani, peggiori)', fontsize=11)
    fig.tight_layout(rect=[0, 0.022, 1, 0.975])

    out = PROGETTO / 'reports' / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    f = out / f'predizioni_{args.dataset}_{args.fold}.png'
    fig.savefig(f, dpi=110, facecolor='white')
    print(f'{len(scelti)} casi -> {f.relative_to(PROGETTO)}')


if __name__ == '__main__':
    main()
