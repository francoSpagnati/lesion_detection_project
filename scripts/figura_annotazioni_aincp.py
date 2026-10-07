#!/usr/bin/env python
"""Le annotazioni dei 17 pazienti AInCP, una fetta per paziente, e il contrasto per fenotipo.

    python scripts/figura_annotazioni_aincp.py
    # -> reports/figures/aincp/annotazioni_17.png        (sostituisce panoramica_maschere.png, primo lotto)
    # -> reports/figures/aincp/contrasto_fenotipi_17.png (sostituisce rapporti_intensita.png, primo lotto)

PERCHE'. Le due figure del capitolo sui dati erano del notebook del primo lotto (10 pazienti). Qui
le stesse due cose su tutti i 17, in inglese, per la tesi.

COME. Griglia T1 nativa, FLAIR e ROI portate con la rigida della pipeline (come in
figura_t1_flair_intro.py, ROI lineare con soglia 0.5). Per ogni paziente la fetta assiale con piu'
lesione, ritagliata sul cervello, FLAIR con i contorni dei fenotipi (il ventricolo in grigio:
annotato ma fuori dal bersaglio). Orientamento dall'affine, convenzione radiologica. Il contrasto
legge reports/contrasto_fenotipi_aincp.json (scripts/contrasto_fenotipi_aincp.py): niente ricalcoli.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_aincp as pa  # noqa: E402
import postprocessing_aincp as pp  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
FIG = PROGETTO / 'reports/figures/aincp'
COL = {'ROI_1': '#2a78d6', 'ROI_2': '#eb6834', 'ROI_4': '#1baf7a', 'ROI_3': '#c3c2b7'}
NOME = {'ROI_1': 'gliosis', 'ROI_2': 'cystic lesion', 'ROI_4': 'encephalomalacia',
        'ROI_3': 'enlarged ventricle (not in the target)'}
INK, INK2, GRIGLIA = '#1f2328', '#59636e', '#e6e5e1'


def carica(paz):
    import ants
    import nibabel as nib
    s = pp.sessione(paz)
    c = pa.trova_casi()[paz]
    t1 = ants.image_read(str(s / 'T1_N4.nii.gz'))
    mat = [str(s / 'transforms' / 'FLAIR_to_T1_0.mat')]
    flair = ants.apply_transforms(fixed=t1, moving=ants.image_read(str(s / 'FLAIR_N4.nii.gz')),
                                  transformlist=mat, interpolator='linear').numpy()
    rif = ants.image_read(str(c['FLAIR']))
    roi = {}
    for r in COL:
        if r in c['roi']:
            m = np.squeeze(ants.image_read(str(c['roi'][r])).numpy()) > 0
            a = ants.apply_transforms(fixed=t1, moving=rif.new_image_like(m.astype('float32')),
                                      transformlist=mat, interpolator='linear')
            roi[r] = a.numpy() >= 0.5
    cer = np.asanyarray(nib.load(str(s / 'cervello_inT1.nii.gz')).dataobj) > 0.5
    assi = nib.aff2axcodes(nib.load(str(s / 'T1_N4.nii.gz')).affine)
    lesione = sum(roi[r].sum((0, 1)) for r in ('ROI_1', 'ROI_2', 'ROI_4') if r in roi)
    k = int(np.argmax(lesione))
    m = cer[:, :, k]
    xs, ys = np.where(m)
    x0, x1 = max(xs.min() - 6, 0), min(xs.max() + 6, m.shape[0])
    y0, y1 = max(ys.min() - 6, 0), min(ys.max() + 6, m.shape[1])

    def fetta(v):
        a = v[x0:x1, y0:y1, k].T
        if assi[1] == 'A':
            a = np.flipud(a)
        if assi[0] == 'R':
            a = np.fliplr(a)
        return a
    lo, hi = np.percentile(flair[cer], [0.5, 99.5])
    img = fetta(np.clip((flair - lo) / (hi - lo), 0, 1))
    sp = t1.spacing
    return paz, {'img': img, 'roi': {r: fetta(v.astype(float)) for r, v in roi.items()},
                 'estensione': (0, (x1 - x0) * sp[0], 0, (y1 - y0) * sp[1])}


def figura_annotazioni(dati, info):
    casi = sorted(dati, key=lambda p: -info[p]['percentuale_cervello_lesionato'])
    nc, nr = 6, 3
    fig, ax = plt.subplots(nr, nc, figsize=(2.6 * nc, 3.05 * nr + 0.6), dpi=170)
    for a in ax.ravel():
        a.set_axis_off()
    for a, p in zip(ax.ravel(), casi):
        d = dati[p]
        a.imshow(d['img'], cmap='gray', extent=d['estensione'], interpolation='lanczos')
        for r, f in d['roi'].items():
            if f.any():
                a.contour(np.flipud(f), levels=[0.5], colors=[COL[r]], linewidths=0.9 if r == 'ROI_3' else 1.1,
                          extent=d['estensione'], origin='upper')
        a.set_title(f"{p} · {pp.mm3(info[p]['lesione_mm3_nativo'])}\n"
                    f"{info[p]['percentuale_cervello_lesionato']:.2f}% of the brain",
                    fontsize=8.5, color=INK, loc='left', linespacing=1.3)
        a.text(0.03, 0.97, 'R', transform=a.transAxes, color='white', fontsize=7, va='top', fontweight='bold')
        a.text(0.97, 0.97, 'L', transform=a.transAxes, color='white', fontsize=7, va='top', ha='right',
               fontweight='bold')
    leg = [Line2D([], [], color=COL[r], lw=2, label=NOME[r]) for r in COL]
    fig.legend(handles=leg, loc='lower center', ncol=4, frameon=False, fontsize=10, labelcolor=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.06, wspace=0.04, hspace=0.25)
    out = FIG / 'annotazioni_17.png'
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    return out


def figura_contrasto():
    R = json.loads((PROGETTO / 'reports/contrasto_fenotipi_aincp.json').read_text())
    fen = [('gliosis', 'ROI_1'), ('cystic lesion', 'ROI_2'), ('encephalomalacia', 'ROI_4')]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), dpi=170, sharey=True)
    rng = np.random.default_rng(0)
    for a, seq in zip(ax, ('T1', 'FLAIR')):
        for i, (f, r) in enumerate(fen):
            v = np.array([x[f][seq] for x in R.values() if f in x])
            a.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=34, color=COL[r], edgecolor='white',
                      linewidth=1.2, zorder=3)
            a.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=2.2, zorder=4)
            a.text(i + 0.29, np.median(v), f'{np.median(v):.2f}', va='center', fontsize=9, color=INK)
        a.axhline(1, color=INK2, lw=1.1, zorder=2)
        a.set_xticks(range(3), [f"{f}\n({sum(f in x for x in R.values())} patients)" for f, _ in fen],
                     fontsize=9.5, color=INK)
        a.set_title(f'{seq}-weighted' if seq == 'T1' else 'FLAIR', loc='left', fontsize=12, color=INK)
        a.grid(axis='y', color=GRIGLIA, zorder=0)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        a.spines['left'].set_color(GRIGLIA)
        a.spines['bottom'].set_color(GRIGLIA)
        a.tick_params(colors=INK2, length=0)
        a.set_xlim(-0.5, 2.7)
    ax[0].set_ylabel('lesion / healthy brain intensity', color=INK2)
    ax[0].text(2.65, 1.03, 'same as healthy tissue', ha='right', va='bottom', fontsize=8.5, color=INK2)
    fig.tight_layout()
    out = FIG / 'contrasto_fenotipi_17.png'
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    return out


def main():
    info = {}
    for f in sorted((PROGETTO / 'data/processed/AINCP').glob('*/pipeline.json')):
        d = json.loads(f.read_text())
        info[d['paziente']] = d
    with Pool(6) as p:
        dati = dict(p.map(carica, sorted(pa.trova_casi())))
    print(figura_annotazioni(dati, info))
    print(figura_contrasto())


if __name__ == '__main__':
    main()
