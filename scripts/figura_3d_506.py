#!/usr/bin/env python
"""Viste 3D del modello della tesi (Z) su pazienti di test: lesioni trovate, mancate, falsi positivi.

    python scripts/figura_3d_506.py            # -> reports/figures/aincp/3d_506_<paziente>.png

Legge l'esportazione per Slicer (esportazioni/506_test/<paziente>/: annotazione e predizione di Z
nella griglia T1 nativa, gia' con il postprocessing congelato) e la maschera del cervello di
SynthStrip. Nessuna metrica nuova: la figura mostra in tre dimensioni le stesse categorie dei file
.seg.nrrd, con gli stessi colori:

- verde, **lesione trovata**: componente annotata toccata dalla predizione;
- blu, **lesione mancata**: componente annotata che la predizione non tocca;
- arancione, **falso positivo**: componente predetta che non tocca nessuna lesione annotata.

Il cervello e' una superficie grigia trasparente (marching cubes sulla maschera). Le componenti
sotto 50 mm³ in superficie sarebbero puntini invisibili a scala di cervello intero: sono disegnate
come sfere al loro baricentro, con area proporzionale al volume, e la didascalia lo dice. Assi in
mm nello spazio del paziente (RAS).

Solo matplotlib e scikit-image, cosi' gira sul server senza schermo e senza VTK. Le immagini sono
di pazienti: stanno in reports/figures/, che e' fuori da git.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import ndimage
from skimage.measure import marching_cubes

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
ESP = PROGETTO / 'esportazioni/506_test'
FIG = PROGETTO / 'reports/figures/aincp'
# stessi colori dei segmenti per Slicer (esporta_slicer_506.py), palette validata con la skill dataviz
COL = {'trovata': '#1baf7a', 'mancata': '#2a78d6', 'fp': '#eb6834'}
NOME = {'trovata': 'lesione trovata', 'mancata': 'lesione mancata', 'fp': 'falso positivo'}
INCHIOSTRO, GRIGIO = '#1f2328', '#59636e'
PICCOLA_MM3 = 50
CASI = {'FSM_049': 'lesione estesa, segmentata bene',
        'FSM_030': 'la lesione principale trovata, le altre dieci mancate (da 1.5 a 912 mm³)',
        'FSM_046': 'falso positivo nel ventricolo dilatato'}


def superficie(maschera: np.ndarray, affine: np.ndarray, passo: int = 1, liscia: float = 0.0):
    m = maschera.astype(np.float32)
    if liscia:
        m = ndimage.gaussian_filter(m, liscia)
    v, f, _, _ = marching_cubes(np.pad(m, 1), 0.5, step_size=passo)
    v = nib.affines.apply_affine(affine, v - 1)
    return v[f]


def categorie(ref: np.ndarray, pred: np.ndarray, vox: float):
    """Componenti per categoria: lista di (maschera, volume in mm³)."""
    out = {k: [] for k in COL}
    lr, nr = ndimage.label(ref, structure=pp.STRUTTURA)
    for i in range(1, nr + 1):
        c = lr == i
        v = c.sum() * vox
        if v < pp.SOGLIA_LESIONE_MM3:
            continue
        out['trovata' if (c & pred).any() else 'mancata'].append((c, v))
    lp, np_ = ndimage.label(pred, structure=pp.STRUTTURA)
    for j in range(1, np_ + 1):
        c = lp == j
        v = c.sum() * vox
        if v >= pp.SOGLIA_LESIONE_MM3 and not (c & ref).any():
            out['fp'].append((c, v))
    return out


def figura(paz: str):
    d = ESP / paz
    ann = nib.load(str(d / 'annotazione.nii.gz'))
    ref = np.asanyarray(ann.dataobj) > 0.5
    pred = np.asanyarray(nib.load(str(d / 'predizione_Z.nii.gz')).dataobj) > 0.5
    cer_img = nib.load(str(pp.sessione(paz) / 'cervello_inT1.nii.gz'))
    cer = np.asanyarray(cer_img.dataobj) > 0.5
    assert cer.shape == ref.shape == pred.shape and np.allclose(cer_img.affine, ann.affine)
    aff, vox = ann.affine, float(np.prod(ann.header.get_zooms()[:3]))
    riep = json.loads((d / 'riepilogo.json').read_text())['Z']
    cat = categorie(ref, pred, vox)

    cervello = superficie(cer, aff, passo=2, liscia=1.0)
    lo, hi = cervello.reshape(-1, 3).min(0), cervello.reshape(-1, 3).max(0)
    fig = plt.figure(figsize=(13, 5.0), dpi=150)
    viste = [('vista laterale sinistra', 8, 180, 'APSI'), ('vista laterale destra', 8, 0, 'APSI'),
             ('vista dall\'alto', 89, -90, 'APLR')]
    centro = (lo + hi) / 2
    # lettere di orientamento poste fuori dal cervello lungo gli assi RAS: la proiezione le mette
    # nel punto giusto di ogni vista
    margine = 12
    LETTERE = {'A': (centro[0], hi[1] + margine, centro[2]), 'P': (centro[0], lo[1] - margine, centro[2]),
               'S': (centro[0], centro[1], hi[2] + margine), 'I': (centro[0], centro[1], lo[2] - margine),
               'L': (lo[0] - margine, centro[1], centro[2]), 'R': (hi[0] + margine, centro[1], centro[2])}
    for k, (titolo, elev, azim, lettere) in enumerate(viste):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        ax.add_collection3d(Poly3DCollection(cervello, facecolor='#9aa4ae', edgecolor='none', alpha=0.06))
        for nome, comp in cat.items():
            grandi = [c for c, v in comp if v >= PICCOLA_MM3]
            piccole = [(c, v) for c, v in comp if v < PICCOLA_MM3]
            if grandi:
                m = np.zeros_like(ref)
                for c in grandi:
                    m |= c
                ax.add_collection3d(Poly3DCollection(superficie(m, aff), facecolor=COL[nome], edgecolor='none',
                                                     alpha=0.9))
            if piccole:
                xyz = nib.affines.apply_affine(aff, np.array([ndimage.center_of_mass(c) for c, _ in piccole]))
                ax.scatter(*xyz.T, s=[30 + 12 * v for _, v in piccole], color=COL[nome], edgecolor='white',
                           linewidth=0.8, depthshade=False, alpha=1.0)
        for l in lettere:
            ax.text(*LETTERE[l], l, color=GRIGIO, fontsize=11, ha='center', va='center', fontweight='bold')
        ax.set_xlim(lo[0] - margine, hi[0] + margine); ax.set_ylim(lo[1] - margine, hi[1] + margine)
        ax.set_zlim(lo[2] - margine, hi[2] + margine)
        ax.set_box_aspect(hi - lo + 2 * margine)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(titolo, fontsize=10, color=GRIGIO)
    n = {k: len(v) for k, v in cat.items()}
    fig.suptitle(f'{paz} (test) — {CASI[paz]}\nmodello Z: Dice {riep["dice"]:.2f}, lesioni trovate '
                 f'{riep["lesioni_rilevate"]}, falsi positivi {riep["falsi_positivi"]}; lesione annotata '
                 f'{pp.mm3(riep["mm3_vero"])}, predetta {pp.mm3(riep["mm3_predetto"])}',
                 fontsize=12, color=INCHIOSTRO, y=0.98)
    leg = [Line2D([], [], marker='s', ls='', markersize=10, color=COL[k], label=f'{NOME[k]} ({n[k]})') for k in COL]
    leg.append(Line2D([], [], marker='o', ls='', markersize=7, color=GRIGIO, markeredgecolor='white',
                      label=f'sfera: componente sotto {PICCOLA_MM3} mm³, ingrandita'))
    fig.legend(handles=leg, loc='lower center', ncol=4, frameon=False, fontsize=10, labelcolor=INCHIOSTRO)
    fig.subplots_adjust(left=0, right=1, top=0.84, bottom=0.07, wspace=0)
    out = FIG / f'3d_506_{paz}.png'
    fig.savefig(out, facecolor='white')
    plt.close(fig)
    print(out, n)


if __name__ == '__main__':
    for p in (sys.argv[1:] or CASI):
        figura(p)
