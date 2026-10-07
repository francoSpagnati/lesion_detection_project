#!/usr/bin/env python
"""Figura introduttiva della tesi: la stessa fetta in T1 e in FLAIR, con i tre fenotipi di lesione.

    python scripts/figura_t1_flair_intro.py [--paziente FSM_032]
    # -> reports/figures/aincp/t1_flair_fenotipi.png

PERCHE'. Tutta la tesi parte da un fatto visivo: le due sequenze vedono lesioni diverse. La gliosi e'
chiara in FLAIR e quasi invisibile in T1; le cavita' (lesione cistica, encefalomalacia) sono nette e
scure in T1. Questa figura lo mostra prima di qualunque numero. Testo in inglese: va nella tesi.

COME. Un paziente di sviluppo (mai un paziente di test), nella griglia T1 nativa, dove la FLAIR e le
ROI arrivano con la rigida FLAIR -> T1 gia' calcolata dalla pipeline (immagine lineare, ROI lineare
con soglia 0.5, come l'etichetta pulita). La fetta assiale e il paziente si scelgono in automatico:
quella in cui il fenotipo meno rappresentato fra gliosi e cavita' ha piu' voxel, cosi' i tre
fenotipi stanno nella stessa immagine. Sopra le immagini pulite, sotto le stesse con i contorni dei
fenotipi. Convenzione radiologica (la destra del paziente a sinistra), come le altre figure.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
OUT = PROGETTO / 'reports/figures/aincp/t1_flair_fenotipi.png'
# stessi colori per fenotipo di figura_fenotipi_504.py (palette validata con la skill dataviz)
COL = {'ROI_1': '#2a78d6', 'ROI_2': '#eb6834', 'ROI_4': '#1baf7a'}
NOME = {'ROI_1': 'gliosis', 'ROI_2': 'cystic lesion', 'ROI_4': 'encephalomalacia'}
INCHIOSTRO, GRIGIO = '#1f2328', '#59636e'
CANDIDATI = ['FSM_028', 'FSM_032', 'FSM_035', 'FSM_036', 'FSM_040', 'FSM_045', 'FSM_056']  # solo sviluppo


def carica(paz: str):
    import ants
    import pipeline_aincp as pa
    s = pp.sessione(paz)
    c = pa.trova_casi()[paz]
    t1 = ants.image_read(str(s / 'T1_N4.nii.gz'))
    mat = [str(s / 'transforms' / 'FLAIR_to_T1_0.mat')]
    flair = ants.apply_transforms(fixed=t1, moving=ants.image_read(str(s / 'FLAIR_N4.nii.gz')),
                                  transformlist=mat, interpolator='linear')
    rif = ants.image_read(str(c['FLAIR']))
    roi = {}
    for r in COL:
        if r in c['roi']:
            m = np.squeeze(ants.image_read(str(c['roi'][r])).numpy()) > 0
            a = ants.apply_transforms(fixed=t1, moving=rif.new_image_like(m.astype('float32')),
                                      transformlist=mat, interpolator='linear')
            roi[r] = a.numpy() >= 0.5
    import nibabel as nib
    img = nib.load(str(s / 'cervello_inT1.nii.gz'))
    assert img.shape == t1.shape
    cer = np.asanyarray(img.dataobj) > 0.5
    # direzione degli assi del paziente: cambia fra pazienti (FSM_036 e' LAS, FSM_046 RAS)
    assi = nib.aff2axcodes(nib.load(str(s / 'T1_N4.nii.gz')).affine)
    assert assi[2] in 'SI'
    return t1.numpy(), flair.numpy(), roi, cer, np.array(t1.spacing), assi


def punteggio(roi):
    """Per ogni fetta assiale: voxel del fenotipo meno presente fra gliosi e cavita'."""
    if 'ROI_1' not in roi or not ({'ROI_2', 'ROI_4'} & set(roi)):
        return None
    glio = roi['ROI_1'].sum((0, 1))
    cav = sum(roi[r].sum((0, 1)) for r in ('ROI_2', 'ROI_4') if r in roi)
    return np.minimum(glio, cav)


def finestra(img, cer):
    lo, hi = np.percentile(img[cer], [0.5, 99.5])
    return np.clip((img - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--paziente')
    a = ap.parse_args()
    if a.paziente:
        paz = a.paziente
        dati = carica(paz)
        k = int(np.argmax(punteggio(dati[2])))
    else:
        migliore = None
        for p in CANDIDATI:
            d = carica(p)
            sc = punteggio(d[2])
            if sc is not None and (migliore is None or sc.max() > migliore[0]):
                migliore = (sc.max(), p, d, int(np.argmax(sc)))
            print(p, None if sc is None else int(sc.max()), flush=True)
        _, paz, dati, k = migliore
    t1, flair, roi, cer, sp, assi = dati
    # riquadro attorno al cervello nella fetta, con un margine
    m = cer[:, :, k]
    xs, ys = np.where(m)
    x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 8, m.shape[0])
    y0, y1 = max(ys.min() - 20, 0), min(ys.max() + 20, m.shape[1])
    # righe = anteriore in alto, colonne = destra del paziente a sinistra (radiologica), vedi fetta()
    def fetta(v):
        # righe = asse y, colonne = asse x; si ribalta in base all'affine del paziente, cosi' che
        # l'anteriore stia in alto e la destra del paziente a sinistra (convenzione radiologica)
        a = v[x0:x1, y0:y1, k].T
        if assi[1] == 'A':
            a = np.flipud(a)
        if assi[0] == 'R':
            a = np.fliplr(a)
        return a
    estensione = (0, (x1 - x0) * sp[0], 0, (y1 - y0) * sp[1])
    fig, ax = plt.subplots(2, 2, figsize=(8.6, 10.2), dpi=200)
    for j, (nome, img) in enumerate((('T1-weighted', t1), ('FLAIR', flair))):
        g = fetta(finestra(img, cer))
        for i in range(2):
            ax[i, j].imshow(g, cmap='gray', extent=estensione, interpolation='lanczos')
            ax[i, j].set_axis_off()
            if i == 1:
                for r, mk in roi.items():
                    f = fetta(mk.astype(float))
                    if f.any():
                        ax[i, j].contour(np.flipud(f), levels=[0.5], colors=[COL[r]], linewidths=1.3,
                                         extent=estensione, origin='upper')
            ax[i, j].text(0.02, 0.98, 'R', transform=ax[i, j].transAxes, color='white', fontsize=10,
                          va='top', ha='left', fontweight='bold')
            ax[i, j].text(0.98, 0.98, 'L', transform=ax[i, j].transAxes, color='white', fontsize=10,
                          va='top', ha='right', fontweight='bold')
        ax[0, j].set_title(nome, fontsize=13, color=INCHIOSTRO, pad=6)
    for i, et in enumerate(('a', 'b')):
        ax[i, 0].text(-0.06, 1.0, et, transform=ax[i, 0].transAxes, fontsize=14, fontweight='bold',
                      color=INCHIOSTRO, va='top')
    leg = [Line2D([], [], color=COL[r], lw=2.2, label=NOME[r]) for r in roi if fetta(roi[r]).any()]
    fig.legend(handles=leg, loc='lower center', ncol=len(leg), frameon=False, fontsize=11, labelcolor=INCHIOSTRO)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.96, bottom=0.05, wspace=0.03, hspace=0.04)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor='white')
    print(OUT, paz, 'fetta', k, {r: int(fetta(mk).sum()) for r, mk in roi.items()})


if __name__ == '__main__':
    main()
