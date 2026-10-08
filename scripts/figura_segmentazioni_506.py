#!/usr/bin/env python
"""Come segmenta il modello della tesi (Z): la predizione accanto all'annotazione, sui 12 pazienti
di addestramento.

    python scripts/figura_segmentazioni_506.py           # -> reports/figures/aincp/segmentazioni_506_*.png
    python scripts/figura_segmentazioni_506.py --test    # i 5 di test: solo DOPO il test, una volta fatto

PERCHE'. I numeri (Dice 0.58, 27/68 lesioni) non dicono *dove* il modello sbaglia: se manca il
bordo di una lesione grande o intere lesioni piccole, se inventa macchie. Le figure lo fanno vedere.

QUALI PREDIZIONI. Solo fuori-fold: ogni paziente e' segmentato dai modelli del fold che non l'ha
visto in addestramento, gia' riportati nel nativo (cache `prob_nativa` di postprocessing_aincp.py)
e con il postprocessing scelto per Z (soglia e volume minimo dal JSON). Il test non si tocca.

COME. Tutto nella griglia T1 nativa, dove si misura: la FLAIR ci viene portata con la rigida
FLAIR->T1 della pipeline. Ogni volume e' riorientato in LAS (convenzione radiologica: la destra del
paziente a sinistra dell'immagine), perche' i pazienti del lotto 2 arrivano in RAS o LAS.
Blu continuo = annotazione, arancio tratteggiato = predizione: lo stile della linea distingue le
due anche senza colore (stessa coppia, validata, di figura_roi_fsm030.py).

1. `panoramica`: un paziente per riquadro, FLAIR e T1 nella fetta assiale con piu' lesione,
   ritagliata attorno a lesione e predizione (le lesioni piccole risultano ingrandite: la scala e'
   scritta nel titolo), ordinati per percentuale di cervello lesionato.
2. `dettaglio_<paziente>`: quattro casi tipici in tre orientamenti, FLAIR sopra e T1 sotto.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import ants
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from scipy import ndimage

import postprocessing_aincp as pp

PROGETTO = Path(__file__).resolve().parent.parent
RUN = (PROGETTO / 'nnUNet/nnUNet_results/Dataset506_AINCP_FINALE'
       / 'nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres')
OUT = PROGETTO / 'reports/figures/aincp'
BLU, ARANCIO = '#2a78d6', '#eb6834'
INK, INK2, FONDO = '#0b0b0b', '#52514e', '#fcfcfb'
DETTAGLIO = {'FSM_028': 'extensive lesion: nearly exact outline; the 7 missed lesions are fragments of 2 to 23 mm³',
             'FSM_056': 'medium lesion, cavity plus gliosis: the bulk is found, the disagreement is at the borders',
             'FSM_041': 'thin periventricular gliosis: almost entirely missed',
             'FSM_033': 'the only lesion missed completely'}
# sul test: il caso in cui Z inventa una lesione (centrato sulla componente falsa piu' grande) e il
# caso dell'altro centro, dove Z fa meglio delle ricette pre-addestrate
DETTAGLIO_TEST = {'FSM_046': ('the large false positive of Z on the test set', 'fp'),
                  'FSM_049': ('the only test patient from another centre: a 196,847 mm³ lesion', 'ref')}
# cv (fuori-fold sui 12) oppure test (insieme dei 5 fold sui 5 di test): impostato in main()
MODO = {'prob': RUN / 'prob_nativa', 'val': 'cv', 'prefisso': 'segmentazioni_506',
        'titolo': 'on the 12 development patients, out-of-fold predictions'}
# asse della fetta -> etichette (sinistra, destra, basso, alto) dopo il riorientamento in LAS
ETICHETTE = {0: ('P', 'A', 'I', 'S'), 1: ('R', 'L', 'I', 'S'), 2: ('R', 'L', 'P', 'A')}
NOME_ASSE = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
STRATO_EN = {'altri centri': 'other centre'}


def las(vol, aff):
    """Riorienta un volume in LAS: RAS canonico, poi si inverte l'asse x."""
    img = nib.as_closest_canonical(nib.Nifti1Image(np.asarray(vol), aff))
    return np.asanyarray(img.dataobj)[::-1]


def carica(caso, par):
    s = pp.sessione(caso)
    t1i = ants.image_read(str(s / 'T1_N4.nii.gz'))
    fl = ants.apply_transforms(fixed=t1i, moving=ants.image_read(str(s / 'FLAIR_N4.nii.gz')),
                               transformlist=[str(s / 'transforms' / 'FLAIR_to_T1_0.mat')], interpolator='linear')
    aff = nib.load(str(s / 'T1_N4.nii.gz')).affine
    ref, cer, vox = pp.carica_caso(caso)
    prob = np.load(MODO['prob'] / f'{caso}.npy').astype(np.float32)
    pred = pp.postprocessa(prob, par['t'], par['vmin'], par['cervello'], cer, vox)
    v = {k: las(x, aff) for k, x in (('t1', t1i.numpy()), ('flair', fl.numpy()), ('ref', ref.astype(np.uint8)),
                                      ('pred', pred.astype(np.uint8)), ('cer', cer.astype(np.uint8)))}
    for k in ('ref', 'pred', 'cer'):
        v[k] = v[k] > 0
    # spaziatura dei voxel nell'orientamento LAS: la T1 del paziente e' anisotropa (1.0 x 0.43 x 0.43 mm),
    # senza questa i pannelli vengono stirati di 2.3 volte lungo un asse
    v['sp'] = np.array(nib.as_closest_canonical(nib.load(str(s / 'T1_N4.nii.gz'))).header.get_zooms()[:3])
    return v


def finestra(img, cer):
    return 0, float(np.percentile(img[cer], 99.5))


def riquadro(m2d, cer2d, margine=22, minimo=70):
    """Ritaglio 2D attorno a lesione+predizione; se non c'e' nulla, il cervello intero."""
    idx = np.argwhere(m2d if m2d.any() else cer2d)
    lo, hi = idx.min(0) - margine, idx.max(0) + margine
    c, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, minimo / 2)
    lo = np.maximum(0, (c - half).astype(int))
    hi = np.minimum(m2d.shape, (c + half).astype(int) + 1)
    return slice(lo[0], hi[0]), slice(lo[1], hi[1])


def pannello(ax, v, chiave, asse, i, rq, vmax, etichette=True, lw=1.8, fs=8):
    img = np.take(v[chiave], i, axis=asse)[rq]
    a, b = [d for d in range(3) if d != asse]
    asp = float(v['sp'][b] / v['sp'][a]) if 'sp' in v else 1.0     # proporzioni in mm, non in voxel
    ax.imshow(img.T, cmap='gray', origin='lower', vmin=0, vmax=vmax, interpolation='nearest', aspect=asp)
    for m, col, ls in (('ref', BLU, '-'), ('pred', ARANCIO, '--')):
        s = np.take(v[m], i, axis=asse)[rq]
        if s.any():
            ax.contour(s.T.astype(float), levels=[0.5], colors=col, linewidths=lw, linestyles=ls, origin='lower')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if etichette:
        e = ETICHETTE[asse]
        for testo, pos, ha, va in ((e[0], (0.015, 0.5), 'left', 'center'), (e[1], (0.985, 0.5), 'right', 'center'),
                                   (e[2], (0.5, 0.015), 'center', 'bottom'), (e[3], (0.5, 0.985), 'center', 'top')):
            ax.text(*pos, testo, transform=ax.transAxes, ha=ha, va=va, color='white', fontsize=fs, fontweight='bold')


def legenda(fig, y=0.995, x=0.995, loc='upper right', ncol=1):
    fig.legend(handles=[Line2D([], [], color=BLU, lw=2.4, label='annotation (reference)'),
                        Line2D([], [], color=ARANCIO, lw=2.4, ls='--', label='prediction of Z, after post-processing')],
               loc=loc, bbox_to_anchor=(x, y), frameon=False, fontsize=11, labelcolor=INK, ncol=ncol)


def descrizione(q):
    ril = sum(l['rilevata'] for l in q['lesioni'])
    return f"Dice {q['dice']:.2f} · lesions {ril}/{len(q['lesioni'])} · false positives {q['fp']}"


def quadrato(rq, forma, sp):
    """Il riquadro reso quadrato in mm (lato = il maggiore dei due), spostato dentro l'immagine invece
    che tagliato: in una griglia di pannelli uguali un ritaglio stretto spreca meta' dello spazio.
    Ritorna anche il lato in mm."""
    mm = max((sl.stop - sl.start) * s for sl, s in zip(rq, sp))
    mm = min(mm, *(n * s for n, s in zip(forma, sp)))
    out = []
    for sl, n, s in zip(rq, forma, sp):
        lato = int(round(mm / s))
        c = (sl.start + sl.stop) // 2
        lo = min(max(c - lato // 2, 0), n - lato)
        out.append(slice(lo, lo + lato))
    return tuple(out), mm


def panoramica(dati, val, par):
    # formato della tesi: 16 cm a 300 dpi, tre pazienti per riga (FLAIR e T1 ciascuno), in ordine di
    # percentuale di cervello lesionato; titolo, soglie e convenzioni vanno nella didascalia
    casi = sorted(dati, key=lambda c: -val[c]['post']['perc_vera'])
    per_riga = 3
    righe = (len(casi) + per_riga - 1) // per_riga
    H = 1.2 * righe + 0.3
    fig, ax = plt.subplots(righe, 2 * per_riga, figsize=(6.3, H), facecolor='white', squeeze=False)
    for x in ax.ravel():
        x.axis('off')
    for n, c in enumerate(casi):
        v, q = dati[c], val[c]['post']
        z = int(np.argmax(v['ref'].sum((0, 1))))           # fetta assiale con piu' lesione
        rq, lato = quadrato(riquadro(v['ref'][:, :, z] | v['pred'][:, :, z], v['cer'][:, :, z]),
                            v['ref'].shape[:2], v['sp'][:2])
        r, col = divmod(n, per_riga)
        ril = sum(l['rilevata'] for l in q['lesioni'])
        for k, chiave in enumerate(('flair', 't1')):
            x = ax[r, 2 * col + k]
            x.axis('on')
            pannello(x, v, chiave, 2, z, rq, finestra(v[chiave], v['cer'])[1], etichette=False, lw=0.8)
            for t, xx, ha in (('R', 0.03, 'left'), ('L', 0.97, 'right')):
                x.text(xx, 0.5, t, transform=x.transAxes, ha=ha, va='center', color='white', fontsize=4.5)
            if k == 0:
                x.set_title(f"{c} · {q['perc_vera']:.2f}% of brain\nDice {q['dice']:.2f} · lesions {ril}/"
                            f"{len(q['lesioni'])} · FP {q['fp']}", fontsize=5.8, color=INK, loc='left', pad=2,
                            linespacing=1.15)
            x.text(0.03, 0.97, 'FLAIR' if k == 0 else 'T1', transform=x.transAxes, ha='left', va='top',
                   color='#ffd84d', fontsize=5, fontweight='bold')
            x.text(0.97, 0.03, f'{lato:.0f} mm', transform=x.transAxes, ha='right', va='bottom', color='white', fontsize=4.5)
    fig.legend(handles=[Line2D([], [], color=BLU, lw=1.2, label='annotation (reference)'),
                        Line2D([], [], color=ARANCIO, lw=1.2, ls='--', label='prediction of model Z, after post-processing')],
               loc='lower center', frameon=False, fontsize=7, labelcolor=INK, ncol=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.28 / H, bottom=0.22 / H, wspace=0.03, hspace=0.32)
    f = OUT / f"{MODO['prefisso']}_panoramica.png"
    fig.savefig(f, dpi=300, facecolor='white')
    plt.close(fig)
    return f


def dettaglio(c, v, q, strato, nota, centro_su='ref'):
    # centro: il voxel piu' vicino al baricentro (il baricentro puo' cadere fuori) della lesione
    # annotata, oppure della componente predetta piu' grande che non tocca l'annotazione
    m = v['ref']
    if centro_su == 'fp':
        lab, n = ndimage.label(v['pred'], structure=np.ones((3, 3, 3), bool))
        false = [j for j in range(1, n + 1) if not v['ref'][lab == j].any()]
        m = lab == max(false, key=lambda j: (lab == j).sum())
    idx = np.argwhere(m)
    centro = idx[np.argmin(((idx - idx.mean(0)) ** 2).sum(1))]
    # formato della tesi: tre viste (assiale, coronale, sagittale) x FLAIR/T1, pannelli quadrati in mm;
    # paziente, volumi e descrizione vanno nella didascalia
    H = 4.55
    fig, ax = plt.subplots(2, 3, figsize=(6.3, H), facecolor='white')
    for col, asse in enumerate((2, 1, 0)):
        i = int(centro[asse])
        assi = [d for d in range(3) if d != asse]
        rq, _ = quadrato(riquadro(np.take(v['ref'] | v['pred'], i, axis=asse), np.take(v['cer'], i, axis=asse),
                                  margine=30, minimo=90), np.take(v['ref'], i, axis=asse).shape, v['sp'][assi])
        for r, chiave in enumerate(('flair', 't1')):
            x = ax[r, col]
            pannello(x, v, chiave, asse, i, rq, finestra(v[chiave], v['cer'])[1], lw=0.9, fs=5)
            x.set_title(f"{'FLAIR' if r == 0 else 'T1'} · {NOME_ASSE[asse]}", fontsize=7, color=INK2, loc='left', pad=2)
    fig.legend(handles=[Line2D([], [], color=BLU, lw=1.2, label='annotation (reference)'),
                        Line2D([], [], color=ARANCIO, lw=1.2, ls='--', label='prediction of model Z, after post-processing')],
               loc='lower center', frameon=False, fontsize=7, labelcolor=INK, ncol=2)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.2 / H, bottom=0.25 / H, wspace=0.04, hspace=0.12)
    f = OUT / f"{MODO['prefisso']}_dettaglio_{c}.png"
    fig.savefig(f, dpi=300, facecolor='white')
    plt.close(fig)
    return f


def main():
    global DETTAGLIO
    if '--test' in sys.argv:
        MODO.update(prob=RUN / 'test_insieme' / 'prob_nativa', val='test', prefisso='segmentazioni_506_test',
                    titolo='on the 5 test patients, ensemble of the 5 fold models')
        DETTAGLIO = DETTAGLIO_TEST
    else:
        DETTAGLIO = {c: (n, 'ref') for c, n in DETTAGLIO.items()}
    j = json.loads((PROGETTO / 'reports' / f"valutazione_{MODO['val']}_{RUN.parent.name}__{RUN.name}.json").read_text())
    par, val = j['parametri'], j['casi']
    OUT.mkdir(parents=True, exist_ok=True)
    dati = {c: carica(c, par) for c in sorted(val)}
    for c, v in dati.items():
        # controllo: la predizione ricostruita qui e' quella valutata nel report
        d = 2 * (v['ref'] & v['pred']).sum() / max(1, v['ref'].sum() + v['pred'].sum())
        assert abs(d - val[c]['post']['dice']) < 1e-3, (c, d, val[c]['post']['dice'])
    print('scritto', panoramica(dati, val, par))
    for c, (nota, centro) in DETTAGLIO.items():
        print('scritto', dettaglio(c, dati[c], val[c]['post'], val[c]['strato'], nota, centro))


if __name__ == '__main__':
    main()
