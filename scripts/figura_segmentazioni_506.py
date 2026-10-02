#!/usr/bin/env python
"""Come segmenta il modello della tesi (Z): la predizione accanto all'annotazione, sui 12 pazienti
di addestramento.

    python scripts/figura_segmentazioni_506.py    # -> reports/figures/aincp/segmentazioni_506_*.png

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
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import ants
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D

import postprocessing_aincp as pp

PROGETTO = Path(__file__).resolve().parent.parent
RUN = (PROGETTO / 'nnUNet/nnUNet_results/Dataset506_AINCP_FINALE'
       / 'nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres')
VAL = PROGETTO / 'reports' / f'valutazione_cv_{RUN.parent.name}__{RUN.name}.json'
OUT = PROGETTO / 'reports/figures/aincp'
BLU, ARANCIO = '#2a78d6', '#eb6834'
INK, INK2, FONDO = '#0b0b0b', '#52514e', '#fcfcfb'
DETTAGLIO = {'FSM_028': 'lesione estesa: contorno quasi esatto; le 7 lesioni mancate sono frammenti da 2 a 23 mm³',
             'FSM_056': 'lesione media, cavità più gliosi: il grosso c\'è, il disaccordo è sui bordi',
             'FSM_041': 'gliosi sottile attorno al ventricolo: quasi tutta mancata',
             'FSM_033': 'l\'unica lesione mancata del tutto'}
# asse della fetta -> etichette (sinistra, destra, basso, alto) dopo il riorientamento in LAS
ETICHETTE = {0: ('P', 'A', 'I', 'S'), 1: ('R', 'L', 'I', 'S'), 2: ('R', 'L', 'P', 'A')}
NOME_ASSE = {0: 'sagittale', 1: 'coronale', 2: 'assiale'}


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
    prob = np.load(RUN / 'prob_nativa' / f'{caso}.npy').astype(np.float32)
    pred = pp.postprocessa(prob, par['t'], par['vmin'], par['cervello'], cer, vox)
    v = {k: las(x, aff) for k, x in (('t1', t1i.numpy()), ('flair', fl.numpy()), ('ref', ref.astype(np.uint8)),
                                      ('pred', pred.astype(np.uint8)), ('cer', cer.astype(np.uint8)))}
    for k in ('ref', 'pred', 'cer'):
        v[k] = v[k] > 0
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


def pannello(ax, v, chiave, asse, i, rq, vmax, etichette=True):
    img = np.take(v[chiave], i, axis=asse)[rq]
    ax.imshow(img.T, cmap='gray', origin='lower', vmin=0, vmax=vmax, interpolation='nearest')
    for m, col, ls in (('ref', BLU, '-'), ('pred', ARANCIO, '--')):
        s = np.take(v[m], i, axis=asse)[rq]
        if s.any():
            ax.contour(s.T.astype(float), levels=[0.5], colors=col, linewidths=1.8, linestyles=ls, origin='lower')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if etichette:
        e = ETICHETTE[asse]
        for testo, pos, ha, va in ((e[0], (0.015, 0.5), 'left', 'center'), (e[1], (0.985, 0.5), 'right', 'center'),
                                   (e[2], (0.5, 0.015), 'center', 'bottom'), (e[3], (0.5, 0.985), 'center', 'top')):
            ax.text(*pos, testo, transform=ax.transAxes, ha=ha, va=va, color='white', fontsize=8, fontweight='bold')


def legenda(fig, y=0.995, x=0.995, loc='upper right', ncol=1):
    fig.legend(handles=[Line2D([], [], color=BLU, lw=2.4, label='annotazione (verità)'),
                        Line2D([], [], color=ARANCIO, lw=2.4, ls='--', label='predizione di Z, dopo il postprocessing')],
               loc=loc, bbox_to_anchor=(x, y), frameon=False, fontsize=11, labelcolor=INK, ncol=ncol)


def descrizione(q):
    ril = sum(l['rilevata'] for l in q['lesioni'])
    return f"Dice {q['dice']:.2f} · lesioni {ril}/{len(q['lesioni'])} · falsi positivi {q['fp']}"


def panoramica(dati, val, par):
    casi = sorted(dati, key=lambda c: -val[c]['post']['perc_vera'])
    fig, ax = plt.subplots(6, 4, figsize=(16, 27), facecolor=FONDO)
    for n, c in enumerate(casi):
        v, q = dati[c], val[c]['post']
        z = int(np.argmax(v['ref'].sum((0, 1))))           # fetta assiale con piu' lesione
        rq = riquadro(v['ref'][:, :, z] | v['pred'][:, :, z], v['cer'][:, :, z])
        lato = max(rq[0].stop - rq[0].start, rq[1].stop - rq[1].start)
        r, col = divmod(n, 2)
        for k, chiave in enumerate(('flair', 't1')):
            x = ax[r, 2 * col + k]
            x.set_facecolor(FONDO)
            pannello(x, v, chiave, 2, z, rq, finestra(v[chiave], v['cer'])[1])
            if k == 0:
                x.set_title(f"{c} · {val[c]['strato']} · {q['ml_vero']:.1f} ml ({q['perc_vera']:.2f}% del cervello)\n"
                            f"{descrizione(q)}", fontsize=10, color=INK, loc='left')
            x.text(0.02, 0.98 if k else 0.98, 'FLAIR' if k == 0 else 'T1', transform=x.transAxes, ha='left', va='top',
                   color='#ffd84d', fontsize=9, fontweight='bold')
            x.text(0.98, 0.02, f'{lato} mm', transform=x.transAxes, ha='right', va='bottom', color='white', fontsize=8)
    fig.suptitle('Il modello della tesi (Z) sui 12 pazienti di addestramento, predizioni fuori-fold',
                 x=0.012, ha='left', fontsize=16, color=INK, fontweight='bold', y=0.997)
    fig.text(0.012, 0.983, f"Fetta assiale con più lesione, ritagliata attorno a lesione e predizione (lato del riquadro in "
             f"basso a destra: i casi piccoli sono ingranditi). Postprocessing: soglia {par['t']}, volume minimo "
             f"{par['vmin']} mm³.\nOrdinati per percentuale di cervello lesionato. Convenzione radiologica: la destra del "
             'paziente è a sinistra.', ha='left', va='top', fontsize=10.5, color=INK2)
    legenda(fig, 0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.962), h_pad=2.0, w_pad=0.8)
    f = OUT / 'segmentazioni_506_panoramica.png'
    fig.savefig(f, dpi=100, facecolor=FONDO)
    plt.close(fig)
    return f


def dettaglio(c, v, q, strato, nota):
    # centro: il voxel di lesione piu' vicino al baricentro (il baricentro puo' cadere fuori)
    idx = np.argwhere(v['ref'])
    centro = idx[np.argmin(((idx - idx.mean(0)) ** 2).sum(1))]
    fig, ax = plt.subplots(2, 3, figsize=(15, 11.5), facecolor=FONDO)
    for col, asse in enumerate((2, 1, 0)):
        i = int(centro[asse])
        rq = riquadro(np.take(v['ref'] | v['pred'], i, axis=asse), np.take(v['cer'], i, axis=asse), margine=30, minimo=90)
        for r, chiave in enumerate(('flair', 't1')):
            x = ax[r, col]
            pannello(x, v, chiave, asse, i, rq, finestra(v[chiave], v['cer'])[1])
            x.set_title(f"{'FLAIR' if r == 0 else 'T1'} · {NOME_ASSE[asse]}, fetta {i + 1}", fontsize=10.5,
                        color=INK2, loc='left')
    fig.suptitle(f'{c} — {nota}', x=0.012, ha='left', fontsize=15, color=INK, fontweight='bold', y=0.995)
    fig.text(0.012, 0.952, f"{strato} · lesione {q['ml_vero']:.2f} ml ({q['perc_vera']:.2f}% del cervello), predetta "
             f"{q['ml_pred']:.2f} ml · {descrizione(q)}.\nFette passanti per la lesione, ritagliate attorno a essa. "
             'Convenzione radiologica: la destra del paziente è a sinistra.',
             ha='left', va='top', fontsize=10.5, color=INK2)
    legenda(fig, y=0.9, x=0.005, loc='upper left', ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.86), h_pad=2.5, w_pad=1.0)
    f = OUT / f'segmentazioni_506_dettaglio_{c}.png'
    fig.savefig(f, dpi=100, facecolor=FONDO)
    plt.close(fig)
    return f


def main():
    j = json.loads(VAL.read_text())
    par, val = j['parametri'], j['casi']
    OUT.mkdir(parents=True, exist_ok=True)
    dati = {c: carica(c, par) for c in sorted(val)}
    for c, v in dati.items():
        # controllo: la predizione ricostruita qui e' quella valutata nel report
        d = 2 * (v['ref'] & v['pred']).sum() / max(1, v['ref'].sum() + v['pred'].sum())
        assert abs(d - val[c]['post']['dice']) < 1e-3, (c, d, val[c]['post']['dice'])
    print('scritto', panoramica(dati, val, par))
    for c, nota in DETTAGLIO.items():
        print('scritto', dettaglio(c, dati[c], val[c]['post'], val[c]['strato'], nota))


if __name__ == '__main__':
    main()
