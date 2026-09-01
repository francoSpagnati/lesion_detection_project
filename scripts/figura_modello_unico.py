#!/usr/bin/env python
"""Che cosa produce il modello unico: lesione e sei tessuti nella stessa uscita.

    python scripts/figura_modello_unico.py

`plot_predictions.py` ordina i casi per Dice della lesione e mostra quella sola classe, quindi su
Dataset531 non fa vedere la cosa che rende interessante quel modello: che **un solo passaggio**
produce insieme la lesione e la parcellazione in sei tessuti. Serve al secondo stadio della
pipeline prevista dal progetto, dove sul "gemello sano" va fatta una parcellazione anatomica.

Tre righe, una per sorgente: un caso DBB (dove il riferimento sono i tessuti), un caso ATLAS e uno
WMH (dove il riferimento e' la sola lesione). Tre colonne: immagine, riferimento manuale,
predizione del modello unico.

**Quello che la figura mostra non e' quello che ci si aspettava.** Il modello non produce entrambe
le cose su tutte le immagini: sui casi adulti predice **solo** la lesione e zero tessuti, sui casi
pediatrici **solo** i tessuti e zero lesione. Ha risolto la contraddizione fra le etichette
imparando a riconoscere la sorgente. La misura su tutti i 151 casi e' in
`reports/risultato_531.md`.

I colori sono scelti perche' le classi anatomicamente confinanti non siano confondibili: la
sostanza bianca e la grigia profonda si toccano, cosi' come la bianca e i ventricoli.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

PROGETTO = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get('nnUNet_results', PROGETTO / 'nnUNet' / 'nnUNet_results'))
PREP = Path(os.environ.get('nnUNet_preprocessed', PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'))
RAW = PROGETTO / 'data/processed/nnUNet_raw'
D = 'Dataset531_UNION_D'
TR = 'nnUNetTrainerWandb__nnUNetPlans__3d_fullres'

CLASSI = [
    (1, 'lesione',         '#e34948'),
    (2, 'liquido (CSF)',   '#2a78d6'),
    (3, 'sostanza grigia', '#1baf7a'),
    (4, 'sostanza bianca', '#eda100'),
    (5, 'grigia profonda', '#e87ba4'),
    (6, 'tronco',          '#4a3aa7'),
    (7, 'cervelletto',     '#eb6834'),
]
CMAP = ListedColormap(['#00000000'] + [c for _, _, c in CLASSI])
NORM = BoundaryNorm(np.arange(-0.5, 8.5), CMAP.N)


def carica(p):
    return np.asanyarray(nib.load(str(p)).dataobj)


def finestra(a):
    lo, hi = np.percentile(a[a > 0], (1, 99)) if (a > 0).any() else (0, 1)
    return lo, max(hi, lo + 1e-6)


def scegli():
    """Un caso per sorgente: per DBB quello con piu' tessuto, per gli altri il Dice mediano
    della lesione, cosi' la figura non mostra ne' il caso migliore ne' il peggiore."""
    j = json.loads((RES / D / TR / 'fold_0' / 'validation' / 'summary.json').read_text())
    per = {}
    for c in j['metric_per_case']:
        nome = Path(c['prediction_file']).name.replace('.nii.gz', '')
        per[nome] = c['metrics']
    fuori = []
    for pref in ('dbb', 'atlas', 'wmh'):
        casi = [n for n in per if n.startswith(pref + '_')]
        if pref == 'dbb':
            fuori.append(sorted(casi)[0])
        else:
            casi = sorted(casi, key=lambda n: per[n]['1']['Dice'])
            fuori.append(casi[len(casi) // 2])
    return fuori


def main():
    casi = scegli()
    fig, assi = plt.subplots(len(casi), 3, figsize=(11, 3.9 * len(casi)), facecolor='white')
    for r, caso in enumerate(casi):
        img = carica(RAW / D / 'imagesTr' / f'{caso}_0000.nii.gz').astype(np.float32)
        gt = carica(PREP / D / 'gt_segmentations' / f'{caso}.nii.gz')
        pred = carica(RES / D / TR / 'fold_0' / 'validation' / f'{caso}.nii.gz')
        # Per DBB si guarda al centro del cervello; per gli altri dove c'e' la lesione.
        z = (int(np.argmax((gt == 1).sum(axis=(0, 1)))) if (gt == 1).any()
             else int(np.argmax((gt > 0).sum(axis=(0, 1)))))
        lo, hi = finestra(img[:, :, z])
        for c, (dati, titolo) in enumerate(((None, 'immagine T1'),
                                            (gt, 'riferimento manuale'),
                                            (pred, 'predizione del modello unico'))):
            ax = assi[r, c]
            ax.imshow(np.rot90(img[:, :, z]), cmap='gray', vmin=lo, vmax=hi)
            if dati is not None:
                m = np.rot90(dati[:, :, z]).astype(float)
                ax.imshow(np.ma.masked_where(m == 0, m), cmap=CMAP, norm=NORM,
                          alpha=.55, interpolation='nearest')
            ax.set_title(titolo, fontsize=10)
            ax.axis('off')
        sorgente = {'dbb': 'DBB — pediatrico, riferimento a 6 tessuti',
                    'atlas': 'ATLAS — riferimento: solo la lesione',
                    'wmh': 'WMH — riferimento: solo la lesione'}[caso.split('_')[0]]
        assi[r, 0].annotate(sorgente, xy=(0, 0.5), xycoords='axes fraction', xytext=(-13, 0),
                            textcoords='offset points', rotation=90, va='center', ha='center',
                            fontsize=9.5, fontweight='bold', color='#333')

    fig.legend(handles=[Patch(facecolor=c, label=n) for _, n, c in CLASSI],
               loc='lower center', ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.004))
    fig.suptitle('Il modello unico ha imparato a riconoscere da quale dataset viene l\'immagine\n'
                 'sui casi pediatrici predice i tessuti, su quelli adulti soltanto la lesione: '
                 'mai entrambi',
                 fontsize=12.5)
    fig.tight_layout(rect=(0.012, 0.05, 1, 0.95), h_pad=1.3)
    out = PROGETTO / 'reports' / 'figures' / 'modello_unico_7_classi.png'
    fig.savefig(out, dpi=140, facecolor='white')
    print(f'scritta {out}  —  casi: {", ".join(casi)}')


if __name__ == '__main__':
    main()
