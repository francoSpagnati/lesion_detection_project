#!/usr/bin/env python
"""Figura prima/dopo dello skull stripping su WMH e ATLAS, con la lesione sovrapposta.

    python scripts/figura_skullstrip_prima_dopo.py

Le figure gia' esistenti (`reports/figures/synthstrip/`) mostrano SynthStrip su DBB, dove esiste una
maschera di riferimento con cui misurarlo. Servono anche su WMH e ATLAS, che sono i dataset su cui lo
skull stripping e' stato poi applicato davvero e dove **nessun riferimento esiste**: li' l'unica
verifica possibile e' guardare, e il senso di questa figura e' rendere guardabile la decisione.

Ogni riga e' un caso, tre pannelli:
    1. immagine originale, con il contorno della maschera di SynthStrip
    2. immagine dopo il ritaglio, quella che la rete vede davvero
    3. la lesione annotata, in verde quella dentro la maschera e in rosso quella persa

I casi non sono scelti a caso: si prendono i due che perdono **piu'** lesione su ATLAS (i casi
peggiori, non quelli che fanno bella figura) piu' un caso tipico per dataset.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
RAW = PROGETTO / 'data/processed/nnUNet_raw'
MASCHERE = PROGETTO / 'data/processed/synthstrip'

VERDE, ROSSO, GIALLO = '#00d67d', '#ff4d4d', '#ffd93d'

# (etichetta, dataset originale, dataset strippato, caso). La perdita indicata nell'etichetta e'
# quella sull'intero volume, misurata da lesioni_fuori_maschera.py; quella stampata sul pannello e'
# della singola fetta mostrata, e le due non coincidono.
CASI = [
    ('ATLAS: worst case\nof all 556 — 6.5%', 'Dataset502_ATLAS',
     'Dataset523_ATLAS_SS', 'sub-r038s061'),
    ('ATLAS: second\nworst — 5.2%', 'Dataset502_ATLAS', 'Dataset523_ATLAS_SS',
     'sub-r040s044'),
    ('ATLAS: typical case\nno loss', 'Dataset502_ATLAS', 'Dataset523_ATLAS_SS',
     'sub-r001s002'),
    ('WMH: no loss\nin all 79 cases', 'Dataset501_WMH', 'Dataset522_WMH_SS',
     'Amsterdam_GE3T_100'),
]


def carica(p: Path):
    img = nib.load(str(p))
    return np.asanyarray(img.dataobj).astype(np.float32)


# La cache di run_synthstrip.py e' organizzata per sorgente, non per nome di dataset nnU-Net.
CARTELLA = {'Dataset502_ATLAS': 'ATLAS', 'Dataset501_WMH': 'WMH', 'Dataset503_DBB': 'DBB'}


def maschera_di(caso: str, dataset: str):
    for c in (MASCHERE / CARTELLA[dataset], MASCHERE):
        f = c / f'{caso}_mask.nii.gz'
        if f.exists():
            return carica(f) > 0
    raise FileNotFoundError(f'maschera mancante per {caso} (cercata in {MASCHERE})')


def fetta_migliore(lesione: np.ndarray) -> int:
    """L'asse assiale e' l'ultimo in ATLAS e in WMH dopo la conversione; si sceglie la fetta con
    piu' lesione, perche' una fetta centrale per convenzione spesso non ne contiene affatto."""
    per_fetta = lesione.sum(axis=(0, 1))
    return int(np.argmax(per_fetta))


def normalizza(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a[a > 0], [1, 99]) if (a > 0).any() else (0, 1)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    fig, assi = plt.subplots(len(CASI), 3, figsize=(11, 3.1 * len(CASI)), facecolor='white')
    for r, (titolo, d_orig, d_strip, caso) in enumerate(CASI):
        img = carica(RAW / d_orig / 'imagesTr' / f'{caso}_0000.nii.gz')
        les = carica(RAW / d_orig / 'labelsTr' / f'{caso}.nii.gz') > 0
        strip = carica(RAW / d_strip / 'imagesTr' / f'{caso}_0000.nii.gz')
        msk = maschera_di(caso, d_orig)

        z = fetta_migliore(les)
        a, b, m, l = (normalizza(img[:, :, z]), normalizza(strip[:, :, z]),
                      msk[:, :, z], les[:, :, z])
        dentro, fuori = l & m, l & ~m
        persa = fuori.sum() / max(l.sum(), 1)

        etichette = ('original, with the mask boundary',
                     'after stripping: what the network sees',
                     f'annotated lesion — {persa:.1%} lost in this slice')
        for c, (dati, sotto) in enumerate(zip((a, b, a), etichette)):
            ax = assi[r, c]
            ax.imshow(np.rot90(dati), cmap='gray', vmin=0, vmax=1)
            if c == 0:
                ax.contour(np.rot90(m), levels=[0.5], colors=[GIALLO], linewidths=1.1)
            if c == 2:
                for maschera_l, colore in ((dentro, VERDE), (fuori, ROSSO)):
                    if maschera_l.any():
                        ax.contour(np.rot90(maschera_l), levels=[0.5], colors=[colore],
                                   linewidths=1.4)
            ax.set_title(sotto, fontsize=9, pad=4)
            ax.axis('off')
        # L'etichetta di riga va fuori dagli assi, altrimenti finisce sopra al titolo del pannello.
        assi[r, 0].annotate(titolo, xy=(0, 0.5), xycoords='axes fraction', xytext=(-14, 0),
                            textcoords='offset points', rotation=90, va='center', ha='center',
                            fontsize=9.5, fontweight='bold', linespacing=1.5)

    fig.suptitle('SynthStrip on WMH and ATLAS: what is removed, and at what cost\n'
                 'yellow = mask boundary    green = lesion kept    '
                 'red = lesion lost', fontsize=12.5)
    fig.tight_layout(rect=(0.015, 0, 1, 0.955), h_pad=1.2)
    out = PROGETTO / 'reports' / 'figures' / 'skullstrip_prima_dopo.png'
    fig.savefig(out, dpi=140, facecolor='white')
    print(f'scritta {out}')


if __name__ == '__main__':
    main()
