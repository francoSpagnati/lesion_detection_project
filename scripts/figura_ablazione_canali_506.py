#!/usr/bin/env python
"""Figura dell'ablazione dei canali su Z: che cosa perde il modello senza FLAIR o senza T1.

    python scripts/figura_ablazione_canali_506.py   # legge reports/ablazione_canali_506.json
                                                   # -> reports/figures/aincp/ablazione_canali_506.png

Due pannelli, perche' sono due misure diverse (nessun doppio asse):
- a sinistra la copertura per fenotipo (voxel coperti su voxel annotati, sommati sui 12 pazienti):
  dice *quale* lesione dipende da quale sequenza;
- a destra le lesioni trovate, tutte e sotto 50 mm³: dice quante se ne perdono.

Colori: la palette categoriale gia' validata per il progetto (CVD ΔE 9.2 sulla coppia peggiore,
avviso di contrasto -> valori scritti sulle barre e legenda), assegnata in ordine fisso alle tre
condizioni.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
R = json.loads((PROGETTO / 'reports/ablazione_canali_506.json').read_text())['totali']
OUT = PROGETTO / 'reports/figures/aincp/ablazione_canali_506.png'
COND = [('entrambe', 'FLAIR + T1', '#2a78d6'), ('solo_FLAIR', 'solo FLAIR', '#1baf7a'),
        ('solo_T1', 'solo T1', '#eb6834')]
FEN = [('ROI_1', 'gliosi'), ('ROI_2', 'lesione cistica'), ('ROI_4', 'encefalomalacia')]
INCHIOSTRO, GRIGIO, GRIGLIA = '#1f2328', '#59636e', '#d8dee4'


def barre(ax, gruppi, valori, formato, massimo):
    n = len(COND)
    w = 0.8 / n
    x = np.arange(len(gruppi))
    for i, (_, nome, col) in enumerate(COND):
        v = [valori[g][i] for g in range(len(gruppi))]
        b = ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.92, color=col, label=nome, zorder=2)
        for r, t in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + massimo * 0.015, formato(t),
                    ha='center', va='bottom', fontsize=8.5, color=INCHIOSTRO)
    ax.set_xticks(x, gruppi, color=INCHIOSTRO, fontsize=10)
    ax.set_ylim(0, massimo * 1.12)
    ax.grid(axis='y', color=GRIGLIA, lw=0.8, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRIGIO)
    ax.tick_params(axis='y', colors=GRIGIO, length=0)
    ax.tick_params(axis='x', length=0)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, gridspec_kw={'width_ratios': [3, 2]})
    cop = [[100 * R[c]['copertura'][f][0] / R[c]['copertura'][f][1] for c, _, _ in COND] for f, _ in FEN]
    barre(a, [f'{n}\n({R["entrambe"]["copertura"][f][1]:,} voxel)'.replace(',', ' ') for f, n in FEN],
          cop, lambda t: f'{t:.0f}%', 100)
    a.set_ylabel('voxel della lesione coperti (%)', color=GRIGIO)
    a.set_title('Copertura per fenotipo', loc='left', color=INCHIOSTRO, fontsize=12)

    vol = np.array(R['entrambe']['volumi'])
    piccole = vol < 50
    trov = [[int(np.sum(R[c]['trovate'])) for c, _, _ in COND],
            [int(np.sum(np.array(R[c]['trovate'])[piccole])) for c, _, _ in COND]]
    barre(b, [f'tutte\n({len(vol)} lesioni)', f'sotto 50 mm³\n({piccole.sum()} lesioni)'], trov, str, len(vol) / 2)
    b.set_ylabel('lesioni trovate', color=GRIGIO)
    b.set_title('Lesioni trovate', loc='left', color=INCHIOSTRO, fontsize=12)

    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=10, labelcolor=INCHIOSTRO,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, 'Modello Z, 12 pazienti fuori-fold, sequenza mancante azzerata in ingresso; postprocessing '
             'congelato (soglia 0.3, volume minimo 50 mm³). Fonte: reports/ablazione_canali_506.md',
             fontsize=8, color=GRIGIO)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUT, facecolor='white')
    print(OUT)


if __name__ == '__main__':
    main()
