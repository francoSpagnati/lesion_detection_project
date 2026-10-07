#!/usr/bin/env python
"""Le due figure dell'analisi per fenotipo su Dataset504.

    python scripts/figura_fenotipi_504.py
    # legge reports/analisi_rilevamento_504.json (da scripts/analisi_rilevamento_504.py)

Produce:
    reports/figures/aincp/copertura_fenotipi_504.png   quanto di ciascun fenotipo copre la
                                                       predizione, paziente per paziente, in
                                                       tre modelli: da zero, pre-addestrato con
                                                       la FLAIR spenta, e con la FLAIR attiva
    reports/figures/aincp/sopravvivenza_1mm_504.png    ogni lesione annotata, per volume e
                                                       fenotipo, e se esiste ancora nel
                                                       bersaglio ricampionato a 1 mm

PERCHE' QUESTE TRE VARIANTI. Il progetto usa T1 e FLAIR perche' vedono fenotipi diversi. La
variante `zero` a passo 1e-4 ha il filtro FLAIR a 0.05 della norma del T1: e' di fatto un
modello a sola T1 pre-addestrato su ATLAS. Quella a passo differenziato e' lo stesso modello con
la FLAIR attiva. Il modello da zero usa la FLAIR in pieno ma non ha il pre-addestramento. Le tre
insieme separano i due contributi senza addestrare nulla di nuovo.

La linea colorata e' la copertura di tutti i pazienti insieme (voxel coperti su voxel totali:
pesa di piu' chi ha piu' lesione, come il Dice); le linee grigie sono i singoli pazienti, che
dicono se l'effetto e' di tutti o di uno solo.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROGETTO = Path(__file__).resolve().parent.parent
FIGURE = PROGETTO / 'reports/figures/aincp'
# Colori dei fenotipi come in figura_lesioni_piccole.py; palette validata (tutte le coppie,
# anche per daltonismo) con lo script della skill dataviz.
COL = {'ROI_1': '#2a78d6', 'ROI_2': '#eb6834', 'ROI_4': '#1baf7a'}
NOME = {'ROI_1': 'gliosis', 'ROI_2': 'cystic lesion', 'ROI_4': 'encephalomalacia'}
VISTA = {'ROI_1': 'hyperintense on FLAIR', 'ROI_2': 'cavity, dark on T1', 'ROI_4': 'cavity, dark on T1'}
INK, INK2, MUTED, GRIGLIA, CONTESTO = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e1', '#c3c7cc'
TRE = [('from scratch', 'nnUNetTrainerWandb_250ep_arresto50'),
       ('ATLAS,\nFLAIR inactive', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50'),
       ('ATLAS,\nFLAIR active', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_arresto50')]
# un paziente per pannello da nominare: quello in cui l'effetto e' piu' grande
DA_NOMINARE = {'ROI_1': 'FSM_039', 'ROI_2': None, 'ROI_4': 'FSM_045'}


def stile(ax):
    for lato in ('top', 'right', 'left'):
        ax.spines[lato].set_visible(False)
    ax.spines['bottom'].set_color(GRIGLIA)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis='y', color=GRIGLIA, lw=0.8)
    ax.set_axisbelow(True)


def figura_copertura(r):
    cop = r['copertura']
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.6), sharey=True)
    fig.subplots_adjust(left=.07, right=.985, top=.80, bottom=.2, wspace=.12)
    x = np.arange(len(TRE))
    for ax, f in zip(axes, COL):
        stile(ax)
        casi = [p['caso'] for p in cop[TRE[0][1]][f]]
        serie = {c: [next(p for p in cop[t][f] if p['caso'] == c) for _, t in TRE] for c in casi}
        for c, v in serie.items():
            y = [p['coperti'] / p['voxel'] for p in v]
            ax.plot(x, y, color=CONTESTO, lw=1.1, zorder=2, solid_capstyle='round')
            ax.scatter(x, y, s=14, color=CONTESTO, zorder=2, linewidths=0)
            if c == DA_NOMINARE[f]:
                ax.annotate(c, (x[-1], y[-1]), xytext=(7, 0), textcoords='offset points',
                            va='center', fontsize=8.5, color=INK2)
        tot = [sum(p['coperti'] for p in cop[t][f]) / sum(p['voxel'] for p in cop[t][f]) for _, t in TRE]
        ax.plot(x, tot, color=COL[f], lw=2.2, zorder=3, solid_capstyle='round', solid_joinstyle='round')
        ax.scatter(x, tot, s=64, color=COL[f], zorder=4, edgecolors='white', linewidths=2)
        for xi, yi in zip(x, tot):
            ax.annotate(f'{yi:.2f}', (xi, yi), xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=9.5, color=INK, fontweight='bold')
        ax.set_xticks(x, [e for e, _ in TRE])
        ax.set_xlim(-.35, len(TRE) - .55)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, .25, .5, .75, 1], ['0', '25%', '50%', '75%', '100%'])
        ax.set_title(f'{NOME[f]}', loc='left', fontsize=12, color=INK, fontweight='bold', pad=20)
        ax.text(0, 1.035, f'{VISTA[f]} · {len(casi)} patients', transform=ax.transAxes,
                fontsize=9, color=INK2)
    axes[0].set_ylabel('phenotype voxels covered by the prediction', fontsize=9.5, color=INK2)
    fig.legend(handles=[Line2D([], [], color=CONTESTO, lw=1.1, marker='o', ms=4, mec='none', label='one patient'),
                        Line2D([], [], color=INK2, lw=2.2, marker='o', ms=7, mec='white', mew=2,
                               label='all patients pooled')],
               frameon=False, ncol=2, loc='lower center', fontsize=9.5, bbox_to_anchor=(.5, .0),
               labelcolor=INK2)
    fig.savefig(FIGURE / 'copertura_fenotipi_504.png', dpi=130, facecolor='white')
    plt.close(fig)


def figura_sopravvivenza(r):
    L = r['lesioni']
    fig, ax = plt.subplots(figsize=(11.2, 4.1))
    fig.subplots_adjust(left=.15, right=.8, top=.86, bottom=.2)
    stile(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRIGLIA, lw=0.8, which='major')
    rng = np.random.default_rng(3)
    righe = list(COL)[::-1]
    for i, f in enumerate(righe):
        sel = [x for x in L if x['fenotipo'] == f]
        v = np.array([x['mm3'] for x in sel])
        ok = np.array([x['voxel_a_1mm'] > 0 for x in sel])
        y = i + rng.uniform(-.22, .22, len(sel))
        ax.scatter(v[ok], y[ok], s=58, color=COL[f], edgecolors='white', linewidths=1.6, zorder=3)
        ax.scatter(v[~ok], y[~ok], s=52, facecolors='white', edgecolors=COL[f], linewidths=1.8, zorder=3)
        ax.text(1.02, i, f'{int((~ok).sum())} di {len(sel)} scompaiono', transform=ax.get_yaxis_transform(),
                va='center', fontsize=9.5, color=INK)
    ax.axvline(1, color=INK2, lw=1.1, zorder=2)
    ax.text(1.12, len(righe) - .45, '1 mm³: un voxel della griglia\nsu cui lavora la rete', fontsize=8.5,
            color=INK2, va='top')
    ax.set_xscale('log')
    ax.set_xlim(.08, 3e5)
    ax.set_xticks([.1, 1, 10, 100, 1e3, 1e4, 1e5], ['0.1', '1', '10', '100', '1 000', '10 000', '100 000'])
    ax.set_yticks(range(len(righe)), [NOME[f] for f in righe], fontsize=10.5, color=INK)
    ax.set_ylim(-.6, len(righe) - .4)
    ax.set_xlabel('volume della lesione annotata, mm³ (scala logaritmica)', fontsize=9.5, color=INK2)
    ax.set_title('Le 71 lesioni annotate: quali esistono ancora nel bersaglio di addestramento a 1 mm',
                 loc='left', fontsize=11.5, color=INK, fontweight='bold', pad=12)
    fig.legend(handles=[Line2D([], [], ls='none', marker='o', ms=8, color=INK2, mec='white', mew=1.6,
                               label='presente nel bersaglio a 1 mm'),
                        Line2D([], [], ls='none', marker='o', ms=8, mfc='white', mec=INK2, mew=1.8,
                               label='scompare nel ricampionamento')],
               frameon=False, ncol=2, loc='lower center', fontsize=9.5, bbox_to_anchor=(.47, .0),
               labelcolor=INK2)
    fig.savefig(FIGURE / 'sopravvivenza_1mm_504.png', dpi=130, facecolor='white')
    plt.close(fig)


def main():
    r = json.load(open(PROGETTO / 'reports/analisi_rilevamento_504.json'))
    FIGURE.mkdir(parents=True, exist_ok=True)
    figura_copertura(r)
    figura_sopravvivenza(r)
    print('scritte', FIGURE / 'copertura_fenotipi_504.png', FIGURE / 'sopravvivenza_1mm_504.png')


if __name__ == '__main__':
    main()
