#!/usr/bin/env python
"""Quali lesioni annotate esistono ancora nel bersaglio di addestramento a 1 mm (17 pazienti, Dataset506).

    python scripts/sopravvivenza_1mm_506.py
    # -> reports/sopravvivenza_1mm_506.{md,json}, reports/figures/aincp/sopravvivenza_1mm_17.png

PERCHE'. La figura sopravvivenza_1mm_504.png era sui 10 pazienti del primo lotto e sulla griglia
nativa di 504. Il modello della tesi (506) si addestra invece nel template pediatrico a 1 mm: qui la
stessa domanda sui 17 pazienti e con la pipeline finale.

COME. Si rifa' la strada di pipeline_aincp.py una componente alla volta: etichetta ROI_1 | ROI_2 | ROI_4
nella griglia della FLAIR originale, residui <= 1 mm³ tolti, componenti a 26 vicini; ogni componente
va nel template con le stesse trasformazioni (rigida FLAIR -> T1, poi SyN verso MNIPediatricAsym),
interpolazione lineare e soglia 0.5. Sopravvive se le resta almeno un voxel. Fenotipo = la ROI che
copre piu' voxel della componente. Controllo: l'unione delle componenti portate una per una deve
coincidere quasi del tutto con il bersaglio salvato (lesione_PED.nii.gz); la differenza e' riportata.
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_aincp as pa  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
TPL = PROGETTO / 'data/templates/tpl-MNIPediatricAsym_cohort-1_res-1_T1w.nii.gz'
OUT = PROGETTO / 'reports/sopravvivenza_1mm_506'
FIG = PROGETTO / 'reports/figures/aincp/sopravvivenza_1mm_17.png'
COL = {'ROI_1': '#2a78d6', 'ROI_2': '#eb6834', 'ROI_4': '#1baf7a'}
NOME = {'ROI_1': 'gliosis', 'ROI_2': 'cystic lesion', 'ROI_4': 'encephalomalacia'}
INK, INK2, GRIGLIA = '#0b0b0b', '#52514e', '#e6e5e1'


def caso(paz):
    import ants
    c = pa.trova_casi()[paz]
    dest = pa.PROC / c['sessione']
    tpl = ants.image_read(str(TPL))
    tp = dest / 'transforms_ped'
    trasf = [str(tp / 'T1_to_PED_fwd_0.nii.gz'), str(tp / 'T1_to_PED_fwd_1.mat'),
             str(dest / 'transforms' / 'FLAIR_to_T1_0.mat')]
    rif = ants.image_read(str(c['FLAIR']))
    vox = float(np.prod(rif.spacing))
    roi = {r: np.squeeze(ants.image_read(str(c['roi'][r])).numpy()) > 0 for r in COL if r in c['roi']}
    m = np.zeros(rif.shape, bool)
    for v in roi.values():
        m |= v
    m, _ = pa.togli_residui(m, vox)
    lab, n = ndimage.label(m, structure=pa.STRUTTURA)
    unione = np.zeros(tpl.shape, bool)
    lesioni = []
    for i in range(1, n + 1):
        comp = lab == i
        quote = {r: int((comp & v).sum()) for r, v in roi.items()}
        w = ants.apply_transforms(fixed=tpl, moving=rif.new_image_like(comp.astype('float32')),
                                  transformlist=trasf, interpolator='linear').numpy() >= 0.5
        unione |= w
        lesioni.append({'paziente': paz, 'mm3': float(comp.sum() * vox), 'fenotipo': max(quote, key=quote.get),
                        'voxel_a_1mm': int(w.sum())})
    salvato = np.asanyarray(__import__('nibabel').load(str(dest / 'lesione_PED.nii.gz')).dataobj) > 0.5
    controllo = float((unione != salvato).sum() / max(salvato.sum(), 1))
    return paz, {'lesioni': lesioni, 'voxel_diversi_dal_bersaglio_%': 100 * controllo}


def figura(L):
    fig, ax = plt.subplots(figsize=(11.2, 4.1), dpi=150)
    fig.subplots_adjust(left=.15, right=.8, top=.86, bottom=.2)
    rng = np.random.default_rng(3)
    righe = list(COL)[::-1]
    for i, f in enumerate(righe):
        sel = [x for x in L if x['fenotipo'] == f]
        v = np.array([x['mm3'] for x in sel])
        ok = np.array([x['voxel_a_1mm'] > 0 for x in sel])
        y = i + rng.uniform(-.22, .22, len(sel))
        ax.scatter(v[ok], y[ok], s=58, color=COL[f], edgecolors='white', linewidths=1.6, zorder=3)
        ax.scatter(v[~ok], y[~ok], s=52, facecolors='white', edgecolors=COL[f], linewidths=1.8, zorder=3)
        ax.text(1.02, i, f'{int((~ok).sum())} of {len(sel)} disappear', transform=ax.get_yaxis_transform(),
                va='center', fontsize=9.5, color=INK)
    ax.axvline(1, color=INK2, lw=1.1, zorder=2)
    ax.text(1.12, len(righe) - .45, '1 mm³: one voxel of the grid\nthe network works on', fontsize=8.5,
            color=INK2, va='top')
    ax.set_xscale('log')
    ax.set_xlim(.8, 3e5)
    ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5], ['1', '10', '100', '1,000', '10,000', '100,000'])
    ax.set_yticks(range(len(righe)), [NOME[f] for f in righe], fontsize=10.5, color=INK)
    ax.set_ylim(-.6, len(righe) - .4)
    ax.grid(axis='x', color=GRIGLIA, lw=0.8)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRIGLIA)
    ax.tick_params(colors=INK2, length=0)
    ax.set_xlabel('annotated lesion volume, mm³ (logarithmic scale)', fontsize=9.5, color=INK2)
    ax.set_title(f'The {len(L)} annotated lesions of the 17 patients: which still exist in the 1 mm training target',
                 loc='left', fontsize=11.5, color=INK, fontweight='bold', pad=12)
    fig.legend(handles=[Line2D([], [], ls='none', marker='o', ms=8, color=INK2, mec='white', mew=1.6,
                               label='present in the 1 mm target'),
                        Line2D([], [], ls='none', marker='o', ms=8, mfc='white', mec=INK2, mew=1.8,
                               label='lost in resampling')],
               frameon=False, ncol=2, loc='lower center', fontsize=9.5, bbox_to_anchor=(.47, .0), labelcolor=INK2)
    fig.savefig(FIG, facecolor='white')
    plt.close(fig)


def main():
    with Pool(6) as p:
        R = dict(p.map(caso, sorted(pa.trova_casi())))
    L = [l for x in R.values() for l in x['lesioni']]
    r = ['# Lesioni annotate e bersaglio a 1 mm nel template pediatrico (17 pazienti)', '',
         'Prodotto da `scripts/sopravvivenza_1mm_506.py`, **non modificare a mano**. Componenti a 26 vicini '
         'nella griglia della FLAIR originale, residui <= 1 mm³ esclusi, portate una per una nel template con '
         'la catena della pipeline (lineare, soglia 0.5).', '',
         '| fenotipo | lesioni | scompaiono | sotto 10 mm³ | di cui scompaiono |', '| --- | --- | --- | --- | --- |']
    for f in COL:
        sel = [x for x in L if x['fenotipo'] == f]
        via = [x for x in sel if x['voxel_a_1mm'] == 0]
        p10 = [x for x in sel if x['mm3'] < 10]
        r.append(f"| {NOME[f]} | {len(sel)} | {len(via)} | {len(p10)} | {sum(x['voxel_a_1mm'] == 0 for x in p10)} |")
    via = [x for x in L if x['voxel_a_1mm'] == 0]
    r += [f"| **totale** | {len(L)} | {len(via)} | | |", '',
          f"Volume massimo di una lesione persa: {max((x['mm3'] for x in via), default=0):.1f} mm³.", '',
          'Controllo: voxel in cui l\'unione delle componenti differisce dal bersaglio salvato, in % del bersaglio: '
          + ', '.join(f"{p} {x['voxel_diversi_dal_bersaglio_%']:.2f}" for p, x in R.items()) + '.']
    OUT.with_suffix('.md').write_text('\n'.join(r) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps(R, indent=1))
    figura(L)
    print('\n'.join(r))


if __name__ == '__main__':
    main()
