#!/usr/bin/env python
"""Dice per singola lesione, per fascia di volume: come si comportano le tre ricette di 506.

    python scripts/dice_per_dimensione_506.py    # -> reports/dice_per_dimensione_506.{md,json}
                                                 #    reports/figures/aincp/dice_per_dimensione_506.png

PERCHE'. Il Dice per paziente mescola tutto: un paziente con una cavita' da 150 000 mm³ e venti frammenti
da 3 voxel ha Dice 0.89 anche se i frammenti sono tutti mancati. Il rilevamento per fascia
(postprocessing_aincp.py valuta) dice se una lesione e' stata *toccata*, non *quanto bene* e' stata
segmentata. Qui si misura la seconda cosa, lesione per lesione.

COME (CC-Dice, Jaus et al. 2024, "Every Component Counts", arXiv 2410.18684). Ogni voxel del volume viene assegnato alla lesione
annotata piu' vicina (regioni di Voronoi, distanza euclidea in mm). Il Dice della lesione j e'
calcolato solo dentro la sua regione:

    Dice_j = 2 |L_j ∩ P ∩ V_j| / (|L_j| + |P ∩ V_j|)

cosi' la predizione di una lesione grande non "paga" per i frammenti vicini e viceversa, e una
lesione mancata ha Dice 0. Lesioni sopra 1 mm³ come nel resto della valutazione, connettivita' 26,
predizioni nel nativo con il postprocessing congelato di ciascuna ricetta. Cross-validation
(fuori-fold, 12 pazienti) e test (insieme dei 5 fold, 5 pazienti) separati: il test e' gia' stato
usato una volta, qui lo si rilegge soltanto.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
R = PROGETTO / 'nnUNet/nnUNet_results/Dataset506_AINCP_FINALE'
RICETTE = {'Z': 'nnUNetTrainerWandb_250ep_dropmod30_arresto50',
           'A': 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_dropmod30_arresto50',
           'D': 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_dropmod30_arresto50_dapre2c'}
NOMI = {'Z': 'Z, da zero', 'A': 'A, da ATLAS', 'D': 'D, da ATLAS + WMH'}
# piu' fini di pp.FASCE sopra 200 mm³: le lesioni grandi arrivano a 196 847 mm³
FASCE = [(1, 10, '1–10 mm³'), (10, 50, '10–50 mm³'), (50, 200, '50–200 mm³'), (200, 1000, '200–1 000 mm³'),
         (1000, 10000, '1 000–10 000 mm³'), (10000, np.inf, '> 10 000 mm³')]
COLORI = {'Z': '#2a78d6', 'A': '#eb6834', 'D': '#1baf7a'}      # validati: CVD ΔE >= 9.2, etichette dirette
INK, INK2, GRIGIO, FONDO = '#0b0b0b', '#52514e', '#c9c8c3', '#fcfcfb'
OUT = PROGETTO / 'reports/dice_per_dimensione_506'
FIG = PROGETTO / 'reports/figures/aincp/dice_per_dimensione_506.png'


def dice_per_lesione(pred, ref, vox, zooms):
    lab, n = ndimage.label(ref, structure=pp.STRUTTURA)
    if n == 0:
        return []
    # regione di Voronoi: per ogni voxel, l'etichetta della lesione annotata piu' vicina
    _, idx = ndimage.distance_transform_edt(lab == 0, sampling=zooms, return_indices=True)
    regione = lab[tuple(idx)]
    vol = np.bincount(lab.ravel(), minlength=n + 1)
    inter = np.bincount(lab[pred], minlength=n + 1)
    pred_reg = np.bincount(regione[pred], minlength=n + 1)
    out = []
    for j in range(1, n + 1):
        mm3 = vol[j] * vox
        if mm3 >= pp.SOGLIA_LESIONE_MM3:
            out.append({'mm3': float(mm3), 'dice': float(2 * inter[j] / (vol[j] + pred_reg[j])),
                        'rilevata': bool(inter[j] > 0)})
    return out


def valuta(k, insieme):
    run = R / f'{RICETTE[k]}__nnUNetPlansDa502__3d_fullres'
    par = json.loads((PROGETTO / 'reports' / f'postprocessing_{R.name}__{run.name}.json').read_text())['scelto']
    cache = run / 'prob_nativa' if insieme == 'cv' else run / 'test_insieme' / 'prob_nativa'
    import nibabel as nib
    ris = {}
    for f in sorted(cache.glob('*.npy')):
        c = f.stem
        ref, cer, vox = pp.carica_caso(c)
        zooms = nib.load(str(pp.sessione(c) / 'lesione_pulita_inT1.nii.gz')).header.get_zooms()[:3]
        pred = pp.postprocessa(np.load(f).astype(np.float32), par['t'], par['vmin'], par['cervello'], cer, vox)
        ris[c] = dice_per_lesione(pred, ref, vox, zooms)
    return ris


def riassunto(ris):
    les = [l for v in ris.values() for l in v]
    righe = []
    for lo, hi, et in FASCE:
        s = [l for l in les if lo <= l['mm3'] < hi]
        if s:
            d = np.array([l['dice'] for l in s])
            righe.append({'fascia': et, 'n': len(s), 'media': float(d.mean()), 'mediana': float(np.median(d)),
                          'rilevate': int(sum(l['rilevata'] for l in s)),
                          'media_rilevate': float(d[d > 0].mean()) if (d > 0).any() else None})
    return righe


def figura(tab):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True, facecolor=FONDO)
    for x, (insieme, titolo) in zip(ax, (('cv', 'Cross-validation, 12 pazienti (68 lesioni)'),
                                          ('test', 'Test, 5 pazienti (40 lesioni)'))):
        x.set_facecolor(FONDO)
        etichette = [r['fascia'] for r in tab[insieme]['Z']]
        pos = np.arange(len(etichette))
        fine = {}
        for i, k in enumerate(('Z', 'A', 'D')):
            y = [r['media'] for r in tab[insieme][k]]
            xs = pos + (i - 1) * 0.12          # piccolo scarto: i punti di valore uguale non si coprono
            x.plot(xs, y, color=COLORI[k], lw=2, marker='o', ms=8, mec=FONDO, mew=2, zorder=3)
            fine[k] = y[-1]
        # etichette dirette a destra, distanziate di almeno 0.05 quando i valori finali coincidono
        ultimo = -1
        for k in sorted(fine, key=fine.get):
            yy = max(fine[k], ultimo + 0.05)
            x.text(pos[-1] + 0.3, yy, k, color=COLORI[k], fontsize=11, fontweight='bold', va='center')
            ultimo = yy
        x.set_xticks(pos)
        # l'unita' una volta sola nel titolo dell'asse: ripetuta su ogni fascia le etichette si toccano
        x.set_xticklabels([f"{e.replace(' mm³', '')}\n{r['n']} lesioni" for e, r in zip(etichette, tab[insieme]['Z'])],
                          fontsize=9, color=INK2)
        x.set_xlabel('volume della lesione (mm³)', fontsize=10, color=INK2)
        x.set_ylim(-0.02, 1.0)
        x.set_xlim(-0.5, len(pos) - 0.1)
        x.grid(axis='y', color=GRIGIO, lw=0.6, alpha=0.6)
        for s in ('top', 'right'):
            x.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            x.spines[s].set_color(GRIGIO)
        x.tick_params(colors=INK2)
        x.set_title(titolo, loc='left', fontsize=12, color=INK)
    ax[0].set_ylabel('Dice medio per lesione (mancate = 0)', color=INK, fontsize=10.5)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=COLORI[k], lw=2, marker='o', label=NOMI[k]) for k in ('Z', 'A', 'D')],
               loc='upper left', bbox_to_anchor=(0.005, 0.9), ncol=3, frameon=False, fontsize=10.5, labelcolor=INK)
    fig.suptitle('Quanto bene viene segmentata ogni lesione, per dimensione', x=0.008, ha='left', fontsize=15,
                 color=INK, fontweight='bold', y=0.99)
    fig.text(0.008, 0.935, 'Dice di ciascuna lesione annotata dentro la sua regione di Voronoi (CC-Dice); fasce di volume '
             'in scala logaritmica. Sotto 50 mm³ le fasce sono dominate dalle lesioni mancate.', ha='left', va='top',
             fontsize=10.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=2)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=100, facecolor=FONDO)


def main():
    tab, grezzo = {'cv': {}, 'test': {}}, {'cv': {}, 'test': {}}
    for insieme in ('cv', 'test'):
        for k in RICETTE:
            grezzo[insieme][k] = valuta(k, insieme)
            tab[insieme][k] = riassunto(grezzo[insieme][k])
            print(insieme, k, 'fatto', flush=True)
    righe = ['# Dice per singola lesione, per fascia di volume (Dataset506)', '',
             'Prodotto da `scripts/dice_per_dimensione_506.py`, **non modificare a mano**. CC-Dice: Dice di ogni lesione '
             'annotata dentro la sua regione di Voronoi, 0 se mancata; ogni ricetta con il suo postprocessing congelato. '
             '"Dice delle rilevate" esclude le mancate: dice quanto bene e\' segmentata una lesione *quando* viene trovata.', '']
    for insieme, titolo in (('cv', 'Cross-validation, fuori-fold (12 pazienti)'), ('test', 'Test (5 pazienti)')):
        righe += [f'## {titolo}', '', '| fascia | lesioni | ' + ' | '.join(f'{k}: Dice medio (mediana) · rilevate · Dice delle rilevate'
                                                                         for k in RICETTE) + ' |',
                  '| --- | --- |' + ' --- |' * len(RICETTE)]
        for i, r in enumerate(tab[insieme]['Z']):
            celle = []
            for k in RICETTE:
                q = tab[insieme][k][i]
                mr = f"{q['media_rilevate']:.2f}" if q['media_rilevate'] is not None else '—'
                celle.append(f"{q['media']:.2f} ({q['mediana']:.2f}) · {q['rilevate']}/{q['n']} · {mr}")
            righe.append(f"| {r['fascia']} | {r['n']} | " + ' | '.join(celle) + ' |')
        righe.append('')
    OUT.with_suffix('.md').write_text('\n'.join(righe))
    OUT.with_suffix('.json').write_text(json.dumps({'riassunto': tab, 'lesioni': grezzo}, indent=1))
    figura(tab)
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
