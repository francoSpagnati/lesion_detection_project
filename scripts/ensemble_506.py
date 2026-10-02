#!/usr/bin/env python
"""Ensemble delle ricette A, D, Z di Dataset506, sulle predizioni fuori-fold (mai sul test).

    python scripts/ensemble_506.py    # -> reports/ensemble_506.{md,json}

PERCHE'. Le tre ricette sono equivalenti sui 12 pazienti di addestramento (A-Z p 0.46, D-Z p 0.83),
ma sbagliano in modo diverso: il pretraining toglie falsi positivi prima del postprocessing, Z trova
qualche lesione piccola in piu'. Mediare le probabilita' di modelli diversi e' il modo piu' economico
di unire i due vantaggi (non serve addestrare nulla), e nella letteratura su ATLAS e' una delle
poche cose che spostano il risultato (MAPPING, arXiv 2211.15486).

COME. Per ogni paziente si mediano le probabilita' gia' riportate nel nativo (cache `prob_nativa`
di postprocessing_aincp.py: ogni paziente e' predetto solo dai modelli del fold che non l'ha visto,
quindi la media resta fuori-fold). Poi la stessa griglia e la stessa regola di scelta del
postprocessing delle ricette singole, cosi' il confronto con Z e' alla pari: ognuna col suo
postprocessing migliore. Confronto appaiato del Dice per paziente con Wilcoxon.

Oltre al Dice medio per paziente si riportano la mediana e il Dice **globale** (tutti i voxel dei
12 pazienti insieme): il medio pesa allo stesso modo un paziente con 3 voxel di lesione e uno con
200 000 mm³, il globale pesa i voxel. Sono le tre forme in cui la letteratura riporta il Dice.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
R = PROGETTO / 'nnUNet/nnUNet_results/Dataset506_AINCP_FINALE'
RUN = {'A': 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_dropmod30_arresto50',
       'D': 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_dropmod30_arresto50_dapre2c',
       'Z': 'nnUNetTrainerWandb_250ep_dropmod30_arresto50'}
RUN = {k: R / f'{v}__nnUNetPlansDa502__3d_fullres' for k, v in RUN.items()}
COMBO = {'Z': 'Z', 'A+D+Z': 'ADZ', 'A+D': 'AD', 'D+Z': 'DZ'}
OUT = PROGETTO / 'reports/ensemble_506'


def valuta_combo(nome):
    casi = sorted(p.stem for p in (RUN['Z'] / 'prob_nativa').glob('*.npy'))
    dati = {c: pp.carica_caso(c) for c in casi}
    prob = {c: np.mean([np.load(RUN[k] / 'prob_nativa' / f'{c}.npy') for k in COMBO[nome]], 0) for c in casi}
    griglia = []
    for t in pp.SOGLIE:
        for v in pp.VOLUMI_MIN:
            for cb in pp.CERVELLO:
                ris = [pp.misure(pp.postprocessa(prob[c], t, v, cb, dati[c][1], dati[c][2]), *dati[c]) for c in casi]
                F, _, _ = pp.f1(ris)
                griglia.append({'t': t, 'vmin': v, 'cervello': cb, 'f1': F,
                                'dice': float(np.mean([r['dice'] for r in ris])), 'ris': ris})
    # stessa regola di postprocessing_aincp.scegli
    top = max(g['f1'] for g in griglia)
    g = max([g for g in griglia if g['f1'] >= top - 0.01],
            key=lambda g: (round(g['dice'], 3), -abs(g['t'] - 0.5), -g['vmin'], not g['cervello']))
    ris = g['ris']
    # Dice globale: 2|A∩B| / (|A|+|B|) sommando i volumi di tutti i pazienti
    inter = sum(r['dice'] * (r['mm3_vero'] + r['mm3_pred']) for r in ris)
    tot = sum(r['mm3_vero'] + r['mm3_pred'] for r in ris)
    les = [l for r in ris for l in r['lesioni']]
    return nome, {'t': g['t'], 'vmin': g['vmin'], 'cervello': g['cervello'], 'f1': g['f1'], 'dice': g['dice'],
                  'mediana': float(np.median([r['dice'] for r in ris])), 'globale': inter / tot,
                  'rilevate': sum(l['rilevata'] for l in les), 'lesioni': len(les), 'fp': sum(r['fp'] for r in ris),
                  'per_paziente': {c: r['dice'] for c, r in zip(casi, ris)}}


def main():
    with Pool(len(COMBO)) as p:
        res = dict(p.map(valuta_combo, COMBO))
    z = np.array(list(res['Z']['per_paziente'].values()))
    righe = ['# Ensemble delle ricette di Dataset506 (fuori-fold)', '',
             'Prodotto da `scripts/ensemble_506.py`, **non modificare a mano**. 12 pazienti di addestramento, '
             'nel nativo; ogni combinazione col suo postprocessing scelto con la regola di `postprocessing_aincp.py`.', '',
             '| | postprocessing | F1 lesioni | Dice medio | mediana | globale | rilevate | FP | Δ Dice vs Z | meglio/peggio | p (Wilcoxon) |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for n, r in res.items():
        e = np.array(list(r['per_paziente'].values()))
        conf = ('—', '—', '—') if n == 'Z' else (f'{(e - z).mean():+.3f}', f'{(e > z).sum()}/{(e < z).sum()}',
                                                 f'{wilcoxon(e, z).pvalue:.2f}')
        righe.append(f"| {n} | t {r['t']}, vmin {r['vmin']} | {r['f1']:.3f} | {r['dice']:.3f} | {r['mediana']:.3f} | "
                     f"{r['globale']:.3f} | {r['rilevate']}/{r['lesioni']} | {r['fp']} | {conf[0]} | {conf[1]} | {conf[2]} |")
    OUT.with_suffix('.md').write_text('\n'.join(righe) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps(res, indent=1))
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
