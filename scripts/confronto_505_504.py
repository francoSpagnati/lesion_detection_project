#!/usr/bin/env python
"""Stadio 3 nel template (505) contro lo spazio nativo (504), misurati nello stesso posto.

    python scripts/porta_505_nativo.py        # prima: predizioni di 505 riportate nel nativo
    python scripts/confronto_505_504.py       # -> reports/confronto_505_504.{md,json}

Tutte le varianti si misurano nella griglia T1 nativa contro lo stesso riferimento
(gt_segmentations di 504), con le predizioni fuori-fold: ogni paziente dal fold in cui era in
validazione. Gli split di 504 e 505 sono identici, quindi il confronto e' appaiato paziente per
paziente. Le lesioni e i falsi positivi si contano con la soglia di 1 mm3 decisa il 23/9
(`metriche_per_lesione.analizza`, soglia 1.001 perche' il confronto li' e' stretto).

Il riferimento dei confronti e' la ricetta scelta su 504 (passo differenziato, freno corretto):
A e' la stessa ricetta nel template, quindi A contro di lei misura lo spazio; B e D partono da pesi
diversi (stadio 2 in sequenza, stadio 1+2 congiunto) e contro A misurano il pretraining.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metriche_per_lesione import analizza  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
RES = Path(os.environ['nnUNet_results'])
GT = Path(os.environ['nnUNet_preprocessed']) / 'Dataset504_AINCP' / 'gt_segmentations'
SOGLIA = 1.001
P = '__nnUNetPlansDa502__3d_fullres'
VARIANTI = [
    ('504 da zero', RES / 'Dataset504_AINCP' / f'nnUNetTrainerWandb_250ep_arresto50{P}', 'validation'),
    ('504 ricetta (nativo)', RES / 'Dataset504_AINCP' / f'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_arresto50{P}', 'validation'),
    ('505 A: ricetta nel template', RES / 'Dataset505_AINCP_MNI' / f'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_arresto50{P}', 'validation_nativo'),
    ('505 B: da 542 (sequenza)', RES / 'Dataset505_AINCP_MNI' / f'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50_da542{P}', 'validation_nativo'),
    ('505 D: da 543 (congiunto)', RES / 'Dataset505_AINCP_MNI' / f'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50_da543{P}', 'validation_nativo'),
]
RIF = '504 ricetta (nativo)'


def dice(caso, f_pred):
    import nibabel as nib
    g = np.asanyarray(nib.load(str(GT / f'{caso}.nii.gz')).dataobj) > 0
    p = np.asanyarray(nib.load(str(f_pred)).dataobj) > 0
    return 2 * float((g & p).sum()) / (g.sum() + p.sum())


def misura(cartella: Path, sotto: str) -> dict | None:
    pred = {}
    for f in range(5):
        for p in (cartella / f'fold_{f}' / sotto).glob('*.nii.gz'):
            pred[p.name[:-7]] = p
    if len(pred) != 10:
        return None
    out = {}
    for c, p in sorted(pred.items()):
        r = analizza((c, p, GT / f'{c}.nii.gz', 1, SOGLIA))
        out[c] = {'dice': dice(c, p), 'lesioni': len(r['lesioni']),
                  'rilevate': sum(l['rilevata'] for l in r['lesioni']), 'fp': r['falsi_positivi']}
    return out


def main():
    ris = {}
    for nome, cartella, sotto in VARIANTI:
        m = misura(cartella, sotto)
        if m is None:
            print(f'{nome}: fold incompleti, salto')
            continue
        ris[nome] = m
    casi = sorted(ris[RIF])
    righe = ['# Stadio 3: template (505) contro nativo (504)', '',
             'Prodotto da `scripts/confronto_505_504.py`, **non modificare a mano**.', '',
             'Tutto nella griglia T1 nativa, contro gli stessi riferimenti, predizioni fuori-fold, '
             f'soglia {SOGLIA:g} mm³ su lesioni e falsi positivi. Differenze appaiate contro «{RIF}», '
             'Wilcoxon (con dieci pazienti p < 0.05 richiede un effetto quasi sempre nello stesso verso).', '',
             '| variante | Dice medio | Δ Dice (p) | lesioni rilevate | falsi positivi | Δ FP (p) |',
             '| --- | --- | --- | --- | --- | --- |']
    r0 = ris[RIF]
    for nome, m in ris.items():
        d = np.array([m[c]['dice'] for c in casi])
        fp = np.array([m[c]['fp'] for c in casi])
        ril, les = sum(m[c]['rilevate'] for c in casi), sum(m[c]['lesioni'] for c in casi)
        if nome == RIF:
            dd = dfp = '—'
        else:
            a = d - np.array([r0[c]['dice'] for c in casi])
            b = fp - np.array([r0[c]['fp'] for c in casi])
            dd = f'{a.mean():+.3f} ({wilcoxon(a).pvalue:.2f})' if a.any() else '0'
            dfp = f'{b.mean():+.1f} ({wilcoxon(b).pvalue:.2f})' if b.any() else '0'
        righe.append(f'| {nome} | {d.mean():.3f} | {dd} | {ril}/{les} | {fp.sum()} ({fp.mean():.1f}/paz.) | {dfp} |')
    righe += ['', '## Dice per paziente', '', '| paziente | ' + ' | '.join(ris) + ' |',
              '| --- |' + ' --- |' * len(ris)]
    for c in casi:
        righe.append(f'| {c} | ' + ' | '.join(f'{ris[n][c]["dice"]:.3f}' for n in ris) + ' |')
    (PROGETTO / 'reports' / 'confronto_505_504.md').write_text('\n'.join(righe) + '\n')
    (PROGETTO / 'reports' / 'confronto_505_504.json').write_text(json.dumps(ris, indent=1))
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
