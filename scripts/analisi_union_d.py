#!/usr/bin/env python
"""Legge il risultato della variante D separandolo per sorgente e per classe.

    python scripts/analisi_union_d.py     # -> reports/risultato_531.md

Perche' serve uno script dedicato invece del solito confronto appaiato. Il `summary.json` di D
riporta una media per classe calcolata su **tutti** i 151 casi di validation, ma le classi non
esistono ovunque: la lesione e' annotata solo nei casi WMH e ATLAS, i sei tessuti solo in quelli
DBB. Una media che mescola le tre sorgenti non descrive niente. Qui ogni classe viene misurata
**sui soli casi che possono contenerla**, e il confronto con le baseline e' appaiato caso per caso
grazie al fatto che i fold di D sono l'unione dei fold di origine.

Il confronto giusto per la componente WMH e' 511 (T1) e non 501 (FLAIR): dentro D, WMH entra con
il solo T1, perche' ATLAS e DBB la FLAIR non ce l'hanno.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROGETTO = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get('nnUNet_results', PROGETTO / 'nnUNet' / 'nnUNet_results'))
TR = 'nnUNetTrainerWandb__nnUNetPlans__3d_fullres'

NOMI = {'1': 'lesione', '2': 'csf', '3': 'grigia', '4': 'bianca',
        '5': 'grigia profonda', '6': 'tronco', '7': 'cervelletto'}
# In D i sei tessuti sono 2..7; nella baseline 503 sono 1..6.
TESSUTI_D_A_503 = {'2': '1', '3': '2', '4': '3', '5': '4', '6': '5', '7': '6'}


def per_caso(dataset: str):
    """{nome del caso: {classe: Dice}} dalla validation del fold 0."""
    s = RES / dataset / TR / 'fold_0' / 'validation' / 'summary.json'
    j = json.loads(s.read_text())
    fuori = {}
    for c in j['metric_per_case']:
        nome = Path(c['prediction_file']).name.replace('.nii.gz', '')
        fuori[nome] = {k: v['Dice'] for k, v in c['metrics'].items()}
    return fuori


def appaiato(a: dict, b: dict, casi, classe_a: str, classe_b: str):
    """Media di a e b sugli stessi casi, differenza e Wilcoxon. Scarta i NaN."""
    x, y = [], []
    for c_a, c_b in casi:
        va, vb = a[c_a].get(classe_a), b[c_b].get(classe_b)
        if va is None or vb is None or np.isnan(va) or np.isnan(vb):
            continue
        x.append(va)
        y.append(vb)
    x, y = np.array(x), np.array(y)
    p = wilcoxon(x, y).pvalue if len(x) > 1 and np.any(x != y) else float('nan')
    return len(x), x.mean(), y.mean(), y.mean() - x.mean(), p, int((y > x).sum()), int((y < x).sum())


def main():
    d = per_caso('Dataset531_UNION_D')
    righe = ['# Variante D: il modello unico su tutti e tre i dataset', '',
             'Prodotto da `scripts/analisi_union_d.py`, **non modificare a mano**.', '',
             'Fold 0, 151 casi di validation: 112 da ATLAS, 23 da DBB, 16 da WMH. Il Dice medio che',
             'nnU-Net stampa a fine training (**0.8966**) e\' la media sulle sette classi ed e\' dominato',
             'dai sei tessuti, che sono densi e facili: non descrive il compito che ci interessa.', '']

    # --- la lesione, separata per sorgente --------------------------------
    righe += ['## La classe che conta: la lesione', '',
              '| sorgente | casi | Dice della lesione in D |', '| --- | --- | --- |']
    for pref, etichetta in (('atlas', 'ATLAS'), ('wmh', 'WMH')):
        v = [m['1'] for n, m in d.items() if n.startswith(pref + '_') and not np.isnan(m.get('1', np.nan))]
        righe.append(f'| {etichetta} | {len(v)} | **{np.mean(v):.4f}** |')
    righe.append('')

    # --- confronto appaiato con le baseline -------------------------------
    righe += ['## Confronto appaiato con le baseline, sugli stessi casi', '',
              '| confronto | classe | casi | baseline | D | differenza | migliora / peggiora | p |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']

    confronti = [
        ('Dataset502_ATLAS', 'atlas_', 'lesione', '1', '1'),
        ('Dataset511_WMH_T1', 'wmh_', 'lesione', '1', '1'),
    ]
    for base_nome, pref, etichetta, cl_base, cl_d in confronti:
        b = per_caso(base_nome)
        casi = [(c, pref + c) for c in b if pref + c in d]
        n, mb, md, diff, p, su, giu = appaiato(b, d, casi, cl_base, cl_d)
        righe.append(f'| D contro `{base_nome}` | {etichetta} | {n} | {mb:.4f} | {md:.4f} | '
                     f'**{diff:+.4f}** | {su} / {giu} | {p:.4g} |')

    b503 = per_caso('Dataset503_DBB')
    casi503 = [(c, 'dbb_' + c) for c in b503 if 'dbb_' + c in d]
    for cl_d, cl_b in TESSUTI_D_A_503.items():
        n, mb, md, diff, p, su, giu = appaiato(b503, d, casi503, cl_b, cl_d)
        righe.append(f'| D contro `Dataset503_DBB` | {NOMI[cl_d]} | {n} | {mb:.4f} | {md:.4f} | '
                     f'**{diff:+.4f}** | {su} / {giu} | {p:.4g} |')
    righe.append('')

    out = PROGETTO / 'reports' / 'risultato_531.md'
    out.write_text('\n'.join(righe) + '\n')
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
