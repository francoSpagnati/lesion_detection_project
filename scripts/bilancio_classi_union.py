"""Conta i voxel di ogni classe in Dataset531_UNION_D, separando i tre sorgenti.

    python scripts/bilancio_classi_union.py    # -> reports/bilancio_classi_531.md

Perche' serve. La variante D mette insieme dataset che annotano cose diverse: WMH e ATLAS marcano solo
la lesione e lasciano a `0` tutto il resto del cervello, DBB marca sei tessuti e riempie l'encefalo.
Quanto pesi davvero questo sbilanciamento non e' un'intuizione da citare, e' un numero da misurare:
decide quanto il campionamento del foreground di nnU-Net (33% delle patch) finira' a pescare tessuto
DBB invece che lesione, ed e' la chiave per leggere il risultato di D.

Si leggono le segmentazioni **gia' preprocessate** (`nnUNetPlans_3d_fullres`), non quelle originali:
sono quelle che la rete vedra' davvero, dopo il ricampionamento a 1 mm.
"""
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import blosc2
import numpy as np

PROGETTO = Path('/home/ccantini/lesion_detection_project')
PRE = PROGETTO / 'nnUNet' / 'nnUNet_preprocessed' / 'Dataset531_UNION_D' / 'nnUNetPlans_3d_fullres'
NOMI = {1: 'lesione', 2: 'csf', 3: 'grigia', 4: 'bianca', 5: 'grigia profonda',
        6: 'tronco', 7: 'cervelletto'}


def conta(f: Path):
    seg = blosc2.open(str(f), mode='r')[:]
    valori, n = np.unique(seg, return_counts=True)
    # -1 e' l'etichetta che nnU-Net usa per i voxel fuori dal ritaglio, non e' una classe.
    return f.name[:-len('_seg.b2nd')], seg.size, {int(v): int(c) for v, c in zip(valori, n) if v >= 0}


def main():
    file = sorted(PRE.glob('*_seg.b2nd'))
    if not file:
        raise SystemExit(f'nessuna segmentazione preprocessata in {PRE}')
    with ProcessPoolExecutor(max_workers=12) as pool:
        risultati = list(pool.map(conta, file, chunksize=8))

    per_sorgente = defaultdict(lambda: {'casi': 0, 'voxel': 0, 'classi': defaultdict(int),
                                        'fg_per_caso': []})
    for caso, tot, classi in risultati:
        s = caso.split('_')[0]
        d = per_sorgente[s]
        d['casi'] += 1
        d['voxel'] += tot
        fg = 0
        for v, c in classi.items():
            if v > 0:
                d['classi'][v] += c
                fg += c
        d['fg_per_caso'].append(fg)

    righe = ['# Bilancio delle classi in Dataset531_UNION_D', '',
             'Misurato sulle segmentazioni preprocessate a 1 mm, cioe' + "' " +
             'su cio' + "' " + 'che la rete vede davvero.', '',
             '| sorgente | casi | foreground mediano per caso (voxel) | frazione del volume |',
             '| --- | --- | --- | --- |']
    for s in ('wmh', 'atlas', 'dbb'):
        d = per_sorgente[s]
        med = float(np.median(d['fg_per_caso']))
        fraz = sum(d['classi'].values()) / d['voxel']
        righe.append(f'| {s} | {d["casi"]} | {med:,.0f} | {fraz:.4%} |')

    med_les = np.median(per_sorgente['wmh']['fg_per_caso'] + per_sorgente['atlas']['fg_per_caso'])
    med_dbb = np.median(per_sorgente['dbb']['fg_per_caso'])
    righe += ['', f'**Un caso DBB porta {med_dbb / med_les:.0f} volte il foreground di un caso '
                  f'lesionale** (mediane: {med_dbb:,.0f} contro {med_les:,.0f} voxel).', '']

    totale = defaultdict(int)
    for d in per_sorgente.values():
        for v, c in d['classi'].items():
            totale[v] += c
    somma = sum(totale.values())
    righe += ['## Composizione del foreground totale', '',
              '| classe | voxel | quota del foreground |', '| --- | --- | --- |']
    for v in sorted(totale):
        righe.append(f'| {v} — {NOMI[v]} | {totale[v]:,} | {totale[v] / somma:.2%} |')

    quota_les = totale[1] / somma
    righe += ['', f'La lesione e' + "' " + f'il **{quota_les:.2%}** di tutto il foreground annotato. '
              'nnU-Net forza il 33% delle patch a contenere foreground, ma non bilancia fra le classi: '
              'quelle patch andranno quindi quasi tutte su tessuto DBB. E' + "' " + 'il motivo per cui '
              'il risultato di D va letto per classe e per sorgente, mai come un Dice medio unico.']

    out = PROGETTO / 'reports' / 'bilancio_classi_531.md'
    out.write_text('\n'.join(righe) + '\n')
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
