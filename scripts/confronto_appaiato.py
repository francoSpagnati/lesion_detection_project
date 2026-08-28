#!/usr/bin/env python
"""Confronta due training sugli **stessi casi**, non sulle rispettive medie.

Serve ogni volta che si cambia una cosa sola (canale, campo magnetico, skull
stripping) e si vuole sapere se il Dice e' cambiato davvero. Confrontare le due
medie complessive non basta: se i due dataset non hanno gli stessi casi di
validation, la differenza puo' venire dalla composizione invece che dal
cambiamento. Qui si intersecano i casi e si confronta caso per caso, con un
Wilcoxon appaiato che non assume normalita' — il Dice su lesioni piccole ha una
distribuzione molto asimmetrica.

Uso:
    python scripts/confronto_appaiato.py Dataset502_ATLAS Dataset521_ATLAS_3T
    python scripts/confronto_appaiato.py Dataset501_WMH Dataset522_WMH_SS --out reports/skull_stripping_wmh.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

RISULTATI = Path(__file__).resolve().parent.parent / 'nnUNet' / 'nnUNet_results'


def per_caso(ds, classe='1'):
    """Dice per caso del fold 0, piu' la media riportata da nnU-Net."""
    s = next((RISULTATI / ds).glob('*/fold_0/validation/summary.json'))
    j = json.load(open(s))
    dice = {Path(m['reference_file']).name.replace('.nii.gz', ''): m['metrics'][classe]['Dice']
            for m in j['metric_per_case']}
    return dice, j['foreground_mean']['Dice']


def confronta(base, variante, classe='1'):
    a, ma = per_caso(base, classe)
    b, mb = per_caso(variante, classe)
    comuni = sorted(set(a) & set(b))
    x = np.array([a[c] for c in comuni])
    y = np.array([b[c] for c in comuni])
    w = stats.wilcoxon(y, x) if len(comuni) > 1 else None

    r = [f'# {variante} contro {base}', '',
         f'| | casi | Dice medio |', '| --- | --- | --- |',
         f'| {base}, tutta la sua validation | {len(a)} | {ma:.4f} |',
         f'| {variante}, tutta la sua validation | {len(b)} | {mb:.4f} |', '',
         f'## Confronto appaiato sugli stessi {len(comuni)} casi', '',
         f'| | Dice medio |', '| --- | --- |',
         f'| {base} | {x.mean():.4f} |',
         f'| {variante} | {y.mean():.4f} |',
         f'| **differenza** | **{y.mean() - x.mean():+.4f}** |', '',
         f'Migliora in **{(y > x).sum()}** casi, peggiora in **{(y < x).sum()}**, '
         f'pari in {(y == x).sum()}.',
         f'Wilcoxon appaiato: **p = {w.pvalue:.4g}**.' if w else '']

    soli = sorted(set(a) - set(b))
    if soli:
        e = np.array([a[c] for c in soli])
        r += ['', f'I {len(soli)} casi presenti solo in {base} avevano Dice **{e.mean():.4f}** '
                  f'(mediana {np.median(e):.4f}).']

    peggiori = sorted(zip(comuni, x, y), key=lambda t: t[2] - t[1])[:5]
    r += ['', '## I cinque casi che peggiorano di piu\'', '',
          '| caso | prima | dopo | delta |', '| --- | --- | --- | --- |']
    r += [f'| {c} | {p:.4f} | {d:.4f} | {d - p:+.4f} |' for c, p, d in peggiori]
    return '\n'.join(r) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base')
    p.add_argument('variante')
    p.add_argument('--classe', default='1')
    p.add_argument('--out')
    a = p.parse_args()
    testo = confronta(a.base, a.variante, a.classe)
    print(testo)
    if a.out:
        Path(a.out).write_text(testo)


if __name__ == '__main__':
    main()
