#!/usr/bin/env python
"""Rapporto d'intensita' lesione / cervello per fenotipo, in T1 e in FLAIR, sui 17 pazienti AInCP.

    python scripts/contrasto_fenotipi_aincp.py   # -> reports/contrasto_fenotipi_aincp.{md,json}

PERCHE'. Su ATLAS la stessa misura (inspect_failures.py, reports/failure_analysis_502.md) separava i
casi risolti (lesione / cervello 0.62: cavita' scure in T1) dai falliti (1.07: lesione con la stessa
intensita' del tessuto attorno). Qui si chiede lo stesso ai pazienti, fenotipo per fenotipo: la
gliosi e' visibile in T1 o solo in FLAIR? E' il numero dietro la figura t1_flair_fenotipi.png e
dietro l'ablazione dei canali.

COME. Griglia T1 nativa dopo N4; FLAIR e ROI portate li' con la rigida FLAIR -> T1 della pipeline
(come in figura_t1_flair_intro.py). Rapporto = media dentro la ROI / media nel cervello (SynthStrip)
fuori da tutte le ROI di lesione. Differenza da ATLAS: li' il cervello era una soglia d'intensita',
qui e' la maschera di SynthStrip. Etichette usate solo per descrivere i dati, nessun modello
coinvolto: entrano tutti i 17 pazienti, test compreso.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_aincp as pa  # noqa: E402
from figura_t1_flair_intro import NOME, carica  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
OUT = PROGETTO / 'reports/contrasto_fenotipi_aincp'


def caso(paz):
    t1, flair, roi, cer, sp, _ = carica(paz)
    lesione = np.zeros_like(cer)
    for m in roi.values():
        lesione |= m
    sano = cer & ~lesione
    out = {}
    for r, m in roi.items():
        m = m & cer
        if m.sum() * np.prod(sp) < 10:   # sotto 10 mm³ la media e' fatta di bordi
            continue
        out[NOME[r]] = {'mm3': float(m.sum() * np.prod(sp)),
                        'T1': float(t1[m].mean() / t1[sano].mean()),
                        'FLAIR': float(flair[m].mean() / flair[sano].mean())}
    return paz, out


def main():
    from multiprocessing import Pool
    with Pool(6) as p:
        R = dict(p.map(caso, sorted(pa.trova_casi())))
    r = ['# Rapporto d\'intensita\' lesione / cervello per fenotipo (17 pazienti AInCP)', '',
         'Prodotto da `scripts/contrasto_fenotipi_aincp.py`, **non modificare a mano**. Media dentro la ROI '
         'divisa per la media nel cervello fuori dalla lesione, griglia T1 nativa dopo N4. 1 = stessa '
         'intensita\' del tessuto attorno; sotto 1 piu\' scura, sopra 1 piu\' chiara. Su ATLAS (T1): '
         '0.62 nei casi risolti, 1.07 nei falliti (`failure_analysis_502.md`).', '',
         '## Mediana (min–max) sui pazienti che hanno il fenotipo', '',
         '| fenotipo | pazienti | T1 | FLAIR |', '| --- | --- | --- | --- |']
    for f in NOME.values():
        v = [x[f] for x in R.values() if f in x]
        cel = [f"{np.median([a[s] for a in v]):.2f} ({min(a[s] for a in v):.2f}–{max(a[s] for a in v):.2f})"
               for s in ('T1', 'FLAIR')]
        r.append(f'| {f} | {len(v)} | {cel[0]} | {cel[1]} |')
    r += ['', '## Per paziente', '', '| paziente | fenotipo | volume | T1 | FLAIR |', '| --- | --- | --- | --- | --- |']
    for paz, x in R.items():
        for f, a in x.items():
            r.append(f"| {paz} | {f} | {a['mm3']:,.0f} mm³ | {a['T1']:.2f} | {a['FLAIR']:.2f} |".replace(',', ' '))
    OUT.with_suffix('.md').write_text('\n'.join(r) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps(R, indent=1))
    print('\n'.join(r))


if __name__ == '__main__':
    main()
