#!/usr/bin/env python
"""Due tabelle per la tesi: Z prima e dopo il postprocessing, e che cosa toglie il filtro dei 50 mm³.

    python scripts/tabelle_postprocessing_506.py     # -> reports/tabelle_postprocessing_506.{md,json}

PERCHE'. Il responsabile vuole vedere quanto cambia il postprocessing e quante componenti vere e
false toglie il volume minimo. I numeri esistono gia' sparsi (valutazione_cv/test_*, analisi_volume_
minimo_506.md); qui stanno in due tabelle leggibili, cross-validation e test affiancati.

COME. Stesse funzioni di postprocessing_aincp.py (postprocessa, misure), quindi stessi numeri dei
report ufficiali. Tre passi: l'uscita di nnU-Net (soglia 0.5, nessun filtro), la sola soglia 0.3, la
soglia 0.3 con il volume minimo di 50 mm³ (il postprocessing di Z, scelto sulla cross-validation e
congelato prima del test). Per il filtro: componenti predette fra 1 e 50 mm³ a soglia 0.3, divise in
quelle che toccano una lesione annotata (vere) e quelle che non ne toccano (falsi positivi), e le
lesioni annotate che restano senza nessuna componente (perse davvero). Sul test e' descrizione.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402
from analisi_volume_minimo_506 import RUN, componenti  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
OUT = PROGETTO / 'reports/tabelle_postprocessing_506'
PASSI = [('uscita di nnU-Net (soglia 0.5, nessun filtro)', 0.5, 0),
         ('soglia 0.3, nessun filtro', 0.3, 0),
         ('**soglia 0.3 + volume minimo 50 mm³ (Z)**', 0.3, 50)]
FASCE_TOLTE = [(1, 10, '1–10 mm³'), (10, 20, '10–20 mm³'), (20, 50, '20–50 mm³')]


def insieme(nome):
    if nome == 'cv':
        return pp.predizioni_fuori_fold(RUN), RUN / 'prob_nativa'
    return {p.name[:-4]: p for p in sorted((RUN / 'test_insieme').glob('*.npz'))}, RUN / 'test_insieme/prob_nativa'


def analizza(nome):
    npz, cache = insieme(nome)
    passi = {i: [] for i in range(len(PASSI))}
    tolte, perse = [], []
    for c, f in sorted(npz.items()):
        prob = pp.prob_nativa(f, cache)
        ref, cer, vox = pp.carica_caso(c)
        for i, (_, t, v) in enumerate(PASSI):
            passi[i].append(pp.misure(pp.postprocessa(prob, t, v, False, cer, vox), ref, cer, vox))
        lesioni, comp, _ = componenti(prob, ref, vox)               # soglia 0.3
        tenute = set().union(*[q['lesioni'] for q in comp if q['mm3'] >= 50]) if comp else set()
        prima = set().union(*[q['lesioni'] for q in comp if q['mm3'] >= pp.SOGLIA_LESIONE_MM3]) if comp else set()
        tolte += [{'caso': c, **q} for q in comp if pp.SOGLIA_LESIONE_MM3 <= q['mm3'] < 50]
        perse += [{'caso': c, 'mm3': l['mm3']} for l in lesioni if l['id'] in prima and l['id'] not in tenute]
    righe = []
    for i, (et, t, v) in enumerate(PASSI):
        ris = passi[i]
        F, se, pc = pp.f1(ris)
        les = [l for r in ris for l in r['lesioni']]
        pic = [l for l in les if l['mm3'] < 50]
        righe.append({'passo': et, 'dice': float(np.mean([r['dice'] for r in ris])),
                      'trovate': sum(l['rilevata'] for l in les), 'lesioni': len(les),
                      'piccole_trovate': sum(l['rilevata'] for l in pic), 'piccole': len(pic),
                      'componenti': sum(r['componenti'] for r in ris), 'fp': sum(r['fp'] for r in ris),
                      'fp_paz': float(np.mean([r['fp'] for r in ris])), 'prec': pc, 'sens': se, 'f1': F,
                      'errore_carico': float(np.mean([abs(r['perc_pred'] - r['perc_vera']) for r in ris]))})
    return {'pazienti': len(npz), 'passi': righe, 'tolte': tolte, 'perse': perse}


def main():
    R = {k: analizza(k) for k in ('cv', 'test')}
    md = ['# Z prima e dopo il postprocessing; che cosa toglie il filtro dei 50 mm³', '',
          'Prodotto da `scripts/tabelle_postprocessing_506.py`, **non modificare a mano**. Griglia T1 nativa; lesione '
          'trovata se toccata da almeno un voxel; componente vera se tocca una lesione annotata, falso positivo se non '
          'ne tocca nessuna (componenti sotto 1 mm³ non contate). Cross-validation: fuori-fold dei 12 pazienti, dove il '
          'postprocessing è stato scelto. Test: 5 pazienti, solo descrittivo.', '']
    for k, tit in (('cv', 'Cross-validation (12 pazienti)'), ('test', 'Test (5 pazienti)')):
        md += [f'## Prima e dopo il postprocessing: {tit}', '',
               '| passo | Dice medio | lesioni trovate | di cui < 50 mm³ | componenti predette | falsi positivi '
               '(per paziente) | precisione | F1 per lesione | errore sul % di cervello lesionato |',
               '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for r in R[k]['passi']:
            md.append(f"| {r['passo']} | {r['dice']:.3f} | {r['trovate']}/{r['lesioni']} | {r['piccole_trovate']}/{r['piccole']} | "
                      f"{r['componenti']} | {r['fp']} ({r['fp_paz']:.1f}) | {r['prec']:.2f} | {r['f1']:.3f} | "
                      f"{r['errore_carico']:.2f} punti |")
        md.append('')
    md += ['## Che cosa toglie il filtro dei 50 mm³ (a soglia 0.3)', '',
           '| | cross-validation (12) | test (5) |', '| --- | --- | --- |']

    def riga(et, f):
        md.append(f"| {et} | {f(R['cv'])} | {f(R['test'])} |")
    riga('componenti predette tolte (1–50 mm³)', lambda x: len(x['tolte']))
    riga('di cui vere (toccano una lesione annotata)', lambda x: sum(bool(q['lesioni']) for q in x['tolte']))
    riga('di cui falsi positivi', lambda x: sum(not q['lesioni'] for q in x['tolte']))
    riga('falsi positivi tolti, sul totale prima del filtro',
         lambda x: f"{sum(not q['lesioni'] for q in x['tolte'])} su {x['passi'][1]['fp']}")
    riga('**lesioni annotate perse davvero**', lambda x: f"**{len(x['perse'])}**")
    riga('volume delle lesioni perse', lambda x: ', '.join(pp.mm3(p['mm3']) for p in sorted(x['perse'], key=lambda p: p['mm3'])) or '—')
    md += ['', 'Una componente vera tolta non vuol dire una lesione persa: quasi sempre la stessa lesione è toccata '
           'anche da una componente più grande, che il filtro lascia. Persa davvero = la lesione era trovata senza '
           'filtro e non lo è più dopo.', '', '### Componenti tolte per dimensione', '',
           '| volume della componente | CV: vere | CV: falsi positivi | test: vere | test: falsi positivi |',
           '| --- | --- | --- | --- | --- |']
    for lo, hi, et in FASCE_TOLTE:
        c = [[q for q in R[k]['tolte'] if lo <= q['mm3'] < hi] for k in ('cv', 'test')]
        md.append(f"| {et} | {sum(bool(q['lesioni']) for q in c[0])} | {sum(not q['lesioni'] for q in c[0])} | "
                  f"{sum(bool(q['lesioni']) for q in c[1])} | {sum(not q['lesioni'] for q in c[1])} |")
    OUT.with_suffix('.md').write_text('\n'.join(md) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps(R, indent=1, default=float))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
