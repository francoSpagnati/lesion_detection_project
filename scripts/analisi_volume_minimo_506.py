#!/usr/bin/env python
"""Che cosa toglie il volume minimo di 50 mm³ dal postprocessing di Z, e che cosa si puo' usare al suo posto.

    python scripts/analisi_volume_minimo_506.py
    # -> reports/analisi_volume_minimo_506.{md,json}

PERCHE'. Il responsabile della tesi trova 50 mm³ troppo grandi come soglia: una sfera di 4.6 mm di
diametro. La domanda: quante lesioni vere si perdono per colpa del filtro, e se un altro passo di
postprocessing tiene le lesioni piccole senza riportare i falsi positivi.

COME. Probabilita' di Z gia' riportate nel nativo (cache prob_nativa di postprocessing_aincp.py),
etichetta pulita, stesse misure di postprocessing_aincp.py (lesione rilevata se toccata da almeno un
voxel; falso positivo = componente predetta >= 1 mm³ che non tocca nessuna lesione). Il filtro agisce
sulle **componenti predette**, non sulle lesioni annotate: una lesione piccola resta trovata se la
macchia che la tocca supera la soglia.

Alternative, tutte a soglia 0.3 come Z:
  - volume minimo piu' basso (0, 5, 10, 20 mm³);
  - **isteresi**: si tiene una componente se il suo voxel piu' sicuro supera h (0.5-0.9). Decide la
    sicurezza della rete, non la dimensione: una macchia piccola ma netta resta;
  - **un voxel della griglia della rete**: la rete lavora a 1 mm nel template, quindi una componente
    nativa sotto 1 mm³ e' un artefatto del ritorno al nativo, non una predizione. Soglia fissata dalla
    geometria, senza guardare i dati.
Sul test (5 pazienti, gia' usato il 2/10) i numeri si riportano solo come descrizione: nessuna scelta.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
RUN = PROGETTO / 'nnUNet/nnUNet_results/Dataset506_AINCP_FINALE/nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres'
OUT = PROGETTO / 'reports/analisi_volume_minimo_506'
T = 0.3
CONFIG = ([('volume', v) for v in (0, 1, 5, 10, 20, 50)] +
          [('isteresi', h) for h in (0.5, 0.6, 0.7, 0.8, 0.9)])


def componenti(prob, ref, vox):
    """Le componenti a soglia T con volume, probabilita' massima e lesioni annotate che toccano."""
    m = prob >= T
    lp, n = ndimage.label(m, structure=pp.STRUTTURA)
    lr, nr = ndimage.label(ref, structure=pp.STRUTTURA)
    vr = np.bincount(lr.ravel(), minlength=nr + 1)[1:] * vox
    idx = np.arange(1, n + 1)
    vol = np.bincount(lp.ravel(), minlength=n + 1)[1:] * vox
    pmax = ndimage.maximum(prob, lp, idx) if n else np.array([])
    tocca = [set() for _ in range(n)]
    for a, b in set(zip(lp[ref & m].ravel(), lr[ref & m].ravel())):
        tocca[a - 1].add(int(b))
    lesioni = [{'id': i + 1, 'mm3': float(vr[i])} for i in range(nr) if vr[i] >= pp.SOGLIA_LESIONE_MM3]
    comp = [{'mm3': float(vol[j]), 'pmax': float(pmax[j]), 'lesioni': sorted(tocca[j])} for j in range(n)]
    return lesioni, comp, lp


def tieni(c, tipo, x):
    return c['mm3'] >= x if tipo == 'volume' else (c['pmax'] >= x and c['mm3'] >= pp.SOGLIA_LESIONE_MM3)


def valuta(casi, tipo, x):
    tot = {'lesioni': 0, 'trovate': 0, 'piccole': 0, 'piccole_trovate': 0, 'fp': 0, 'comp': 0, 'dice': []}
    perse = []
    for c, d in casi.items():
        k = [j for j, q in enumerate(d['comp']) if tieni(q, tipo, x)]
        trovate = set().union(*[d['comp'][j]['lesioni'] for j in k]) if k else set()
        for l in d['lesioni']:
            tot['lesioni'] += 1
            tot['trovate'] += l['id'] in trovate
            if l['mm3'] < 50:
                tot['piccole'] += 1
                tot['piccole_trovate'] += l['id'] in trovate
        conta = [j for j in k if d['comp'][j]['mm3'] >= pp.SOGLIA_LESIONE_MM3]
        tot['comp'] += len(conta)
        tot['fp'] += sum(not d['comp'][j]['lesioni'] for j in conta)
        pred = np.isin(d['lp'], np.array(k) + 1) if k else np.zeros_like(d['ref'])
        s = d['ref'].sum() + pred.sum()
        tot['dice'].append(2 * float((d['ref'] & pred).sum()) / s if s else 1.0)
        perse += [(c, l['mm3']) for l in d['lesioni'] if l['id'] in d['trovate_senza_filtro'] and l['id'] not in trovate]
    sens = tot['trovate'] / tot['lesioni']
    prec = (tot['comp'] - tot['fp']) / tot['comp'] if tot['comp'] else 0.0
    tot['f1'] = 2 * sens * prec / (sens + prec) if sens + prec else 0.0
    tot['dice'] = float(np.mean(tot['dice']))
    tot['perse_rispetto_a_nessun_filtro'] = perse
    return tot


def carica(npz, cache):
    casi = {}
    for c, p in sorted(npz.items()):
        prob = pp.prob_nativa(p, cache)
        ref, _, vox = pp.carica_caso(c)
        lesioni, comp, lp = componenti(prob, ref, vox)
        trovate = set().union(*[q['lesioni'] for q in comp if q['mm3'] >= pp.SOGLIA_LESIONE_MM3]) if comp else set()
        casi[c] = {'lesioni': lesioni, 'comp': comp, 'lp': lp, 'ref': ref, 'trovate_senza_filtro': trovate}
    return casi


def piccole(casi):
    """Le lesioni annotate sotto 50 mm³: quante la rete vede, e con quale macchia predetta le trova.
    Se la macchia che tocca una lesione piccola e' grande, la lesione e' un satellite di una lesione
    grande e il filtro non la puo' togliere."""
    r = {'n': 0, 'trovate': 0, 'da_macchia_sotto_50': 0}
    for d in casi.values():
        for l in d['lesioni']:
            if l['mm3'] >= 50:
                continue
            r['n'] += 1
            v = [q['mm3'] for q in d['comp'] if l['id'] in q['lesioni']]
            if v:
                r['trovate'] += 1
                r['da_macchia_sotto_50'] += max(v) < 50
    return r


def frase_piccole(r):
    return (f"Lesioni annotate sotto 50 mm³: {r['n']}. Toccate dalla predizione a soglia 0.3, senza filtro: "
            f"{r['trovate']}; di queste, trovate **solo** da una macchia sotto 50 mm³ (le uniche che il filtro può "
            f"far perdere): **{r['da_macchia_sotto_50']}**. Le altre stanno dentro una macchia predetta più grande, "
            'quasi sempre quella di una lesione grande vicina.')


def tabella(casi, titolo):
    righe = [f'## {titolo}', '',
             '| filtro (soglia 0.3) | lesioni trovate | di cui < 50 mm³ | falsi positivi | F1 per lesione | Dice medio |',
             '| --- | --- | --- | --- | --- | --- |']
    ris = {}
    for tipo, x in CONFIG:
        r = valuta(casi, tipo, x)
        ris[f'{tipo}_{x}'] = r
        et = f'volume >= {x:g} mm³' if tipo == 'volume' else f'isteresi: picco >= {x:g}'
        if tipo == 'volume' and x == 50:
            et = f'**{et} (Z, scelto)**'
        righe.append(f"| {et} | {r['trovate']}/{r['lesioni']} | {r['piccole_trovate']}/{r['piccole']} | {r['fp']} | "
                     f"{r['f1']:.3f} | {r['dice']:.3f} |")
    return righe, ris


def main():
    cv = carica(pp.predizioni_fuori_fold(RUN), RUN / 'prob_nativa')
    te = carica({p.name[:-4]: p for p in sorted((RUN / 'test_insieme').glob('*.npz'))}, RUN / 'test_insieme/prob_nativa')
    md = ['# Il volume minimo di 50 mm³ nel postprocessing di Z: che cosa toglie', '',
          'Prodotto da `scripts/analisi_volume_minimo_506.py`, **non modificare a mano**. Soglia di probabilità 0.3 '
          'in tutte le righe; griglia T1 nativa; lesione trovata se toccata da almeno un voxel; falso positivo = '
          'componente predetta >= 1 mm³ che non tocca lesioni annotate.', '']
    r_cv, ris_cv = tabella(cv, 'Cross-validation, fuori-fold (12 pazienti)')
    md += r_cv + ['', frase_piccole(piccole(cv))]
    # le componenti che il filtro dei 50 toglie, e le lesioni che si perdono con loro
    tolte = [(c, q) for c, d in cv.items() for q in d['comp'] if pp.SOGLIA_LESIONE_MM3 <= q['mm3'] < 50]
    vere = [(c, q) for c, q in tolte if q['lesioni']]
    md += ['', f"Componenti predette fra 1 e 50 mm³ (quelle che il filtro toglie): **{len(tolte)}**, di cui "
           f"**{len(vere)} toccano una lesione vera** e {len(tolte) - len(vere)} sono falsi positivi.", '',
           'Lesioni vere perse passando da nessun filtro a 50 mm³: ' +
           (', '.join(f'{c} {pp.mm3(v)}' for c, v in ris_cv['volume_50']['perse_rispetto_a_nessun_filtro']) or 'nessuna') + '.',
           '', '| componente tolta | mm³ | probabilità massima | tocca una lesione |', '| --- | --- | --- | --- |']
    for c, q in sorted(tolte, key=lambda t: -t[1]['pmax']):
        md.append(f"| {c} | {q['mm3']:.1f} | {q['pmax']:.2f} | {'sì' if q['lesioni'] else 'no'} |")
    r_te, ris_te = tabella(te, 'Test (5 pazienti): solo descrittivo, nessuna scelta fatta qui')
    md += [''] + r_te + ['', frase_piccole(piccole(te))]
    md += ['', 'Lesioni vere del test perse passando da nessun filtro a 50 mm³: ' +
           (', '.join(f'{c} {pp.mm3(v)}' for c, v in ris_te['volume_50']['perse_rispetto_a_nessun_filtro']) or 'nessuna') + '.']
    OUT.with_suffix('.md').write_text('\n'.join(md) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps({'cv': ris_cv, 'test': ris_te,
                                                    'componenti_tolte_cv': [{'caso': c, **q} for c, q in tolte]}, indent=1))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
