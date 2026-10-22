#!/usr/bin/env python
"""Confronto fuori-fold fra il training misto (M, Dataset532) e il modello della tesi (Z, Dataset506).

    python scripts/postprocessing_aincp.py scegli --run <cartella di M>   # prima: il postprocessing di M
    python scripts/confronto_misto_532.py                                  # -> reports/confronto_misto_532.{md,json}
    python scripts/confronto_misto_532.py B                                # -> reports/confronto_blob_506.{md,json}

IL CRITERIO E' FISSATO PRIMA DEI RISULTATI (scritto il 5/10/2026, prima di lanciare M), cosi' che
non lo si possa scegliere guardando i numeri:

1. **misura principale: le lesioni sotto 50 mm³ trovate.** E' l'errore di Z (10 su 46 nella
   cross-validation) ed e' cio' che i dati pubblici dovrebbero insegnare. Confronto appaiato
   lesione per lesione (le stesse 46 lesioni annotate per i due modelli), test esatto di McNemar
   sulle lesioni trovate da uno solo dei due;
2. **vincolo: i falsi positivi.** Trovare piu' lesioni piccole inventandone molte di piu' non e'
   un miglioramento; si riportano i totali e l'F1 a livello di lesione, che tiene insieme le due cose;
3. **secondarie**: Dice per paziente (Wilcoxon appaiato sui 12), mediana e Dice globale, lesioni
   trovate per fascia di volume, copertura per fenotipo.

M vince se trova piu' lesioni piccole con p < 0.05 senza peggiorare l'F1. Con 12 pazienti un
pareggio sul Dice e' il risultato atteso e non e' un fallimento. Ognuno dei due con il proprio
postprocessing, scelto sui fuori-fold con la stessa regola (postprocessing_aincp.py): e' cosi' che
sono state confrontate A, D e Z. I pazienti in validazione sono gli stessi fold per fold (gli split
di 532 sono quelli di 506), quindi ogni paziente e' predetto in entrambi da un modello che non
l'ha visto e il confronto e' appaiato.

Il test non entra: se M vince qui, si misura sul test come analisi aggiuntiva accanto a Z, che
resta il modello dichiarato prima di aprire il test.

RICETTA B (blob loss, 12/10/2026). Stesso confronto, stesso criterio, scritto qui prima di lanciare
B: misura principale le lesioni sotto 50 mm³ trovate (McNemar appaiato sulle stesse 46 lesioni),
vincolo l'F1 per lesione e i falsi positivi, secondarie come sopra. B vince se trova piu' lesioni
piccole con p < 0.05 senza peggiorare l'F1. B e' su Dataset506 con gli stessi split di Z.

RICETTE C E F (21/10/2026, scritto prima di lanciarle; scripts/dataset_classi_aincp.py). Stessi split,
stesso trainer di Z, la lesione e' l'unione delle classi di lesione. C (ventricolo dilatato come classe)
vuole togliere falsi positivi: misura principale i **falsi positivi** e l'F1 per lesione, vincolo le
lesioni trovate (McNemar, non devono scendere). F (fenotipi come classi) vuole dire il tipo di lesione:
qui conta che il rilevamento **non peggiori** (F1, lesioni trovate, Dice); quanto indovina il tipo si
misura a parte. Le stesse tabelle servono per tutte e due.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402
from ablazione_canali_506 import FASCE, FENOTIPI, fenotipi_in_t1, fp_, mcnemar  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
RES = Path(os.environ['nnUNet_results'])
RUN = {'Z': RES / 'Dataset506_AINCP_FINALE' / 'nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres',
       'M': RES / 'Dataset532_MISTO_AINCP' / 'nnUNetTrainerMisto_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres',
       'B': RES / 'Dataset506_AINCP_FINALE' / 'nnUNetTrainerBlob_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres',
       'C': RES / 'Dataset507_AINCP_VENTRICOLO' / 'nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres',
       'F': RES / 'Dataset508_AINCP_FENOTIPI' / 'nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres'}
CAND = sys.argv[1] if len(sys.argv) > 1 else 'M'
TITOLO = {'M': 'Training misto (M, 532)', 'B': 'Blob loss (B, 506)', 'C': 'Lesione + ventricolo dilatato (C, 507)',
          'F': 'Fenotipi come classi (F, 508)'}[CAND]
RUN = {'Z': RUN['Z'], CAND: RUN[CAND]}
CRITERIO = {k: 'misura principale le lesioni sotto 50 mm³ trovate (McNemar appaiato), vincolo i falsi positivi e l\'F1.'
            for k in ('M', 'B')}
CRITERIO['C'] = 'misura principale i falsi positivi e l\'F1 per lesione, vincolo le lesioni trovate (McNemar, non devono scendere).'
CRITERIO['F'] = ('il rilevamento non deve peggiorare (F1, lesioni trovate, Dice); il riconoscimento del tipo di lesione '
                 'è in reports/fenotipi_508.md.')
OUT = PROGETTO / 'reports' / {'M': 'confronto_misto_532', 'B': 'confronto_blob_506', 'C': 'confronto_ventricolo_507',
                              'F': 'confronto_fenotipi_508'}[CAND]


def parametri(run: Path) -> dict:
    f = PROGETTO / 'reports' / f'postprocessing_{run.parent.name}__{run.name}.json'
    if not f.exists():
        sys.exit(f'manca {f.name}: prima postprocessing_aincp.py scegli --run {run}')
    return json.loads(f.read_text())['scelto']


def valuta_caso(c: str):
    ref, cer, vox = pp.carica_caso(c)
    fen = {r: m & ref for r, m in fenotipi_in_t1(c, pp.sessione(c)).items()}
    out = {}
    for nome, run in RUN.items():
        par = parametri(run)
        cache = run / 'prob_nativa'
        cache.mkdir(exist_ok=True)
        pred = pp.postprocessa(pp.prob_nativa(pp.predizioni_fuori_fold(run)[c], cache, pp.canali_lesione(run)),
                               par['t'], par['vmin'], par['cervello'], cer, vox)
        m = pp.misure(pred, ref, cer, vox)
        m['copertura'] = {r: [int((mk & pred).sum()), int(mk.sum())] for r, mk in fen.items()}
        out[nome] = m
    return c, out


def main():
    casi = sorted(pp.predizioni_fuori_fold(RUN['Z']))
    assert sorted(pp.predizioni_fuori_fold(RUN[CAND])) == casi, f'{CAND} non ha tutti i fold'
    with Pool(6) as p:
        R = dict(p.map(valuta_caso, casi))
    tot = {}
    for n in RUN:
        ris = [R[c][n] for c in casi]
        F, se, pc = pp.f1(ris)
        les = [l for r in ris for l in r['lesioni']]
        inter = sum(r['dice'] * (r['mm3_vero'] + r['mm3_pred']) for r in ris)
        tot[n] = {'postprocessing': parametri(RUN[n]), 'f1': F, 'sens': se, 'prec': pc,
                  'dice': float(np.mean([r['dice'] for r in ris])), 'mediana': float(np.median([r['dice'] for r in ris])),
                  'globale': inter / sum(r['mm3_vero'] + r['mm3_pred'] for r in ris), 'fp': sum(r['fp'] for r in ris),
                  'trovate': [l['rilevata'] for l in les], 'volumi': [l['mm3'] for l in les],
                  'per_paziente': {c: R[c][n]['dice'] for c in casi},
                  'copertura': {f: [sum(R[c][n]['copertura'].get(f, [0, 0])[i] for c in casi) for i in (0, 1)]
                                for f in FENOTIPI}}
    z, m = tot['Z'], tot[CAND]
    M = CAND
    v = np.array(z['volumi'])
    piccole = v < 50
    tz, tm = np.array(z['trovate']), np.array(m['trovate'])
    sm, sz, p_picc = mcnemar(tm[piccole], tz[piccole])
    sm_all, sz_all, p_all = mcnemar(tm, tz)
    dz, dm = np.array(list(z['per_paziente'].values())), np.array(list(m['per_paziente'].values()))
    r = [f'# {TITOLO} contro il modello della tesi (Z, 506), fuori-fold', '',
         f'Prodotto da `scripts/confronto_misto_532.py{" " + CAND if CAND != "M" else ""}`, **non modificare a mano**. Criterio fissato prima dei risultati: '
         + CRITERIO[CAND], '',
         '## Misura principale', '',
         f'| | Z | {M} | trovate solo da {M} / solo da Z | p (McNemar) |', '| --- | --- | --- | --- | --- |',
         f'| lesioni < 50 mm³ trovate | {tz[piccole].sum()}/{piccole.sum()} | {tm[piccole].sum()}/{piccole.sum()} | '
         f'{sm} / {sz} | {fp_(p_picc)} |',
         f'| tutte le lesioni trovate | {tz.sum()}/{len(tz)} | {tm.sum()}/{len(tm)} | {sm_all} / {sz_all} | {fp_(p_all)} |', '',
         '## Vincolo e misure secondarie', '',
         '| | postprocessing | F1 lesioni | falsi positivi | Dice medio | mediana | globale |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for n, t in tot.items():
        pp_ = t['postprocessing']
        r.append(f"| {n} | soglia {pp_['t']}, volume minimo {pp_['vmin']} mm³{', cervello' if pp_['cervello'] else ''} | "
                 f"{t['f1']:.3f} | {t['fp']} | {t['dice']:.3f} | {t['mediana']:.3f} | {t['globale']:.3f} |")
    r += ['', f'Dice per paziente, {M} meno Z: {(dm - dz).mean():+.3f} in media, {M} meglio in {(dm > dz).sum()} pazienti e '
          f'peggio in {(dm < dz).sum()}, p = {fp_(wilcoxon(dm, dz).pvalue)} (Wilcoxon).', '',
          '## Lesioni trovate per fascia di volume', '', f'| fascia | lesioni | Z | {M} |', '| --- | --- | --- | --- |']
    for lo, hi, et in FASCE:
        sel = (v >= lo) & (v < hi)
        r.append(f'| {et} | {sel.sum()} | {tz[sel].sum()} | {tm[sel].sum()} |')
    r += ['', '## Copertura per fenotipo', '', f'| fenotipo | voxel | Z | {M} |', '| --- | --- | --- | --- |']
    for f, nome in FENOTIPI.items():
        nv = z['copertura'][f][1]
        r.append(f"| {nome} | {nv} | " + ' | '.join(f"{tot[k]['copertura'][f][0] / nv:.0%}" if nv else '—' for k in RUN) + ' |')
    r += ['', '## Dice per paziente', '', f'| paziente | Z | {M} |', '| --- | --- | --- |']
    r += [f"| {c} | {z['per_paziente'][c]:.3f} | {m['per_paziente'][c]:.3f} |" for c in casi]
    OUT.with_suffix('.md').write_text('\n'.join(r) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps({'totali': tot, 'per_caso': R}, indent=1, default=float))
    print('\n'.join(r))


if __name__ == '__main__':
    main()
