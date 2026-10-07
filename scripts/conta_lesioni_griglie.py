#!/usr/bin/env python
"""Perche' le lesioni annotate sono 90 nella griglia della FLAIR e 108 nella griglia T1 nativa.

    python scripts/conta_lesioni_griglie.py   # -> reports/conta_lesioni_griglie.{md,json}

La stessa annotazione vive in due griglie. Il bersaglio di addestramento si costruisce dalla griglia
della FLAIR originale, dove il clinico ha disegnato (sopravvivenza_1mm_506.py: 90 componenti dopo
aver tolto i residui <= 1 mm3). Le metriche si calcolano nella griglia T1 nativa, dove l'etichetta
arriva con la rigida FLAIR -> T1, interpolazione lineare e soglia 0.5, e dove si contano le
componenti >= 1 mm3 (postprocessing_aincp.misure: 68 + 40 = 108). Qui ogni componente della FLAIR
viene portata da sola nella T1 con la stessa trasformazione, e si conta in quanti pezzi >= 1 mm3
arriva; poi si confronta la somma con le componenti dell'etichetta intera, dove pezzi di lesioni
diverse possono toccarsi e fondersi.
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent))
PROGETTO = Path(__file__).resolve().parent.parent
import pipeline_aincp as pa
S = pa.STRUTTURA

def caso(paz):
    import ants, nibabel as nib
    c = pa.trova_casi()[paz]; dest = pa.PROC / c['sessione']
    t1 = ants.image_read(str(dest / 'T1_N4.nii.gz'))
    rif = ants.image_read(str(c['FLAIR'])); vf = float(np.prod(rif.spacing)); vt = float(np.prod(t1.spacing))
    m = np.zeros(rif.shape, bool)
    for r in pa.ROI_LESIONE:
        if r in c['roi']:
            m |= np.squeeze(ants.image_read(str(c['roi'][r])).numpy()) > 0
    m, _ = pa.togli_residui(m, vf)
    lab, n = ndimage.label(m, structure=S)
    mat = [str(dest / 'transforms' / 'FLAIR_to_T1_0.mat')]
    pezzi = []; vols = []
    for i in range(1, n + 1):
        w = ants.apply_transforms(fixed=t1, moving=rif.new_image_like((lab == i).astype('float32')),
                                  transformlist=mat, interpolator='linear').numpy() >= 0.5
        l2, n2 = ndimage.label(w, structure=S)
        v = np.bincount(l2.ravel())[1:] * vt
        pezzi.append(int((v >= 1.001).sum())); vols.append(float((lab == i).sum() * vf))
    ref = np.asanyarray(nib.load(str(dest / 'lesione_pulita_inT1.nii.gz')).dataobj) > 0.5
    lr, nr = ndimage.label(ref, structure=S)
    vr = np.bincount(lr.ravel())[1:] * vt
    return paz, {'flair': n, 'pezzi_in_T1': pezzi, 'vol_flair': vols, 'T1_tutte': int(nr),
                 'T1_valutate': int((vr >= 1.001).sum()), 'T1_sotto_1mm3': int((vr < 1.001).sum())}

if __name__ == '__main__':
    with Pool(6) as p:
        R = dict(p.map(caso, sorted(pa.trova_casi())))
    (PROGETTO / 'reports/conta_lesioni_griglie.json').write_text(json.dumps(R, indent=1))
    tot = {k: 0 for k in ('flair', 'T1_valutate', 'T1_tutte', 'somma_pezzi')}
    for k, x in R.items():
        sp = sum(x['pezzi_in_T1'])
        print(k, 'FLAIR', x['flair'], 'T1 valutate', x['T1_valutate'], 'T1 tutte', x['T1_tutte'], 'somma pezzi', sp,
              'spezzate', sum(p > 1 for p in x['pezzi_in_T1']), 'sparite', sum(p == 0 for p in x['pezzi_in_T1']))
        tot['flair'] += x['flair']; tot['T1_valutate'] += x['T1_valutate']; tot['T1_tutte'] += x['T1_tutte']; tot['somma_pezzi'] += sp
    print(tot)
    P = [(p, v) for x in R.values() for p, v in zip(x['pezzi_in_T1'], x['vol_flair'])]
    print('spezzate', sum(p > 1 for p, _ in P), 'pezzi extra', sum(p - 1 for p, _ in P if p > 1),
          'sparite', sum(p == 0 for p, _ in P), 'volumi sparite', [round(v, 1) for p, v in P if p == 0])
    print('volumi delle spezzate', sorted(round(v) for p, v in P if p > 1))
    sp = [(p, v) for p, v in P if p > 1]; via = [v for p, v in P if p == 0]
    fusioni = tot['somma_pezzi'] - tot['T1_valutate']
    r = ['# Lesioni annotate: 90 nella griglia FLAIR, 108 nella griglia T1 nativa', '',
         'Prodotto da `scripts/conta_lesioni_griglie.py`, **non modificare a mano**. Componenti a 26 vicini; '
         'nella FLAIR residui <= 1 mm³ tolti (come la pipeline), nella T1 contano le componenti >= 1 mm³ (come le metriche).', '',
         f"- componenti nella griglia FLAIR: **{tot['flair']}**",
         f"- spezzate dal passaggio alla T1: **{len(sp)}**, che diventano {sum(p for p, _ in sp)} pezzi (**+{sum(p - 1 for p, _ in sp)}**); "
         f"volumi {min(v for _, v in sp):.0f}–" + f"{max(v for _, v in sp):,.0f}".replace(',', '\u202f') + ' mm³',
         f"- senza nessun pezzo >= 1 mm³ nella T1: **{len(via)}** (**−{len(via)}**), tutte fra {min(via):.1f} e {max(via):.1f} mm³",
         f"- pezzi di lesioni diverse che nella T1 si toccano e si fondono: **−{fusioni}**",
         f"- totale: {tot['flair']} − {len(via)} + {sum(p - 1 for p, _ in sp)} − {fusioni} = **{tot['T1_valutate']}** "
         f"(nella T1 ci sono altre {tot['T1_tutte'] - tot['T1_valutate']} componenti sotto 1 mm³, non contate)", '',
         '| paziente | FLAIR | T1 (>= 1 mm³) | spezzate | senza pezzi >= 1 mm³ |', '| --- | --- | --- | --- | --- |']
    for k, x in R.items():
        r.append(f"| {k} | {x['flair']} | {x['T1_valutate']} | {sum(p > 1 for p in x['pezzi_in_T1'])} | {sum(p == 0 for p in x['pezzi_in_T1'])} |")
    (PROGETTO / 'reports/conta_lesioni_griglie.md').write_text('\n'.join(r) + '\n')
