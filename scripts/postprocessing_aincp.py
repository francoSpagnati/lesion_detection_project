#!/usr/bin/env python
"""Postprocessing e valutazione del modello AInCP, nello spazio del paziente.

    # 1. scelta dei parametri sulle predizioni fuori-fold della cross-validation (mai sul test)
    python scripts/postprocessing_aincp.py scegli  --run <cartella trainer__piano__config>
    # 2. valutazione con i parametri scelti: sui fuori-fold, o una volta sola sul test
    python scripts/postprocessing_aincp.py valuta  --run <cartella> [--test <cartella predizioni>]

CHE COSA FA IL POSTPROCESSING. nnU-Net restituisce, voxel per voxel, la probabilita' di lesione
nel template. Da li' alla maschera finale ci sono tre scelte, che nnU-Net fa di default in un modo
solo (soglia 0.5, nessun filtro) e che qui si scelgono sui dati:

1. **ritorno al nativo delle probabilita'**, non della maschera: si interpola la mappa continua
   con la trasformazione inversa e si taglia dopo, cosi' il bordo della lesione cade dove la
   probabilita' passa la soglia nella griglia del paziente, invece di ereditare i gradini da 1 mm;
2. **soglia di probabilita'** t: sotto 0.5 si trovano piu' lesioni e si inventano piu' falsi
   positivi, sopra il contrario;
3. **volume minimo** di una componente predetta, in mm3 nativi: i falsi positivi dei modelli su
   504 sono in maggioranza macchie piccole (reports/analisi_rilevamento_504.md);
4. **maschera del cervello** (SynthStrip sulla T1): si scarta una componente con meno di meta' dei
   voxel nel cervello.

LA SCELTA. Griglia piccola (5 x 6 x 2 = 60 combinazioni) valutata sulle predizioni fuori-fold di
tutti i pazienti di addestramento, obiettivo **F1 a livello di lesione** (media armonica di
sensibilita' per lesione e precisione per componente predetta), perche' e' la misura che tiene
insieme le due cose che interessano al clinico: le lesioni trovate e quelle inventate. A parita'
di F1 entro 0.01 si preferisce il Dice, poi la combinazione piu' vicina al default. I parametri
scelti si scrivono su file e **si congelano prima di guardare il test**.

LE MISURE (`valuta`). Nella griglia T1 nativa, contro l'etichetta pulita dai residui
(lesione_pulita_inT1 di pipeline_aincp.py): Dice; lesioni rilevate per fascia di volume (soglia
1 mm3, criterio "almeno un voxel"); falsi positivi; **volume e percentuale di cervello lesionato**,
predetti contro veri (la grandezza che il responsabile ha indicato come quella che conta), con
correlazione e differenza media; tutto anche per strato di acquisizione (centro e campo), con
intervalli di confidenza bootstrap sui pazienti.
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROGETTO = Path(__file__).resolve().parent.parent
PROC = PROGETTO / 'data/processed/AINCP'
STRUTTURA = np.ones((3, 3, 3), dtype=bool)
SOGLIA_LESIONE_MM3 = 1.001
SOGLIE = [0.3, 0.4, 0.5, 0.6, 0.7]
VOLUMI_MIN = [0, 5, 10, 20, 50, 100]
CERVELLO = [False, True]
DEFAULT = {'t': 0.5, 'vmin': 0, 'cervello': False}
FASCE = [(1, 10, '1–10 mm³'), (10, 50, '10–50'), (50, 200, '50–200'), (200, np.inf, '> 200')]


def mm3(v: float) -> str:
    """Un volume in mm³ da leggere: cifre a gruppi di tre con uno spazio sottile, nessun decimale
    sopra 10 mm³. Nel progetto i volumi si scrivono sempre e solo in mm³."""
    if v >= 10:
        return f'{v:,.0f}'.replace(',', '\u202f') + ' mm³'
    return f'{v:.1f} mm³'


def sessione(caso: str) -> Path:
    return next(d for d in PROC.iterdir() if d.is_dir() and '_'.join(d.name.split('_')[:2]) == caso)


def canali_lesione(run: Path) -> tuple:
    """I canali di uscita che sono lesione: tutte le classi tranne lo sfondo e il ventricolo dilatato.
    506: (1,); 507 (lesione + ventricolo): (1,); 508 (gliosi, cistica, encefalomalacia): (1, 2, 3)."""
    lab = json.loads((run / 'dataset.json').read_text())['labels']
    return tuple(sorted(v for k, v in lab.items() if k not in ('background', 'ventricolo_dilatato')))


def prob_nativa(npz: Path, cache: Path, canali: tuple = (1,)) -> np.ndarray:
    """Probabilita' di lesione nella griglia T1 nativa del paziente (float16, in cache). Con piu' classi
    di lesione e' la somma delle loro probabilita' (softmax: la probabilita' che il voxel sia lesione)."""
    f = cache / (npz.name[:-4] + '.npy')
    if f.exists():
        return np.load(f).astype(np.float32)
    import ants
    import nibabel as nib
    caso = npz.name[:-4]
    s = sessione(caso)
    rif = nib.load(str(npz.with_suffix('.nii.gz')))              # la maschera esportata: geometria del template
    p = np.load(npz)['probabilities'][list(canali)].sum(0).transpose(2, 1, 0)   # SimpleITK (z,y,x) -> nibabel (x,y,z)
    assert p.shape == rif.shape, (caso, p.shape, rif.shape)
    tmp = cache / f'{caso}_tpl.nii.gz'
    nib.save(nib.Nifti1Image(p.astype(np.float32), rif.affine), str(tmp))
    t = s / 'transforms_ped'
    nat = ants.apply_transforms(fixed=ants.image_read(str(s / 'T1_N4.nii.gz')), moving=ants.image_read(str(tmp)),
                                transformlist=[str(t / 'T1_to_PED_inv_0.mat'), str(t / 'T1_to_PED_inv_1.nii.gz')],
                                whichtoinvert=[True, False], interpolator='linear')
    tmp.unlink()
    # ANTs e nibabel leggono lo stesso file con lo stesso ordine degli assi: verificato contro gt
    a = nat.numpy().astype(np.float16)
    np.save(f, a)
    return a.astype(np.float32)


def carica_caso(caso: str):
    import nibabel as nib
    s = sessione(caso)
    g = nib.load(str(s / 'lesione_pulita_inT1.nii.gz'))
    ref = np.asanyarray(g.dataobj) > 0.5
    cer = np.asanyarray(nib.load(str(s / 'cervello_inT1.nii.gz')).dataobj) > 0.5
    return ref, cer, float(np.prod(g.header.get_zooms()[:3]))


def postprocessa(prob, t, vmin, cervello, cer, vox):
    m = prob >= t
    if vmin == 0 and not cervello:
        return m
    lab, n = ndimage.label(m, structure=STRUTTURA)
    if n == 0:
        return m
    idx = np.arange(1, n + 1)
    vol = ndimage.sum(np.ones_like(m), lab, idx) * vox
    tieni = vol >= vmin if vmin else np.ones(n, bool)
    if cervello:
        tieni &= ndimage.mean(cer, lab, idx) >= 0.5
    return np.isin(lab, idx[tieni])


def misure(pred, ref, cer, vox):
    lr, nr = ndimage.label(ref, structure=STRUTTURA)
    lp, np_ = ndimage.label(pred, structure=STRUTTURA)
    vr = np.bincount(lr.ravel(), minlength=nr + 1)[1:] * vox
    vp = np.bincount(lp.ravel(), minlength=np_ + 1)[1:] * vox
    toccate = set(np.unique(lr[pred])) - {0}
    lesioni = [{'mm3': float(vr[i]), 'rilevata': (i + 1) in toccate} for i in range(nr) if vr[i] >= SOGLIA_LESIONE_MM3]
    colpite = set(np.unique(lp[ref])) - {0}
    fp = sum(1 for j in range(np_) if vp[j] >= SOGLIA_LESIONE_MM3 and (j + 1) not in colpite)
    comp = int((vp >= SOGLIA_LESIONE_MM3).sum())
    s = ref.sum() + pred.sum()
    cer_mm3 = cer.sum() * vox
    return {'dice': 2 * float((ref & pred).sum()) / s if s else 1.0, 'lesioni': lesioni, 'fp': int(fp),
            'componenti': comp, 'mm3_vero': float(ref.sum() * vox), 'mm3_pred': float(pred.sum() * vox),
            'perc_vera': float(100 * ref.sum() * vox / cer_mm3), 'perc_pred': float(100 * pred.sum() * vox / cer_mm3)}


def f1(ris):
    les = [l for r in ris for l in r['lesioni']]
    sens = sum(l['rilevata'] for l in les) / max(1, len(les))
    comp = sum(r['componenti'] for r in ris)
    prec = (comp - sum(r['fp'] for r in ris)) / comp if comp else 0.0
    return (2 * sens * prec / (sens + prec) if sens + prec else 0.0), sens, prec


def predizioni_fuori_fold(run: Path) -> dict:
    out = {}
    for f in sorted(run.glob('fold_*')):
        for p in (f / 'validation').glob('*.npz'):
            out[p.name[:-4]] = p
    return out


def scegli(run: Path):
    cache = run / 'prob_nativa'
    cache.mkdir(exist_ok=True)
    npz = predizioni_fuori_fold(run)
    casi = {c: (prob_nativa(p, cache, canali_lesione(run)), *carica_caso(c)) for c, p in sorted(npz.items())}
    griglia = []
    for t in SOGLIE:
        for v in VOLUMI_MIN:
            for cb in CERVELLO:
                ris = [misure(postprocessa(pr, t, v, cb, cer, vox), ref, cer, vox) for pr, ref, cer, vox in casi.values()]
                F, se, pc = f1(ris)
                griglia.append({'t': t, 'vmin': v, 'cervello': cb, 'f1': F, 'sens': se, 'prec': pc,
                                'dice': float(np.mean([r['dice'] for r in ris])), 'fp': sum(r['fp'] for r in ris)})
                print(f"t={t} vmin={v} cervello={cb}: F1 {F:.3f} sens {se:.3f} prec {pc:.3f} Dice {griglia[-1]['dice']:.3f}", flush=True)
    top = max(g['f1'] for g in griglia)
    vicini = [g for g in griglia if g['f1'] >= top - 0.01]
    scelto = max(vicini, key=lambda g: (round(g['dice'], 3), -abs(g['t'] - 0.5), -g['vmin'], not g['cervello']))
    default = next(g for g in griglia if all(g[k] == DEFAULT[k] for k in DEFAULT))
    nome = f'{run.parent.name}__{run.name}'
    out = PROGETTO / 'reports' / f'postprocessing_{nome}'
    out.with_suffix('.json').write_text(json.dumps({'run': str(run), 'casi': sorted(casi), 'scelto':
                                                    {k: scelto[k] for k in DEFAULT}, 'griglia': griglia}, indent=1))
    righe = [f'# Postprocessing scelto per `{nome}`', '', 'Prodotto da `scripts/postprocessing_aincp.py scegli`, '
             f'**non modificare a mano**. Predizioni fuori-fold di {len(casi)} pazienti, nel nativo.', '',
             '| | soglia | volume minimo (mm³) | maschera cervello | F1 lesioni | sensibilita\' | precisione | Dice | falsi positivi |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for et, g in (('default nnU-Net', default), ('**scelto**', scelto)):
        righe.append(f"| {et} | {g['t']} | {g['vmin']} | {'si' if g['cervello'] else 'no'} | {g['f1']:.3f} | {g['sens']:.3f} | "
                     f"{g['prec']:.3f} | {g['dice']:.3f} | {g['fp']} |")
    righe += ['', 'Le dieci combinazioni migliori per F1:', '', '| soglia | vmin | cervello | F1 | sens | prec | Dice | FP |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for g in sorted(griglia, key=lambda g: -g['f1'])[:10]:
        righe.append(f"| {g['t']} | {g['vmin']} | {'si' if g['cervello'] else 'no'} | {g['f1']:.3f} | {g['sens']:.3f} | "
                     f"{g['prec']:.3f} | {g['dice']:.3f} | {g['fp']} |")
    out.with_suffix('.md').write_text('\n'.join(righe) + '\n')
    print('\n'.join(righe))


def bootstrap(v, n=2000, seme=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seme)
    m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def valuta(run: Path, test: Path | None):
    nome = f'{run.parent.name}__{run.name}'
    par = json.loads((PROGETTO / 'reports' / f'postprocessing_{nome}.json').read_text())['scelto']
    if test is None:
        npz, cache, etichetta = predizioni_fuori_fold(run), run / 'prob_nativa', 'fuori-fold (cross-validation)'
    else:
        npz = {p.name[:-4]: p for p in sorted(test.glob('*.npz'))}
        cache, etichetta = test / 'prob_nativa', f'**test** ({test})'
    cache.mkdir(exist_ok=True)
    split = next(iter(Path(os.environ['nnUNet_raw']).glob(f'{run.parent.name}/split_aincp.json')), None)
    strati = {}
    if split:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from pipeline_aincp import strato
        strati = {p: strato(m) for p, m in json.loads(split.read_text())['pazienti'].items()}
    ris = {}
    for c, p in sorted(npz.items()):
        ref, cer, vox = carica_caso(c)
        pr = prob_nativa(p, cache, canali_lesione(run))
        ris[c] = {'post': misure(postprocessa(pr, par['t'], par['vmin'], par['cervello'], cer, vox), ref, cer, vox),
                  'default': misure(pr >= 0.5, ref, cer, vox), 'strato': strati.get(c, '—')}
    righe = [f'# Valutazione di `{nome}`', '', f'Prodotto da `scripts/postprocessing_aincp.py valuta`, **non modificare a mano**. '
             f'Predizioni {etichetta}, {len(ris)} pazienti, nella griglia T1 nativa. Postprocessing: soglia {par["t"]}, '
             f'volume minimo {par["vmin"]} mm³, maschera del cervello {"si" if par["cervello"] else "no"}.', '',
             '| | Dice (IC 95%) | lesioni rilevate | falsi positivi / paziente | F1 lesioni |', '| --- | --- | --- | --- | --- |']
    for k, et in (('default', 'senza postprocessing'), ('post', 'con postprocessing')):
        d = [r[k]['dice'] for r in ris.values()]
        les = [l for r in ris.values() for l in r[k]['lesioni']]
        F, _, _ = f1([r[k] for r in ris.values()])
        lo, hi = bootstrap(d)
        righe.append(f"| {et} | {np.mean(d):.3f} ({lo:.3f}–{hi:.3f}) | {sum(l['rilevata'] for l in les)}/{len(les)} | "
                     f"{np.mean([r[k]['fp'] for r in ris.values()]):.1f} | {F:.3f} |")
    righe += ['', '## Per fascia di volume (con postprocessing)', '', '| fascia | lesioni | rilevate |', '| --- | --- | --- |']
    les = [l for r in ris.values() for l in r['post']['lesioni']]
    for lo_, hi_, et in FASCE:
        s = [l for l in les if lo_ <= l['mm3'] < hi_]
        if s:
            righe.append(f"| {et} | {len(s)} | {sum(l['rilevata'] for l in s)} ({sum(l['rilevata'] for l in s) / len(s):.0%}) |")
    v = np.array([r['post']['perc_vera'] for r in ris.values()])
    w = np.array([r['post']['perc_pred'] for r in ris.values()])
    righe += ['', '## Percentuale di cervello lesionato: predetta contro vera', '',
              # Pearson e' dominata dai pochi pazienti con lesioni grandi; Spearman guarda solo l'ordine
              f'Correlazione di Pearson {np.corrcoef(v, w)[0, 1]:.3f}, per ranghi (Spearman) '
              f'{__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(v, w)[0]:.3f}; differenza media (predetta − vera) '
              f'{np.mean(w - v):+.2f} punti percentuali, limiti di accordo ±{1.96 * np.std(w - v, ddof=1):.2f}.', '',
              '| paziente | strato | % vera | % predetta | Dice | lesioni rilevate | falsi positivi |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for c, r in ris.items():
        q = r['post']
        righe.append(f"| {c} | {r['strato']} | {q['perc_vera']:.2f} | {q['perc_pred']:.2f} | {q['dice']:.3f} | "
                     f"{sum(l['rilevata'] for l in q['lesioni'])}/{len(q['lesioni'])} | {q['fp']} |")
    if len(set(r['strato'] for r in ris.values())) > 1:
        righe += ['', '## Per strato di acquisizione', '', '| strato | pazienti | Dice | lesioni rilevate | FP / paziente |',
                  '| --- | --- | --- | --- | --- |']
        for s in sorted(set(r['strato'] for r in ris.values())):
            g = [r['post'] for r in ris.values() if r['strato'] == s]
            ls = [l for q in g for l in q['lesioni']]
            righe.append(f"| {s} | {len(g)} | {np.mean([q['dice'] for q in g]):.3f} | {sum(l['rilevata'] for l in ls)}/{len(ls)} | "
                         f"{np.mean([q['fp'] for q in g]):.1f} |")
    suff = 'test' if test else 'cv'
    out = PROGETTO / 'reports' / f'valutazione_{suff}_{nome}'
    out.with_suffix('.md').write_text('\n'.join(righe) + '\n')
    out.with_suffix('.json').write_text(json.dumps({'parametri': par, 'casi': ris}, indent=1))
    print('\n'.join(righe))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('azione', choices=['scegli', 'valuta'])
    ap.add_argument('--run', required=True, type=Path)
    ap.add_argument('--test', type=Path, help='cartella delle predizioni del test (npz + nii.gz)')
    a = ap.parse_args()
    scegli(a.run) if a.azione == 'scegli' else valuta(a.run, a.test)


if __name__ == '__main__':
    main()
