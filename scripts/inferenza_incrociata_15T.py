#!/usr/bin/env python
"""Applica il modello addestrato **solo su 3 T** ai casi a 1.5 T che non ha mai visto.

**La domanda, che e' diversa da quella gia' misurata.** Il test 521 contro 502 ha
risposto a "escludere i 1.5 T dal training migliora la segmentazione dei 3 T?", e la
risposta e' no (+0.0093, p=0.67). Qui si chiede il rovescio: **quanto costa
l'esclusione?** Se un giorno arrivasse una risonanza a 1.5 T, un modello addestrato
solo su 3 T saprebbe ancora segmentarla, o avremmo costruito un modello che su quel
dominio non funziona?

Due insiemi, con potere statistico diverso, e la differenza va tenuta presente:

  - **9 casi puliti**: erano nella validation del fold 0 di 502 e non compaiono
    affatto in 521. Nessuno dei due modelli li ha visti, quindi il confronto e'
    appaiato e onesto. Sono pochi.
  - **36 casi in piu'**: 521 non li ha mai visti, ma 502 li aveva in training. Qui
    il numero di 521 e' valido da solo (generalizzazione fuori dominio), mentre
    quello di 502 sarebbe ottimistico e **non va confrontato**.

Uso:
    python scripts/inferenza_incrociata_15T.py
"""
import collections
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
RAW = PROGETTO / 'data' / 'processed' / 'nnUNet_raw' / 'Dataset502_ATLAS'
RISULTATI = PROGETTO / 'nnUNet' / 'nnUNet_results'
LAVORO = PROGETTO / 'data' / 'processed' / 'inferenza_15T'
COORTI_15T = {'r029', 'r039', 'r040'}


def coorte(caso):
    return re.match(r'^sub-(r\d+)s\d+$', caso).group(1)


def insiemi():
    dev = sorted(p.name.replace('_0000.nii.gz', '')
                 for p in (RAW / 'imagesTr').glob('*_0000.nii.gz'))
    q15 = [c for c in dev if coorte(c) in COORTI_15T]
    s = json.load(open(PROGETTO / 'nnUNet' / 'nnUNet_preprocessed' /
                       'Dataset502_ATLAS' / 'splits_final.json'))[0]
    val, tr = set(s['val']), set(s['train'])
    return [c for c in q15 if c in val], [c for c in q15 if c in tr]


def dice(a, b):
    a, b = a > 0.5, b > 0.5
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def prevedi(casi, uscita):
    """Un solo `nnUNetv2_predict` su tutti i casi: caricare il modello costa piu'
    dell'inferenza, quindi non si chiama una volta per caso."""
    ingresso = LAVORO / 'input'
    ingresso.mkdir(parents=True, exist_ok=True)
    for c in casi:
        d = ingresso / f'{c}_0000.nii.gz'
        if not d.exists():
            d.symlink_to(os.path.realpath(RAW / 'imagesTr' / f'{c}_0000.nii.gz'))
    uscita.mkdir(parents=True, exist_ok=True)
    modello = RISULTATI / 'Dataset521_ATLAS_3T' / 'nnUNetTrainerWandb__nnUNetPlans__3d_fullres'
    cmd = ['nnUNetv2_predict', '-i', str(ingresso), '-o', str(uscita),
           '-d', '521', '-c', '3d_fullres', '-f', '0',
           '-tr', 'nnUNetTrainerWandb', '-chk', 'checkpoint_final.pth',
           '-npp', '4', '-nps', '4']
    print('  ' + ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-2000:] + '\n' + r.stderr[-2000:])
    assert modello.exists()


def dice_di_502(casi):
    s = next((RISULTATI / 'Dataset502_ATLAS').glob('*/fold_0/validation/summary.json'))
    j = json.load(open(s))
    return {Path(m['reference_file']).name.replace('.nii.gz', ''): m['metrics']['1']['Dice']
            for m in j['metric_per_case']}


def main():
    puliti, extra = insiemi()
    tutti = puliti + extra
    print(f'{len(puliti)} casi puliti + {len(extra)} visti solo da 502 = {len(tutti)}')

    uscita = LAVORO / 'pred_521'
    if len(list(uscita.glob('*.nii.gz'))) < len(tutti):
        prevedi(tutti, uscita)

    d521 = {}
    for c in tutti:
        gt = np.asarray(nib.load(str(RAW / 'labelsTr' / f'{c}.nii.gz')).dataobj)
        pr = np.asarray(nib.load(str(uscita / f'{c}.nii.gz')).dataobj)
        d521[c] = dice(gt, pr)
    d502 = dice_di_502(tutti)

    x = np.array([d502[c] for c in puliti])
    y = np.array([d521[c] for c in puliti])
    ye = np.array([d521[c] for c in extra])
    yt = np.array([d521[c] for c in tutti])

    from scipy import stats
    w = stats.wilcoxon(y, x)

    # Quanto del divario viene da un caso solo? Se la risposta e' "quasi tutto",
    # la media non descrive un effetto sistematico ma un singolo fallimento.
    delta = y - x
    i = int(np.argmin(delta))
    peggiore = puliti[i]
    senza_peggiore = float(np.delete(delta, i).mean())

    r = ['# Il modello solo-3 T applicato ai casi a 1.5 T', '',
         'Risponde a "quanto **costa** escludere i 1.5 T", non a "quanto si guadagna":',
         'la seconda domanda ha gia\' risposto `reports/test_esclusione_1_5T.md` (+0.0093, p=0.67).', '',
         f'## Confronto appaiato sui {len(puliti)} casi mai visti da nessuno dei due modelli', '',
         '| | Dice medio | mediana |', '| --- | --- | --- |',
         f'| `502`, addestrato **con** i 1.5 T | {x.mean():.4f} | {np.median(x):.4f} |',
         f'| `521`, addestrato **senza** | {y.mean():.4f} | {np.median(y):.4f} |',
         f'| **differenza** | **{y.mean() - x.mean():+.4f}** | |', '',
         f'Migliora in {(y > x).sum()}, peggiora in {(y < x).sum()}. Wilcoxon appaiato: '
         f'**p = {w.pvalue:.4g}**.', '',
         '### Tre limiti da tenere presenti leggendo questo numero', '',
         f'1. **Potere statistico basso**: con {len(puliti)} casi la misura serve a escludere un '
         'crollo, non a stimare una differenza fine.',
         f'2. **Una sola coorte**: i casi puliti vengono tutti da `{sorted(set(coorte(c) for c in puliti))}`. '
         'Quindi si sta misurando la generalizzazione a **uno scanner**, non ai 1.5 T in generale.',
         f'3. **Un solo caso pesa per due terzi**: togliendo `{peggiore}` la differenza media passa da '
         f'{(y - x).mean():+.4f} a {senza_peggiore:+.4f}. Non e\' un effetto diffuso.', '',
         f'## `521` da solo su tutti i {len(tutti)} casi a 1.5 T mai visti', '',
         f'Dice medio **{yt.mean():.4f}**, mediana {np.median(yt):.4f}, '
         f'minimo {yt.min():.4f}, massimo {yt.max():.4f}. '
         f'Composizione per coorte: {dict(collections.Counter(coorte(c) for c in tutti))}.',
         f'Sui {len(extra)} casi che 502 aveva in training il solo `521` vale {ye.mean():.4f}; '
         'il numero di 502 su quei casi sarebbe ottimistico e non va confrontato.', '',
         f'Per riferimento, `521` sui suoi 103 casi di validation a **3 T** vale **0.6160**.', '',
         '## Dettaglio dei casi puliti', '',
         '| caso | 502 (con 1.5 T) | 521 (senza) | delta |', '| --- | --- | --- | --- |']
    r += [f'| {c} | {d502[c]:.4f} | {d521[c]:.4f} | {d521[c] - d502[c]:+.4f} |' for c in puliti]
    testo = '\n'.join(r) + '\n'
    (PROGETTO / 'reports' / 'inferenza_incrociata_15T.md').write_text(testo)
    print(testo)


if __name__ == '__main__':
    main()
