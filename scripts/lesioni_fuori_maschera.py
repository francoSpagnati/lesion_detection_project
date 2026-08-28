#!/usr/bin/env python
"""Misura quanta parte delle lesioni annotate cadrebbe FUORI dalla maschera di
SynthStrip, su WMH e ATLAS.

**Perche' serve.** Su DBB SynthStrip e' risultato affidabile (Dice 0.966), ma nei
casi con ventricoli dilatati esclude volume che e' quasi tutto CSF. Nella paralisi
cerebrale la dilatazione ventricolare e' comune e le lesioni tipiche (leucomalacia
periventricolare) stanno proprio intorno ai ventricoli. Quindi prima di usare la
maschera per **ritagliare** le immagini va risposto a una domanda sola: quanti
voxel di lesione annotata perderemmo?

La risposta cambia la decisione:
  - vicino a zero  -> si puo' costruire un dataset skull-stripped ritagliato
  - non zero       -> la maschera si usa solo come canale o come `ignore` label,
                      dove un voxel escluso per errore non toglie annotazione

Uso:
    python scripts/lesioni_fuori_maschera.py --n-atlas 100
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

PROGETTO = Path(__file__).resolve().parent.parent
RAW = PROGETTO / 'data' / 'processed' / 'nnUNet_raw'
SCRIPT = PROGETTO / 'scripts' / 'synthstrip' / 'synthstrip.py'
PESI = PROGETTO / 'synthstrip.1.pt'


def casi(dataset, n=None, seme=0):
    """Coppie (immagine canale 0, label) del dataset nnU-Net."""
    img = sorted((RAW / dataset / 'imagesTr').glob('*_0000.nii.gz'))
    if n is not None and n < len(img):
        img = list(np.random.default_rng(seme).choice(img, size=n, replace=False))
        img = sorted(img)
    out = []
    for i in img:
        nome = i.name.replace('_0000.nii.gz', '')
        lab = RAW / dataset / 'labelsTr' / f'{nome}.nii.gz'
        if lab.exists():
            out.append({'caso': nome, 'img': str(i), 'lab': str(lab)})
    return out


def misura(dataset, n, uscita):
    uscita.mkdir(parents=True, exist_ok=True)
    lista = casi(dataset, n)
    print(f'{dataset}: {len(lista)} casi', flush=True)
    righe = []
    t0 = time.time()
    for k, c in enumerate(lista, 1):
        maschera = uscita / f'{c["caso"]}_mask.nii.gz'
        if not maschera.exists():
            r = subprocess.run(
                [sys.executable, str(SCRIPT), '-i', c['img'], '-m', str(maschera),
                 '--model', str(PESI)],
                capture_output=True, text=True)
            if r.returncode != 0:
                print(f'  {c["caso"]}: FALLITO\n{r.stderr[-400:]}', flush=True)
                continue
        gt = nib.load(c['lab']).get_fdata() > 0.5
        mb = nib.load(str(maschera)).get_fdata() > 0.5
        vox = np.prod(nib.load(c['lab']).header.get_zooms()[:3]) / 1000.0  # ml
        n_les = int(gt.sum())
        if n_les == 0:
            continue
        fuori = int((gt & ~mb).sum())
        righe.append({'caso': c['caso'], 'dataset': dataset,
                      'lesione_ml': n_les * vox, 'fuori_ml': fuori * vox,
                      'frazione_fuori': fuori / n_les})
        if k % 20 == 0:
            print(f'  {k}/{len(lista)} ({time.time()-t0:.0f}s)', flush=True)
    return pd.DataFrame(righe)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-atlas', type=int, default=100)
    p.add_argument('--n-wmh', type=int, default=None)
    a = p.parse_args()

    base = PROGETTO / 'data' / 'processed' / 'synthstrip'
    d = pd.concat([
        misura('Dataset501_WMH', a.n_wmh, base / 'WMH'),
        misura('Dataset502_ATLAS', a.n_atlas, base / 'ATLAS'),
    ], ignore_index=True)
    d.to_csv(PROGETTO / 'reports' / 'lesioni_fuori_maschera.csv', index=False)

    r = ['# Lesioni perse dallo skull stripping', '',
         'Frazione di voxel di lesione annotata che cade **fuori** dalla maschera',
         'di SynthStrip. Decide se la maschera puo' + "'" + ' essere usata per ritagliare le',
         'immagini o solo come canale/`ignore` label.', '',
         '| dataset | casi | frazione media | mediana | massima | casi >1% | casi >5% |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for ds, g in d.groupby('dataset'):
        r.append(f'| {ds} | {len(g)} | {g.frazione_fuori.mean():.4f} | '
                 f'{g.frazione_fuori.median():.4f} | {g.frazione_fuori.max():.4f} | '
                 f'{(g.frazione_fuori > 0.01).sum()} | {(g.frazione_fuori > 0.05).sum()} |')
    r += ['', '## I dieci casi che perderebbero di piu' + "'", '',
          '| caso | dataset | lesione (ml) | fuori (ml) | frazione |',
          '| --- | --- | --- | --- | --- |']
    for _, x in d.nlargest(10, 'frazione_fuori').iterrows():
        r.append(f'| {x.caso} | {x.dataset} | {x.lesione_ml:.1f} | {x.fuori_ml:.2f} | '
                 f'{x.frazione_fuori:.4f} |')
    (PROGETTO / 'reports' / 'lesioni_fuori_maschera.md').write_text('\n'.join(r) + '\n')
    print('\n'.join(r[5:12]))


if __name__ == '__main__':
    main()
