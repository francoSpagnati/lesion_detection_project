#!/usr/bin/env python
"""Riporta le predizioni fuori-fold di Dataset505 (template pediatrico) nello spazio T1 del paziente.

    python scripts/porta_505_nativo.py                 # tutti i trainer di 505 con 5 fold finiti

PERCHE'. Il modello lavora nel template, ma la misura che conta e' nello spazio del paziente: e'
li' che il clinico guarda e li' che sta l'annotazione originale. Misurare nel template darebbe
lesioni ingrandite di 1.1-1.6 volte (reports/aincp_mni_manifesto.json) e numeri non confrontabili
con 504. Riportando le predizioni nella griglia T1 nativa, 505 e 504 si misurano contro **lo
stesso riferimento** (gt_segmentations di 504) con gli stessi script, e il confronto e' appaiato.

E' anche il pezzo di pipeline che servira' per ogni paziente nuovo: template -> modello -> nativo.

COME. Trasformazione inversa T1->template salvata in transforms_ped/ (affine invertita, poi il
campo inverso), interpolazione lineare della maschera predetta e soglia 0.5: lo stesso criterio
usato all'andata per l'etichetta. Scrive, accanto a validation/, una cartella validation_nativo/
con un file per caso e un summary.json con il Dice nella griglia nativa.
"""
import json
import os
from pathlib import Path

import ants
import nibabel as nib
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
PROC = PROGETTO / 'data/processed/AINCP'
RES = Path(os.environ['nnUNet_results']) / 'Dataset505_AINCP_MNI'
GT = Path(os.environ['nnUNet_preprocessed']) / 'Dataset504_AINCP' / 'gt_segmentations'


def al_nativo(pred_template: Path, soggetto: Path) -> ants.ANTsImage:
    t = soggetto / 'transforms_ped'
    fisso = ants.image_read(str(soggetto / 'T1_N4.nii.gz'))
    p = ants.image_read(str(pred_template))
    p = p.new_image_like((p.numpy() > 0).astype('float32'))
    r = ants.apply_transforms(fixed=fisso, moving=p,
                              transformlist=[str(t / 'T1_to_PED_inv_0.mat'), str(t / 'T1_to_PED_inv_1.nii.gz')],
                              whichtoinvert=[True, False], interpolator='linear')
    return r.new_image_like((r.numpy() >= 0.5).astype('uint8'))


def dice(a, b):
    s = a.sum() + b.sum()
    return float('nan') if s == 0 else 2 * float((a & b).sum()) / s


def main():
    soggetti = {'_'.join(p.name.split('_')[:2]): p for p in PROC.iterdir() if (p / 'manifesto.json').exists()}
    for run in sorted(d for d in RES.iterdir() if d.is_dir()):
        folds = [run / f'fold_{f}' for f in range(5)]
        if not all((f / 'validation' / 'summary.json').exists() for f in folds):
            print(f'{run.name}: fold incompleti, salto')
            continue
        for f in folds:
            out = f / 'validation_nativo'
            out.mkdir(exist_ok=True)
            per_caso = []
            for p in sorted((f / 'validation').glob('*.nii.gz')):
                c = p.name[:-7]
                nat = al_nativo(p, soggetti[c])
                ants.image_write(nat, str(out / p.name))
                # l'immagine scritta da ANTs deve cadere sulla griglia del riferimento di 504
                g = nib.load(str(GT / p.name))
                n = nib.load(str(out / p.name))
                assert g.shape == n.shape and np.allclose(g.affine, n.affine, atol=1e-3), c
                d = dice(np.asanyarray(g.dataobj) > 0, np.asanyarray(n.dataobj) > 0)
                per_caso.append({'reference_file': str(GT / p.name), 'prediction_file': str(out / p.name),
                                 'metrics': {'1': {'Dice': d}}})
                print(f'{run.name[:60]} {f.name} {c}: Dice nativo {d:.3f}', flush=True)
            (out / 'summary.json').write_text(json.dumps({'metric_per_case': per_caso}, indent=1))


if __name__ == '__main__':
    main()
