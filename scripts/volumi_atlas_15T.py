#!/usr/bin/env python
"""Volume delle lesioni ATLAS nei due insiemi su cui si misura il modello solo-3 T (Dataset521).

    python scripts/volumi_atlas_15T.py      # -> reports/volumi_atlas_15T.{md,json}

PERCHE'. reports/inferenza_incrociata_15T.md da' il Dice di 521 sui 45 casi a 1.5 T che non ha mai
visto (0.672) e sui suoi 103 casi di validazione a 3 T (0.616). Il Dice cresce con il volume della
lesione: senza il volume dei due insiemi non si puo' dire se la differenza viene dal campo o dalle
lesioni. Il report non lo riporta; qui lo si misura dalle etichette.

COME. I 45 casi a 1.5 T sono quelli di inferenza_incrociata_15T.insiemi() (9 nella validazione del
fold 0 di 502 + 36 nel suo training; coorti r029, r039, r040); i 103 sono la validazione del fold 0
di Dataset521_ATLAS_3T (splits_final.json). Etichette in labelsTr, voxel da 1 mm3 (verificato).
Due misure: il volume totale di lesione per caso e il volume di ogni componente connessa
(connettivita' 26), con quartili.
"""
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inferenza_incrociata_15T as ii  # noqa: E402
from postprocessing_aincp import mm3  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
SPLIT521 = Path(os.environ['nnUNet_preprocessed']) / 'Dataset521_ATLAS_3T' / 'splits_final.json'
OUT = PROGETTO / 'reports' / 'volumi_atlas_15T'
STRUTTURA = np.ones((3, 3, 3), dtype=bool)


def volumi(caso: str):
    im = nib.load(str(ii.RAW / 'labelsTr' / f'{caso}.nii.gz'))
    m = np.asanyarray(im.dataobj) > 0
    vox = float(np.prod(im.header.get_zooms()[:3]))
    lab, _ = ndimage.label(m, structure=STRUTTURA)
    return float(m.sum() * vox), (np.bincount(lab.ravel())[1:] * vox).tolist(), vox


def main():
    pulita, in_train = ii.insiemi()
    insiemi = [('1.5 T, mai visti da 521', pulita + in_train),
               ('3 T, validazione di 521', json.loads(SPLIT521.read_text())[0]['val'])]
    R = {}
    for nome, casi in insiemi:
        v = [volumi(c) for c in casi]
        assert {x[2] for x in v} == {1.0}, 'voxel diversi da 1 mm3'
        tot = np.array([x[0] for x in v])
        comp = np.concatenate([x[1] for x in v])
        R[nome] = {'casi': len(casi), 'per_caso_mm3': dict(zip(casi, tot.tolist())),
                   'quartili_caso': np.percentile(tot, [25, 50, 75]).tolist(), 'min': tot.min(), 'max': tot.max(),
                   'componenti': len(comp), 'quartili_componente': np.percentile(comp, [25, 50, 75]).tolist()}
    num = lambda x: mm3(x).replace(' mm³', '')                # l'unita' sta nell'intestazione  # noqa: E731
    q = lambda xs: ' / '.join(num(x) for x in xs)  # noqa: E731
    md = ['# Volume delle lesioni ATLAS: 1.5 T contro 3 T (insiemi del modello 521)', '',
          'Prodotto da `scripts/volumi_atlas_15T.py`, **non modificare a mano**. Etichette di Dataset502 '
          '(`labelsTr`, voxel da 1 mm³); connettività 26. Il Dice di 521 sui due insiemi è in '
          '`reports/inferenza_incrociata_15T.md`.', '',
          '| insieme | casi | volume per caso: Q1 / mediana / Q3 (mm³) | minimo – massimo (mm³) | componenti | '
          'volume per componente: Q1 / mediana / Q3 (mm³) |', '| --- | --- | --- | --- | --- | --- |']
    for nome, r in R.items():
        md.append(f'| {nome} | {r["casi"]} | {q(r["quartili_caso"])} | {num(r["min"])} – {num(r["max"])} | '
                  f'{r["componenti"]} | {q(r["quartili_componente"])} |')
    OUT.with_suffix('.md').write_text('\n'.join(md) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps(R, indent=1, default=float))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
