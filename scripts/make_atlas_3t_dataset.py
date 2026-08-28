#!/usr/bin/env python
"""Costruisce `Dataset521_ATLAS_3T`: ATLAS senza le coorti a 1.5 Tesla.

Risponde a: **il modello migliora se lo si addestra solo su dati a 3 T?**

Il campo magnetico non era censito. Lo dichiarano i metadati del dataset
(`20211112_ATLAS_2.0_SupplementaryInfo.csv`, prima riga di ogni colonna): tre
coorti su 33 sono a 1.5 T - `R029`, `R039`, `R040` - per 52 casi sui 655 annotati.

**Come va misurato il confronto, altrimenti non significa niente.** Rimuovendo i
casi a 1.5 T cambia anche l'insieme su cui si valuta, e un Dice piu' alto potrebbe
venire solo dall'aver tolto casi difficili dalla validation invece che da un
training migliore. Per questo i fold **non vengono rigenerati**: si copiano quelli
di `Dataset502_ATLAS` togliendo i casi a 1.5 T. Cosi' la validation del fold 0 e'
esattamente il sottoinsieme 3 T di quella della baseline, e il confronto si fa sugli
**stessi identici casi**, uno per uno.

Uso:
    python scripts/make_atlas_3t_dataset.py
"""
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

PROGETTO = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get('nnUNet_raw', PROGETTO / 'nnUNet' / 'nnUNet_raw'))
PREP = Path(os.environ.get('nnUNet_preprocessed', PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'))
SORGENTE, DESTINAZIONE = 'Dataset502_ATLAS', 'Dataset521_ATLAS_3T'
META = '/data01/atlas_mri_data/ATLAS_2/20211112_ATLAS_2.0_SupplementaryInfo.csv'


def coorti_per_campo():
    """Dai metadati ufficiali: {coorte minuscola -> '3T' | '1.5T'}."""
    c = pd.read_csv(META)
    out = {}
    for coorte in c.columns:
        m = re.search(r'(\d+(?:\.\d+)?)\s*T\b', str(c[coorte].iloc[0]))
        out[coorte.lower()] = (m.group(1) + 'T') if m else '?'
    return out


def coorte_di(caso):
    m = re.match(r'^sub-(r\d+)s\d+$', caso)
    return m.group(1) if m else None


def main():
    campo = coorti_per_campo()
    esclusi = {k for k, v in campo.items() if v != '3T'}
    print(f'coorti escluse ({len(esclusi)}): {sorted(esclusi)}')

    src, dst = RAW / SORGENTE, RAW / DESTINAZIONE
    if dst.exists():
        shutil.rmtree(dst)
    tenuti, scartati = [], []
    for sub in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        if not (src / sub).exists():
            continue
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for f in sorted((src / sub).iterdir()):
            caso = f.name.split('.')[0].replace('_0000', '')
            if coorte_di(caso) in esclusi:
                if sub == 'imagesTr':
                    scartati.append(caso)
                continue
            # symlink al symlink originale: nessun dato duplicato
            os.symlink(os.path.realpath(f), dst / sub / f.name)
            if sub == 'imagesTr':
                tenuti.append(caso)

    dj = json.loads((src / 'dataset.json').read_text())
    dj['numTraining'] = len(tenuti)
    dj['name'] = DESTINAZIONE
    dj['description'] = ('ATLAS 2.0 senza le coorti a 1.5 T (R029, R039, R040). '
                         'Variante di Dataset502_ATLAS: cambia solo quali casi sono presenti.')
    (dst / 'dataset.json').write_text(json.dumps(dj, indent=2))

    # --- fold copiati, non rigenerati
    splits = json.loads((PREP / SORGENTE / 'splits_final.json').read_text())
    nuovi = [{k: [c for c in fold[k] if coorte_di(c) not in esclusi] for k in ('train', 'val')}
             for fold in splits]
    (PREP / DESTINAZIONE).mkdir(parents=True, exist_ok=True)
    (PREP / DESTINAZIONE / 'splits_final.json').write_text(json.dumps(nuovi, indent=2))

    print(f'casi di training: {len(tenuti)} tenuti, {len(scartati)} scartati')
    for i, (v, n) in enumerate(zip(splits, nuovi)):
        print(f"  fold {i}: train {len(v['train'])} -> {len(n['train'])}, "
              f"val {len(v['val'])} -> {len(n['val'])}")
    print(f'\n-> {dst}\n-> {PREP / DESTINAZIONE / "splits_final.json"}')


if __name__ == '__main__':
    main()
