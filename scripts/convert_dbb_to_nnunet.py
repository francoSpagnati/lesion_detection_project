#!/usr/bin/env python3
"""Converte il Distorted Brain Benchmark in formato nnU-Net (`Dataset503_DBB`).

Compito: segmentazione di **6 tessuti** (non lesioni) su T1 pediatrici, con lo
split ufficiale del benchmark: si addestra su cervelli sani e si valuta su
cervelli malformati. Il test set resta quindi fuori dalla cross-validation.

Le immagini sono esposte come link simbolici ai file scaricati in `data/raw/DBB/`;
le label (parcellazioni con valori 1..6 gia' contigui) non richiedono riscrittura e
sono anch'esse link simbolici. Il dataset convertito occupa qualche centinaio di KB.

Due dettagli non ovvi, entrambi conseguenza dell'analisi in `reports/dbb_analysis.md`:

- 31 casi su 244 non sono in spazio MNI e 12 di questi sono orientati RAS invece che
  LAS. nnU-Net legge gli array cosi' come sono: mescolare orientamenti significa dare
  in pasto alla rete volumi specchiati fra loro. Si imposta quindi
  `overwrite_image_reader_writer: NibabelIOWithReorient`, che riorienta al volo
  immagini e label in modo coerente, senza riscrivere i dati su disco.
- il fold 0 riproduce **esattamente** lo split ufficiale del benchmark (184 train /
  23 validation), cosi' un run a fold singolo e' confrontabile con i risultati
  pubblicati. I fold 1-4 sono partizioni aggiuntive degli stessi 207 casi sani,
  per una stima piu' stabile in cross-validation.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DBB = ROOT / 'data' / 'raw' / 'DBB'
MANIFEST = ROOT / 'data' / 'processed' / 'manifests' / 'DBB_cases.csv'

CLASSI = {
    'background': 0,
    'csf': 1,
    'gray_matter': 2,
    'white_matter': 3,
    'deep_gray_matter': 4,
    'brain_stem': 5,
    'cerebellum': 6,
}


def percorsi_per_soggetto():
    """subject -> {slug: path del file NIfTI}, dal manifest del download."""
    per_sub = {}
    with open(DBB / 'manifest.csv') as fh:
        for r in csv.DictReader(fh):
            d = DBB / r['path']
            nii = sorted(p for p in d.glob('*.nii*'))
            if nii:
                per_sub.setdefault(r['subject'], {})[r['datatype_slug']] = nii[0]
    return per_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', type=int, default=503)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--n-folds', type=int, default=5)
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f'Manca {MANIFEST}: eseguire prima scripts/analyze_dbb.py')
    with open(MANIFEST) as fh:
        casi = list(csv.DictReader(fh))

    per_sub = percorsi_per_soggetto()
    nome = f'Dataset{args.id}_DBB'
    dst = ROOT / 'data' / 'processed' / 'nnUNet_raw' / nome
    if dst.exists():
        shutil.rmtree(dst)
    for sub in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        (dst / sub).mkdir(parents=True)

    sviluppo, test = [], []
    for c in casi:
        (test if c['source_split'] == 'test' else sviluppo).append(c)

    def collega(caso, cartella_img, cartella_lab):
        subject = caso['case_id'].replace('sub-', '')
        f = per_sub[subject]
        (dst / cartella_img / f"{caso['case_id']}_0000.nii.gz").symlink_to(f['anat-t1w'].resolve())
        (dst / cartella_lab / f"{caso['case_id']}.nii.gz").symlink_to(f['parcellation-volume'].resolve())

    for c in sviluppo:
        collega(c, 'imagesTr', 'labelsTr')
    for c in test:
        collega(c, 'imagesTs', 'labelsTs')

    # --- fold 0 = split ufficiale, fold 1..n = partizioni aggiuntive -------------
    ufficiale_val = sorted(c['case_id'] for c in sviluppo if c['source_split'] == 'validation')
    ufficiale_train = sorted(c['case_id'] for c in sviluppo if c['source_split'] == 'train')
    splits = [{'train': ufficiale_train, 'val': ufficiale_val}]

    # stratificazione sulla conformita' allo spazio MNI: e' l'eterogeneita' reale
    # del training set (213 casi MNI contro 31 in spazio nativo)
    def conforme(c):
        return f"{c['nx']}x{c['ny']}x{c['nz']}" == '182x218x182'

    rng = np.random.default_rng(args.seed)
    strati = {True: [], False: []}
    for c in sviluppo:
        strati[conforme(c)].append(c['case_id'])
    assegnazione = {}
    for strato, ids in strati.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        for i, case_id in enumerate(ids):
            assegnazione[case_id] = i % args.n_folds
    tutti = sorted(c['case_id'] for c in sviluppo)
    for k in range(1, args.n_folds):
        val = sorted(c for c in tutti if assegnazione[c] == k)
        splits.append({'train': [c for c in tutti if c not in set(val)], 'val': val})

    pre = ROOT / 'nnUNet' / 'nnUNet_preprocessed' / nome
    pre.mkdir(parents=True, exist_ok=True)
    (pre / 'splits_final.json').write_text(json.dumps(splits, indent=2) + '\n')

    (dst / 'dataset.json').write_text(json.dumps({
        'name': nome,
        'description': ('Distorted Brain Benchmark: segmentazione di 6 tessuti su T1 pediatrici. '
                        'Training su cervelli sani, test su cervelli malformati. '
                        'Preparato da scripts/convert_dbb_to_nnunet.py'),
        'channel_names': {'0': 'T1'},
        'labels': CLASSI,
        'numTraining': len(sviluppo),
        'file_ending': '.nii.gz',
        # 31 casi su 244 non sono in MNI e 12 sono RAS invece che LAS: senza
        # riorientamento la rete vedrebbe volumi specchiati fra loro
        'overwrite_image_reader_writer': 'NibabelIOWithReorient',
        'dataset_preparation': {
            'seed': args.seed,
            'n_folds': args.n_folds,
            'fold_0': 'split ufficiale del benchmark (184 train / 23 validation)',
            'fold_1_4': 'partizioni aggiuntive, stratificate su conformita\' allo spazio MNI',
            'test_set': 'i 37 casi patologici del benchmark, esclusi dalla cross-validation',
        },
    }, indent=2) + '\n')

    link = ROOT / 'nnUNet' / 'nnUNet_raw' / nome
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(dst)

    from collections import Counter
    print(f'{nome}')
    print(f'  sviluppo (imagesTr): {len(sviluppo)}  |  test (imagesTs): {len(test)}')
    print(f'  test per patologia: {dict(Counter(c["condition"] for c in test))}')
    print(f'  fold (train/val): ' + ', '.join(f"{len(s['train'])}/{len(s['val'])}" for s in splits))
    print(f'  splits_final.json -> {pre / "splits_final.json"}')
    print(f'  esposto come {link}')


if __name__ == '__main__':
    main()
