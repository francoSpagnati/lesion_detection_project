#!/usr/bin/env python
"""Costruisce Dataset543_ATLAS_WMH_MNI: gli stadi 1 e 2 fusi in un solo addestramento.

    python scripts/make_union_atlas_wmh_mni.py
    nnUNetv2_extract_fingerprint -d 543
    python scripts/make_union_atlas_wmh_mni.py --piano
    nnUNetv2_preprocess -d 543 -plans_name nnUNetPlansDa502 -c 3d_fullres

PERCHE'. Lo stadio 2 in sequenza (542: WMH dai pesi di ATLAS) ha fatto dimenticare le cavita':
con la FLAIR vuota il Dice sulla validazione di ATLAS scende da 0.610 a 0.217
(reports/oblio_atlas_stadio2.md). Le cavita' sono proprio il sapere che ATLAS porta ai pazienti
(encefalomalacia e lesioni cistiche). Addestrando ATLAS e WMH **insieme** la rete non puo'
dimenticare nessuno dei due, perche' li rivede entrambi a ogni epoca.

IL CANALE MANCANTE. ATLAS non ha FLAIR, e nnU-Net vuole tutti i canali per tutti i casi: per
questo il training congiunto era stato scartato (docs/obiettivo-e-piano.md). Con il modality
dropout l'ostacolo cade: ai casi di ATLAS si da' una FLAIR tutta zero, che dopo la
normalizzazione z-score e' identica a cio' che la rete vede quando il dropout spegne la FLAIR di
un caso WMH. La rete impara cosi' due regimi espliciti -- T1 sola, T1 piu' FLAIR -- ed e' quello
che serve sui pazienti, dove una delle due sequenze puo' mancare (i 17 a 1.5 T).

LE ETICHETTE. Una classe sola, `lesione`: la lesione ischemica di ATLAS e l'iperintensita' di WMH
sono entrambe "tessuto patologico" e non si sovrappongono mai nello stesso caso. Nessun conflitto
di semantica come quello con i tessuti di DBB che aveva frenato 531.

GRIGLIA E SPLIT. Tutti e due sono gia' nella griglia di MNI152NLin2009aSym a 1 mm (ATLAS
nativamente, WMH da make_wmh_mni_dataset.py). I casi sono collegati, non copiati; i nomi hanno il
prefisso della sorgente. I fold sono l'unione dei fold k delle due sorgenti: la validazione del
fold 0 contiene i 112 casi di ATLAS e i 16 di WMH gia' usati per 502 e 542, e il confronto con
quei modelli resta appaiato. I test set delle due sorgenti restano fuori.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

RAW = Path(os.environ['nnUNet_raw'])
PREP = Path(os.environ['nnUNet_preprocessed'])
NOME = 'Dataset543_ATLAS_WMH_MNI'
DST = RAW / NOME
ATLAS, WMH = 'Dataset502_ATLAS', 'Dataset542_WMH_MNI'


def costruisci():
    if DST.exists():
        sys.exit(f'{DST} esiste gia\': un dataset non si sovrascrive')
    (DST / 'imagesTr').mkdir(parents=True)
    (DST / 'labelsTr').mkdir()
    n = 0
    for f in sorted((RAW / ATLAS / 'labelsTr').glob('*.nii.gz')):
        c = f.name[:-7]
        t1 = RAW / ATLAS / 'imagesTr' / f'{c}_0000.nii.gz'
        img = nib.load(str(t1))
        nib.save(nib.Nifti1Image(np.zeros(img.shape, np.uint8), img.affine),
                 str(DST / 'imagesTr' / f'atlas_{c}_0000.nii.gz'))
        (DST / 'imagesTr' / f'atlas_{c}_0001.nii.gz').symlink_to(t1)
        (DST / 'labelsTr' / f'atlas_{c}.nii.gz').symlink_to(f)
        n += 1
    for f in sorted((RAW / WMH / 'labelsTr').glob('*.nii.gz')):
        c = f.name[:-7]
        for k in ('0000', '0001'):
            (DST / 'imagesTr' / f'wmh_{c}_{k}.nii.gz').symlink_to(RAW / WMH / 'imagesTr' / f'{c}_{k}.nii.gz')
        (DST / 'labelsTr' / f'wmh_{c}.nii.gz').symlink_to(f)
        n += 1
    (DST / 'dataset.json').write_text(json.dumps({
        'name': NOME,
        'description': ('ATLAS (FLAIR vuota) + WMH in MNI152NLin2009aSym a 1 mm, due canali, una classe. '
                        'Preparato da scripts/make_union_atlas_wmh_mni.py.'),
        'channel_names': {'0': 'FLAIR', '1': 'T1'},
        'labels': {'background': 0, 'lesione': 1},
        'numTraining': n,
        'file_ending': '.nii.gz',
    }, indent=2))
    print(n, 'casi in', DST)


def piano():
    d = PREP / NOME
    if not (d / 'dataset_fingerprint.json').exists():
        sys.exit('prima: nnUNetv2_extract_fingerprint -d 543')
    if not (d / 'nnUNetPlansDa502.json').exists():
        os.system('nnUNetv2_move_plans_between_datasets -s 502 -t 543 -sp nnUNetPlans -tp nnUNetPlansDa502')
    p = json.load(open(d / 'nnUNetPlansDa502.json'))
    fp = json.load(open(d / 'dataset_fingerprint.json'))
    for c in p['configurations'].values():
        if 'normalization_schemes' in c:
            c['normalization_schemes'] = ['ZScoreNormalization'] * 2
            c['use_mask_for_norm'] = [False] * 2
    p['foreground_intensity_properties_per_channel'] = fp['foreground_intensity_properties_per_channel']
    json.dump(p, open(d / 'nnUNetPlansDa502.json', 'w'), indent=2)
    shutil.copy(DST / 'dataset.json', d / 'dataset.json')
    if not (d / 'splits_final.json').exists():
        sa = json.load(open(PREP / ATLAS / 'splits_final.json'))
        sw = json.load(open(PREP / WMH / 'splits_final.json'))
        split = [{k: [f'atlas_{c}' for c in a[k]] + [f'wmh_{c}' for c in w[k]] for k in ('train', 'val')}
                 for a, w in zip(sa, sw)]
        (d / 'splits_final.json').write_text(json.dumps(split, indent=2))
    print('piano e split pronti in', d)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--piano', action='store_true')
    piano() if ap.parse_args().piano else costruisci()
