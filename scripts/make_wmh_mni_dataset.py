#!/usr/bin/env python
"""Costruisce `Dataset542_WMH_MNI`: WMH a due canali portato nello spazio MNI di ATLAS.

    python scripts/make_wmh_mni_dataset.py [--processi 12]   # registrazione, ~15 min su 96 CPU
    python scripts/make_wmh_mni_dataset.py --piano           # piano di 502 + correzioni per canale
    nnUNetv2_preprocess -d 542 -plans_name nnUNetPlansDa502 -c 3d_fullres

PERCHE'. E' lo stadio 2 della Proposta B: la rete espansa, pre-addestrata su ATLAS a un canale,
impara la FLAIR sui 79 casi di WMH prima di arrivare ai pazienti. All'incontro del 23 settembre il
responsabile ha indicato di lavorare in uno spazio standard: ATLAS e' gia' in MNI152NLin2009aSym, e
WMH ci viene portato qui, cosi' i due stadi di pre-addestramento vedono cervelli nello stesso
spazio e con lo stesso orientamento. Il template adulto va bene: i pazienti di WMH sono adulti.

COME. Stessa registrazione dei pazienti AInCP (ANTs SyN: rigida, affine, non lineare), calcolata
sulla T1 e applicata alla FLAIR e all'etichetta. In WMH la T1 e' gia' co-registrata alla FLAIR
dagli organizzatori (verificato in make_wmh_two_channel_dataset.py), quindi una sola
trasformazione vale per tutti e tre. L'etichetta si trasporta al vicino piu' prossimo.

Una variabile rispetto a `Dataset512_WMH_T1_FLAIR`: lo spazio. Stessi casi, stessi canali nello
stesso ordine (0 = FLAIR, 1 = T1, come Dataset504), stesse etichette, stesso split di 512
(copiato, non rigenerato): il Dice di validazione resta confrontabile con 0.753.
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROGETTO = Path(__file__).resolve().parent.parent
SRC = Path(os.environ['nnUNet_raw']) / 'Dataset512_WMH_T1_FLAIR'
DST = Path(os.environ['nnUNet_raw']) / 'Dataset542_WMH_MNI'
PREP = Path(os.environ['nnUNet_preprocessed'])
TEMPLATE = '/data01/atlas_mri_data/ATLAS_2/MNI152NLin2009aSym.nii.gz'


def registra(caso):
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '8'
    import ants
    import numpy as np
    out_l = DST / 'labelsTr' / f'{caso}.nii.gz'
    if out_l.exists():
        return caso, None
    tpl = ants.image_read(TEMPLATE)
    fl = ants.image_read(str(SRC / 'imagesTr' / f'{caso}_0000.nii.gz'))
    t1 = ants.image_read(str(SRC / 'imagesTr' / f'{caso}_0001.nii.gz'))
    lab = ants.image_read(str(SRC / 'labelsTr' / f'{caso}.nii.gz'))
    reg = ants.registration(fixed=tpl, moving=t1, type_of_transform='SyN')
    tr = reg['fwdtransforms']
    ants.image_write(ants.apply_transforms(fixed=tpl, moving=fl, transformlist=tr, interpolator='linear'),
                     str(DST / 'imagesTr' / f'{caso}_0000.nii.gz'))
    ants.image_write(reg['warpedmovout'], str(DST / 'imagesTr' / f'{caso}_0001.nii.gz'))
    lt = ants.apply_transforms(fixed=tpl, moving=lab, transformlist=tr, interpolator='nearestNeighbor')
    ants.image_write(lt.new_image_like(lt.numpy().round().astype('float32')), str(out_l))
    vn, vt = np.prod(lab.spacing) / 1000, np.prod(tpl.spacing) / 1000
    return caso, {'ml_nativo': float((lab.numpy() > 0).sum() * vn), 'ml_mni': float((lt.numpy() > .5).sum() * vt)}


def costruisci(processi):
    for sub in ('imagesTr', 'labelsTr'):
        (DST / sub).mkdir(parents=True, exist_ok=True)
    casi = sorted(p.name[:-7] for p in (SRC / 'labelsTr').glob('*.nii.gz'))
    manif = {}
    with ProcessPoolExecutor(processi) as ex:
        for caso, m in ex.map(registra, casi):
            if m:
                manif[caso] = m
                print(caso, f"{m['ml_nativo']:.2f} -> {m['ml_mni']:.2f} ml", flush=True)
    dj = json.load(open(SRC / 'dataset.json'))
    dj['name'] = 'WMH_MNI'
    dj['description'] = 'WMH a due canali (0 FLAIR, 1 T1) registrato su MNI152NLin2009aSym (lo spazio di ATLAS).'
    json.dump(dj, open(DST / 'dataset.json', 'w'), indent=1)
    vecchio = json.load(open(PROGETTO / 'reports/wmh_mni_manifesto.json')) if (PROGETTO / 'reports/wmh_mni_manifesto.json').exists() else {}
    vecchio.update(manif)
    json.dump(vecchio, open(PROGETTO / 'reports/wmh_mni_manifesto.json', 'w'), indent=1)
    print(len(casi), 'casi')


def piano():
    """Il piano di 502 (1 mm isotropo, patch 128^3) e lo split di 512, con le tre correzioni per
    canale documentate in fondo a make_aincp_dataset.py."""
    d = PREP / 'Dataset542_WMH_MNI'
    if not (d / 'dataset_fingerprint.json').exists():
        sys.exit('prima: nnUNetv2_extract_fingerprint -d 542')
    if not (d / 'nnUNetPlansDa502.json').exists():
        os.system('nnUNetv2_move_plans_between_datasets -s 502 -t 542 -sp nnUNetPlans -tp nnUNetPlansDa502')
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
        shutil.copy(PREP / 'Dataset512_WMH_T1_FLAIR' / 'splits_final.json', d / 'splits_final.json')
    print('piano e split pronti in', d)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--processi', type=int, default=12)
    ap.add_argument('--piano', action='store_true')
    a = ap.parse_args()
    piano() if a.piano else costruisci(a.processi)
