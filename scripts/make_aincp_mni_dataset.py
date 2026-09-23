#!/usr/bin/env python
"""Costruisce Dataset505_AINCP_MNI: i dieci pazienti nel template pediatrico, due canali.

    python scripts/make_aincp_mni_dataset.py            # immagini ed etichette in nnUNet_raw
    nnUNetv2_extract_fingerprint -d 505
    python scripts/make_aincp_mni_dataset.py --piano    # piano di 502 + split di 504
    nnUNetv2_preprocess -d 505 -plans_name nnUNetPlansDa502 -c 3d_fullres

PERCHE' IN MNI. Deciso all'incontro del 23/9: si lavora nello spazio del template e, per un
paziente nuovo, si torna al nativo con la trasformazione inversa. La proporzione di cervello
lesionato cambia poco nel passaggio (mediana -10%, reports/confronto_template_aincp.md).

PERCHE' IL TEMPLATE PEDIATRICO. MNIPediatricAsym cohort-1 (4.5-18.5 anni) allinea i cervelli dei
bambini meglio di quello adulto (Dice del cervello 0.95-0.98 contro 0.82-0.93), e ha **la stessa
griglia** di ATLAS e di Dataset542 (197x233x189 a 1 mm, stessa origine): i pesi degli stadi 1 e 2
vedono i pazienti esattamente nello spazio in cui sono stati addestrati. Le eta' del foglio Excel
(docs/AINCP_SERIES_TYPE_dataset_FSM_v1.xlsx) vanno da 3 a 15 anni.

UNA SOLA INTERPOLAZIONE. La FLAIR e l'etichetta partono dalla griglia originale della FLAIR e
arrivano al template componendo FLAIR->T1 (rigida) e T1->template (SyN) in una sola
applicazione, invece di passare per FLAIR_inT1: ogni ricampionamento in piu' smussa l'immagine e
assottiglia le lesioni piccole. Le trasformazioni T1->template sono quelle gia' calcolate da
scripts/confronto_template_aincp.py (transforms_ped/), la rigida quella di
scripts/preprocess_aincp.py (transforms/FLAIR_to_T1_0.mat).

L'ETICHETTA. ROI_1 | ROI_2 | ROI_4 nello spazio FLAIR originale, **meno le componenti fino a
1 mm3** (26-connessione): sono i 22 residui del tracciato elencati in
reports/residui_annotazione.md, che il responsabile ha indicato come rimasugli. Si tolgono prima
del trasporto perche' nel template un voxel da 0.2 mm3 diventerebbe una macchia indistinguibile
da una lesione vera. Il trasporto e' lineare sulla maschera con soglia 0.5, lo stesso criterio
del ricampionamento one-hot di nnU-Net, invece del vicino piu' prossimo, che su un passaggio da
0.2 a 1 mm3 per voxel decide ogni voxel sul campione di un solo punto.

Lo split e' quello di 504 (stessi pazienti negli stessi fold), cosi' 505 e 504 si confrontano
appaiati paziente per paziente.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import ants
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_aincp_dataset import caso  # noqa: E402
from preprocess_aincp import ROI_LESIONE, elenca_soggetti  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
PROC = PROGETTO / 'data/processed/AINCP'
TEMPLATE = PROGETTO / 'data/templates/tpl-MNIPediatricAsym_cohort-1_res-1_T1w.nii.gz'
NOME = 'Dataset505_AINCP_MNI'
RAW = Path(os.environ['nnUNet_raw']) / NOME
PREP = Path(os.environ['nnUNet_preprocessed']) / NOME
SOGLIA_MM3 = 1.0
STRUTTURA = np.ones((3, 3, 3), dtype=bool)


def trasformazioni(s: Path) -> list[str]:
    """FLAIR -> template. ANTs applica la lista dall'ultima alla prima: prima la rigida."""
    t = s / 'transforms_ped'
    return [str(t / 'T1_to_PED_fwd_0.nii.gz'), str(t / 'T1_to_PED_fwd_1.mat'),
            str(s / 'transforms' / 'FLAIR_to_T1_0.mat')]


def etichetta_pulita(roi: dict, flair: ants.ANTsImage) -> tuple[ants.ANTsImage, dict]:
    m = np.zeros(flair.shape, bool)
    for r in ROI_LESIONE:
        if r in roi:
            m |= ants.image_read(str(roi[r])).numpy() > 0
    mm3 = float(np.prod(flair.spacing))
    lab, n = ndimage.label(m, structure=STRUTTURA)
    vol = np.bincount(lab.ravel())[1:] * mm3
    piccole = np.flatnonzero(vol <= SOGLIA_MM3) + 1
    m[np.isin(lab, piccole)] = False
    info = {'componenti': int(n), 'tolte_fino_a_1mm3': int(len(piccole)),
            'ml_prima': round(float(vol.sum()) / 1000, 3), 'ml_dopo': round(float(m.sum()) * mm3 / 1000, 3)}
    return flair.new_image_like(m.astype('float32')), info


def elabora(s: Path, roi: dict, tpl: ants.ANTsImage) -> dict:
    tr = trasformazioni(s)
    fl = ants.image_read(str(s / 'FLAIR_N4.nii.gz'))
    t1 = ants.image_read(str(s / 'T1_N4.nii.gz'))
    lez, info = etichetta_pulita(roi, fl)
    t1_t = ants.apply_transforms(fixed=tpl, moving=t1, transformlist=tr[:2], interpolator='linear')
    fl_t = ants.apply_transforms(fixed=tpl, moving=fl, transformlist=tr, interpolator='linear')
    lez_t = ants.apply_transforms(fixed=tpl, moving=lez, transformlist=tr, interpolator='linear')
    lez_t = lez_t.new_image_like((lez_t.numpy() >= 0.5).astype('uint8'))
    c = caso(s.name)
    ants.image_write(fl_t, str(RAW / 'imagesTr' / f'{c}_0000.nii.gz'))
    ants.image_write(t1_t, str(RAW / 'imagesTr' / f'{c}_0001.nii.gz'))
    ants.image_write(lez_t, str(RAW / 'labelsTr' / f'{c}.nii.gz'))
    # Quanto sopravvive al trasporto: componenti e volume nel template (1 mm3 per voxel)
    _, n_t = ndimage.label(lez_t.numpy() > 0, structure=STRUTTURA)
    info.update({'componenti_nel_template': int(n_t), 'ml_nel_template': round(float(lez_t.numpy().sum()) / 1000, 3)})
    return info


def costruisci():
    if RAW.exists():
        sys.exit(f'{RAW} esiste gia\': un dataset non si sovrascrive, si crea con un ID nuovo')
    (RAW / 'imagesTr').mkdir(parents=True)
    (RAW / 'labelsTr').mkdir()
    tpl = ants.image_read(str(TEMPLATE))
    soggetti = sorted(p for p in PROC.iterdir() if (p / 'manifesto.json').exists())
    assert len(soggetti) == 10
    # le ROI si rileggono dai sorgenti in /data01 (sola lettura), con la stessa ricerca del preprocessing
    sorgenti = elenca_soggetti()
    manif = {}
    for s in soggetti:
        manif[caso(s.name)] = elabora(s, sorgenti[s.name]['roi'], tpl)
        print(caso(s.name), manif[caso(s.name)], flush=True)
    (RAW / 'dataset.json').write_text(json.dumps({
        'name': NOME,
        'description': ('AInCP primo lotto, 10 pazienti Stella Maris 3 T, nel template MNIPediatricAsym '
                        'cohort-1 (stessa griglia di ATLAS). Canale 0 FLAIR, 1 T1, dopo N4. Label = '
                        'ROI_1+ROI_2+ROI_4 senza le componenti fino a 1 mm3. '
                        'Preparato da scripts/make_aincp_mni_dataset.py.'),
        'channel_names': {'0': 'FLAIR', '1': 'T1'},
        'labels': {'background': 0, 'lesione': 1},
        'numTraining': len(soggetti),
        'file_ending': '.nii.gz',
    }, indent=2))
    (PROGETTO / 'reports' / 'aincp_mni_manifesto.json').write_text(json.dumps(manif, indent=1))


def piano():
    """Piano di 502 con le tre correzioni per canale (vedi fondo di make_aincp_dataset.py) e split di 504."""
    if not (PREP / 'dataset_fingerprint.json').exists():
        sys.exit('prima: nnUNetv2_extract_fingerprint -d 505')
    if not (PREP / 'nnUNetPlansDa502.json').exists():
        os.system('nnUNetv2_move_plans_between_datasets -s 502 -t 505 -sp nnUNetPlans -tp nnUNetPlansDa502')
    p = json.load(open(PREP / 'nnUNetPlansDa502.json'))
    fp = json.load(open(PREP / 'dataset_fingerprint.json'))
    for c in p['configurations'].values():
        if 'normalization_schemes' in c:
            c['normalization_schemes'] = ['ZScoreNormalization'] * 2
            c['use_mask_for_norm'] = [False] * 2
    p['foreground_intensity_properties_per_channel'] = fp['foreground_intensity_properties_per_channel']
    json.dump(p, open(PREP / 'nnUNetPlansDa502.json', 'w'), indent=2)
    shutil.copy(RAW / 'dataset.json', PREP / 'dataset.json')
    dest = PREP / 'splits_final.json'
    if not dest.exists():
        shutil.copy(Path(os.environ['nnUNet_preprocessed']) / 'Dataset504_AINCP' / 'splits_final.json', dest)
    print('piano e split pronti in', PREP)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--piano', action='store_true')
    piano() if ap.parse_args().piano else costruisci()
