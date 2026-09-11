#!/usr/bin/env python
"""Costruisce Dataset504_AINCP: i dieci pazienti nello spazio T1 nativo, due canali.

    python scripts/make_aincp_dataset.py

Parte dall'output di scripts/preprocess_aincp.py e prende, per ogni paziente:
    canale 0  FLAIR_inT1   la FLAIR corretta con N4 e registrata rigidamente sulla T1
    canale 1  T1_N4        la T1 corretta con N4, nella sua griglia originale
    label     lesione_inT1 ROI_1 + ROI_2 + ROI_4 trasportate sulla griglia della T1

Lo spazio e' quello nativo della T1 e non MNI: il passaggio al template ingrandisce le
lesioni di un fattore fra 1.6 e 2.5 perche' il template e' adulto (reports/aincp_scala_mni.json),
e nnU-Net normalizza la spaziatura da solo, quindi i pesi di ATLAS non richiedono MNI.

L'ordine dei canali e' quello di Dataset512 ({0: FLAIR, 1: T1}): il trainer di espansione
abbina i canali per nome, e cosi' un modello passato per lo stadio 2 su WMH trova gli stessi
nomi nello stesso ordine.

Scrive anche lo split a 5 fold, per paziente, deciso a mano e una volta sola:
    - ogni fold di validazione ha un paziente con molta lesione e uno con poca;
    - i tre pazienti con FLAIR a fette da 1.0 mm (039, 054, 056) stanno su fold diversi,
      perche' sono un'acquisizione diversa dagli altri sette (0.8 mm).
Lo split va nella cartella preprocessata, che deve gia' esistere: si lancia quindi dopo
nnUNetv2_preprocess, oppure si rilancia con --solo-split.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
PROC = PROGETTO / 'data/processed/AINCP'
NOME = 'Dataset504_AINCP'
RAW = Path(os.environ['nnUNet_raw']) / NOME
PRE = Path(os.environ['nnUNet_preprocessed']) / NOME

# (validazione) per fold. Volumi di lesione in ml dal notebook, per giustificare l'accoppiamento:
# 032 123.5 | 040 55.7 | 054 33.8 | 056 12.5 | 045 9.5 || 052 4.9 | 041 2.0 | 039 1.4 | 053 0.6 | 046 0.5
FOLD = [
    ['FSM_032_MR_1_anon', 'FSM_039_MR_2_anon'],   # 039: fette da 1.0 mm
    ['FSM_054_MR_1_anon', 'FSM_046_MR_1_anon'],   # 054: fette da 1.0 mm
    ['FSM_056_MR_1_anon', 'FSM_053_MR_1_anon'],   # 056: fette da 1.0 mm
    ['FSM_040_MR_1_anon', 'FSM_041_MR_1_anon'],
    ['FSM_045_MR_1_anon', 'FSM_052_MR_1_anon'],
]


def caso(s: str) -> str:
    """FSM_032_MR_1_anon -> FSM_032. Il paziente e' l'unita', la sessione e' una sola."""
    return '_'.join(s.split('_')[:2])


def scrivi_split():
    if not PRE.exists():
        sys.exit(f'{PRE} non esiste: lanciare prima nnUNetv2_preprocess')
    dest = PRE / 'splits_final.json'
    if dest.exists():
        sys.exit(f'{dest} esiste gia\' e non si rigenera: cancellarlo a mano se e\' davvero voluto')
    tutti = sorted(caso(s) for f in FOLD for s in f)
    assert len(tutti) == len(set(tutti)) == 10
    split = [{'train': [c for c in tutti if c not in map(caso, val)],
              'val': sorted(map(caso, val))} for val in FOLD]
    dest.write_text(json.dumps(split, indent=2))
    print('split scritto:', dest)
    for i, s in enumerate(split):
        print(f"  fold {i}: val {s['val']}")


def costruisci():
    if RAW.exists():
        sys.exit(f'{RAW} esiste gia\': un dataset non si sovrascrive, si crea con un ID nuovo')
    (RAW / 'imagesTr').mkdir(parents=True)
    (RAW / 'labelsTr').mkdir()
    soggetti = sorted(p.name for p in PROC.iterdir() if (p / 'manifesto.json').exists())
    assert len(soggetti) == 10, soggetti
    for s in soggetti:
        c = PROC / s
        t1, fl, les = (nib.load(str(c / f)) for f in ('T1_N4.nii.gz', 'FLAIR_inT1.nii.gz',
                                                       'lesione_inT1.nii.gz'))
        # tutto deve stare sulla griglia della T1, altrimenti il canale 0 e la label non
        # corrispondono voxel per voxel a quello che la rete vede nel canale 1
        for img in (fl, les):
            assert img.shape == t1.shape and np.allclose(img.affine, t1.affine, atol=1e-3), s
        lab = np.asanyarray(les.dataobj)
        assert set(np.unique(lab)) <= {0, 1}, s
        shutil.copy(c / 'FLAIR_inT1.nii.gz', RAW / 'imagesTr' / f'{caso(s)}_0000.nii.gz')
        shutil.copy(c / 'T1_N4.nii.gz', RAW / 'imagesTr' / f'{caso(s)}_0001.nii.gz')
        nib.save(nib.Nifti1Image(lab.astype(np.uint8), les.affine), RAW / 'labelsTr' / f'{caso(s)}.nii.gz')
        print(f'{caso(s)}: {t1.shape} {tuple(round(float(z), 3) for z in t1.header.get_zooms())}')
    (RAW / 'dataset.json').write_text(json.dumps({
        'name': NOME,
        'description': ('AInCP primo lotto, 10 pazienti Stella Maris, spazio T1 nativo. '
                        'FLAIR N4 registrata rigidamente sulla T1 N4; label = ROI_1+ROI_2+ROI_4 '
                        '(ventricolo ROI_3 escluso). Preparato da scripts/make_aincp_dataset.py '
                        'dall\'output di scripts/preprocess_aincp.py.'),
        'channel_names': {'0': 'FLAIR', '1': 'T1'},
        'labels': {'background': 0, 'lesione': 1},
        'numTraining': len(soggetti),
        'file_ending': '.nii.gz',
        'dataset_preparation': {
            'spazio': 'T1 nativo del paziente (non MNI: vedi reports/aincp_scala_mni.json)',
            'split': '5 fold per paziente, fissati a mano in make_aincp_dataset.py',
            'soglia_minima_lesione_mm3': 'nessuna sulle label; 10 mm3 nelle metriche per lesione',
        },
    }, indent=2))
    print('scritto', RAW)


if __name__ == '__main__':
    if '--solo-split' in sys.argv:
        scrivi_split()
    else:
        costruisci()
        if PRE.exists():
            scrivi_split()


# --- nota sul piano trasferito da 502 ---------------------------------------
#
# Dopo `nnUNetv2_move_plans_between_datasets -s 502 -t 504 -sp nnUNetPlans -tp nnUNetPlansDa502`
# il piano copiato porta con se' `normalization_schemes` e `use_mask_for_norm` con UNA voce,
# perche' 502 ha un canale. Il preprocessore li indicizza per canale
# (default_preprocessor.py: `normalization_schemes[c]`), quindi sul secondo canale andrebbe in
# IndexError. Vanno portati a due voci a mano, cosa che il comando di nnU-Net non fa:
#
#   for c in plans['configurations'].values():
#       c['normalization_schemes'] = ['ZScoreNormalization'] * 2
#       c['use_mask_for_norm'] = [False] * 2
#
# Lo stesso vale per Dataset512 quando riceve il piano di 502 per lo stadio 2.
#
# Stessa cosa per `foreground_intensity_properties_per_channel`: il comando copia quello di 502
# (un canale) invece di prenderlo dal fingerprint del dataset di destinazione, e il preprocessore
# lo indicizza per canale anche quando la normalizzazione (z-score) non lo usa -> KeyError '1'.
# Si sostituisce con la voce omonima di dataset_fingerprint.json del dataset di destinazione.
