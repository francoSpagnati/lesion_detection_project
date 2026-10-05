#!/usr/bin/env python
"""Costruisce Dataset532_MISTO_AINCP: i pazienti AInCP di addestramento insieme ad ATLAS e WMH.

    python scripts/make_misto_dataset.py          # raw e preprocessed, tutto per collegamento

PERCHE'. Il pretraining seguito dal fine-tuning (ricette A e D di 506) non si distingue
dall'addestramento da zero (Z): nel fine-tuning, a passo 1e-4, i 12 pazienti contano quasi da
soli, e quello che ATLAS e WMH avevano insegnato puo' venire sovrascritto. Qui i dati pubblici
restano presenti per tutto l'addestramento: un modello solo, da zero come Z, che a ogni epoca
vede pazienti e casi pubblici insieme. E' l'unione di 543 (ATLAS + WMH) e 506 (i pazienti),
con una regola di campionamento per sorgente nel trainer (nnUNetTrainerMisto.py), perche' in
proporzione ai casi i pazienti sarebbero 12 su 647.

PERCHE' SI PUO' FARE SENZA PREPROCESSARE DI NUOVO. 506 e 543 hanno lo stesso piano
(nnUNetPlansDa502: 1 mm isotropo, patch 128, z-score per caso su ciascun canale, stessa rete):
verificato configurazione per configurazione il 5/10/2026, le uniche differenze sono il nome
del dataset e le statistiche d'intensita', che nnU-Net usa solo per la TC. Il preprocessing
di ogni caso dipende solo dal caso e dal piano, quindi i file gia' preprocessati di 506 e 543
sono esattamente quelli che nnUNetv2_preprocess produrrebbe qui. Si collegano: zero GB in piu'.
I pazienti stanno nel template pediatrico MNIPediatricAsym, ATLAS e WMH in quello adulto
MNI152NLin2009aSym, entrambi a 1 mm.

LE ETICHETTE. Una classe, `lesione`, compatibile fra le tre sorgenti: ictus cronico (ATLAS),
iperintensita' della bianca (WMH), lesioni dei pazienti (ROI_1 | ROI_2 | ROI_4). ATLAS ha la FLAIR
tutta zero, come in 543.

GLI SPLIT. La validazione di ogni fold e' **solo** quella di 506, cioe' gli stessi pazienti
fuori-fold di Z: il confronto con Z resta appaiato paziente per paziente, e l'arresto si decide
sui pazienti. Tutti i 635 casi pubblici sono sempre in addestramento. I 5 pazienti di test non
sono in 506 preprocessato e non entrano qui in nessun modo.
"""
import json
import os
import shutil
import sys
from pathlib import Path

RAW = Path(os.environ['nnUNet_raw'])
PREP = Path(os.environ['nnUNet_preprocessed'])
NOME = 'Dataset532_MISTO_AINCP'
PAZIENTI, PUBBLICI = 'Dataset506_AINCP_FINALE', 'Dataset543_ATLAS_WMH_MNI'
PIANO = 'nnUNetPlansDa502'
CONF = f'{PIANO}_3d_fullres'


def collega(sorgente: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(sorgente.resolve(), dest)


def piani_compatibili():
    a = json.loads((PREP / PAZIENTI / f'{PIANO}.json').read_text())
    b = json.loads((PREP / PUBBLICI / f'{PIANO}.json').read_text())
    assert a['configurations'] == b['configurations'], 'piani diversi: i file preprocessati non si possono unire'
    for k in a:
        if k not in ('configurations', 'dataset_name', 'foreground_intensity_properties_per_channel'):
            assert a[k] == b[k], k
    for k in ('channel_names', 'labels', 'file_ending'):
        assert (json.loads((PREP / PAZIENTI / 'dataset.json').read_text())[k]
                == json.loads((PREP / PUBBLICI / 'dataset.json').read_text())[k]), k
    return a


def main():
    if (RAW / NOME).exists() or (PREP / NOME).exists():
        sys.exit(f'{NOME} esiste gia\': un dataset non si sovrascrive')
    piano = piani_compatibili()
    casi = {}
    for ds in (PAZIENTI, PUBBLICI):
        for f in sorted((PREP / ds / CONF).glob('*.pkl')):
            c = f.stem
            assert c not in casi, f'nome duplicato fra le sorgenti: {c}'
            casi[c] = ds
    pazienti = sorted(c for c, d in casi.items() if d == PAZIENTI)
    assert len(pazienti) == 12 and all(c.startswith('FSM_') for c in pazienti), pazienti

    # preprocessed: i file di ogni caso, le etichette per la validazione finale, il piano
    for c, ds in casi.items():
        for suf in ('.b2nd', '_seg.b2nd', '.pkl'):
            collega(PREP / ds / CONF / f'{c}{suf}', PREP / NOME / CONF / f'{c}{suf}')
        collega(PREP / ds / 'gt_segmentations' / f'{c}.nii.gz', PREP / NOME / 'gt_segmentations' / f'{c}.nii.gz')
    piano['dataset_name'] = NOME
    (PREP / NOME / f'{PIANO}.json').write_text(json.dumps(piano, indent=4))
    shutil.copy(PREP / PAZIENTI / 'dataset_fingerprint.json', PREP / NOME / 'dataset_fingerprint.json')
    dj = {'name': NOME,
          'description': 'Pazienti AInCP di addestramento (12, template MNIPediatricAsym) + ATLAS con FLAIR vuota e WMH '
                         '(635, MNI152NLin2009aSym), 1 mm, due canali, una classe. Collegati da 506 e 543, '
                         'preparato da scripts/make_misto_dataset.py.',
          'channel_names': {'0': 'FLAIR', '1': 'T1'}, 'labels': {'background': 0, 'lesione': 1},
          'numTraining': len(casi), 'file_ending': '.nii.gz',
          'sorgenti': {'AInCP': len(pazienti), 'ATLAS': sum(c.startswith('atlas_') for c in casi),
                       'WMH': sum(c.startswith('wmh_') for c in casi)}}
    (PREP / NOME / 'dataset.json').write_text(json.dumps(dj, indent=2, ensure_ascii=False))

    # split: la validazione di 506, fold per fold; i pubblici sempre in addestramento
    pubblici = sorted(c for c, d in casi.items() if d == PUBBLICI)
    split_506 = json.loads((PREP / PAZIENTI / 'splits_final.json').read_text())
    split = [{'train': sorted(f['train']) + pubblici, 'val': sorted(f['val'])} for f in split_506]
    for f in split:
        assert not set(f['train']) & set(f['val']) and len(f['train']) + len(f['val']) == len(casi)
    (PREP / NOME / 'splits_final.json').write_text(json.dumps(split, indent=4))

    # raw: solo per completezza e ispezione (il training legge il preprocessed), sempre collegamenti
    for c, ds in casi.items():
        nome_raw = PAZIENTI if ds == PAZIENTI else PUBBLICI
        for i in (0, 1):
            collega(RAW / nome_raw / 'imagesTr' / f'{c}_000{i}.nii.gz', RAW / NOME / 'imagesTr' / f'{c}_000{i}.nii.gz')
        collega(RAW / nome_raw / 'labelsTr' / f'{c}.nii.gz', RAW / NOME / 'labelsTr' / f'{c}.nii.gz')
    (RAW / NOME / 'dataset.json').write_text(json.dumps(dj, indent=2, ensure_ascii=False))
    print(f"{NOME}: {dj['sorgenti']}, split di 506 sui pazienti "
          f"({', '.join(str(len(f['val'])) for f in split)} in validazione)")


if __name__ == '__main__':
    main()
