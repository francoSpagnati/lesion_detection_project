#!/usr/bin/env python3
"""Costruisce `Dataset511_WMH_T1`: WMH con il solo canale T1 al posto della FLAIR.

Serve a rispondere a una domanda precisa: **quanto della prestazione su WMH viene
dalla FLAIR?** Il dataset e' identico a `Dataset501_WMH` in tutto - stessi casi,
stesse label (stessa politica sulla label 2), stessi fold - tranne il canale di
input. Una sola variabile cambia, quindi il confronto 501 contro 511 e'
attribuibile.

Nota clinica: le iperintensita' della sostanza bianca sono **definite** sulla
FLAIR, dove risaltano; sul T1 sono molto meno visibili e alcune sono di fatto
invisibili. Un Dice sensibilmente piu' basso e' il risultato atteso, non un bug.
Questo dataset e' un esperimento di ablazione, non il modello di produzione.

Il T1 usato e' `*_T1w_BiasCorr_2FLAIR_Aligned.nii.gz`, cioe' il T1 gia'
co-registrato nello spazio della FLAIR dagli organizzatori: condivide quindi la
geometria delle label. La verifica e' nel manifest (`ch1_geom_ok`), True su 99/99.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'processed' / 'nnUNet_raw'
PRE = ROOT / 'nnUNet' / 'nnUNet_preprocessed'
SORGENTE = Path('/data01/miccai_mri_data/miccai_nifti')
SUFFISSO_T1 = '_T1w_BiasCorr_2FLAIR_Aligned.nii.gz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', type=int, default=511)
    args = ap.parse_args()

    src = RAW / 'Dataset501_WMH'
    if not src.exists():
        raise SystemExit(f'Manca {src}: eseguire prima notebooks/dataset_preparation.ipynb')
    src_json = json.loads((src / 'dataset.json').read_text())

    nome = f'Dataset{args.id}_WMH_T1'
    dst = RAW / nome
    if dst.exists():
        shutil.rmtree(dst)

    mancanti = []
    conteggi = {}
    for sub_img, sub_lab in (('imagesTr', 'labelsTr'), ('imagesTs', 'labelsTs')):
        if not (src / sub_lab).exists():
            continue
        (dst / sub_img).mkdir(parents=True)
        (dst / sub_lab).mkdir(parents=True)
        casi = sorted(p.name[:-7] for p in (src / sub_lab).glob('*.nii.gz'))
        for caso in casi:
            t1 = SORGENTE / caso / f'{caso}{SUFFISSO_T1}'
            if not t1.exists():
                mancanti.append(caso)
                continue
            (dst / sub_img / f'{caso}_0000.nii.gz').symlink_to(t1)
            # le label sono quelle gia' prodotte per 501: stessa politica sulla label 2
            (dst / sub_lab / f'{caso}.nii.gz').symlink_to((src / sub_lab / f'{caso}.nii.gz').resolve())
        conteggi[sub_img] = len(casi) - len([c for c in mancanti if c in casi])

    if mancanti:
        raise SystemExit(f'T1 allineato mancante per {len(mancanti)} casi: {mancanti[:5]}')

    (dst / 'dataset.json').write_text(json.dumps({
        'name': nome,
        'description': ('WMH con il solo canale T1 (ablazione della FLAIR). Identico a '
                        'Dataset501_WMH tranne il canale di input. '
                        'Preparato da scripts/make_wmh_t1_dataset.py'),
        'channel_names': {'0': 'T1'},
        'labels': src_json['labels'],
        'numTraining': conteggi['imagesTr'],
        'file_ending': '.nii.gz',
        'ablazione': {
            'riferimento': 'Dataset501_WMH',
            'variabile_cambiata': 'canale di input: FLAIR -> T1 co-registrato',
            'immagine_usata': f'*{SUFFISSO_T1}',
            'atteso': 'Dice piu\' basso: le WMH sono definite sulla FLAIR',
        },
        'dataset_preparation': src_json.get('dataset_preparation'),
    }, indent=2) + '\n')

    # stessi fold di 501: senza questo il confronto non sarebbe attribuibile al canale
    splits_501 = PRE / 'Dataset501_WMH' / 'splits_final.json'
    pre = PRE / nome
    pre.mkdir(parents=True, exist_ok=True)
    if splits_501.exists():
        shutil.copy(splits_501, pre / 'splits_final.json')
        fold_copiati = 'si'
    else:
        fold_copiati = f'NO: {splits_501} non esiste, nnU-Net ne generera\' di propri'

    link = ROOT / 'nnUNet' / 'nnUNet_raw' / nome
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(dst)

    print(f'{nome}: {conteggi}')
    print(f'  fold ereditati da Dataset501_WMH: {fold_copiati}')
    print(f'  esposto come {link}')


if __name__ == '__main__':
    main()
