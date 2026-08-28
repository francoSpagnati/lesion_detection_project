#!/usr/bin/env python
"""Costruisce `Dataset512_WMH_T1_FLAIR`: WMH con **due** canali di input.

**La domanda.** L'ablazione ha gia' misurato i due canali da soli: FLAIR 0.774
contro T1 0.626, con la FLAIR migliore in 16 casi su 16. Resta aperta la domanda
che conta per il progetto: dare alla rete **entrambi** i canali aggiunge qualcosa
rispetto alla sola FLAIR, o il T1 e' informazione ridondante?

Non e' una curiosita' accademica. I dati AInCP della Stella Maris arriveranno con
piu' sequenze, e bisogna sapere in anticipo se conviene usarle tutte o se un
canale solo basta — con dieci casi, ogni canale in piu' e' anche un modo in piu'
di andare in overfitting.

Il dataset e' identico a `Dataset501_WMH` in tutto (stessi casi, stesse label,
stessi fold) tranne che ha un canale in piu'. Confronto attribuibile: 512 contro
501 isola l'aggiunta del T1.

Il T1 usato e' `*_T1w_BiasCorr_2FLAIR_Aligned.nii.gz`, gia' co-registrato nello
spazio della FLAIR dagli organizzatori: condivide quindi geometria e affine con la
FLAIR e con le label. Lo script lo **verifica** caso per caso invece di fidarsi:
nnU-Net richiede che tutti i canali siano allineati e un disallineamento silenzioso
produrrebbe un modello sbagliato senza mai dare errore.

Uso:
    python scripts/make_wmh_two_channel_dataset.py
"""
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
RAW = PROGETTO / 'data' / 'processed' / 'nnUNet_raw'
PREP = PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'
FLAIR = 'Dataset501_WMH'
T1 = 'Dataset511_WMH_T1'
NUOVO = 'Dataset512_WMH_T1_FLAIR'


def collega(sorgente: Path, destinazione: Path):
    """Symlink al file vero, non al symlink: i sorgenti stanno su /data01."""
    if not destinazione.exists():
        destinazione.symlink_to(os.path.realpath(sorgente))


def costruisci_split(split_img, split_lab):
    src_f = RAW / FLAIR / split_img
    src_t = RAW / T1 / split_img
    dst = RAW / NUOVO / split_img
    if not src_f.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    (RAW / NUOVO / split_lab).mkdir(parents=True, exist_ok=True)

    n = 0
    for f in sorted(src_f.glob('*_0000.nii.gz')):
        caso = f.name.replace('_0000.nii.gz', '')
        t = src_t / f.name
        if not t.exists():
            raise FileNotFoundError(f'manca il T1 di {caso}')

        # Verifica esplicita: stessa griglia, altrimenti i due canali non sono
        # sovrapponibili e nnU-Net non se ne accorgerebbe.
        a, b = nib.load(str(f)), nib.load(str(t))
        if a.shape != b.shape or not np.allclose(a.affine, b.affine, atol=1e-4):
            raise ValueError(f'{caso}: FLAIR e T1 non condividono la geometria')

        collega(f, dst / f.name)
        collega(t, dst / f'{caso}_0001.nii.gz')
        collega(RAW / FLAIR / split_lab / f'{caso}.nii.gz',
                RAW / NUOVO / split_lab / f'{caso}.nii.gz')
        n += 1
    return n


def main():
    n_tr = costruisci_split('imagesTr', 'labelsTr')
    n_ts = costruisci_split('imagesTs', 'labelsTs')
    print(f'{n_tr} casi di sviluppo, {n_ts} di test, geometria verificata su tutti')

    meta = json.loads((RAW / FLAIR / 'dataset.json').read_text())
    meta['name'] = NUOVO
    meta['description'] = ('WMH con due canali di input, FLAIR e T1 co-registrato. '
                           'Identico a Dataset501_WMH tranne il canale aggiunto. '
                           'Preparato da scripts/make_wmh_two_channel_dataset.py')
    meta['channel_names'] = {'0': 'FLAIR', '1': 'T1'}
    meta['ablazione'] = {
        'riferimento': 'Dataset501_WMH',
        'variabile_cambiata': 'aggiunto il canale T1 accanto alla FLAIR',
        'immagine_usata': '*_T1w_BiasCorr_2FLAIR_Aligned.nii.gz',
        'contesto': 'da solo il T1 vale 0.626 contro 0.774 della FLAIR (511 contro 501)',
        'domanda': 'il T1 aggiunge informazione alla FLAIR o e\' ridondante?',
    }
    (RAW / NUOVO / 'dataset.json').write_text(json.dumps(meta, indent=2) + '\n')

    # I fold non si rigenerano mai: stessi casi, stessi fold del dataset di
    # riferimento, altrimenti il confronto con 501 perde validita'.
    s = PREP / FLAIR / 'splits_final.json'
    (PREP / NUOVO).mkdir(parents=True, exist_ok=True)
    if not (PREP / NUOVO / 'splits_final.json').exists():
        shutil.copy2(s, PREP / NUOVO / 'splits_final.json')

    collegamento = PROGETTO / 'nnUNet' / 'nnUNet_raw' / NUOVO
    if not collegamento.exists():
        collegamento.symlink_to(RAW / NUOVO)
    print(f'pronto: {RAW / NUOVO}')


if __name__ == '__main__':
    main()
