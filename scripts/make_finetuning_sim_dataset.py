#!/usr/bin/env python
"""Costruisce il dataset che simula AInCP: pochi casi di target, mai visti da nessun pretraining.

    python scripts/make_finetuning_sim_dataset.py            # -> Dataset541_SIM_AINCP
    python scripts/make_finetuning_sim_dataset.py --verifica

LA DOMANDA. L'intero progetto poggia su una premessa mai misurata: che pre-addestrare sui dataset
pubblici serva davvero quando il bersaglio e' di circa dieci pazienti. Se fosse falsa, lo si
scoprirebbe il giorno in cui arrivano i dati veri, cioe' troppo tardi. Questo dataset permette di
verificarla adesso.

IL DISEGNO, E PERCHE' PROPRIO QUESTI CASI. Servono casi di target che **nessun modello di
pretraining abbia mai visto**, altrimenti la misura e' priva di senso. I 45 casi di ATLAS acquisiti
a 1.5 T sono esattamente questo:

    - `Dataset521_ATLAS_3T` li esclude per costruzione: verificato, zero di essi compare in un
      qualsiasi fold di 521;
    - `Dataset503_DBB` e' un'altra popolazione, quindi non li ha visti;
    - `Dataset502_ATLAS` e `Dataset531_UNION_D` invece li **hanno** visti, e per questo non possono
      fare da sorgente di pretraining in questo esperimento.

Non si tocca alcun test set: i 45 casi vengono dal pool di sviluppo di ATLAS.

LA PARTIZIONE. Fold 0: **10 casi di addestramento** (quanti saranno i pazienti AInCP) e **35 di
valutazione**. I 10 sono scelti stratificando per volume di lesione, in modo che non siano per caso
tutti minuscoli o tutti enormi: su questo compito il Dice dipende quasi solo dal volume del
bersaglio, e una scelta sbilanciata renderebbe il risultato non interpretabile.

IL PIANO. Il piano di nnU-Net **non** va ricalcolato su 45 casi: va copiato da 502 con
`nnUNetv2_move_plans_between_datasets`. Il trasferimento dei pesi funziona solo fra architetture
identiche, e un fingerprint calcolato su un campione diverso produrrebbe patch e batch diversi,
rendendo il confronto impossibile prima ancora di cominciare.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

PROGETTO = Path('/home/ccantini/lesion_detection_project')
RAW = PROGETTO / 'nnUNet' / 'nnUNet_raw'
PRE = PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'
DEST = PROGETTO / 'data/processed/nnUNet_raw'
N_TRAIN = 10
SEME = 20260901


def casi_di(dataset: str, split: str = 'Tr'):
    return {p.name[:-len('_0000.nii.gz')]
            for p in (RAW / dataset / f'images{split}').glob('*_0000.nii.gz')}


def volume_lesione(caso: str) -> float:
    lab = RAW / 'Dataset502_ATLAS' / 'labelsTr' / f'{caso}.nii.gz'
    img = nib.load(str(lab))
    n = int((np.asanyarray(img.dataobj) > 0).sum())
    return n * float(np.prod(img.header.get_zooms()))   # mm³


def scegli_train(casi, volumi):
    """Dieci casi distribuiti su tutto lo spettro dei volumi, non dieci a caso.

    Si ordina per volume e si prende un caso ogni len/10: il campione copre l'intervallo invece di
    concentrarsi dove i casi sono piu' densi. Il seme fissa lo scarto iniziale, cosi' la scelta e'
    riproducibile ma non e' sempre il primo di ogni gruppo.
    """
    ordinati = sorted(casi, key=lambda c: volumi[c])
    rng = np.random.default_rng(SEME)
    passo = len(ordinati) / N_TRAIN
    scelti = []
    for k in range(N_TRAIN):
        lo, hi = int(k * passo), max(int(k * passo) + 1, int((k + 1) * passo))
        scelti.append(ordinati[int(rng.integers(lo, min(hi, len(ordinati))))])
    if len(set(scelti)) != N_TRAIN:
        sys.exit('la stratificazione ha prodotto duplicati: cambiare seme')
    return sorted(scelti)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nome', default='Dataset541_SIM_AINCP')
    ap.add_argument('--verifica', action='store_true')
    a = ap.parse_args()

    a502, a521 = casi_di('Dataset502_ATLAS'), casi_di('Dataset521_ATLAS_3T')
    target = sorted(a502 - a521)
    if len(target) != 45:
        sys.exit(f'attesi 45 casi a 1.5 T, trovati {len(target)}')

    # La verifica che rende valido tutto il resto: nessuno dei 45 in un fold delle sorgenti.
    for sorgente in ('Dataset521_ATLAS_3T', 'Dataset503_DBB'):
        visti = set()
        for f in json.loads((PRE / sorgente / 'splits_final.json').read_text()):
            visti |= set(f['train']) | set(f['val'])
        intrusi = set(target) & visti
        if intrusi:
            sys.exit(f'{sorgente} ha gia\' visto {len(intrusi)} casi di target: esperimento invalido')

    volumi = {c: volume_lesione(c) for c in target}
    train = scegli_train(target, volumi)
    val = sorted(set(target) - set(train))

    coorti = dict(Counter(re.match(r'sub-(r\d+)', c).group(1) for c in target))
    print(f'target: {len(target)} casi a 1.5 T   coorti: {coorti}')
    print(f'train {len(train)}  lesione mediana {np.median([volumi[c] for c in train]):.0f} mm³ '
          f'(da {min(volumi[c] for c in train):.0f} a {max(volumi[c] for c in train):.0f})')
    print(f'val   {len(val)}  lesione mediana {np.median([volumi[c] for c in val]):.0f} mm³ '
          f'(da {min(volumi[c] for c in val):.0f} a {max(volumi[c] for c in val):.0f})')
    if a.verifica:
        return

    dst = DEST / a.nome
    for sub in ('imagesTr', 'labelsTr'):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for caso in target:
        for sorgente, destinazione, suffisso in (
                ('imagesTr', 'imagesTr', '_0000.nii.gz'), ('labelsTr', 'labelsTr', '.nii.gz')):
            s = (RAW / 'Dataset502_ATLAS' / sorgente / f'{caso}{suffisso}').resolve()
            d = dst / destinazione / f'{caso}{suffisso}'
            if d.is_symlink() or d.exists():
                d.unlink()
            d.symlink_to(s)

    (dst / 'dataset.json').write_text(json.dumps({
        'channel_names': {'0': 'T1'},
        'labels': {'background': 0, 'lesion': 1},
        'numTraining': len(target),
        'file_ending': '.nii.gz',
        'description': ('Simulazione del caso AInCP: i 45 casi ATLAS a 1.5 T, mai visti da '
                        'Dataset521_ATLAS_3T ne da Dataset503_DBB. Fold 0: 10 casi di training, '
                        '35 di validation. Costruito da scripts/make_finetuning_sim_dataset.py.'),
    }, indent=2) + '\n')

    (PRE / a.nome).mkdir(parents=True, exist_ok=True)
    # Cinque fold identici: l'esperimento usa solo il fold 0, ma nnU-Net si aspetta la lista.
    (PRE / a.nome / 'splits_final.json').write_text(
        json.dumps([{'train': train, 'val': val}] * 5, indent=2) + '\n')

    collegamento = RAW / a.nome
    if not collegamento.exists():
        collegamento.symlink_to(dst)
    print(f'\n{a.nome} costruito.')


if __name__ == '__main__':
    main()
