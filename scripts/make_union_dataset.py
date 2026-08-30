"""Costruisce la variante D: un solo dataset con tutti e tre i sorgenti e 7 classi.

    python scripts/make_union_dataset.py            # -> Dataset531_UNION_D
    python scripts/make_union_dataset.py --verifica  # solo controlli, non scrive

COSA C'E' DENTRO

    canale 0   T1 per tutti e tre i sorgenti
    label 1    lesione, da WMH e da ATLAS
    label 2-7  i sei tessuti di DBB (CSF, grigia, bianca, grigia profonda, tronco, cervelletto)

WMH entra con il **T1** (Dataset511_WMH_T1) e non con la FLAIR, perche' nnU-Net pretende gli stessi
canali per tutti i casi e ATLAS e DBB la FLAIR non ce l'hanno. Il prezzo e' misurato: 0.626 contro
0.774 (vedi docs/modelli-ed-esperimenti.md). Il test 512 ha escluso l'alternativa dei due canali.

SCELTE DI COSTRUZIONE, E PERCHE'

1. **Le immagini non si copiano, si collegano.** Sono identiche ai sorgenti: un symlink costa zero byte
   e rende impossibile che una copia divergga dall'originale. Idem per le label di WMH e ATLAS, dove
   `1` significa gia' "lesione". Si riscrivono soltanto le 244 label di DBB, che vanno rimappate da
   `1..6` a `2..7`.

2. **I nomi dei casi vengono prefissati** con `wmh_`, `atlas_`, `dbb_`. Serve a due cose: ATLAS e DBB
   usano entrambi nomi che iniziano per `sub-`, quindi senza prefisso una collisione sarebbe possibile
   e silenziosa; e con il prefisso la provenienza di ogni caso resta leggibile nel `summary.json`, che
   e' l'unico modo di riportare il Dice separatamente per sorgente.

3. **Il "tutto a 1 mm" non si fa qui.** I file restano nella loro griglia nativa: il ricampionamento e'
   compito del preprocessing di nnU-Net, che porta tutto alla target spacing del piano. Duplicare i
   volumi gia' ricampionati significherebbe decine di GB e un secondo ricampionamento a valle.

4. **I fold si compongono, non si rigenerano.** Il fold k dell'unione e' l'unione dei fold k dei tre
   sorgenti. Cosi' un caso che era in validation nel fold 0 di ATLAS resta in validation nel fold 0 di
   D, e il confronto con la baseline e' appaiato caso per caso invece che fra due medie complessive.
   Rigenerare gli split renderebbe l'intero registro degli esperimenti incomparabile.

IL CONFLITTO CHE QUESTO DATASET CONTIENE DI PROPOSITO

Il tessuto cerebrale sano vale `0` in WMH e ATLAS (che annotano solo la lesione) e `2..7` in DBB. La
rete riceve quindi segnali opposti sullo stesso ingresso, e DBB porta 266 volte piu' foreground per
caso (misurato, vedi reports/bilancio_classi_531.md). E' noto e accettato: D e' la versione diretta della richiesta "tutti i dati insieme", e il
suo costo si **misura** invece di assumerlo. La variante C, che marca `ignore` quei voxel, esiste
proprio per toglierlo, e si potra' costruire ora che SynthStrip da' le maschere cerebrali.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np

PROGETTO = Path('/home/ccantini/lesion_detection_project')
RAW = PROGETTO / 'data/processed/nnUNet_raw'

# sorgente -> (prefisso, rimappa le label?)
SORGENTI = {
    'Dataset511_WMH_T1': ('wmh', False),
    'Dataset502_ATLAS':  ('atlas', False),
    'Dataset503_DBB':    ('dbb', True),
}

# DBB usa 1..6 per i tessuti; in D il valore 1 e' occupato dalla lesione.
RIMAPPA_DBB = {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}

ETICHETTE = {
    'background': 0,
    'lesion': 1,
    'csf': 2,
    'gray_matter': 3,
    'white_matter': 4,
    'deep_gray_matter': 5,
    'brain_stem': 6,
    'cerebellum': 7,
}


def casi_di(sorgente: Path, split: str):
    """Nomi dei casi (senza `_0000.nii.gz`) in imagesTr o imagesTs."""
    d = sorgente / f'images{split}'
    return sorted(f.name[:-len('_0000.nii.gz')] for f in d.glob('*_0000.nii.gz'))


def collega(sorgente: Path, destinazione: Path, relativo: str):
    """Symlink assoluto: la destinazione sta altrove nell'albero e i relativi si romperebbero."""
    dst = destinazione / relativo.split('/')[-1]
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(sorgente.resolve())


def rimappa(src: Path, dst: Path):
    """Riscrive una label DBB portando 1..6 a 2..7, senza toccare geometria o header."""
    img = nib.load(str(src))
    dati = np.asanyarray(img.dataobj)
    presenti = set(np.unique(dati).tolist())
    ignote = presenti - set(RIMAPPA_DBB)
    if ignote:
        raise ValueError(f'{src.name}: valori di label non previsti {sorted(ignote)}')
    fuori = np.zeros_like(dati)
    for vecchio, nuovo in RIMAPPA_DBB.items():
        fuori[dati == vecchio] = nuovo
    # Scrittura atomica: un file troncato da un'interruzione sarebbe indistinguibile da uno valido.
    tmp = dst.with_name(dst.name.replace('.nii.gz', '.parziale.nii.gz'))
    nib.save(nib.Nifti1Image(fuori.astype(np.uint8), img.affine, img.header), str(tmp))
    tmp.replace(dst)


def costruisci(dst_nome: str, verifica_soltanto: bool):
    dst = RAW / dst_nome
    for sub in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        if not verifica_soltanto:
            (dst / sub).mkdir(parents=True, exist_ok=True)

    mappa_nomi = {}   # sorgente -> {caso originale: caso nuovo}
    conteggi = Counter()

    for nome, (prefisso, va_rimappata) in SORGENTI.items():
        src = RAW / nome
        if not src.exists():
            sys.exit(f'sorgente mancante: {src}')
        mappa_nomi[nome] = {}
        for split in ('Tr', 'Ts'):
            for caso in casi_di(src, split):
                nuovo = f'{prefisso}_{caso}'
                mappa_nomi[nome][caso] = nuovo
                conteggi[f'{prefisso}/{split}'] += 1
                if verifica_soltanto:
                    continue
                collega(src / f'images{split}' / f'{caso}_0000.nii.gz',
                        dst / f'images{split}', f'{nuovo}_0000.nii.gz')
                lab_src = src / f'labels{split}' / f'{caso}.nii.gz'
                lab_dst = dst / f'labels{split}' / f'{nuovo}.nii.gz'
                if va_rimappata:
                    if not lab_dst.exists():
                        rimappa(lab_src, lab_dst)
                else:
                    collega(lab_src, dst / f'labels{split}', f'{nuovo}.nii.gz')

    n_tr = sum(v for k, v in conteggi.items() if k.endswith('/Tr'))
    n_ts = sum(v for k, v in conteggi.items() if k.endswith('/Ts'))

    # I nomi devono essere unici: una collisione sovrascriverebbe un caso in silenzio.
    tutti = [n for m in mappa_nomi.values() for n in m.values()]
    if len(tutti) != len(set(tutti)):
        sys.exit('collisione fra nomi di caso dopo il prefisso')

    if verifica_soltanto:
        print(f'verifica: {n_tr} casi di sviluppo, {n_ts} di test', dict(conteggi))
        return

    (dst / 'dataset.json').write_text(json.dumps({
        'channel_names': {'0': 'T1'},
        'labels': ETICHETTE,
        'numTraining': n_tr,
        'file_ending': '.nii.gz',
        'description': ('Variante D: unione di WMH (T1), ATLAS e DBB. Classe 1 = lesione da WMH e '
                        'ATLAS, classi 2-7 = i sei tessuti di DBB. Costruita da '
                        'scripts/make_union_dataset.py.'),
    }, indent=2) + '\n')

    # --- fold: unione dei fold k dei sorgenti -------------------------------
    # nnUNet_preprocessed non e' un albero di symlink come nnUNet_raw: e' una cartella vera.
    pre = PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'
    fold_sorgente = {n: json.loads((pre / n / 'splits_final.json').read_text()) for n in SORGENTI}
    n_fold = {len(v) for v in fold_sorgente.values()}
    if n_fold != {5}:
        sys.exit(f'i sorgenti non hanno tutti 5 fold: {n_fold}')

    split = []
    for k in range(5):
        tr, val = [], []
        for nome in SORGENTI:
            m = mappa_nomi[nome]
            tr += [m[c] for c in fold_sorgente[nome][k]['train']]
            val += [m[c] for c in fold_sorgente[nome][k]['val']]
        # Un caso in entrambi gli insiemi renderebbe la validation contaminata.
        if set(tr) & set(val):
            sys.exit(f'fold {k}: casi presenti sia in train sia in val')
        split.append({'train': sorted(tr), 'val': sorted(val)})

    (pre / dst_nome).mkdir(parents=True, exist_ok=True)
    (pre / dst_nome / 'splits_final.json').write_text(json.dumps(split, indent=2) + '\n')

    # nnU-Net cerca i dataset in $nnUNet_raw, che per convenzione del progetto contiene
    # symlink a data/processed/nnUNet_raw/.
    collegamento = PROGETTO / 'nnUNet' / 'nnUNet_raw' / dst_nome
    if not collegamento.exists():
        collegamento.symlink_to(dst)
    print(f'{dst_nome}: {n_tr} casi di sviluppo, {n_ts} di test')
    for k, v in sorted(conteggi.items()):
        print(f'   {k}: {v}')
    print(f'fold 0: {len(split[0]["train"])} train, {len(split[0]["val"])} val')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nome', default='Dataset531_UNION_D')
    ap.add_argument('--verifica', action='store_true')
    a = ap.parse_args()
    costruisci(a.nome, a.verifica)
