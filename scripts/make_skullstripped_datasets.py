#!/usr/bin/env python
"""Costruisce le versioni skull-stripped di WMH e ATLAS come dataset nnU-Net a se'.

**Perche' un dataset separato e non un'opzione di preprocessing.** Cosi' il confronto
resta pulito: 522 contro 501 e 523 contro 502 differiscono per una cosa sola, la
rimozione del cranio. Stessi casi, stessi fold, stessa architettura.

**Perche' potrebbe funzionare, in concreto.** nnU-Net normalizza z-score sull'intero
volume e accende `use_mask_for_norm` da solo quando il ritaglio al non-zero scende
sotto i 3/4 del volume originale. Oggi e' `False` su tutti e tre i dataset, perche'
le immagini sono teste intere. Azzerando fuori dal cervello il ritaglio crolla e la
normalizzazione viene calcolata **solo sui voxel cerebrali**, invece che su una
distribuzione che include grasso sottocutaneo, diploe e aria.

**La maschera non viene dilatata.** La misura in `reports/lesioni_fuori_maschera.md`
dice che si perde lo 0.00% delle lesioni su WMH e lo 0.07% su ATLAS. Dilatare di
2 mm porterebbe il caso peggiore da 1.9% a 0.8%, ma rimetterebbe dentro ~160 ml di
guscio non cerebrale per ogni caso: si pagherebbe l'intero beneficio per recuperare
tre casi su cento.

**La geometria non viene toccata**: nessun resampling, nessun crop, stesse affine.
Le label restano valide senza rimapparle, e la risoluzione di destinazione si sceglie
al momento del planning con `nnUNetv2_plan_experiment -overwrite_target_spacing`.
Portare tutto a 1 mm e' quindi una decisione di planning, non un secondo dataset.

**DBB non e' incluso, di proposito.** Li' il compito e' segmentare i tessuti, e
l'unione delle sei classi coincide con la maschera cerebrale (verificato: Dice
0.9999). Azzerare fuori dalla maschera regalerebbe alla rete l'estensione del
foreground, cioe' buona parte della risposta.

Uso:
    python scripts/make_skullstripped_datasets.py            # WMH e ATLAS
    python scripts/make_skullstripped_datasets.py --solo WMH
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
RAW = PROGETTO / 'data' / 'processed' / 'nnUNet_raw'
PREP = PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'
MASCHERE = PROGETTO / 'data' / 'processed' / 'synthstrip'
SCRIPT = PROGETTO / 'scripts' / 'synthstrip' / 'synthstrip.py'
PESI = PROGETTO / 'synthstrip.1.pt'

LAVORI = {
    'WMH':   ('Dataset501_WMH',   'Dataset522_WMH_SS',   'WMH'),
    'ATLAS': ('Dataset502_ATLAS', 'Dataset523_ATLAS_SS', 'ATLAS'),
}


# Impostato da --gpu. Su CPU SynthStrip costa 6 s per caso a macchina scarica, ma
# sotto contesa (altri utenti, data loader di un training in corso) crolla a oltre
# 100 s. Su GPU e' circa 1 s e non toglie CPU al training.
GPU = False


def maschera_di(img: Path, cache: Path) -> Path:
    """Maschera cerebrale del caso, calcolata solo se non e' gia' in cache.

    La scrittura e' atomica: il file compare solo a calcolo finito, cosi' se il
    processo viene interrotto non resta in cache una maschera troncata che al
    riavvio verrebbe scambiata per buona.
    """
    caso = img.name.replace('_0000.nii.gz', '')
    m = cache / f'{caso}_mask.nii.gz'
    if not m.exists():
        cache.mkdir(parents=True, exist_ok=True)
        tmp = m.with_suffix('.parziale.nii.gz')
        cmd = [sys.executable, str(SCRIPT), '-i', str(img), '-m', str(tmp),
               '--model', str(PESI)] + (['-g'] if GPU else [])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f'SynthStrip fallito su {caso}:\n{r.stderr[-500:]}')
        tmp.replace(m)
    return m


def calcola_maschere(immagini, cache: Path, jobs: int):
    """Calcola in parallelo le maschere mancanti.

    Ogni caso costa circa 1 s di inferenza ma 8-9 s di avvio del processo (import
    di torch, caricamento dei pesi). Pagare quell'avvio 500 volte in fila e' il
    grosso del tempo: N processi in volo lo ammortizzano. I thread bastano, perche'
    l'attesa e' su `subprocess` e non tiene il GIL.
    """
    mancanti = [i for i in immagini
                if not (cache / f'{i.name.replace("_0000.nii.gz", "")}_mask.nii.gz').exists()]
    if not mancanti:
        return
    print(f'    {len(mancanti)} maschere da calcolare, {jobs} in parallelo', flush=True)
    t0 = time.time()
    fatte = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futuri = {pool.submit(maschera_di, i, cache): i for i in mancanti}
        for f in as_completed(futuri):
            f.result()  # rilancia l'eccezione se SynthStrip e' fallito
            fatte += 1
            if fatte % 25 == 0:
                v = fatte / (time.time() - t0)
                print(f'    {fatte}/{len(mancanti)} ({v:.2f} casi/s, '
                      f'restano ~{(len(mancanti) - fatte) / v / 60:.0f} min)', flush=True)


def strippa(sorgente: Path, destinazione: Path, cache: Path, jobs: int):
    """Azzera fuori dal cervello, conservando geometria, header e affine."""
    destinazione.mkdir(parents=True, exist_ok=True)
    immagini = sorted(sorgente.glob('*_0000.nii.gz'))
    da_fare = [i for i in immagini if not (destinazione / i.name).exists()]
    calcola_maschere(da_fare, cache, jobs)
    frazioni = []
    for k, img in enumerate(da_fare, 1):
        m = maschera_di(img, cache)
        i = nib.load(str(img))
        dati = i.get_fdata()
        cervello = np.asarray(nib.load(str(m)).dataobj) > 0.5
        frazioni.append(cervello.sum() / cervello.size)
        dati[~cervello] = 0
        # Il temporaneo deve comunque finire in .nii.gz: nibabel deduce il formato
        # dall'estensione e su un nome arbitrario solleva ImageFileError.
        tmp = destinazione / img.name.replace('.nii.gz', '.parziale.nii.gz')
        nib.save(nib.Nifti1Image(dati.astype(np.float32), i.affine, i.header), str(tmp))
        tmp.replace(destinazione / img.name)
        if k % 100 == 0:
            print(f'    scritte {k}/{len(da_fare)}', flush=True)
    return frazioni


def collega(sorgente: Path, destinazione: Path):
    """Le label non cambiano: symlink al file vero, non al symlink."""
    if not sorgente.exists():
        return 0
    destinazione.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(sorgente.glob('*.nii.gz')):
        d = destinazione / f.name
        if not d.exists():
            d.symlink_to(os.path.realpath(f))
        n += 1
    return n


def costruisci(chiave, jobs: int):
    src_nome, dst_nome, cache_nome = LAVORI[chiave]
    src, dst = RAW / src_nome, RAW / dst_nome
    cache = MASCHERE / cache_nome
    print(f'\n=== {src_nome} -> {dst_nome} ===', flush=True)

    frazioni = []
    for split in ('imagesTr', 'imagesTs'):
        if (src / split).exists():
            print(f'  {split}: skull stripping', flush=True)
            frazioni += strippa(src / split, dst / split, cache, jobs)
    for split in ('labelsTr', 'labelsTs'):
        n = collega(src / split, dst / split)
        print(f'  {split}: {n} label collegate', flush=True)

    meta = json.loads((src / 'dataset.json').read_text())
    meta['name'] = dst_nome
    meta['description'] = (
        f'{src_nome} con skull stripping SynthStrip. Geometria e label invariate: '
        'i voxel fuori dalla maschera cerebrale sono azzerati, nient\'altro cambia.')
    meta['skull_stripping'] = {
        'strumento': 'SynthStrip (FreeSurfer), modello principale synthstrip.1.pt',
        'validazione': 'reports/synthstrip_dbb.md — Dice 0.966 su 244 casi DBB, '
                       '0.957 sui cervelli malformati',
        'lesione_persa': 'reports/lesioni_fuori_maschera.md',
        'dilatazione_mm': 0,
        'resampling': 'nessuno: la spacing di destinazione si sceglie al planning',
    }
    (dst / 'dataset.json').write_text(json.dumps(meta, indent=2) + '\n')

    # I fold non si rigenerano mai: si copiano dal dataset di partenza, cosi' il
    # confronto appaiato regge. I nomi dei casi sono identici.
    s = PREP / src_nome / 'splits_final.json'
    if s.exists():
        (PREP / dst_nome).mkdir(parents=True, exist_ok=True)
        d = PREP / dst_nome / 'splits_final.json'
        if not d.exists():
            shutil.copy2(s, d)
        print(f'  splits_final.json copiato da {src_nome}', flush=True)

    if frazioni:
        f = np.array(frazioni)
        print(f'  frazione di volume tenuta: mediana {np.median(f):.3f} '
              f'(min {f.min():.3f}, max {f.max():.3f})', flush=True)
    print(f'  fatto: {dst}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--solo', choices=sorted(LAVORI), default=None)
    p.add_argument('--gpu', action='store_true', help='inferenza SynthStrip su GPU')
    p.add_argument('--jobs', type=int, default=6, help='casi in parallelo')
    a = p.parse_args()
    global GPU
    GPU = a.gpu
    for k in ([a.solo] if a.solo else sorted(LAVORI)):
        costruisci(k, a.jobs)


if __name__ == '__main__':
    main()
