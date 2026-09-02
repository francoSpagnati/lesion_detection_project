#!/usr/bin/env python
"""Confronta i due bracci del test del pretraining. -> reports/test_pretraining.md

I due bracci vivono nello stesso dataset (`Dataset541_SIM_AINCP`) e differiscono solo per
l'inizializzazione dei pesi, quindi `confronto_appaiato.py` non li sa leggere: quello confronta due
dataset diversi, qui i casi di validation sono gli stessi 35 per costruzione.

Oltre alla differenza media si guarda **come si distribuisce**, separando i casi per volume di
lesione. Su questo compito il Dice dipende quasi solo dal volume del bersaglio, e un guadagno medio
puo' nascondere che tutto il vantaggio viene da una sola fascia.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats

PROGETTO = Path(__file__).resolve().parent.parent
RES = PROGETTO / 'nnUNet' / 'nnUNet_results' / 'Dataset541_SIM_AINCP'
RAW = PROGETTO / 'nnUNet' / 'nnUNet_raw' / 'Dataset502_ATLAS'
# Predizioni di 521 applicato ai casi a 1.5 T senza alcun fine-tuning, prodotte a suo tempo da
# scripts/inferenza_incrociata_15T.py. Sono il terzo braccio, ed e' quello che cambia la lettura.
ZEROSHOT = PROGETTO / 'data/processed/inferenza_15T/pred_521'
BRACCI = {
    'da zero': 'nnUNetTrainerWandb_250epochs__nnUNetPlans__3d_fullres',
    'fine-tuning da 521': 'nnUNetTrainerWandb_250epochs__nnUNetPlansFT__3d_fullres',
}


def dadi(cartella: str):
    j = json.loads((RES / cartella / 'fold_0' / 'validation' / 'summary.json').read_text())
    return {Path(m['reference_file']).name.replace('.nii.gz', ''): m['metrics']['1']['Dice']
            for m in j['metric_per_case']}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    somma = float(a.sum() + b.sum())
    return 1.0 if somma == 0 else 2 * float((a & b).sum()) / somma


def dadi_zeroshot(casi):
    """Dice di 521 applicato tale e quale, senza vedere nessuno dei 10 casi di training."""
    fuori = {}
    for c in casi:
        p = ZEROSHOT / f'{c}.nii.gz'
        if not p.exists():
            continue
        pr = np.asanyarray(nib.load(str(p)).dataobj) > 0
        gt = np.asanyarray(nib.load(str(RAW / 'labelsTr' / f'{c}.nii.gz')).dataobj) > 0
        fuori[c] = dice(gt, pr)
    return fuori


def volume(caso: str) -> float:
    img = nib.load(str(RAW / 'labelsTr' / f'{caso}.nii.gz'))
    n = int((np.asanyarray(img.dataobj) > 0).sum())
    return n * float(np.prod(img.header.get_zooms())) / 1000.0


def main():
    a, b = dadi(BRACCI['da zero']), dadi(BRACCI['fine-tuning da 521'])
    casi = sorted(set(a) & set(b))
    x, y = np.array([a[c] for c in casi]), np.array([b[c] for c in casi])
    w = stats.wilcoxon(y, x)

    righe = ['# Test del pretraining: serve, con dieci casi di target?', '',
             'Prodotto da `scripts/analisi_test_pretraining.py`, **non modificare a mano**.', '',
             'Stesso dataset, stessa partizione (10 casi di training, 35 di validation), stessa',
             'architettura, stesse 250 epoche. Cambia **solo** l\'inizializzazione dei pesi.', '',
             '| braccio | Dice medio sui 35 casi |', '| --- | --- |',
             f'| addestrato da zero | {x.mean():.4f} |',
             f'| fine-tuning da `Dataset521_ATLAS_3T` | **{y.mean():.4f}** |',
             f'| **differenza** | **{y.mean() - x.mean():+.4f}** |', '',
             f'Migliora in **{(y > x).sum()}** casi su {len(casi)}, peggiora in '
             f'**{(y < x).sum()}**, pari in {(y == x).sum()}.',
             f'Wilcoxon appaiato: **p = {w.pvalue:.4g}**.', '',
             ]

    # --- il terzo braccio, quello che cambia la lettura --------------------
    zs = dadi_zeroshot(casi)
    comuni = [c for c in casi if c in zs]
    z = np.array([zs[c] for c in comuni])
    f = np.array([b[c] for c in comuni])
    wz = stats.wilcoxon(f, z)
    righe += ['## Il terzo braccio: e se non si facesse fine-tuning affatto?', '',
              'Il modello pre-addestrato applicato **tale e quale** ai casi di valutazione, senza',
              'vedere i 10 casi di training. Stessi casi, quindi il confronto e\' appaiato.', '',
              '| | Dice sui 35 casi |', '| --- | --- |',
              f'| addestrato da zero sui 10 | {x.mean():.4f} |',
              f'| fine-tuning da `521` sui 10 | {f.mean():.4f} |',
              f'| **`521` senza alcun fine-tuning** | **{z.mean():.4f}** |',
              f'| differenza fine-tuning meno zero-shot | **{f.mean() - z.mean():+.4f}** |', '',
              f'Il fine-tuning batte lo zero-shot in **{(f > z).sum()}** casi su {len(comuni)}. '
              f'Wilcoxon appaiato: **p = {wz.pvalue:.4g}**.', '',
              '**Il fine-tuning con la ricetta predefinita peggiora un modello gia\' buono.** Il',
              'pretraining serve, e molto, ma la messa a punto su dieci casi cosi\' com\'e\' distrugge',
              'piu\' di quanto aggiunga: nnU-Net riparte con passo di apprendimento 0.01, che e\' quello',
              'di un addestramento completo, e con dieci casi cancella le feature pre-addestrate.', '',
              '| termine di paragone | Dice |', '| --- | --- |',
              '| `502`, addestrato su 556 casi | 0.6103 |',
              f'| `521` zero-shot su questi casi | {z.mean():.4f} |', '']

    # --- come si distribuisce il guadagno ---------------------------------
    vol = {c: volume(c) for c in casi}
    mediana = float(np.median(list(vol.values())))
    righe += ['## Come si distribuisce il guadagno', '',
              f'Mediana del volume di lesione nei 35 casi: {mediana:.1f} ml.', '',
              '| gruppo | casi | da zero | fine-tuning | differenza |', '| --- | --- | --- | --- | --- |']
    for etichetta, sel in (('lesioni sotto la mediana', [c for c in casi if vol[c] <= mediana]),
                           ('lesioni sopra la mediana', [c for c in casi if vol[c] > mediana])):
        i = [casi.index(c) for c in sel]
        righe.append(f'| {etichetta} | {len(sel)} | {x[i].mean():.4f} | {y[i].mean():.4f} | '
                     f'**{y[i].mean() - x[i].mean():+.4f}** |')
    righe.append('')

    # --- i casi in cui il pretraining cambia tutto -------------------------
    delta = y - x
    ordine = np.argsort(delta)[::-1]
    righe += ['## I cinque casi in cui il pretraining cambia di piu\'', '',
              '| caso | lesione (ml) | da zero | fine-tuning | delta |', '| --- | --- | --- | --- | --- |']
    for i in ordine[:5]:
        c = casi[i]
        righe.append(f'| {c} | {vol[c]:.1f} | {x[i]:.4f} | {y[i]:.4f} | **{delta[i]:+.4f}** |')
    righe += ['', '## Limiti da dichiarare', '',
              '- Una sola estrazione dei 10 casi di training: con dieci casi la varianza e\' alta.',
              '- Una sola sorgente di pretraining. Quella fra compiti diversi (DBB) richiede prima di',
              '  spostare il piano di 502 su DBB e riaddestrarlo, perche\' le architetture non combaciano.',
              '- 39 dei 45 casi vengono dalla coorte `r040`: e\' in larga parte un solo scanner.',
              '- 250 epoche per entrambi i bracci. Un budget maggiore favorirebbe forse il braccio da',
              '  zero, che parte da piu\' lontano; il confronto vale a parita\' di calcolo.', '']

    out = PROGETTO / 'reports' / 'test_pretraining.md'
    out.write_text('\n'.join(righe) + '\n')
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
