#!/usr/bin/env python
"""Lo stadio 2 ha fatto dimenticare le cavita' di ATLAS? Rete di 542 sulla validazione di 502.

    python scripts/oblio_atlas_stadio2.py --gpu 1

PERCHE'. Lo stadio 2 (Dataset542, WMH in MNI a due canali) parte dai pesi di ATLAS (502) e
insegna alla rete a usare la FLAIR sulle iperintensita' della sostanza bianca. Il rischio e' che,
imparando WMH, perda quello che ATLAS le aveva insegnato: riconoscere in T1 le cavita' degli
ictus, che sono il fenotipo cistico/encefalomalacico dei pazienti AInCP. Il Dice su WMH non lo
vede; questo script si'.

COME. Si applica la rete dello stadio 2 ai 112 casi di validazione del fold 0 di 502, che la rete
non ha mai visto in addestramento ne' allo stadio 1 ne' allo stadio 2. ATLAS non ha FLAIR: il
canale 0 si passa **vuoto** (tutto zero), che e' esattamente cio' che la rete vede quando il
modality dropout spegne la FLAIR, perche' dopo la normalizzazione z-score un canale costante vale
zero. Il riferimento e' la rete di 502 sugli stessi casi (Dice medio 0.610), confronto appaiato.

Le etichette si leggono da labelsTr del raw: i gt_segmentations preprocessati di 502 sono stati
cancellati nella pulizia del disco (docs/cosa-e-stato-fatto.md, sezione 27). ATLAS e' gia' nella
griglia 1 mm di MNI, quindi raw e predizione sono sulla stessa griglia.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import wilcoxon

PROGETTO = Path(__file__).resolve().parent.parent
RAW = Path(os.environ['nnUNet_raw'])
PREP = Path(os.environ['nnUNet_preprocessed'])
RES = Path(os.environ['nnUNet_results'])
LAVORO = PROGETTO / 'nnUNet' / 'oblio_atlas'
RIF = RES / 'Dataset502_ATLAS' / 'nnUNetTrainerWandb__nnUNetPlans__3d_fullres' / 'fold_0' / 'validation'
TRAINER = ['nnUNetTrainerEspansioneCanali_zero_500ep_lr1e3_dropmod30_arresto50',
           'nnUNetTrainerEspansioneCanali_zero_500ep_lr1e4_primo1e2_dropmod30_arresto50']


def dice(a, b):
    s = a.sum() + b.sum()
    # stessa convenzione di nnU-Net: caso senza lesione e senza predizione -> nan, escluso
    return float('nan') if s == 0 else 2 * float((a & b).sum()) / s


def prepara_input(casi):
    inp = LAVORO / 'input'
    inp.mkdir(parents=True, exist_ok=True)
    for c in casi:
        t1 = RAW / 'Dataset502_ATLAS' / 'imagesTr' / f'{c}_0000.nii.gz'
        f0, f1 = inp / f'{c}_0000.nii.gz', inp / f'{c}_0001.nii.gz'
        if not f1.exists():
            f1.symlink_to(t1)
        if not f0.exists():
            img = nib.load(str(t1))
            nib.save(nib.Nifti1Image(np.zeros(img.shape, np.uint8), img.affine), str(f0))
    return inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', default='1')
    a = ap.parse_args()
    casi = json.loads((PREP / 'Dataset502_ATLAS' / 'splits_final.json').read_text())[0]['val']
    inp = prepara_input(casi)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)
    for tr in TRAINER:
        out = LAVORO / tr
        if len(list(out.glob('*.nii.gz'))) < len(casi):
            subprocess.run(['nnUNetv2_predict', '-i', str(inp), '-o', str(out), '-d', '542',
                            '-c', '3d_fullres', '-f', '0', '-tr', tr, '-p', 'nnUNetPlansDa502',
                            '-chk', 'checkpoint_final.pth', '--disable_progress_bar'],
                           env=env, check=True)

    per_caso = {}
    for c in casi:
        gt = np.asanyarray(nib.load(str(RAW / 'Dataset502_ATLAS' / 'labelsTr' / f'{c}.nii.gz')).dataobj) == 1
        r = {'mm3': int(gt.sum()),
             '502': dice(gt, np.asanyarray(nib.load(str(RIF / f'{c}.nii.gz')).dataobj) == 1)}
        for tr in TRAINER:
            r[tr] = dice(gt, np.asanyarray(nib.load(str(LAVORO / tr / f'{c}.nii.gz')).dataobj) == 1)
        per_caso[c] = r

    righe = ['# Oblio delle cavita\' di ATLAS dopo lo stadio 2', '',
             'Prodotto da `scripts/oblio_atlas_stadio2.py`, **non modificare a mano**.', '',
             f'Rete dello stadio 2 (Dataset542, fold 0, `checkpoint_final`) sui {len(casi)} casi di '
             'validazione del fold 0 di 502, con la FLAIR vuota. Riferimento: la rete di 502 sugli '
             'stessi casi. Confronto appaiato, Wilcoxon.', '',
             '| rete | Dice medio | differenza appaiata da 502 | p |', '| --- | --- | --- | --- |']
    rif = np.array([per_caso[c]['502'] for c in casi])
    righe.append(f'| 502 (stadio 1) | {np.nanmean(rif):.3f} | — | — |')
    for tr in TRAINER:
        v = np.array([per_caso[c][tr] for c in casi])
        ok = ~np.isnan(v) & ~np.isnan(rif)
        p = wilcoxon(v[ok], rif[ok]).pvalue
        righe.append(f'| `{tr.replace("nnUNetTrainerEspansioneCanali_", "")}` | {np.nanmean(v):.3f} | '
                     f'{np.mean(v[ok] - rif[ok]):+.3f} | {p:.2g} |')
    # Le cavita' piccole sono quelle che si dimenticano per prime: si guarda anche per volume.
    righe += ['', 'Per volume della lesione (terzili sui casi di validazione):', '',
              '| fascia | casi | 502 | ' + ' | '.join(t.replace('nnUNetTrainerEspansioneCanali_zero_500ep_', '')
                                               for t in TRAINER) + ' |',
              '| --- | --- | --- |' + ' --- |' * len(TRAINER)]
    vol = np.array([per_caso[c]['mm3'] for c in casi])
    q = np.quantile(vol, [1 / 3, 2 / 3])
    for nome, sel in [(f'< {q[0]:.0f} mm3', vol < q[0]),
                      (f'{q[0]:.0f}–{q[1]:.0f}', (vol >= q[0]) & (vol < q[1])),
                      (f'>= {q[1]:.0f}', vol >= q[1])]:
        cs = [c for c, s in zip(casi, sel) if s]
        righe.append(f'| {nome} | {len(cs)} | {np.nanmean([per_caso[c]["502"] for c in cs]):.3f} | ' +
                     ' | '.join(f'{np.nanmean([per_caso[c][t] for c in cs]):.3f}' for t in TRAINER) + ' |')
    (PROGETTO / 'reports' / 'oblio_atlas_stadio2.md').write_text('\n'.join(righe) + '\n')
    (PROGETTO / 'reports' / 'oblio_atlas_stadio2.json').write_text(json.dumps(per_caso, indent=1))
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
