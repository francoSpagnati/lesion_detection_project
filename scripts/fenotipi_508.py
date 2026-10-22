#!/usr/bin/env python
"""Ricetta F (Dataset508): quanto la rete riconosce il tipo di lesione, sui fuori-fold dei 12.

    python scripts/fenotipi_508.py      # -> reports/fenotipi_508.{md,json}

PERCHE'. F impara gliosi, cistica ed encefalomalacia come tre classi. Il confronto con Z
(confronto_misto_532.py F) guarda solo se il rilevamento peggiora; qui la domanda nuova: il tipo che la
rete assegna e' quello del clinico?

COME. Nella griglia T1 nativa, come tutte le metriche. Le probabilita' di ciascuna classe tornano dal
template con la trasformazione inversa (lineare, come prob_nativa); la maschera predetta e' quella del
postprocessing di F (somma delle tre classi, soglia e volume minimo scelti sui fuori-fold); dentro la
maschera, il tipo predetto e' la classe piu' probabile. Il tipo vero: le ROI del clinico nella T1
(ablazione_canali_506.fenotipi_in_t1), ristrette all'etichetta pulita, con la precedenza usata per
costruire 508 (cistica, poi encefalomalacia, poi gliosi). Tre misure:
  1. voxel per voxel, sui voxel di lesione trovati (veri e predetti): matrice di confusione in mm³;
  2. Dice per classe, sommando i voxel di tutti i pazienti che hanno quella classe;
  3. lesione per lesione (componenti dell'etichetta trovate): tipo prevalente vero contro predetto,
     accanto alla regola banale "sempre gliosi", che per numero di lesioni e' gia' quasi sempre giusta.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402
from ablazione_canali_506 import fenotipi_in_t1  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
RUN = PROGETTO / 'nnUNet/nnUNet_results/Dataset508_AINCP_FENOTIPI/nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres'
OUT = PROGETTO / 'reports/fenotipi_508'
CLASSI = {1: 'gliosi', 2: 'cistica', 3: 'encefalomalacia'}
DA_ROI = [('ROI_2', 2), ('ROI_4', 3), ('ROI_1', 1)]          # precedenza di dataset_classi_aincp.py


def classe_nativa(npz: Path, k: int) -> np.ndarray:
    """Probabilita' della classe k nella griglia T1 nativa (cache accanto a prob_nativa)."""
    cache = RUN / 'prob_nativa' / f'{npz.name[:-4]}_classe{k}.npy'
    if cache.exists():
        return np.load(cache).astype(np.float32)
    import ants
    import nibabel as nib
    caso = npz.name[:-4]
    s = pp.sessione(caso)
    rif = nib.load(str(npz.with_suffix('.nii.gz')))
    p = np.load(npz)['probabilities'][k].transpose(2, 1, 0)
    tmp = cache.with_suffix('.nii.gz')
    nib.save(nib.Nifti1Image(p.astype(np.float32), rif.affine), str(tmp))
    t = s / 'transforms_ped'
    nat = ants.apply_transforms(fixed=ants.image_read(str(s / 'T1_N4.nii.gz')), moving=ants.image_read(str(tmp)),
                                transformlist=[str(t / 'T1_to_PED_inv_0.mat'), str(t / 'T1_to_PED_inv_1.nii.gz')],
                                whichtoinvert=[True, False], interpolator='linear')
    tmp.unlink()
    a = nat.numpy().astype(np.float16)
    np.save(cache, a)
    return a.astype(np.float32)


def caso(c: str):
    par = json.loads((PROGETTO / 'reports' / f'postprocessing_{RUN.parent.name}__{RUN.name}.json').read_text())['scelto']
    npz = pp.predizioni_fuori_fold(RUN)[c]
    ref, cer, vox = pp.carica_caso(c)
    (RUN / 'prob_nativa').mkdir(exist_ok=True)
    pred = pp.postprocessa(pp.prob_nativa(npz, RUN / 'prob_nativa', pp.canali_lesione(RUN)),
                           par['t'], par['vmin'], par['cervello'], cer, vox)
    tipo_pred = np.argmax(np.stack([classe_nativa(npz, k) for k in CLASSI]), 0).astype(np.uint8) + 1
    fen = fenotipi_in_t1(c, pp.sessione(c))
    tipo_vero = np.zeros(ref.shape, np.uint8)
    for r, k in DA_ROI[::-1]:                                  # l'ultima scritta vince: cistica
        if r in fen:
            tipo_vero[fen[r] & ref] = k
    conf = np.zeros((4, 4), float)                             # righe vero (0 = non assegnato), colonne predetto
    both = ref & pred
    np.add.at(conf, (tipo_vero[both], tipo_pred[both]), vox)
    dice = {k: [float((tipo_vero == k).sum() * vox), float(((tipo_pred == k) & pred).sum() * vox),
                float(((tipo_vero == k) & (tipo_pred == k) & pred).sum() * vox)] for k in CLASSI}
    lesioni = []
    lr, n = ndimage.label(ref, structure=pp.STRUTTURA)
    for i in range(1, n + 1):
        m = lr == i
        if m.sum() * vox < pp.SOGLIA_LESIONE_MM3 or not (m & pred).any():
            continue
        v = np.bincount(tipo_vero[m], minlength=4)[1:]
        p = np.bincount(tipo_pred[m & pred], minlength=4)[1:]
        if v.sum():
            lesioni.append({'mm3': float(m.sum() * vox), 'vero': int(v.argmax()) + 1, 'predetto': int(p.argmax()) + 1})
    return c, {'confusione_mm3': conf.tolist(), 'dice': dice, 'lesioni': lesioni}


def main():
    casi = sorted(pp.predizioni_fuori_fold(RUN))
    with Pool(4) as p:
        R = dict(p.map(caso, casi))
    conf = sum(np.array(r['confusione_mm3']) for r in R.values())
    nomi = [CLASSI[k] for k in CLASSI]
    md = ['# Ricetta F: il tipo di lesione predetto contro quello del clinico (fuori-fold, 12 pazienti)', '',
          'Prodotto da `scripts/fenotipi_508.py`, **non modificare a mano**. Griglia T1 nativa; maschera dopo il '
          'postprocessing di F; tipo predetto = classe più probabile; tipo vero dalle ROI con la precedenza di 508.', '',
          '## Voxel di lesione trovati: matrice di confusione (mm³; % della riga)', '',
          '| vero \\ predetto | ' + ' | '.join(nomi) + ' |', '| --- | ' + ' | '.join('---' for _ in nomi) + ' |']
    for k in CLASSI:
        riga = conf[k, 1:]
        md.append(f'| {CLASSI[k]} | ' + ' | '.join(f'{pp.mm3(x)} ({x / riga.sum():.0%})' if riga.sum() else '—'
                                                    for x in riga) + ' |')
    acc = np.trace(conf[1:, 1:]) / conf[1:, 1:].sum()
    md += ['', f'Voxel trovati con il tipo giusto: **{acc:.0%}**. Voxel trovati senza tipo vero (bordo '
           f'dell\'interpolazione delle ROI): {pp.mm3(conf[0].sum())}.', '',
           '## Dice per classe (voxel di tutti i pazienti con quella classe)', '',
           '| classe | pazienti con la classe | volume vero | Dice |', '| --- | --- | --- | --- |']
    for k, nome in CLASSI.items():
        d = np.array([r['dice'][str(k)] if str(k) in r['dice'] else r['dice'][k] for r in R.values()])
        pazienti = int((d[:, 0] > 0).sum())
        md.append(f'| {nome} | {pazienti} | {pp.mm3(d[:, 0].sum())} | {2 * d[:, 2].sum() / (d[:, 0].sum() + d[:, 1].sum()):.3f} |')
    les = [l for r in R.values() for l in r['lesioni']]
    giuste = sum(l['vero'] == l['predetto'] for l in les)
    banale = sum(l['vero'] == 1 for l in les)
    md += ['', '## Lesione per lesione (lesioni annotate trovate)', '',
           f'Tipo prevalente giusto in **{giuste} lesioni su {len(les)}**; la regola banale «sempre gliosi» ne '
           f'indovina {banale}.', '', '| vero \\ predetto | ' + ' | '.join(nomi) + ' |',
           '| --- | ' + ' | '.join('---' for _ in nomi) + ' |']
    for k in CLASSI:
        md.append(f'| {CLASSI[k]} | ' + ' | '.join(str(sum(l['vero'] == k and l['predetto'] == j for l in les))
                                                    for j in CLASSI) + ' |')
    OUT.with_suffix('.md').write_text('\n'.join(md) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps(R, indent=1))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
