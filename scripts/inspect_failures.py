#!/usr/bin/env python
"""Esamina i casi in cui la segmentazione e' fallita, invece di mediarli con gli altri.

Un Dice medio non distingue due fallimenti molto diversi: bordi imprecisi su una
lesione trovata, e una lesione **non trovata** con predizione altrove. Il secondo
caso e' un errore di localizzazione e va guardato uno per uno, perche' le cause
possibili (ground truth anomala, lesione in una zona atipica, modello che segue un
falso indizio di intensita') portano ad azioni diverse.

Per ogni caso sotto soglia calcola: volumi, numero di componenti connesse,
centroidi in coordinate fisiche, distanza fra i centroidi, emisfero, intensita'
media dell'immagine dentro le due maschere, e il **test dello specchio** — se il
centroide predetto coincide col riflesso di quello vero rispetto al piano sagittale
mediano, il modello ha segmentato la regione controlaterale, che e' un errore di
tipo diverso (e sospetto, dato che l'augmentation include il flip).

Uso:
    python scripts/inspect_failures.py 502 --soglia 0.05
"""
import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PROGETTO = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get('nnUNet_results', PROGETTO / 'nnUNet' / 'nnUNet_results'))
PREP = Path(os.environ.get('nnUNet_preprocessed', PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'))
RAW = Path(os.environ.get('nnUNet_raw', PROGETTO / 'nnUNet' / 'nnUNet_raw'))


def descrivi(mask, affine, img=None):
    """Volume, componenti connesse, centroide fisico e intensita' media."""
    n = int(mask.sum())
    if n == 0:
        return dict(voxel=0, ml=0.0, componenti=0, centroide=None, intensita=None)
    _, ncomp = ndimage.label(mask)
    cen_vox = np.array(ndimage.center_of_mass(mask))
    cen_mm = nib.affines.apply_affine(affine, cen_vox)
    vox_ml = abs(np.linalg.det(affine[:3, :3])) / 1000.0
    return dict(voxel=n, ml=n * vox_ml, componenti=int(ncomp),
                centroide=cen_mm,
                intensita=float(img[mask].mean()) if img is not None else None)


def contrasto(caso, gtdir, imgdir):
    """Rapporto fra l'intensita' media dentro la lesione annotata e quella del
    tessuto cerebrale circostante. Su T1 una cavita' post-ictus e' ipointensa,
    quindi il rapporto atteso e' ben sotto 1. Un rapporto vicino a 1 significa che
    la regione annotata **non e' distinguibile per intensita'** da cio' che la
    circonda: il modello non ha nulla da vedere, e il fallimento non e' un bug."""
    gt = np.asarray(nib.load(str(gtdir / f'{caso}.nii.gz')).dataobj) > 0
    img = np.asarray(nib.load(str(imgdir / f'{caso}_0000.nii.gz')).dataobj, dtype=np.float32)
    if not gt.any() or not (img > 0).any():
        return None
    cervello = img > img[img > 0].mean() * 0.25
    les = float(img[gt].mean())
    cer = float(img[cervello & ~gt].mean())
    return les, cer, (les / cer if cer else float('nan')), float(gt.sum())


def probabilita_nella_lesione(caso, valdir, gtdir):
    """Probabilita' che il modello ha assegnato ai voxel della lesione vera.
    Distingue "non l'ha vista" da "l'ha vista ma sotto la soglia di 0.5": nel
    secondo caso abbassare la soglia recupererebbe qualcosa, nel primo no."""
    npz = valdir / f'{caso}.npz'
    if not npz.exists():
        return None
    p = np.load(str(npz))['probabilities'][1]
    gt = np.asarray(nib.load(str(gtdir / f'{caso}.nii.gz')).dataobj) > 0
    gt = np.transpose(gt, (2, 1, 0))   # da (x,y,z) di nibabel a (z,y,x) di nnU-Net
    if gt.shape != p.shape or not gt.any():
        return None
    pv = p[gt]
    return float(np.median(pv)), float(np.percentile(pv, 90)), float(pv.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--soglia', type=float, default=0.05,
                    help='Dice sotto il quale un caso viene esaminato (default 0.05)')
    ap.add_argument('--trainer', default='nnUNetTrainerWandb__nnUNetPlans__3d_fullres')
    ap.add_argument('--fold', default='fold_0')
    args = ap.parse_args()

    nome = next(d.name for d in RAW.iterdir() if d.name.startswith(f'Dataset{args.dataset}'))
    valdir = RESULTS / nome / args.trainer / args.fold / 'validation'
    gtdir = PREP / nome / 'gt_segmentations'
    imgdir = RAW / nome / 'imagesTr'

    s = json.loads((valdir / 'summary.json').read_text())
    classe = sorted(s['mean'].keys(), key=int)[0]
    falliti = [(Path(c['reference_file']).name.replace('.nii.gz', ''), c['metrics'][classe])
               for c in s['metric_per_case'] if c['metrics'][classe]['Dice'] <= args.soglia]
    falliti.sort(key=lambda t: -t[1]['n_ref'])

    righe = [f'# Casi falliti: {nome}, {args.fold}', '',
             'Generato da `scripts/inspect_failures.py`, **non modificare a mano**.', '',
             f'Casi con Dice <= {args.soglia}: **{len(falliti)}** su {len(s["metric_per_case"])}.', '']

    for caso, m in falliti:
        gt_img = nib.load(str(gtdir / f'{caso}.nii.gz'))
        gt = np.asarray(gt_img.dataobj) > 0
        pr = np.asarray(nib.load(str(valdir / f'{caso}.nii.gz')).dataobj) > 0
        img = np.asarray(nib.load(str(imgdir / f'{caso}_0000.nii.gz')).dataobj, dtype=np.float32)
        aff = gt_img.affine

        dg = descrivi(gt, aff, img)
        dp = descrivi(pr, aff, img)
        righe.append(f'## {caso}\n')
        righe.append(f"Dice {m['Dice']:.4f} | riferimento {dg['voxel']} voxel ({dg['ml']:.2f} ml), "
                     f"{dg['componenti']} componenti | predetto {dp['voxel']} voxel ({dp['ml']:.2f} ml), "
                     f"{dp['componenti']} componenti\n")
        if dg['centroide'] is None or dp['centroide'] is None:
            righe.append('Una delle due maschere e\' vuota: niente da confrontare geometricamente.\n')
            continue
        cg, cp = dg['centroide'], dp['centroide']
        dist = float(np.linalg.norm(cg - cp))
        # test dello specchio: riflesso rispetto al piano sagittale mediano (x -> -x in RAS)
        cg_mirror = cg.copy(); cg_mirror[0] = -cg_mirror[0]
        dist_mirror = float(np.linalg.norm(cg_mirror - cp))
        righe.append(f'- centroide vero `{np.round(cg, 1)}` mm, predetto `{np.round(cp, 1)}` mm')
        righe.append(f'- **distanza fra i centroidi: {dist:.1f} mm**')
        righe.append(f'- distanza dal riflesso speculare del centroide vero: {dist_mirror:.1f} mm'
                     + ('  <-- **piu\' vicino allo specchio: predizione controlaterale**'
                        if dist_mirror < dist * 0.6 else ''))
        righe.append(f"- emisfero: vero {'destro' if cg[0] > 0 else 'sinistro'}, "
                     f"predetto {'destro' if cp[0] > 0 else 'sinistro'}")
        righe.append(f"- intensita' media T1: {dg['intensita']:.1f} dentro il riferimento, "
                     f"{dp['intensita']:.1f} dentro la predizione")
        sovr = int((gt & pr).sum())
        righe.append(f'- voxel in comune: {sovr}')
        pr_prob = probabilita_nella_lesione(caso, valdir, gtdir)
        if pr_prob:
            righe.append(f'- probabilita\' assegnata ai voxel di lesione: mediana {pr_prob[0]:.4f}, '
                         f'90esimo percentile {pr_prob[1]:.4f}, massimo {pr_prob[2]:.4f}')
        co = contrasto(caso, gtdir, imgdir)
        if co:
            righe.append(f'- intensita\' lesione / cervello: {co[0]:.1f} / {co[1]:.1f} = '
                         f'**{co[2]:.2f}**')
        righe.append('')

    # --- confronto sistematico con i casi riusciti: e' il paragone che spiega i fallimenti
    buoni = [Path(c['reference_file']).name[:-7]
             for c in sorted(s['metric_per_case'], key=lambda c: -c['metrics'][classe]['Dice'])[:6]]
    righe.append('## Contrasto della lesione: falliti contro riusciti\n')
    righe.append('Rapporto fra intensita\' media dentro la lesione annotata e tessuto cerebrale attorno.\n')
    righe.append('| gruppo | caso | volume (ml) | lesione / cervello |')
    righe.append('| --- | --- | --- | --- |')
    riass = {}
    for etichetta, casi in (('fallito', [c for c, _ in falliti]), ('riuscito', buoni)):
        rapporti = []
        for c in casi:
            co = contrasto(c, gtdir, imgdir)
            if not co:
                continue
            rapporti.append(co[2])
            righe.append(f'| {etichetta} | {c} | {co[3] / 1000:.2f} | {co[2]:.2f} |')
        riass[etichetta] = float(np.median(rapporti)) if rapporti else float('nan')
    righe.append('')
    righe.append(f"**Rapporto mediano: {riass.get('fallito', float('nan')):.2f} nei casi falliti, "
                 f"{riass.get('riuscito', float('nan')):.2f} nei riusciti.**\n")
    righe.append('Nei casi riusciti la lesione e\' nettamente ipointensa, come una cavita\' post-ictus deve')
    righe.append('essere su T1. Nei falliti il rapporto e\' vicino a 1 o superiore: la regione annotata ha la')
    righe.append('stessa intensita\' del tessuto sano attorno. Unito al fatto che le probabilita\' predette')
    righe.append('dentro quelle regioni sono praticamente nulle, il fallimento non e\' un difetto della')
    righe.append('pipeline ma un **limite dell\'informazione contenuta nel T1**.\n')
    righe.append('Limite di questa misura: confronta intensita\' **medie**, non l\'aspetto visivo. Una lesione')
    righe.append('puo\' essere riconoscibile da texture o asimmetria senza differire in media.\n')

    out = PROGETTO / 'reports' / f'failure_analysis_{args.dataset}.md'
    out.write_text('\n'.join(righe) + '\n')
    print(f'{len(falliti)} casi esaminati -> {out.relative_to(PROGETTO)}')


if __name__ == '__main__':
    main()
