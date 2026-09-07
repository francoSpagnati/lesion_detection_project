#!/usr/bin/env python
"""Metriche di rilevamento per singola lesione, non per voxel.

    python scripts/metriche_per_lesione.py 502 501 531   # -> reports/metriche_per_lesione.md

PERCHE' SERVONO. Il Dice e' una metrica di **sovrapposizione**: misura quanto bene si ricalca il
contorno di cio' che si e' trovato. Non dice nulla su cio' che non si e' trovato affatto. Un caso
con dieci lesioni in cui il modello ne trova una sola, ma la delinea benissimo, puo' avere un Dice
dignitoso; clinicamente sarebbe un fallimento.

Su questo progetto la domanda vera e' proprio quella: **il modello perde le lesioni piccole?** Le
misure di sovrapposizione l'hanno gia' suggerito per due vie indipendenti (l'ablazione FLAIR e i
casi falliti di ATLAS), ma non l'hanno mai quantificato a livello di lesione. Questo script lo fa.

COME. Si etichettano le componenti connesse della segmentazione di riferimento e della predizione.
Una lesione di riferimento e' **rilevata** se almeno un voxel della predizione ci cade dentro: e'
il criterio piu' permissivo, e quindi il piu' onesto quando la domanda e' "l'ha vista o no?"
invece di "l'ha delineata bene?". Una componente predetta che non tocca alcuna lesione vera e' un
**falso positivo**.

Le lesioni si raggruppano per volume secondo le fasce usate in letteratura sulle lesioni cerebrali
piccole (Component-Adaptive and Lesion-Level Supervision, arXiv 2604.08015): minuscole fino a 10
voxel, piccole fino a 50, medie fino a 200, grandi oltre.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

PROGETTO = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get('nnUNet_results', PROGETTO / 'nnUNet' / 'nnUNet_results'))
PREP = Path(os.environ.get('nnUNet_preprocessed', PROGETTO / 'nnUNet' / 'nnUNet_preprocessed'))

FASCE = [(0, 10, 'minuscole (<=10 voxel)'), (10, 50, 'piccole (<=50)'),
         (50, 200, 'medie (<=200)'), (200, np.inf, 'grandi (>200)')]
# Connettivita' a 26 vicini: due voxel che si toccano solo per uno spigolo appartengono alla
# stessa lesione. Con la connettivita' a 6 una lesione sottile si spezzerebbe in molte componenti.
STRUTTURA = np.ones((3, 3, 3), dtype=bool)


def analizza(args):
    caso, f_pred, f_gt, classe = args
    pred = np.asanyarray(nib.load(str(f_pred)).dataobj) == classe
    gt = np.asanyarray(nib.load(str(f_gt)).dataobj) == classe
    lab_gt, n_gt = ndimage.label(gt, structure=STRUTTURA)
    lab_pr, n_pr = ndimage.label(pred, structure=STRUTTURA)

    lesioni = []
    for i in range(1, n_gt + 1):
        m = lab_gt == i
        lesioni.append((int(m.sum()), bool((pred & m).any())))
    # Un falso positivo e' una componente predetta che non tocca nessuna lesione vera.
    fp = sum(1 for j in range(1, n_pr + 1) if not gt[lab_pr == j].any())
    return caso, lesioni, fp, int(n_pr)


def per_dataset(dataset: str, classe: int = 1):
    nome = next(d.name for d in RES.iterdir() if d.name.startswith(f'Dataset{dataset}'))
    val = next((RES / nome).glob('*/fold_0/validation'))
    gt = PREP / nome / 'gt_segmentations'
    lavori = [(f.name[:-len('.nii.gz')], f, gt / f.name, classe)
              for f in sorted(val.glob('*.nii.gz'))]
    if not lavori:
        raise SystemExit(f'nessuna predizione in {val}')
    with ProcessPoolExecutor(max_workers=12) as pool:
        return nome, list(pool.map(analizza, lavori, chunksize=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', nargs='+')
    ap.add_argument('--classe', type=int, default=1)
    a = ap.parse_args()

    righe = ['# Rilevamento per singola lesione', '',
             'Prodotto da `scripts/metriche_per_lesione.py`, **non modificare a mano**.', '',
             'Il Dice misura quanto bene si ricalca il contorno di cio\' che si e\' trovato, non',
             'quante lesioni si sono perse. Qui una lesione di riferimento conta come **rilevata**',
             'se almeno un voxel della predizione ci cade dentro: il criterio piu\' permissivo, e',
             'quindi quello che risponde a "l\'ha vista?" invece che "l\'ha delineata bene?".', '']

    for d in a.dataset:
        nome, ris = per_dataset(d, a.classe)
        tutte = [(v, ok) for _, lesioni, _, _ in ris for v, ok in lesioni]
        fp_tot = sum(fp for _, _, fp, _ in ris)
        n_casi = len(ris)
        righe += [f'## `{nome}` — {n_casi} casi di validation, {len(tutte)} lesioni annotate', '',
                  '| fascia di volume | lesioni | rilevate | tasso di rilevamento |',
                  '| --- | --- | --- | --- |']
        for lo, hi, etichetta in FASCE:
            sel = [ok for v, ok in tutte if lo < v <= hi]
            if not sel:
                continue
            righe.append(f'| {etichetta} | {len(sel)} | {sum(sel)} | '
                         f'**{sum(sel) / len(sel):.1%}** |')
        rilevate = sum(ok for _, ok in tutte)
        righe += [f'| **totale** | {len(tutte)} | {rilevate} | **{rilevate / len(tutte):.1%}** |', '',
                  f'Falsi positivi (componenti predette che non toccano alcuna lesione vera): '
                  f'**{fp_tot}** in tutto, cioe\' **{fp_tot / n_casi:.1f} per caso**.', '']
        # Il conteggio dei casi in cui non si trova proprio nulla e' il fallimento clinico vero.
        vuoti = sum(1 for _, lesioni, _, npr in ris if lesioni and npr == 0)
        righe += [f'Casi in cui il modello non predice **nessuna** componente pur essendoci '
                  f'lesione annotata: **{vuoti}** su {n_casi}.', '']

    out = PROGETTO / 'reports' / 'metriche_per_lesione.md'
    out.write_text('\n'.join(righe) + '\n')
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
