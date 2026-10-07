#!/usr/bin/env python
"""Metriche di rilevamento per singola lesione, non per voxel.

    python scripts/metriche_per_lesione.py 502 501 531              # fold 0, come prima
    python scripts/metriche_per_lesione.py 504 --trainer EspansioneCanali_zero --fold all --soglia-mm3 10

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

Le fasce di volume sono in **millimetri cubi**, non in voxel: su ATLAS un voxel vale 1 mm3 e le
due cose coincidono, ma sulle FLAIR di AInCP un voxel vale 0.2 mm3 e su WMH 3.5, e contare in
voxel renderebbe incomparabili fasce che differiscono di venti volte. I limiti (10, 50, 200) sono
quelli della letteratura sulle lesioni piccole (arXiv 2604.08015), letti in mm3.

LA SOGLIA MINIMA (`--soglia-mm3`). Sui pazienti AInCP il 59% delle componenti annotate sta sotto
il millimetro cubo, cioe' e' un singolo voxel del tracciato manuale (notebooks/aincp_primo_lotto).
Contare ognuna di quelle come lesione mancata renderebbe il tasso di rilevamento un indice del
rumore di annotazione. La soglia si applica **simmetricamente**: le componenti di riferimento
sotto soglia non contano come lesioni, e le componenti predette sotto soglia non contano come
falsi positivi. Va dichiarata una volta e non aggiustata guardando i risultati; per AInCP e'
10 mm3 (docs/obiettivo-e-piano.md).

LA GRIGLIA. Riferimento e predizione si leggono nella **griglia originale** del caso
(gt_segmentations e la validation esportata da nnU-Net stanno entrambe li'), non in quella
ricampionata del preprocessing: su AInCP il ricampionamento a 1 mm spezza le componenti sottili,
e contarle li' darebbe un numero che non e' quello delle annotazioni.

Oltre al report in Markdown scrive un JSON per caso in reports/lesioni_per_caso/, che serve a
scripts/confronto_fold.py per il confronto appaiato sui fold.
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
RAW = Path(os.environ.get('nnUNet_raw', PROGETTO / 'nnUNet' / 'nnUNet_raw'))
PER_CASO = PROGETTO / 'reports' / 'lesioni_per_caso'

FASCE = [(0, 10, 'minuscole (<=10 mm3)'), (10, 50, 'piccole (<=50)'),
         (50, 200, 'medie (<=200)'), (200, np.inf, 'grandi (>200)')]
# Connettivita' a 26 vicini: due voxel che si toccano solo per uno spigolo appartengono alla
# stessa lesione. Con la connettivita' a 6 una lesione sottile si spezzerebbe in molte componenti.
STRUTTURA = np.ones((3, 3, 3), dtype=bool)


def analizza(args):
    caso, f_pred, f_gt, classe, soglia_mm3 = args
    img_gt = nib.load(str(f_gt))
    img_pr = nib.load(str(f_pred))
    gt = np.asanyarray(img_gt.dataobj) == classe
    pred = np.asanyarray(img_pr.dataobj) == classe
    if gt.shape != pred.shape:
        raise ValueError(f'{caso}: riferimento {gt.shape} e predizione {pred.shape} non sono '
                         f'sulla stessa griglia')
    mm3 = float(np.prod(img_gt.header.get_zooms()[:3]))
    lab_gt, n_gt = ndimage.label(gt, structure=STRUTTURA)
    lab_pr, n_pr = ndimage.label(pred, structure=STRUTTURA)
    vol_gt = np.bincount(lab_gt.ravel())[1:] * mm3
    vol_pr = np.bincount(lab_pr.ravel())[1:] * mm3

    lesioni = []
    for i in range(1, n_gt + 1):
        if vol_gt[i - 1] < soglia_mm3:
            continue
        m = lab_gt == i
        lesioni.append({'mm3': float(vol_gt[i - 1]), 'rilevata': bool((pred & m).any())})
    # Un falso positivo e' una componente predetta, sopra soglia, che non tocca nessuna
    # lesione vera (qui le lesioni vere contano tutte, anche sotto soglia: una predizione che
    # cade su un voxel annotato non e' un'invenzione del modello).
    fp = sum(1 for j in range(1, n_pr + 1)
             if vol_pr[j - 1] >= soglia_mm3 and not gt[lab_pr == j].any())
    n_pr_sopra = int((vol_pr >= soglia_mm3).sum())
    return {'caso': caso, 'lesioni': lesioni, 'falsi_positivi': int(fp),
            'componenti_predette': n_pr_sopra, 'voxel_mm3': mm3}


def cartella_risultati(dataset: str, trainer: str | None) -> tuple[str, Path]:
    nome = next(d.name for d in sorted(RES.iterdir()) if d.name.startswith(f'Dataset{dataset}'))
    cartelle = sorted(d for d in (RES / nome).iterdir() if d.is_dir())
    if trainer:
        cartelle = [d for d in cartelle if trainer in d.name]
    if len(cartelle) != 1:
        raise SystemExit(f'{nome}: {len(cartelle)} cartelle di risultati '
                         f'{[d.name for d in cartelle]}; scegliere con --trainer')
    return nome, cartelle[0]


def per_dataset(dataset: str, trainer: str | None, fold: list[int] | None,
                classe: int, soglia_mm3: float):
    nome, run = cartella_risultati(dataset, trainer)
    gt = PREP / nome / 'gt_segmentations'
    if not any(gt.glob('*.nii.gz')):
        # cartella cancellata per spazio: nnU-Net la crea copiando labelsTr cosi' com'e'
        gt = RAW / nome / 'labelsTr'
    if fold is None:
        folds = sorted(int(d.name.split('_')[1]) for d in run.glob('fold_*')
                       if (d / 'validation').is_dir())
    else:
        folds = fold
    lavori, appartenenza = [], {}
    for f in folds:
        val = run / f'fold_{f}' / 'validation'
        for p in sorted(val.glob('*.nii.gz')):
            caso = p.name[:-len('.nii.gz')]
            lavori.append((caso, p, gt / p.name, classe, soglia_mm3))
            appartenenza[caso] = f
    if not lavori:
        raise SystemExit(f'nessuna predizione in {run} per i fold {folds}')
    with ProcessPoolExecutor(max_workers=12) as pool:
        ris = list(pool.map(analizza, lavori, chunksize=2))
    for r in ris:
        r['fold'] = appartenenza[r['caso']]
    return nome, run.name, folds, ris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', nargs='+')
    ap.add_argument('--trainer', help='sottostringa della cartella trainer__piano__config, '
                                      'obbligatoria se il dataset ne ha piu\' di una')
    ap.add_argument('--fold', default='0', help='"all", oppure indici separati da virgola')
    ap.add_argument('--classe', type=int, default=1)
    ap.add_argument('--soglia-mm3', type=float, default=0.0,
                    help='componenti sotto questo volume ignorate, sia nel riferimento sia '
                         'nella predizione')
    ap.add_argument('--out', default=str(PROGETTO / 'reports' / 'metriche_per_lesione.md'))
    a = ap.parse_args()
    fold = None if a.fold == 'all' else [int(x) for x in a.fold.split(',')]

    righe = ['# Rilevamento per singola lesione', '',
             'Prodotto da `scripts/metriche_per_lesione.py`, **non modificare a mano**.', '',
             'Il Dice misura quanto bene si ricalca il contorno di cio\' che si e\' trovato, non',
             'quante lesioni si sono perse. Qui una lesione di riferimento conta come **rilevata**',
             'se almeno un voxel della predizione ci cade dentro: il criterio piu\' permissivo, e',
             'quindi quello che risponde a "l\'ha vista?" invece che "l\'ha delineata bene?".',
             'Le fasce sono in millimetri cubi, misurati sulla griglia originale di ogni caso.', '']
    if a.soglia_mm3 > 0:
        righe += [f'**Soglia minima: {a.soglia_mm3:g} mm3.** Le componenti piu\' piccole non '
                  f'contano ne\' come lesioni ne\' come falsi positivi.', '']

    PER_CASO.mkdir(parents=True, exist_ok=True)
    for d in a.dataset:
        nome, run, folds, ris = per_dataset(d, a.trainer, fold, a.classe, a.soglia_mm3)
        tutte = [(l['mm3'], l['rilevata']) for r in ris for l in r['lesioni']]
        fp_tot = sum(r['falsi_positivi'] for r in ris)
        n_casi = len(ris)
        etichetta_fold = 'fold ' + ', '.join(map(str, folds)) if len(folds) < 5 else f'{len(folds)} fold'
        righe += [f'## `{nome}` / `{run}` — {n_casi} casi ({etichetta_fold}), '
                  f'{len(tutte)} lesioni annotate', '',
                  '| fascia di volume | lesioni | rilevate | tasso di rilevamento |',
                  '| --- | --- | --- | --- |']
        for lo, hi, etichetta in FASCE:
            sel = [ok for v, ok in tutte if lo < v <= hi]
            if not sel:
                continue
            righe.append(f'| {etichetta} | {len(sel)} | {sum(sel)} | '
                         f'**{sum(sel) / len(sel):.1%}** |')
        rilevate = sum(ok for _, ok in tutte)
        tasso = rilevate / len(tutte) if tutte else float('nan')
        righe += [f'| **totale** | {len(tutte)} | {rilevate} | **{tasso:.1%}** |', '',
                  f'Falsi positivi (componenti predette che non toccano alcuna lesione vera): '
                  f'**{fp_tot}** in tutto, cioe\' **{fp_tot / n_casi:.1f} per caso**.', '']
        # Il conteggio dei casi in cui non si trova proprio nulla e' il fallimento clinico vero.
        vuoti = sum(1 for r in ris if r['lesioni'] and r['componenti_predette'] == 0)
        righe += [f'Casi in cui il modello non predice **nessuna** componente pur essendoci '
                  f'lesione annotata: **{vuoti}** su {n_casi}.', '']

        f_json = PER_CASO / f'{nome}__{run}.json'
        f_json.write_text(json.dumps({'dataset': nome, 'run': run, 'fold': folds,
                                      'soglia_mm3': a.soglia_mm3, 'classe': a.classe,
                                      'casi': ris}, indent=1))
        righe += [f'Dettaglio per caso: `{f_json.relative_to(PROGETTO)}`.', '']

    Path(a.out).write_text('\n'.join(righe) + '\n')
    print('\n'.join(righe))


if __name__ == '__main__':
    main()
