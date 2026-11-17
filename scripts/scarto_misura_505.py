#!/usr/bin/env python
"""Perche' lo stesso modello di 505 ha due serie di numeri "senza postprocessing".

    python scripts/scarto_misura_505.py      # -> reports/scarto_misura_505.{md,json}

PERCHE'. Nella tesi (capitolo 8, "Native space or template") due tabelle sugli stessi modelli
di 505, sui 10 pazienti fuori-fold e senza postprocessing, danno numeri diversi: Pretrain-Joint
(da 543) 25/47 lesioni e 25 falsi positivi in reports/confronto_505_504.md, 27/47 e 22 in
reports/valutazione_cv_Dataset505_...da543....md. Non e' un errore di uno dei due: sono due catene
di misura che differiscono in due punti, e qui si misura quanto pesa ciascuno.

  1. CHE COSA TORNA NEL NATIVO. confronto_505_504.py legge le maschere di porta_505_nativo.py: la
     maschera binaria di nnU-Net (argmax nel template) interpolata linearmente e tagliata a 0.5.
     postprocessing_aincp.py riporta invece la probabilita' (interpolazione lineare) e taglia
     dopo, a 0.5 nel default.
  2. IL RIFERIMENTO. confronto_505_504.py usa gt_segmentations di Dataset504 (l'etichetta della
     prima pipeline, make_aincp_dataset.py); postprocessing_aincp.py usa lesione_pulita_inT1, la
     stessa etichetta ripulita dai residui da pipeline_aincp.py.

COME. Per ogni modello di 505 con predizioni fuori-fold, le quattro combinazioni (maschera o
probabilita') x (gt di 504 o etichetta pulita), tutte con le stesse misure di
postprocessing_aincp.misure: griglia T1 nativa, connettivita' 26, lesioni vere e falsi positivi
contati sopra 1.001 mm3, lesione trovata se tocca almeno un voxel predetto. Nessun riaddestramento,
nessuna nuova predizione: solo i file gia' salvati.
"""
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
RES = Path(os.environ['nnUNet_results']) / 'Dataset505_AINCP_MNI'
GT504 = Path(os.environ['nnUNet_preprocessed']) / 'Dataset504_AINCP' / 'gt_segmentations'
P = '__nnUNetPlansDa502__3d_fullres'
MODELLI = [
    ('Template, passo differenziato (505 A)', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_arresto50'),
    ('Template da Pretrain-Seq (505 B, da 542)', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50_da542'),
    ('Template da Pretrain-Joint (505 D, da 543)', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50_da543'),
]
COMBINAZIONI = [('maschera', 'gt504'), ('maschera', 'pulita'), ('probabilita', 'gt504'), ('probabilita', 'pulita')]
NOTA = {('maschera', 'gt504'): ' (= confronto_505_504)', ('probabilita', 'pulita'): ' (= postprocessing, default)'}
OUT = PROGETTO / 'reports' / 'scarto_misura_505'


def maschera_nativa(run: Path, caso: str) -> Path | None:
    for f in range(5):
        p = run / f'fold_{f}' / 'validation_nativo' / f'{caso}.nii.gz'
        if p.exists():
            return p
    return None


def modello(run: Path) -> dict | None:
    npz = pp.predizioni_fuori_fold(run)
    if len(npz) != 10:
        return None
    (run / 'prob_nativa').mkdir(exist_ok=True)
    per_caso, diversi = {k: [] for k in COMBINAZIONI}, {}
    for c in sorted(npz):
        pulita, cer, vox = pp.carica_caso(c)
        gt504 = np.asanyarray(nib.load(str(GT504 / f'{c}.nii.gz')).dataobj) > 0
        assert gt504.shape == pulita.shape, c
        diversi[c] = float((gt504 ^ pulita).sum() * vox)        # mm3
        f = maschera_nativa(run, c)
        if f is None:
            return None
        pred = {'maschera': np.asanyarray(nib.load(str(f)).dataobj) > 0,
                'probabilita': pp.prob_nativa(npz[c], run / 'prob_nativa') >= 0.5}
        rif = {'gt504': gt504, 'pulita': pulita}
        for k in COMBINAZIONI:
            m = pp.misure(pred[k[0]], rif[k[1]], cer, vox)
            per_caso[k].append({'caso': c, 'dice': m['dice'], 'fp': m['fp'], 'lesioni': len(m['lesioni']),
                                'trovate': sum(l['rilevata'] for l in m['lesioni'])})
    tot = {k: {'dice': float(np.mean([r['dice'] for r in v])), 'trovate': sum(r['trovate'] for r in v),
               'lesioni': sum(r['lesioni'] for r in v), 'fp': sum(r['fp'] for r in v)} for k, v in per_caso.items()}
    return {'totali': tot, 'per_caso': per_caso, 'mm3_diversi_fra_riferimenti': diversi}


def main():
    R = {}
    for nome, tr in MODELLI:
        r = modello(RES / (tr + P))
        if r is None:
            print(f'{nome}: predizioni fuori-fold incomplete, saltato')
            continue
        R[nome] = r
    md = ['# Due catene di misura per 505: da dove viene lo scarto', '',
          'Prodotto da `scripts/scarto_misura_505.py`, **non modificare a mano**. 10 pazienti fuori-fold, '
          'nessun postprocessing (soglia 0.5), griglia T1 nativa, connettività 26, lesioni vere e falsi '
          'positivi contati sopra 1.001 mm³, lesione trovata se tocca almeno un voxel predetto.', '',
          '- **maschera**: maschera binaria di nnU-Net riportata nel nativo (interpolazione lineare, taglio a 0.5; '
          '`porta_505_nativo.py`), come in `confronto_505_504.md`;',
          '- **probabilità**: probabilità riportata nel nativo, taglio a 0.5 dopo (`postprocessing_aincp.py`);',
          '- **gt504**: `gt_segmentations` di Dataset504 (prima pipeline);',
          '- **pulita**: `lesione_pulita_inT1` (etichetta ripulita dai residui, `pipeline_aincp.py`), il riferimento '
          'di tutte le valutazioni di 506.', '',
          '| modello | trasporto | riferimento | Dice medio | lesioni trovate | falsi positivi |',
          '| --- | --- | --- | --- | --- | --- |']
    for nome, r in R.items():
        for i, k in enumerate(COMBINAZIONI):
            t = r['totali'][k]
            md.append(f'| {nome if i == 0 else ""} | {k[0]} | {k[1]}{NOTA.get(k, "")} | {t["dice"]:.3f} | '
                      f'{t["trovate"]}/{t["lesioni"]} | {t["fp"]} |')
    if R:
        d = next(iter(R.values()))['mm3_diversi_fra_riferimenti']
        md += ['', 'Volume in cui i due riferimenti non coincidono (differenza simmetrica), per paziente: ' +
               ', '.join(f'{c.split("_MR")[0]} {pp.mm3(v)}' for c, v in d.items()) + '.', '']
        # la lettura si scrive dai numeri: verso dei falsi positivi passando dalla maschera alla probabilita'
        for nome, r in R.items():
            t = r['totali']
            fp = [t[(tr, rf)]['fp'] for tr in ('maschera', 'probabilita') for rf in ('gt504', 'pulita')]
            les = [t[(tr, rf)]['trovate'] for tr in ('maschera', 'probabilita') for rf in ('gt504', 'pulita')]
            md.append(f'- {nome}: falsi positivi {fp[0]} / {fp[1]} con la maschera, {fp[2]} / {fp[3]} con la '
                      f'probabilità (gt504 / pulita); lesioni trovate {les[0]} → {les[1]} con la maschera e '
                      f'{les[2]} → {les[3]} con la probabilità passando da gt504 all\'etichetta pulita.')
        md += ['', 'Le due tabelle della tesi usano catene diverse: i loro «prima del postprocessing» non vanno '
               'confrontati fra loro.']
    OUT.with_suffix('.md').write_text('\n'.join(md) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps(
        {n: {**r, 'totali': {'+'.join(k): v for k, v in r['totali'].items()},
             'per_caso': {'+'.join(k): v for k, v in r['per_caso'].items()}} for n, r in R.items()}, indent=1))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
