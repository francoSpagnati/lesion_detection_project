#!/usr/bin/env python
"""Ampiezza del campo di bias che N4 ha tolto ai 17 pazienti AInCP, dentro il cervello.

    python scripts/bias_campo_aincp.py   # -> reports/bias_campo_aincp.{md,json}

PERCHE'. Sui dataset pubblici N4 e' stato misurato prima e scartato (reports/bias_field_assessment.md:
p99/p1 del campo 1.10-1.12 a 12 mm dal bordo). Su AInCP invece la pipeline lo applica sempre, perche'
le immagini vengono da tre centri e due campi magnetici. Mancava il numero corrispondente: quanto era
grande il campo tolto.

COME. ANTs scrive corretta = originale / campo, quindi il campo e' originale / corretta, a meno di una
costante che il rapporto p99/p1 ignora. Nessun N4 da rieseguire. Si misura dentro la maschera del
cervello di SynthStrip (nella griglia della sequenza), erosa di 0, 3, 6, 12 mm come per i dati
pubblici; qui la maschera e' gia' il cervello, quindi anche il valore a 0 mm e' pulito. Lettura:
1.00 = campo piatto; 1.20 = fra il punto piu' chiaro e il piu' scuro il campo cambia del 20%.
Nessuna etichetta entra: vale per tutti i 17 pazienti, test compreso.
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_aincp as pa  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
PROC = PROGETTO / 'data/processed/AINCP'
OUT = PROGETTO / 'reports/bias_campo_aincp'
MARGINI = (0, 3, 6, 12)
CAMPO_3T = {'FSM_032', 'FSM_039', 'FSM_040', 'FSM_041', 'FSM_045', 'FSM_046',
            'FSM_052', 'FSM_053', 'FSM_054', 'FSM_056'}


def rapporti(orig: Path, corr: Path, cer: Path):
    o, c = nib.load(str(orig)), nib.load(str(corr))
    a, b = np.asanyarray(o.dataobj, dtype=np.float64).squeeze(), np.asanyarray(c.dataobj, dtype=np.float64).squeeze()
    m = np.asanyarray(nib.load(str(cer)).dataobj).squeeze() > 0.5
    assert a.shape == b.shape == m.shape, (orig, a.shape, b.shape, m.shape)
    dist = ndimage.distance_transform_edt(m, sampling=o.header.get_zooms()[:3])
    out = {}
    for mm in MARGINI:
        dentro = (dist > mm) & (a > 0) & (b > 0)
        if dentro.sum() < 1000:
            out[mm] = None
            continue
        f = a[dentro] / b[dentro]
        p1, p99 = np.percentile(f, [1, 99])
        out[mm] = float(p99 / p1)
    return out


def main():
    R = {}
    for caso, c in sorted(pa.trova_casi().items()):
        s = next(PROC.glob(f'{caso}_MR_*_anon'))
        R[caso] = {'campo': '3 T' if caso in CAMPO_3T else '1.5 T',
                   'T1': rapporti(Path(c['T1']), s / 'T1_N4.nii.gz', s / 'cervello_inT1.nii.gz'),
                   'FLAIR': rapporti(Path(c['FLAIR']), s / 'FLAIR_N4.nii.gz', s / 'cervello_FLAIR_originale.nii.gz')}
        print(caso, R[caso], flush=True)
    r = ['# Campo di bias tolto da N4 sui 17 pazienti AInCP', '',
         'Prodotto da `scripts/bias_campo_aincp.py`, **non modificare a mano**. Rapporto p99/p1 del campo '
         '(originale / N4) dentro la maschera del cervello di SynthStrip, erosa di un margine in mm. '
         '1.00 = campo piatto. Confronto: WMH, ATLAS, DBB a 12 mm = 1.12, 1.12, 1.10 (`bias_field_assessment.md`).', '',
         '## Mediana (min–max) per sequenza e campo magnetico', '',
         '| sequenza | gruppo | n | ' + ' | '.join(f'{mm} mm' for mm in MARGINI) + ' |',
         '| --- | --- | --- | ' + ' | '.join('---' for _ in MARGINI) + ' |']
    for seq in ('T1', 'FLAIR'):
        for gruppo, casi in (('tutti', list(R)), ('3 T', [k for k in R if R[k]['campo'] == '3 T']),
                             ('1.5 T', [k for k in R if R[k]['campo'] == '1.5 T'])):
            celle = []
            for mm in MARGINI:
                v = [R[k][seq][mm] for k in casi if R[k][seq][mm] is not None]
                celle.append(f'{np.median(v):.2f} ({min(v):.2f}–{max(v):.2f})')
            r.append(f'| {seq} | {gruppo} | {len(casi)} | ' + ' | '.join(celle) + ' |')
    r += ['', '## Per paziente (0 mm e 12 mm)', '', '| paziente | campo | T1 0 mm | T1 12 mm | FLAIR 0 mm | FLAIR 12 mm |',
          '| --- | --- | --- | --- | --- | --- |']
    for k, x in R.items():
        r.append(f"| {k} | {x['campo']} | {x['T1'][0]:.2f} | {x['T1'][12]:.2f} | {x['FLAIR'][0]:.2f} | {x['FLAIR'][12]:.2f} |")
    OUT.with_suffix('.md').write_text('\n'.join(r) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps(R, indent=1))
    print('\n'.join(r))


if __name__ == '__main__':
    main()
