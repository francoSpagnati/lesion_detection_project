#!/usr/bin/env python3
"""Confronto a tre fra WMH, ATLAS e DBB a partire dai manifest per caso.

Legge `data/processed/manifests/{WMH,ATLAS,DBB}_cases.csv` e scrive
`reports/dataset_comparison_three.{json,md}`. Non tocca i dati.

La domanda a cui il report deve rispondere e' operativa, non descrittiva: quali
di questi dataset possono condividere un training, e quali no.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / 'data' / 'processed' / 'manifests'

# Cosa non si ricava dai manifest e va dichiarato: semantica del compito e sorgente.
SCHEDE = {
    'WMH': {
        'compito': 'lesioni della sostanza bianca (iperintensita)',
        'canale_usato': 'FLAIR (T1 co-registrato disponibile)',
        'spazio': 'nativo dello scanner',
        'classi': 'binaria (wmh); label 2 "altra patologia" esclusa',
        'popolazione': 'adulti, 4 siti / scanner diversi',
        'nnunet': 'Dataset501_WMH',
    },
    'ATLAS': {
        'compito': 'lesioni da ictus (post-stroke)',
        'canale_usato': 'T1',
        'spazio': 'MNI152NLin2009aSym',
        'classi': 'binaria (lesion)',
        'popolazione': 'adulti, 33 coorti',
        'nnunet': 'Dataset502_ATLAS',
    },
    'DBB': {
        'compito': 'segmentazione di 6 tessuti (non lesioni)',
        'canale_usato': 'T1',
        'spazio': 'MNI152 (213 casi) + nativo (31 casi)',
        'classi': '6 tessuti: CSF, GM, WM, GM profonda, tronco, cervelletto',
        'popolazione': 'pediatrica; training su sani, test su malformazioni',
        'nnunet': 'Dataset503_DBB',
    },
}


def carica(nome):
    p = MAN / f'{nome}_cases.csv'
    if not p.exists():
        return None
    with open(p) as fh:
        return list(csv.DictReader(fh))


def f(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def riassumi(nome, casi_tutti):
    # Un manifest puo' contenere anche immagini senza label (per ATLAS sono i 300 casi
    # del test ufficiale). Il confronto va fatto sui casi appaiati: sono gli unici
    # utilizzabili per addestrare o valutare.
    casi = [c for c in casi_tutti if str(c.get('lesion_ml', '')).strip() != '']
    senza_label = len(casi_tutti) - len(casi)
    spac = np.array([[f(c['sx']), f(c['sy']), f(c['sz'])] for c in casi])
    shape = np.array([[f(c['nx']), f(c['ny']), f(c['nz'])] for c in casi])
    fov = shape * spac
    fg = np.array([f(c['lesion_ml']) for c in casi])
    fg = fg[~np.isnan(fg)]
    vol_totale = np.prod(fov, axis=1) / 1000.0          # ml
    frazione = fg / vol_totale[: len(fg)] if len(fg) else np.array([])

    return {
        'dataset': nome,
        **SCHEDE[nome],
        'n_casi': len(casi),
        'n_senza_label': senza_label,
        'n_gruppi': len(set(c['group'] for c in casi)),
        'orientamenti': dict(Counter(c['orientation'] for c in casi)),
        # arrotondato a 4 decimali: senza, la precisione in virgola mobile degli header
        # conta come "distinte" spacing che sono la stessa risoluzione
        'spacing_distinti': len(set(map(tuple, np.round(spac, 4).tolist()))),
        'spacing_mediana': [round(float(x), 4) for x in np.median(spac, axis=0)],
        'shape_mediana': [int(x) for x in np.median(shape, axis=0)],
        'anisotropia_mediana': round(float(np.median([f(c['anisotropy']) for c in casi])), 2),
        'voxel_mm3_mediano': round(float(np.median([f(c['voxel_vol_mm3']) for c in casi])), 3),
        'fov_mediano_ml': round(float(np.median(vol_totale)), 0),
        'foreground_ml': {
            'p10': round(float(np.percentile(fg, 10)), 2),
            'mediana': round(float(np.median(fg)), 2),
            'p90': round(float(np.percentile(fg, 90)), 2),
        } if len(fg) else None,
        'frazione_foreground_mediana': round(float(np.median(frazione)), 4) if len(frazione) else None,
        'casi_geometria_ko': sum(1 for c in casi if c.get('geom_ok') == 'False'),
    }


def main():
    dati = {}
    for nome in ('WMH', 'ATLAS', 'DBB'):
        casi = carica(nome)
        if casi is None:
            print(f'manifest mancante per {nome}, saltato')
            continue
        dati[nome] = riassumi(nome, casi)

    (ROOT / 'reports' / 'dataset_comparison_three.json').write_text(
        json.dumps(dati, indent=2, default=str) + '\n')

    nomi = list(dati)
    def riga(etichetta, chiave, fmt=str):
        return '| ' + etichetta + ' | ' + ' | '.join(fmt(dati[n][chiave]) for n in nomi) + ' |'

    L = ['# Confronto fra i tre dataset', '',
         'Generato da `scripts/compare_datasets.py` dai manifest per caso. Read-only.', '',
         '## Che problema pone ciascuno', '',
         '| | ' + ' | '.join(nomi) + ' |',
         '| --- |' + ' --- |' * len(nomi),
         riga('dataset nnU-Net', 'nnunet', lambda v: f'`{v}`'),
         riga('compito', 'compito'),
         riga('classi', 'classi'),
         riga('canale usato', 'canale_usato'),
         riga('popolazione', 'popolazione'),
         riga('casi con immagine + label', 'n_casi', str),
         riga('immagini senza label', 'n_senza_label', str),
         '',
         '## Geometria', '',
         '| | ' + ' | '.join(nomi) + ' |',
         '| --- |' + ' --- |' * len(nomi),
         riga('spazio', 'spazio'),
         riga('orientamento', 'orientamenti', lambda v: '`' + ', '.join(f'{k}:{n}' for k, n in v.items()) + '`'),
         riga('spacing mediana (mm)', 'spacing_mediana', lambda v: ' x '.join(f'{x:g}' for x in v)),
         riga('spacing distinti', 'spacing_distinti', str),
         riga('shape mediana', 'shape_mediana', lambda v: 'x'.join(str(x) for x in v)),
         riga('anisotropia', 'anisotropia_mediana', str),
         riga('volume voxel (mm3)', 'voxel_mm3_mediano', str),
         riga('campo di vista (ml)', 'fov_mediano_ml', lambda v: f'{v:.0f}'),
         '',
         '## Carico di foreground', '',
         'Per WMH e ATLAS e\' il volume di lesione; per DBB e\' il volume complessivo dei 6 tessuti,',
         'cioe\' quasi tutto il cervello. La differenza di **tre ordini di grandezza** nella frazione',
         'di foreground e\' la ragione per cui i tre compiti non sono lo stesso problema.', '',
         '| | ' + ' | '.join(nomi) + ' |',
         '| --- |' + ' --- |' * len(nomi),
         riga('foreground mediano (ml)', 'foreground_ml', lambda v: f"{v['mediana']}" if v else '-'),
         riga('p10 - p90 (ml)', 'foreground_ml', lambda v: f"{v['p10']} - {v['p90']}" if v else '-'),
         riga('frazione del campo di vista', 'frazione_foreground_mediana',
              lambda v: f'{v:.4f}' if v is not None else '-'),
         '']
    (ROOT / 'reports' / 'dataset_comparison_three.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
