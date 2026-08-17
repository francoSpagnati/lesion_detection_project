#!/usr/bin/env python3
"""Analisi read-only del terzo dataset (Distorted Brain Benchmark).

Produce un manifest per caso con lo stesso schema di WMH e ATLAS
(`data/processed/manifests/DBB_cases.csv`) e un report di confronto a tre
(`reports/dbb_analysis.{json,md}`).

Nessun dato viene modificato: si leggono solo gli header e i volumi di label.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DBB = ROOT / 'data' / 'raw' / 'DBB'

# parcellation-volume/label.json, uguale per tutti i soggetti
CLASSI = {
    1: 'CSF',
    2: 'Gray Matter',
    3: 'White Matter',
    4: 'Deep Gray Matter',
    5: 'Brain Stem',
    6: 'Cerebellum',
}


def manifest_rows():
    with open(DBB / 'manifest.csv') as fh:
        return list(csv.DictReader(fh))


def indicizza(rows):
    """subject -> {slug: path assoluta del file NIfTI}."""
    per_sub = defaultdict(dict)
    for r in rows:
        d = DBB / r['path']
        nii = sorted(p for p in d.glob('*.nii*'))
        if nii:
            per_sub[r['subject']][r['datatype_slug']] = nii[0]
            per_sub[r['subject']]['_tags'] = r['tags']
    return per_sub


def geometria(img):
    shape = tuple(int(x) for x in img.shape[:3])
    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
    orient = ''.join(nib.aff2axcodes(img.affine))
    return shape, spacing, orient


def stessa_geometria(a, b, tol=1e-3):
    return (a.shape[:3] == b.shape[:3]
            and bool(np.allclose(a.affine, b.affine, atol=tol)))


def analizza_caso(subject, files):
    """Una riga di manifest. Ritorna None se manca il T1."""
    t1_path = files.get('anat-t1w')
    if t1_path is None:
        return None

    t1 = nib.load(str(t1_path), mmap=True)
    shape, spacing, orient = geometria(t1)
    voxel_vol = float(np.prod(spacing))

    tags = files.get('_tags', '').split('|')
    coorte = tags[0] if tags else ''
    condizione = tags[1] if len(tags) > 1 else ''
    split = 'validation' if 'validation' in tags else ('test' if 'test' in tags else 'train')

    riga = {
        'dataset': 'DBB',
        'case_id': f'sub-{subject}',
        'group': f'{coorte}_{condizione}',
        'source_split': split,
        'cohort': coorte,
        'condition': condizione,
        'sx': spacing[0], 'sy': spacing[1], 'sz': spacing[2],
        'nx': shape[0], 'ny': shape[1], 'nz': shape[2],
        'fov_x': shape[0] * spacing[0], 'fov_y': shape[1] * spacing[1], 'fov_z': shape[2] * spacing[2],
        'orientation': orient,
        'voxel_vol_mm3': voxel_vol,
        'anisotropy': max(spacing) / min(spacing),
        'has_ch1': False,
        'ch1_geom_ok': '',
    }

    parc_path = files.get('parcellation-volume')
    mask_path = files.get('mask')
    riga['has_parc'] = parc_path is not None
    riga['has_mask'] = mask_path is not None

    geom_ok = True
    if parc_path is not None:
        parc_img = nib.load(str(parc_path), mmap=True)
        riga['parc_geom_ok'] = stessa_geometria(t1, parc_img)
        geom_ok &= riga['parc_geom_ok']
        parc = np.asanyarray(parc_img.dataobj)
        valori = np.unique(parc)
        valori = valori[valori > 0]
        riga['label_values'] = '/'.join(['0'] + [str(int(v)) for v in valori])
        conteggi = Counter()
        for v in valori:
            conteggi[int(v)] = int((parc == v).sum())
        for k, nome in CLASSI.items():
            n = conteggi.get(k, 0)
            riga[f'vox_{k}'] = n
            riga[f'ml_{k}'] = n * voxel_vol / 1000.0
        totale = sum(conteggi.values())
        riga['n_lesion_vox'] = totale          # colonna comune: qui e' il tessuto etichettato
        riga['lesion_ml'] = totale * voxel_vol / 1000.0
        riga['frazione_etichettata'] = totale / float(np.prod(shape))
    else:
        riga['parc_geom_ok'] = ''
        riga['label_values'] = ''
        riga['n_lesion_vox'] = ''
        riga['lesion_ml'] = ''

    if mask_path is not None:
        mask_img = nib.load(str(mask_path), mmap=True)
        riga['mask_geom_ok'] = stessa_geometria(t1, mask_img)
        geom_ok &= riga['mask_geom_ok']
        mask = np.asanyarray(mask_img.dataobj) > 0
        riga['brain_ml'] = float(mask.sum()) * voxel_vol / 1000.0
    else:
        riga['mask_geom_ok'] = ''
        riga['brain_ml'] = ''

    riga['geom_ok'] = geom_ok

    dati = np.asanyarray(t1.dataobj)
    riga['intensita_min'] = float(dati.min())
    riga['intensita_max'] = float(dati.max())
    riga['frazione_nonzero'] = float((dati != 0).mean())
    return riga


def target_spacing_nnunet(spacings, shapes):
    """Riproduce la regola di nnU-Net v2 per la target spacing."""
    sp = np.median(np.array(spacings), axis=0)
    sh = np.median(np.array(shapes), axis=0)
    peggiore = int(np.argmax(sp))
    altri = [i for i in range(3) if i != peggiore]
    cond_spacing = bool(sp[peggiore] > 3.0 * min(sp[i] for i in altri))
    cond_voxel = bool(sh[peggiore] * 3.0 < min(sh[i] for i in altri))
    override = cond_spacing and cond_voxel
    target = list(sp)
    if override:
        target[peggiore] = float(np.percentile([s[peggiore] for s in spacings], 10))
    return {
        'spacing_mediana': [round(float(x), 4) for x in sp],
        'shape_mediana': [int(x) for x in sh],
        'asse_peggiore': 'xyz'[peggiore],
        'condizione_spacing': cond_spacing,
        'condizione_voxel': cond_voxel,
        'override_anisotropico': override,
        'target_spacing': [round(float(x), 4) for x in target],
        'resampling_separato_z': bool(max(sp) / min(sp) > 3.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='analizza solo i primi N casi (debug)')
    args = ap.parse_args()

    rows = manifest_rows()
    per_sub = indicizza(rows)
    soggetti = sorted(per_sub)
    if args.limit:
        con_t1 = [s for s in soggetti if 'anat-t1w' in per_sub[s]]
        soggetti = con_t1[:args.limit]

    manifest = []
    for i, s in enumerate(soggetti, 1):
        riga = analizza_caso(s, per_sub[s])
        if riga is not None:
            manifest.append(riga)
        if i % 50 == 0:
            print(f'  {i}/{len(soggetti)} soggetti scansionati, {len(manifest)} con T1')

    # --- copertura: quanti soggetti hanno cosa ---
    copertura = Counter()
    for s in per_sub:
        chiavi = {k for k in per_sub[s] if not k.startswith('_')}
        copertura['totale'] += 1
        copertura['con_t1'] += 'anat-t1w' in chiavi
        copertura['con_parc'] += 'parcellation-volume' in chiavi
        copertura['con_mask'] += 'mask' in chiavi
        if 'anat-t1w' not in chiavi and 'parcellation-volume' in chiavi:
            copertura['label_senza_immagine'] += 1

    coorti_senza_t1 = Counter()
    for s in per_sub:
        if 'anat-t1w' not in per_sub[s]:
            coorti_senza_t1[per_sub[s].get('_tags', '').split('|')[0]] += 1

    out_manifest = ROOT / 'data' / 'processed' / 'manifests' / 'DBB_cases.csv'
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    campi = list(manifest[0].keys())
    with open(out_manifest, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=campi)
        w.writeheader()
        w.writerows(manifest)

    spacings = [(r['sx'], r['sy'], r['sz']) for r in manifest]
    shapes = [(r['nx'], r['ny'], r['nz']) for r in manifest]
    ts = target_spacing_nnunet(spacings, shapes)

    per_gruppo = defaultdict(list)
    for r in manifest:
        per_gruppo[r['group']].append(r)

    gruppi = []
    for g, rs in sorted(per_gruppo.items()):
        sp = np.median([[r['sx'], r['sy'], r['sz']] for r in rs], axis=0)
        sh = np.median([[r['nx'], r['ny'], r['nz']] for r in rs], axis=0)
        gruppi.append({
            'gruppo': g,
            'n': len(rs),
            'split': rs[0]['source_split'],
            'spacing': [round(float(x), 4) for x in sp],
            'shape': [int(x) for x in sh],
            'brain_ml_mediana': round(float(np.median([r['brain_ml'] for r in rs if r['brain_ml'] != ''])), 1),
        })

    volumi_classe = {}
    for k, nome in CLASSI.items():
        vals = [r[f'ml_{k}'] for r in manifest if f'ml_{k}' in r]
        volumi_classe[nome] = {
            'valore_voxel': k,
            'mediana_ml': round(float(np.median(vals)), 1),
            'min_ml': round(float(np.min(vals)), 1),
            'max_ml': round(float(np.max(vals)), 1),
            'casi_con_classe_vuota': int(sum(1 for v in vals if v == 0)),
        }

    orientamenti = Counter(r['orientation'] for r in manifest)
    shapes_distinte = Counter(f"{r['nx']}x{r['ny']}x{r['nz']}" for r in manifest)
    spacing_distinti = Counter(f"{r['sx']:.4g} x {r['sy']:.4g} x {r['sz']:.4g}" for r in manifest)
    geom_ko = [r['case_id'] for r in manifest if not r['geom_ok']]
    etichette_distinte = Counter(r['label_values'] for r in manifest)

    report = {
        'sorgente': str(DBB),
        'copertura': dict(copertura),
        'coorti_senza_t1': dict(coorti_senza_t1),
        'casi_analizzati': len(manifest),
        'orientamenti': dict(orientamenti),
        'shapes': dict(shapes_distinte),
        'spacings': dict(spacing_distinti),
        'geometria_incoerente': geom_ko,
        'etichette': dict(etichette_distinte),
        'target_spacing_nnunet': ts,
        'gruppi': gruppi,
        'volumi_per_classe': volumi_classe,
        'split_ufficiale': dict(Counter(r['source_split'] for r in manifest)),
        'frazione_nonzero_mediana': round(float(np.median([r['frazione_nonzero'] for r in manifest])), 4),
        'brain_ml': {
            'mediana': round(float(np.median([r['brain_ml'] for r in manifest])), 1),
            'min': round(float(np.min([r['brain_ml'] for r in manifest])), 1),
            'max': round(float(np.max([r['brain_ml'] for r in manifest])), 1),
        },
    }

    out = ROOT / 'reports'
    (out / 'dbb_analysis.json').write_text(json.dumps(report, indent=2, default=str) + '\n')

    L = ['# DBB - analisi del terzo dataset', '',
         f'Generato da `scripts/analyze_dbb.py`. Read-only: nessun dato modificato.',
         f'Manifest per caso: `data/processed/manifests/DBB_cases.csv`.', '',
         '## Copertura',
         '',
         '| | soggetti |', '| --- | --- |',
         f"| nel progetto brainlife | {copertura['totale']} |",
         f"| con parcellazione (label) | {copertura['con_parc']} |",
         f"| con maschera cerebrale | {copertura['con_mask']} |",
         f"| **con T1 (usabili)** | **{copertura['con_t1']}** |",
         f"| con label ma senza immagine | {copertura['label_senza_immagine']} |",
         '',
         'I soggetti senza T1 appartengono a coorti ad accesso ristretto: '
         + ', '.join(f'{k} ({v})' for k, v in sorted(coorti_senza_t1.items()) if k) + '.',
         'Le loro label sono pubbliche ma le immagini no, quindi sono inutilizzabili.',
         '',
         '## Geometria', '',
         f"- casi analizzati: **{len(manifest)}**",
         f"- orientamento: `{dict(orientamenti)}`",
         f"- shape distinte: `{dict(shapes_distinte)}`",
         f"- spacing distinti: `{dict(spacing_distinti)}`",
         f"- casi con immagine e label non allineate: **{len(geom_ko)}**"
         + (f" ({', '.join(geom_ko[:5])})" if geom_ko else ''),
         f"- frazione mediana di voxel non nulli: **{report['frazione_nonzero_mediana']}**",
         '',
         '### Target spacing prevista (regola nnU-Net v2)', '',
         f"- spacing mediano: `{ts['spacing_mediana']}` mm, shape mediana `{ts['shape_mediana']}`",
         f"- asse peggiore: `{ts['asse_peggiore']}`",
         f"- override anisotropico applicato: **{ts['override_anisotropico']}**",
         f"- **target spacing: `{ts['target_spacing']}` mm**",
         '',
         '## Split ufficiale del benchmark', '',
         '| Gruppo | n | split | spacing (mm) | shape | volume cervello mediano (ml) |',
         '| --- | --- | --- | --- | --- | --- |']
    for g in gruppi:
        L.append(f"| {g['gruppo']} | {g['n']} | {g['split']} | "
                 f"{g['spacing'][0]:g} x {g['spacing'][1]:g} x {g['spacing'][2]:g} | "
                 f"{g['shape'][0]}x{g['shape'][1]}x{g['shape'][2]} | {g['brain_ml_mediana']} |")
    L += ['',
          '## Classi di tessuto', '',
          f"Valori di label osservati: `{dict(etichette_distinte)}`", '',
          '| Valore | Classe | mediana (ml) | min | max | casi con classe vuota |',
          '| --- | --- | --- | --- | --- | --- |']
    for nome, v in volumi_classe.items():
        L.append(f"| {v['valore_voxel']} | {nome} | {v['mediana_ml']} | {v['min_ml']} | "
                 f"{v['max_ml']} | {v['casi_con_classe_vuota']} |")
    L += ['',
          f"Volume cerebrale (maschera fornita): mediana **{report['brain_ml']['mediana']} ml**, "
          f"da {report['brain_ml']['min']} a {report['brain_ml']['max']} ml.",
          '']
    (out / 'dbb_analysis.md').write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
