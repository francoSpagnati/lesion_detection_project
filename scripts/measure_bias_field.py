#!/usr/bin/env python3
"""Fase 0: misura il bias di campo residuo su WMH, ATLAS e DBB.

Non modifica i dati. Per ogni immagine campionata stima il campo di bias con N4
(ANTs, via antspyx) e ne misura l'ampiezza *dentro la testa*, per decidere se una
correzione N4 esplicita sia giustificata prima della conversione nnU-Net.

Il campo di bias e' moltiplicativo e liscio: un campo "piatto" (rapporto p99/p1
vicino a 1.0) significa che l'immagine e' gia' corretta e che rieseguire N4
toglierebbe contrasto anatomico vero invece di artefatto.

Uso:
    python scripts/measure_bias_field.py [--n 10] [--seed 0]
Output:
    reports/bias_field_assessment.json
    reports/bias_field_assessment.md
"""
import argparse
import json
import random
from pathlib import Path

import ants
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]

# Soglie di giudizio sul rapporto p99/p1 del campo stimato dentro la testa.
NEGLIGIBLE = 1.10
MODERATE = 1.30

# Margini di erosione in millimetri a cui si misura il campo.
MARGINS = (0, 3, 6, 12)


def sources():
    """Percorsi campionabili per dataset, piu' l'eventuale maschera cerebrale."""
    raw = ROOT / 'data' / 'processed' / 'nnUNet_raw'
    return {
        # canale 0000 = FLAIR, gia' distribuito come *_BiasCorr dagli organizzatori
        'WMH': {
            'images': sorted((raw / 'Dataset501_WMH' / 'imagesTr').glob('*_0000.nii.gz')),
            'masks': None,
            'nota': "file gia' corretti dagli organizzatori (_BiasCorr)",
        },
        'ATLAS': {
            'images': sorted((raw / 'Dataset502_ATLAS' / 'imagesTr').glob('*_0000.nii.gz')),
            'masks': None,
            'nota': 'T1w in spazio MNI, preprocessing degli autori non documentato come N4',
        },
        'DBB': {
            'images': sorted((ROOT / 'data' / 'raw' / 'DBB').glob('sub-*/anat-t1w*/*/t1.nii.gz')),
            'masks': 'dbb',
            'nota': 'T1 pediatrici in spazio MNI, maschera cerebrale fornita dal dataset',
        },
    }


def dbb_mask_for(image_path):
    """La maschera cerebrale DBB sta in sub-XXXX/mask/<dataset_id>/mask.nii.gz."""
    subject_dir = image_path.parents[2]
    found = sorted(subject_dir.glob('mask/*/mask.nii.gz'))
    return found[0] if found else None


def eroded(mask_bool, spacing, margin_mm):
    """Erode la maschera di un margine *fisico*, non in voxel.

    Serve perche' i tre dataset hanno spacing molto diversi (WMH ha fette da 3 mm):
    erodere di N voxel toglierebbe 3N mm su WMH e N mm su ATLAS, rendendo i numeri
    non confrontabili.
    """
    if margin_mm <= 0:
        return mask_bool
    radii = [max(1, int(round(margin_mm / s))) for s in spacing]
    struct = np.ones([2 * r + 1 for r in radii], dtype=bool)
    return ndimage.binary_erosion(mask_bool, structure=struct)


def measure(image_path, mask_path=None, margins=MARGINS):
    """Stima il campo di bias e ne misura l'ampiezza a profondita' crescenti.

    Restituisce un *profilo di erosione* invece di un numero solo. La forma del
    profilo e' il vero elemento diagnostico:

    - rapporto che resta alto man mano che si entra nella testa  -> bias reale,
      liscio e globale, che N4 puo' correggere;
    - rapporto che crolla appena ci si stacca dal bordo -> l'ampiezza veniva dal
      contorno della maschera (scalpo, grasso sottocutaneo, aria), non da un bias
      interno al cervello: correggere non servirebbe.
    """
    img = ants.image_read(str(image_path))
    if mask_path is not None:
        mask = ants.threshold_image(ants.image_read(str(mask_path)), 0.5, 1e9)
        mask_kind = 'cervello (fornita dal dataset)'
    else:
        # in assenza di maschera cerebrale si usa la testa intera: il campo va
        # comunque misurato dove c'e' segnale, non nell'aria di fondo
        mask = ants.get_mask(img)
        mask_kind = 'testa (soglia di intensita)'

    field = ants.n4_bias_field_correction(img, mask=mask, return_bias_field=True)
    f = field.numpy()
    base = mask.numpy() > 0
    spacing = img.spacing

    profile = {}
    for margin in margins:
        m = eroded(base, spacing, margin)
        vals = f[m]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 1000:
            continue
        p1, p50, p99 = (float(x) for x in np.percentile(vals, [1, 50, 99]))
        profile[margin] = {
            'voxel': int(m.sum()),
            'p1_rel': p1 / p50,
            'p99_rel': p99 / p50,
            'rapporto_p99_p1': p99 / p1,
            'cv': float(vals.std() / vals.mean()),
        }

    if not profile:
        return None
    deepest = max(profile)
    voxel_mm3 = float(np.prod(spacing))
    return {
        'file': str(image_path.relative_to(ROOT)),
        'spacing_mm': [round(float(s), 4) for s in spacing],
        'maschera': mask_kind,
        'volume_maschera_mm3': round(float(base.sum()) * voxel_mm3),
        'profilo': profile,
        'margine_piu_profondo_mm': deepest,
        'rapporto_bordo': profile[min(profile)]['rapporto_p99_p1'],
        'rapporto_interno': profile[deepest]['rapporto_p99_p1'],
    }


def verdict(ratio_interno, ratio_bordo):
    """Il giudizio si basa sul campo *interno*; il bordo serve solo a spiegarlo."""
    calo = ratio_bordo - ratio_interno
    if ratio_interno < NEGLIGIBLE:
        nota = "campo piatto all'interno"
        if calo > 0.15:
            nota += ", l'ampiezza al bordo era artefatto di contorno"
        return 'trascurabile', 'nessuna correzione N4: ' + nota
    if ratio_interno < MODERATE:
        return 'moderato', 'N4 opzionale, solo come variante misurata contro la baseline'
    return 'sostanziale', 'N4 giustificato prima della conversione nnU-Net'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10, help='immagini campionate per dataset')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    report = {'parametri': {'n_per_dataset': args.n, 'seed': args.seed}, 'datasets': {}}

    for name, spec in sources().items():
        images = spec['images']
        entry = {'nota': spec['nota'], 'immagini_disponibili': len(images), 'casi': []}
        if not images:
            entry['errore'] = 'nessuna immagine trovata'
            report['datasets'][name] = entry
            print('%-6s nessuna immagine trovata' % name, flush=True)
            continue

        sample = rng.sample(images, min(args.n, len(images)))
        for path in sample:
            mask_path = dbb_mask_for(path) if spec['masks'] == 'dbb' else None
            try:
                res = measure(path, mask_path)
            except Exception as exc:
                res = {'file': str(path.relative_to(ROOT)), 'errore': repr(exc)}
            if res:
                entry['casi'].append(res)
                print('%-6s %-40s bordo=%.3f  interno(%dmm)=%.3f' % (
                    name, Path(res['file']).name[:40], res['rapporto_bordo'],
                    res['margine_piu_profondo_mm'], res['rapporto_interno'],
                ), flush=True)

        # Aggregare il "margine piu' profondo raggiunto" caso per caso sarebbe
        # sbagliato: le teste piccole si fermano a 6 mm e le grandi arrivano a 12,
        # e il rapporto cresce quanto meno si erode. Si aggrega quindi *a parita' di
        # margine*, tenendo solo i livelli coperti da tutti i casi del dataset.
        per_margine = {}
        for margin in MARGINS:
            vals = [c['profilo'][str(margin)]['rapporto_p99_p1']
                    if str(margin) in c.get('profilo', {})
                    else c['profilo'][margin]['rapporto_p99_p1']
                    for c in entry['casi']
                    if margin in c.get('profilo', {}) or str(margin) in c.get('profilo', {})]
            if vals:
                per_margine[margin] = {
                    'casi': len(vals),
                    'mediano': float(np.median(vals)),
                    'max': float(np.max(vals)),
                }
        entry['per_margine'] = per_margine

        # Confronto **appaiato**: solo i casi che arrivano al margine piu' profondo,
        # seguiti attraverso tutti i margini. Cosi' il calo con la profondita' non e'
        # inquinato dal fatto che i casi con testa piccola escono per strada.
        profondo = max((m for m in MARGINS
                        if any(m in c.get('profilo', {}) for c in entry['casi'])),
                       default=None)
        if profondo is not None:
            appaiati = [c for c in entry['casi'] if profondo in c.get('profilo', {})]
            entry['appaiato'] = {
                'margine_max_mm': profondo,
                'casi': len(appaiati),
                'mediana_per_margine': {
                    m: float(np.median([c['profilo'][m]['rapporto_p99_p1']
                                        for c in appaiati if m in c['profilo']]))
                    for m in MARGINS
                    if all(m in c['profilo'] for c in appaiati)
                },
            }
            med = entry['appaiato']['mediana_per_margine']
            entry['interno_mediano'] = med[profondo]
            entry['bordo_mediano'] = med[min(med)]
            entry['margine_riferimento_mm'] = profondo
            entry['interno_max'] = float(np.max(
                [c['profilo'][profondo]['rapporto_p99_p1'] for c in appaiati]))
            entry['giudizio'], entry['azione'] = verdict(
                entry['interno_mediano'], entry['bordo_mediano'])
            entry['maschera'] = appaiati[0].get('maschera')
            entry['volume_maschera_mm3_mediano'] = float(np.median(
                [c['volume_maschera_mm3'] for c in entry['casi']
                 if 'volume_maschera_mm3' in c]))
        report['datasets'][name] = entry

    out = ROOT / 'reports'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'bias_field_assessment.json').write_text(json.dumps(report, indent=2) + '\n')

    lines = [
        '# Fase 0 - bias di campo residuo', '',
        "Generato da `scripts/measure_bias_field.py`. Nessun dato e' stato modificato.", '',
        "Il campo di bias e' stimato con N4 (ANTs) e misurato dentro la testa. `p99/p1` e' il",
        "rapporto fra il valore alto e quello basso del campo: 1.00 = campo piatto, nessun bias.", '',
        'La misura e\' ripetuta erodendo la maschera di un margine **fisico** crescente. Il valore',
        '"bordo" usa tutta la maschera, il valore "interno" solo la parte piu\' profonda: se il',
        'secondo e\' molto piu\' basso del primo, l\'ampiezza veniva dal contorno (scalpo, grasso)',
        'e non da un bias interno al cervello.', '',
        'Il confronto e\' **appaiato**: la tabella segue i soli casi che arrivano al margine piu\'',
        'profondo, misurati a tutte le profondita\'. Aggregare il "margine massimo raggiunto" caso',
        'per caso falserebbe il confronto, perche\' le teste piccole si fermano prima e mostrano',
        'quindi rapporti sistematicamente piu\' alti.', '',
        '| Dataset | Casi appaiati | Margine max | p99/p1 bordo | p99/p1 interno | max interno | Giudizio | Azione |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for name, e in report['datasets'].items():
        if 'interno_mediano' in e:
            lines.append('| %s | %d | %d mm | %.3f | %.3f | %.3f | %s | %s |' % (
                name, e['appaiato']['casi'], e['margine_riferimento_mm'], e['bordo_mediano'],
                e['interno_mediano'], e['interno_max'], e['giudizio'], e['azione']))
        else:
            lines.append('| %s | 0 | - | - | - | - | - | %s |' % (name, e.get('errore', 'n/d')))

    lines += ['', '## Profilo di erosione appaiato (mediana sugli stessi casi)', '',
              '| Dataset | ' + ' | '.join('%d mm' % m for m in MARGINS) + ' |',
              '| --- | ' + ' | '.join(['---'] * len(MARGINS)) + ' |']
    for name, e in report['datasets'].items():
        med = e.get('appaiato', {}).get('mediana_per_margine', {})
        cells = ['%.3f' % med[m] if m in med else '-' for m in MARGINS]
        lines.append('| %s | %s |' % (name, ' | '.join(cells)))

    lines += [
        '', '## Limite della misura', '',
        'La maschera non ha la stessa natura nei tre dataset, e questo va tenuto presente prima di',
        'confrontare i numeri a profondita\' bassa:', '',
        '| Dataset | Maschera | Volume mediano |',
        '| --- | --- | --- |',
    ]
    for name, e in report['datasets'].items():
        if 'maschera' in e:
            lines.append('| %s | %s | %s mm³ |' % (
                name, e['maschera'], f"{e['volume_maschera_mm3_mediano']:,.0f}".replace(',', '\u202f')))
    lines += [
        '',
        'Dove la maschera e\' la **testa** (WMH, ATLAS) i primi millimetri di erosione cadono ancora',
        'su cranio, scalpo e grasso sottocutaneo, dove N4 estrapola male: il rapporto misurato li\'',
        'sovrastima il bias interno al cervello. Dove la maschera e\' gia\' il **cervello** (DBB) la',
        'misura parte pulita. Il confronto fra dataset e\' quindi affidabile solo al margine piu\'',
        'profondo. Una misura definitiva per WMH e ATLAS richiederebbe prima lo skull stripping.',
        '', '## Note per dataset', '']
    for name, e in report['datasets'].items():
        lines += ['- **%s**: %s' % (name, e['nota'])]
    lines.append('')
    (out / 'bias_field_assessment.md').write_text('\n'.join(lines))
    print('\nscritto reports/bias_field_assessment.{json,md}', flush=True)


if __name__ == '__main__':
    main()
