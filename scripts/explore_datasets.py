#!/usr/bin/env python3
"""Esplorazione read-only di struttura e header NIfTI."""
import argparse, json
from collections import Counter
from pathlib import Path
import nibabel as nib

def classify(name):
    n = name.lower()
    if any(x in n for x in ('mask','label','seg','lesion','wmh')): return 'label_or_lesion'
    if 'flair' in n: return 'FLAIR'
    if any(x in n for x in ('t1','t1w','3dt1')): return 'T1'
    return 'other'

def inspect(name, spec, limit):
    # path puo' essere null: e' il caso di un dataset atteso ma non ancora consegnato
    root = Path(spec.get('path') or '/nonexistent')
    out = {'path': spec.get('path'), 'description': spec.get('description',''), 'exists': root.exists()}
    if not root.exists(): return out
    files = sorted(p for p in root.rglob('*') if p.is_file() and (p.name.endswith('.nii') or p.name.endswith('.nii.gz')))
    out['nifti_files'] = len(files); out['roles'] = dict(Counter(classify(p.name) for p in files))
    out['training_files'] = sum('training' in {x.lower() for x in p.parts} for p in files)
    out['testing_files'] = sum('testing' in {x.lower() for x in p.parts} or 'test' in {x.lower() for x in p.parts} for p in files)
    shapes, spacings, samples = Counter(), Counter(), []
    for p in files[:limit]:
        try:
            img = nib.load(str(p), mmap=True); shape = tuple(int(x) for x in img.shape[:3]); spacing = tuple(round(float(x),4) for x in img.header.get_zooms()[:3])
            shapes[str(shape)] += 1; spacings[str(spacing)] += 1
            samples.append({'file': str(p.relative_to(root)), 'shape': shape, 'spacing_mm': spacing, 'role': classify(p.name)})
        except Exception as e: samples.append({'file': str(p.relative_to(root)), 'error': str(e)})
    out.update({'sampled': len(samples), 'sample_shapes': dict(shapes), 'sample_spacings_mm': dict(spacings), 'samples': samples})
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--sample-limit', type=int, default=50); a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); report = {'datasets': {n: inspect(n,s,a.sample_limit) for n,s in cfg['datasets'].items()}}
    out = Path(cfg['project_root'])/'reports'; out.mkdir(parents=True, exist_ok=True)
    (out/'dataset_exploration.json').write_text(json.dumps(report, indent=2, default=str)+'\n')
    lines = ['# Dataset exploration', '']
    for n,r in report['datasets'].items():
        lines += [f'## {n}', f"- Path: `{r['path']}`", f"- Exists: `{r['exists']}`"]
        if r['exists']: lines += [f"- NIfTI: **{r['nifti_files']}**", f"- Training-like files: **{r['training_files']}**", f"- Testing-like files: **{r['testing_files']}**", f"- Roles: `{r['roles']}`", f"- Sample spacings: `{r['sample_spacings_mm']}`"]
        lines.append('')
    lines += ['> Nota: questo report e\' un inventario grezzo. Conta *tutti* i file NIfTI, inclusi i',
              '> sottoprodotti (`_BiasField`, T1 non allineato), e campiona gli header dei primi',
              '> `--sample-limit` file in ordine alfabetico. Gli spacing qui elencati non sono quindi',
              '> quelli dei canali effettivamente usati per il training: per l\'analisi per caso',
              '> appaiato vedere `reports/dataset_preparation.md`.', '',
              '## Preprocessing candidates', '',
              '- Verify affine, orientation, shape and image/label alignment per case.',
              '- Use nnU-Net standard resampling and foreground z-score normalization as baseline.',
              '- Use nearest-neighbour interpolation for labels and linear/cubic interpolation for images.',
              '- Do not resample manually before nnU-Net: a double interpolation degrades small lesions.',
              '- WMH: start with FLAIR-only and optionally FLAIR+T1 (`*_T1w_BiasCorr_2FLAIR_Aligned`).',
              '- WMH: label 2 is "other pathology" and is excluded from the official evaluation.',
              '  Map it to background (or to the nnU-Net ignore label); do NOT merge it into label 1.',
              '- ATLAS: start with T1w-only. The two mask variants (`_mask`, `_mask_value2`) share the',
              '  same binary support and differ only in the encoded value; use `_mask`.',
              '- ATLAS: binarize masks with `> 0`, not `== 1`: at least one case is encoded 0/0.01.',
              '- ATLAS `Testing/` has no labels: it cannot serve as a local test set.',
              '- DBB: T1-only, 6 tissue classes (not lesions). Only 244 of 955 subjects have an image;',
              '  31 of those are not in MNI space and 12 are RAS instead of LAS, hence',
              '  `NibabelIOWithReorient` in the converted dataset.',
              '- Keep training/test and site splits separate to prevent leakage.', '']
    (out/'dataset_exploration.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
if __name__ == '__main__': main()
