# Dataset exploration

## WMH
- Path: `/data01/miccai_mri_data/miccai_nifti`
- Exists: `True`
- NIfTI: **594**
- Training-like files: **0**
- Testing-like files: **0**
- Roles: `{'FLAIR': 297, 'T1': 198, 'label_or_lesion': 99}`
- Sample spacings: `{'(1.3, 1.2109, 3.0)': 10, '(0.9766, 0.9766, 1.5)': 16, '(1.2999, 1.2109, 3.0)': 4, '(1.3001, 1.2109, 3.0)': 20}`

## ATLAS
- Path: `/data01/atlas_mri_data/ATLAS_2`
- Exists: `True`
- NIfTI: **2266**
- Training-like files: **1965**
- Testing-like files: **300**
- Roles: `{'other': 1, 'T1': 955, 'label_or_lesion': 1310}`
- Sample spacings: `{'(1.0, 1.0, 1.0)': 50}`

## DBB
- Path: `/home/ccantini/lesion_detection_project/data/raw/DBB`
- Exists: `True`
- NIfTI: **2865**
- Training-like files: **0**
- Testing-like files: **0**
- Roles: `{'label_or_lesion': 1666, 'other': 955, 'T1': 244}`
- Sample spacings: `{'(1.0, 1.0, 1.0)': 50}`

## AINCP
- Path: `None`
- Exists: `False`

> Nota: questo report e' un inventario grezzo. Conta *tutti* i file NIfTI, inclusi i
> sottoprodotti (`_BiasField`, T1 non allineato), e campiona gli header dei primi
> `--sample-limit` file in ordine alfabetico. Gli spacing qui elencati non sono quindi
> quelli dei canali effettivamente usati per il training: per l'analisi per caso
> appaiato vedere `reports/dataset_preparation.md`.

## Preprocessing candidates

- Verify affine, orientation, shape and image/label alignment per case.
- Use nnU-Net standard resampling and foreground z-score normalization as baseline.
- Use nearest-neighbour interpolation for labels and linear/cubic interpolation for images.
- Do not resample manually before nnU-Net: a double interpolation degrades small lesions.
- WMH: start with FLAIR-only and optionally FLAIR+T1 (`*_T1w_BiasCorr_2FLAIR_Aligned`).
- WMH: label 2 is "other pathology" and is excluded from the official evaluation.
  Map it to background (or to the nnU-Net ignore label); do NOT merge it into label 1.
- ATLAS: start with T1w-only. The two mask variants (`_mask`, `_mask_value2`) share the
  same binary support and differ only in the encoded value; use `_mask`.
- ATLAS: binarize masks with `> 0`, not `== 1`: at least one case is encoded 0/0.01.
- ATLAS `Testing/` has no labels: it cannot serve as a local test set.
- DBB: T1-only, 6 tissue classes (not lesions). Only 244 of 955 subjects have an image;
  31 of those are not in MNI space and 12 are RAS instead of LAS, hence
  `NibabelIOWithReorient` in the converted dataset.
- Keep training/test and site splits separate to prevent leakage.
