# Dataset preparation

Generato da `notebooks/dataset_preparation.ipynb` il 2026-08-13. Seed `42`, 5 fold.

Report dei dati **appaiati** (immagine + label per caso), a differenza di `dataset_exploration.md` che e' un inventario grezzo di tutti i file.

## WMH (Dataset501)

- Sorgente: `/data01/miccai_mri_data/miccai_nifti`
- Casi appaiati: **99**  |  gruppi: **4**  (sito/scanner)
- Immagini senza label (test ufficiale): **0**
- Orientamento: `LPS`
- Spacing distinti: **7**

### Risoluzione spaziale

| Gruppo | n | spacing (mm) | shape mediana | voxel mm3 | anisotropia |
| --- | --- | --- | --- | --- | --- |
| Amsterdam_GE1T5 | 10 | 1.3 x 1.211 x 3 | 128x256x103 | 4.723 | 2.48 |
| Amsterdam_GE3T | 49 | 1.2 x 0.9766 x 3 | 132x256x83 | 3.516 | 3.07 |
| Singapore | 20 | 1 x 1 x 3 | 232x256x48 | 3.000 | 3.00 |
| Utrecht | 20 | 0.9583 x 0.9583 x 3 | 240x240x48 | 2.755 | 3.13 |

### Target spacing prevista (regola nnU-Net v2)

- spacing mediano: `[1.2, 0.9766, 3.0]` mm, shape mediana `[132, 256, 83]`
- asse peggiore: `z`
- condizione spacing: `3 > 3.0*1.2=3.6 -> False`
- condizione voxel: `83*3.0 < 132 -> False`
- override anisotropico applicato: **False**
- **target spacing: `[1.2, 0.9766, 3.0]` mm**
- resampling separato lungo z (anisotropia > 3.0): **True**

### Carico lesionale

- volume lesione (ml): min 0.786, p10 1.61, mediana 10.93, p90 40.12, max 74.99
- casi con lesione vuota: 0
- casi sotto 100 voxel alla risoluzione target: 0 / 99

### Label

- valori osservati: `{'0/1': 65, '0/1/2': 34}`
- label 2 ("altra patologia") presente in **34/99** casi, volume max 32.90 ml
- politica applicata: `exclude`

### Split

- stratificazione: `group | terzile lesion_ml (bins=3)`
- development (5-fold CV): **79**  |  test interno: **20**  |  test ufficiale senza label: **0**
- fold (train/val): 63/16, 63/16, 63/16, 63/16, 64/15
- output: `data/processed/nnUNet_raw/Dataset501_WMH`

## ATLAS (Dataset502)

- Sorgente: `/data01/atlas_mri_data/ATLAS_2`
- Casi appaiati: **655**  |  gruppi: **33**  (coorte Rxxx)
- Immagini senza label (test ufficiale): **300**
- Orientamento: `RAS`
- Spacing distinti: **1**

### Risoluzione spaziale

Risoluzione **uniforme** su tutti i 33 gruppi: una sola riga riassuntiva.

| Gruppo | n | spacing (mm) | shape mediana | voxel mm3 | anisotropia |
| --- | --- | --- | --- | --- | --- |
| tutti (33 gruppi) | 655 | 1 x 1 x 1 | 197x233x189 | 1.000 | 1.00 |

### Target spacing prevista (regola nnU-Net v2)

- spacing mediano: `[1.0, 1.0, 1.0]` mm, shape mediana `[197, 233, 189]`
- asse peggiore: `x`
- condizione spacing: `1 > 3.0*1=3 -> False`
- condizione voxel: `197*3.0 < 189 -> False`
- override anisotropico applicato: **False**
- **target spacing: `[1.0, 1.0, 1.0]` mm**
- resampling separato lungo z (anisotropia > 3.0): **False**

### Carico lesionale

- volume lesione (ml): min 0.013, p10 0.38, mediana 5.29, p90 106.09, max 496.66
- casi con lesione vuota: 0
- casi sotto 100 voxel alla risoluzione target: 19 / 655

### Label

- valori osservati: `{'0/1': 654, '0/0.01': 1}`
- le due varianti `_mask` / `_mask_value2` condividono lo stesso supporto binario; usata `_mask`
- codifiche anomale: ['sub-r039s002'] (binarizzazione con `> 0`)

### Split

- stratificazione: `group | terzile lesion_ml (bins=3)`
- development (5-fold CV): **556**  |  test interno: **99**  |  test ufficiale senza label: **300**
- fold (train/val): 444/112, 445/111, 445/111, 445/111, 445/111
- output: `data/processed/nnUNet_raw/Dataset502_ATLAS`

## Riproduzione

Eseguire `notebooks/dataset_preparation.ipynb` dall'inizio alla fine. I manifest sono in cache in `data/processed/manifests/`: per ricalcolarli da zero impostare `FORCE_RECOMPUTE = True` nella cella di configurazione.
