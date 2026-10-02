# DBB - analisi del terzo dataset

Generato da `scripts/analyze_dbb.py`. Read-only: nessun dato modificato.
Manifest per caso: `data/processed/manifests/DBB_cases.csv`.

## Copertura

| | soggetti |
| --- | --- |
| nel progetto brainlife | 955 |
| con parcellazione (label) | 955 |
| con maschera cerebrale | 954 |
| **con T1 (usabili)** | **244** |
| con label ma senza immagine | 711 |

I soggetti senza T1 appartengono a coorti ad accesso ristretto: C-MIND (413), NIH-PED (298).
Le loro label sono pubbliche ma le immagini no, quindi sono inutilizzabili.

## Geometria

- casi analizzati: **244**
- orientamento: `{'LAS': 232, 'RAS': 12}`
- shape distinte: `{'182x218x182': 213, '170x256x256': 2, '160x240x240': 6, '155x240x240': 4, '150x224x224': 3, '155x224x224': 3, '160x256x256': 2, '150x240x240': 2, '140x224x224': 4, '151x240x240': 1, '145x224x224': 1, '149x240x240': 1, '150x256x256': 1, '155x256x256': 1}`
- spacing distinti: `{'1 x 1 x 1': 221, '1 x 0.9583 x 0.9583': 5, '1 x 0.9821 x 0.9821': 3, '1 x 0.9375 x 0.9375': 8, '1 x 0.938 x 0.938': 5, '1 x 0.9545 x 0.9545': 1, '1 x 0.958 x 0.958': 1}`
- casi con immagine e label non allineate: **0**
- frazione mediana di voxel non nulli: **0.6074**

### Target spacing prevista (regola nnU-Net v2)

- spacing mediano: `[1.0, 1.0, 1.0]` mm, shape mediana `[182, 218, 182]`
- asse peggiore: `x`
- override anisotropico applicato: **False**
- **target spacing: `[1.0, 1.0, 1.0]` mm**

## Split ufficiale del benchmark

| Gruppo | n | split | spacing (mm) | shape | volume cervello mediano (mm³) |
| --- | --- | --- | --- | --- | --- |
| EMEDEA-PED_ACC | 10 | test | 1 x 1 x 1 | 182x218x182 | 1 298 200 |
| EMEDEA-PED_HD | 6 | test | 1 x 1 x 1 | 182x218x182 | 1 977 800 |
| EMEDEA-PED_Healthy | 207 | train | 1 x 1 x 1 | 182x218x182 | 1 483 300 |
| EMEDEA-PED_MCDs | 10 | test | 1 x 1 x 1 | 182x218x182 | 1 056 700 |
| EMEDEA-PED_PFM | 11 | test | 1 x 1 x 1 | 182x218x182 | 1 268 600 |

## Classi di tessuto

Valori di label osservati: `{'0/1/2/3/4/5/6': 244}`

| Valore | Classe | mediana (mm³) | min (mm³) | max (mm³) | casi con classe vuota |
| --- | --- | --- | --- | --- | --- |
| 1 | CSF | 232 400 | 59 900 | 1 376 300 | 0 |
| 2 | Gray Matter | 609 000 | 261 400 | 820 600 | 0 |
| 3 | White Matter | 412 500 | 146 200 | 606 500 | 0 |
| 4 | Deep Gray Matter | 43 500 | 14 200 | 58 700 | 0 |
| 5 | Brain Stem | 20 700 | 6 300 | 30 900 | 0 |
| 6 | Cerebellum | 159 900 | 31 900 | 232 400 | 0 |

Volume cerebrale (maschera fornita): mediana **1 478 600 mm³**, da 647 400 a 2 196 600 mm³.
