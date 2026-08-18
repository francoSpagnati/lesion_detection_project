# Validation delle baseline (cross-validation)

Generato da `scripts/summarize_validation.py`, **non modificare a mano**.

Sono i risultati della validation di nnU-Net sui **casi interi** ricostruiti con la
finestra scorrevole, non lo "pseudo Dice" stampato durante il training (che e' calcolato
sulle patch di addestramento e serve solo a seguire la convergenza).

Questi numeri vengono dalla **cross-validation**, non dal test holdout, che resta
intoccato fino alla scelta definitiva della configurazione.

## Dataset501_WMH — nnUNetTrainerWandb__nnUNetPlans__3d_fullres

Fold: `fold_0`, 16 casi di validation.

### Dice per classe, sui casi interi

| | n | media | mediana | Q1 | Q3 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classe 1 | 16 | 0.7739 | 0.8369 | 0.7047 | 0.8803 | 0.4264 | 0.9057 |

### Dice per gruppo (sito per WMH, coorte per ATLAS)

| | n | media | mediana | Q1 | Q3 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Amsterdam_GE1T5 | 1 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 |
| Amsterdam_GE3T | 8 | 0.7899 | 0.8476 | 0.7480 | 0.8803 | 0.4955 | 0.8990 |
| Singapore | 3 | 0.8184 | 0.8732 | 0.7747 | 0.8894 | 0.6762 | 0.9057 |
| Utrecht | 4 | 0.7147 | 0.7729 | 0.6293 | 0.8582 | 0.4264 | 0.8866 |

### I cinque casi con il bersaglio piu' piccolo

| caso | voxel di riferimento | Dice |
| --- | --- | --- |
| Amsterdam_GE1T5_156 | 496 | 0.7500 |
| Singapore_58 | 543 | 0.6762 |
| Amsterdam_GE3T_123 | 555 | 0.4955 |
| Amsterdam_GE3T_116 | 613 | 0.7073 |
| Utrecht_11 | 894 | 0.4264 |

Dice medio sulla meta' di casi con bersaglio piu' piccolo: **0.6674**; sulla meta' con bersaglio piu' grande: **0.8805**.


## Dataset502_ATLAS — nnUNetTrainerWandb__nnUNetPlans__3d_fullres

Fold: `fold_0`, 112 casi di validation.

### Dice per classe, sui casi interi

| | n | media | mediana | Q1 | Q3 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classe 1 | 112 | 0.6103 | 0.7211 | 0.4649 | 0.8394 | 0.0000 | 0.9459 |

### Dice per gruppo (sito per WMH, coorte per ATLAS)

| | n | media | mediana | Q1 | Q3 | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r001 | 6 | 0.5574 | 0.6519 | 0.2925 | 0.8068 | 0.1707 | 0.8390 |
| r002 | 3 | 0.8587 | 0.8802 | 0.8221 | 0.9060 | 0.7640 | 0.9319 |
| r003 | 2 | 0.9457 | 0.9457 | 0.9456 | 0.9458 | 0.9455 | 0.9459 |
| r004 | 7 | 0.7563 | 0.8273 | 0.7210 | 0.8863 | 0.3339 | 0.9188 |
| r005 | 3 | 0.7682 | 0.8406 | 0.7088 | 0.8638 | 0.5771 | 0.8870 |
| r009 | 19 | 0.3365 | 0.3029 | 0.1385 | 0.5834 | 0.0000 | 0.7905 |
| r010 | 4 | 0.6961 | 0.6559 | 0.6332 | 0.7188 | 0.6004 | 0.8721 |
| r011 | 5 | 0.7457 | 0.8131 | 0.7031 | 0.8434 | 0.5248 | 0.8440 |
| r015 | 2 | 0.5586 | 0.5586 | 0.3815 | 0.7356 | 0.2045 | 0.9126 |
| r017 | 2 | 0.8158 | 0.8158 | 0.7983 | 0.8333 | 0.7808 | 0.8508 |
| r018 | 1 | 0.8958 | 0.8958 | 0.8958 | 0.8958 | 0.8958 | 0.8958 |
| r019 | 3 | 0.4904 | 0.6922 | 0.3461 | 0.7356 | 0.0000 | 0.7790 |
| r024 | 1 | 0.8283 | 0.8283 | 0.8283 | 0.8283 | 0.8283 | 0.8283 |
| r027 | 3 | 0.6233 | 0.8478 | 0.4994 | 0.8594 | 0.1511 | 0.8709 |
| r028 | 1 | 0.6674 | 0.6674 | 0.6674 | 0.6674 | 0.6674 | 0.6674 |
| r031 | 7 | 0.6066 | 0.7643 | 0.4911 | 0.8029 | 0.0531 | 0.8409 |
| r034 | 5 | 0.5388 | 0.6332 | 0.5090 | 0.7488 | 0.0000 | 0.8031 |
| r035 | 2 | 0.5784 | 0.5784 | 0.4901 | 0.6667 | 0.4018 | 0.7549 |
| r038 | 8 | 0.5966 | 0.7345 | 0.4943 | 0.8061 | 0.0000 | 0.9035 |
| r040 | 9 | 0.6528 | 0.7930 | 0.6983 | 0.8471 | 0.0000 | 0.9084 |
| r042 | 2 | 0.7739 | 0.7739 | 0.7097 | 0.8380 | 0.6456 | 0.9022 |
| r044 | 1 | 0.7228 | 0.7228 | 0.7228 | 0.7228 | 0.7228 | 0.7228 |
| r045 | 1 | 0.8411 | 0.8411 | 0.8411 | 0.8411 | 0.8411 | 0.8411 |
| r046 | 2 | 0.6867 | 0.6867 | 0.6054 | 0.7681 | 0.5240 | 0.8495 |
| r047 | 4 | 0.8104 | 0.8434 | 0.7922 | 0.8616 | 0.6923 | 0.8624 |
| r048 | 3 | 0.5198 | 0.6257 | 0.3160 | 0.7766 | 0.0062 | 0.9274 |
| r049 | 2 | 0.3845 | 0.3845 | 0.1923 | 0.5768 | 0.0000 | 0.7690 |
| r050 | 2 | 0.6349 | 0.6349 | 0.6190 | 0.6508 | 0.6031 | 0.6667 |
| r052 | 2 | 0.7441 | 0.7441 | 0.7373 | 0.7508 | 0.7306 | 0.7576 |

### I cinque casi con il bersaglio piu' piccolo

| caso | voxel di riferimento | Dice |
| --- | --- | --- |
| sub-r009s039 | 23 | 0.5846 |
| sub-r009s035 | 35 | 0.0000 |
| sub-r009s076 | 72 | 0.3403 |
| sub-r009s079 | 82 | 0.2032 |
| sub-r010s023 | 152 | 0.6004 |

Dice medio sulla meta' di casi con bersaglio piu' piccolo: **0.5086**; sulla meta' con bersaglio piu' grande: **0.7120**.

### Casi con Dice esattamente 0

| caso | voxel di riferimento | voxel predetti |
| --- | --- | --- |
| sub-r040s049 | 9586 | 217 |
| sub-r034s012 | 2045 | 97 |
| sub-r009s093 | 1426 | 420 |
| sub-r009s025 | 991 | 923 |
| sub-r049s026 | 922 | 204 |
| sub-r038s026 | 523 | 152 |
| sub-r019s001 | 237 | 647 |
| sub-r009s035 | 35 | 111 |

Un Dice nullo su un bersaglio grande, con voxel predetti in quantita', non e'
una rifinitura imprecisa dei bordi: e' un errore di **localizzazione**, il modello ha
segmentato altrove. Vanno guardati come categoria a se'.

