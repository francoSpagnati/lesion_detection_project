# Rilevamento per singola lesione

Prodotto da `scripts/metriche_per_lesione.py`, **non modificare a mano**.

Il Dice misura quanto bene si ricalca il contorno di cio' che si e' trovato, non
quante lesioni si sono perse. Qui una lesione di riferimento conta come **rilevata**
se almeno un voxel della predizione ci cade dentro: il criterio piu' permissivo, e
quindi quello che risponde a "l'ha vista?" invece che "l'ha delineata bene?".

## `Dataset511_WMH_T1` — 16 casi di validation, 865 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 voxel) | 575 | 120 | **20.9%** |
| piccole (<=50) | 208 | 135 | **64.9%** |
| medie (<=200) | 53 | 46 | **86.8%** |
| grandi (>200) | 29 | 29 | **100.0%** |
| **totale** | 865 | 330 | **38.2%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **90** in tutto, cioe' **5.6 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **0** su 16.

## `Dataset512_WMH_T1_FLAIR` — 16 casi di validation, 865 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 voxel) | 575 | 340 | **59.1%** |
| piccole (<=50) | 208 | 193 | **92.8%** |
| medie (<=200) | 53 | 53 | **100.0%** |
| grandi (>200) | 29 | 29 | **100.0%** |
| **totale** | 865 | 615 | **71.1%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **92** in tutto, cioe' **5.8 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **0** su 16.

