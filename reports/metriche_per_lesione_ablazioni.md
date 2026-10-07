# Rilevamento per singola lesione

Prodotto da `scripts/metriche_per_lesione.py`, **non modificare a mano**.

Il Dice misura quanto bene si ricalca il contorno di cio' che si e' trovato, non
quante lesioni si sono perse. Qui una lesione di riferimento conta come **rilevata**
se almeno un voxel della predizione ci cade dentro: il criterio piu' permissivo, e
quindi quello che risponde a "l'ha vista?" invece che "l'ha delineata bene?".
Le fasce sono in millimetri cubi, misurati sulla griglia originale di ogni caso.

## `Dataset521_ATLAS_3T` / `nnUNetTrainerWandb__nnUNetPlans__3d_fullres` — 103 casi (fold 0), 230 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 mm3) | 23 | 4 | **17.4%** |
| piccole (<=50) | 17 | 2 | **11.8%** |
| medie (<=200) | 35 | 14 | **40.0%** |
| grandi (>200) | 155 | 127 | **81.9%** |
| **totale** | 230 | 147 | **63.9%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **119** in tutto, cioe' **1.2 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **0** su 103.

Dettaglio per caso: `reports/lesioni_per_caso/Dataset521_ATLAS_3T__nnUNetTrainerWandb__nnUNetPlans__3d_fullres.json`.

## `Dataset522_WMH_SS` / `nnUNetTrainerWandb__nnUNetPlans__3d_fullres` — 16 casi (fold 0), 865 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 mm3) | 219 | 114 | **52.1%** |
| piccole (<=50) | 439 | 337 | **76.8%** |
| medie (<=200) | 131 | 127 | **96.9%** |
| grandi (>200) | 76 | 76 | **100.0%** |
| **totale** | 865 | 654 | **75.6%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **101** in tutto, cioe' **6.3 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **0** su 16.

Dettaglio per caso: `reports/lesioni_per_caso/Dataset522_WMH_SS__nnUNetTrainerWandb__nnUNetPlans__3d_fullres.json`.

## `Dataset523_ATLAS_SS` / `nnUNetTrainerWandb__nnUNetPlans__3d_fullres` — 112 casi (fold 0), 244 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 mm3) | 23 | 3 | **13.0%** |
| piccole (<=50) | 17 | 2 | **11.8%** |
| medie (<=200) | 36 | 13 | **36.1%** |
| grandi (>200) | 168 | 137 | **81.5%** |
| **totale** | 244 | 155 | **63.5%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **98** in tutto, cioe' **0.9 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **1** su 112.

Dettaglio per caso: `reports/lesioni_per_caso/Dataset523_ATLAS_SS__nnUNetTrainerWandb__nnUNetPlans__3d_fullres.json`.

## `Dataset531_UNION_D` / `nnUNetTrainerWandb__nnUNetPlans__3d_fullres` — 151 casi (fold 0), 1109 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 mm3) | 242 | 19 | **7.9%** |
| piccole (<=50) | 456 | 101 | **22.1%** |
| medie (<=200) | 167 | 91 | **54.5%** |
| grandi (>200) | 244 | 209 | **85.7%** |
| **totale** | 1109 | 420 | **37.9%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **204** in tutto, cioe' **1.4 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **2** su 151.

Dettaglio per caso: `reports/lesioni_per_caso/Dataset531_UNION_D__nnUNetTrainerWandb__nnUNetPlans__3d_fullres.json`.

