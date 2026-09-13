# Rilevamento per singola lesione

Prodotto da `scripts/metriche_per_lesione.py`, **non modificare a mano**.

Il Dice misura quanto bene si ricalca il contorno di cio' che si e' trovato, non
quante lesioni si sono perse. Qui una lesione di riferimento conta come **rilevata**
se almeno un voxel della predizione ci cade dentro: il criterio piu' permissivo, e
quindi quello che risponde a "l'ha vista?" invece che "l'ha delineata bene?".
Le fasce sono in millimetri cubi, misurati sulla griglia originale di ogni caso.

## `Dataset502_ATLAS` / `nnUNetTrainerWandb__nnUNetPlans__3d_fullres` — 112 casi (fold 0), 244 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 mm3) | 23 | 2 | **8.7%** |
| piccole (<=50) | 17 | 1 | **5.9%** |
| medie (<=200) | 36 | 16 | **44.4%** |
| grandi (>200) | 168 | 138 | **82.1%** |
| **totale** | 244 | 157 | **64.3%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **119** in tutto, cioe' **1.1 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **0** su 112.

Dettaglio per caso: `reports/lesioni_per_caso/Dataset502_ATLAS__nnUNetTrainerWandb__nnUNetPlans__3d_fullres.json`.

## `Dataset501_WMH` / `nnUNetTrainerWandb__nnUNetPlans__3d_fullres` — 16 casi (fold 0), 865 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 mm3) | 219 | 99 | **45.2%** |
| piccole (<=50) | 439 | 331 | **75.4%** |
| medie (<=200) | 131 | 125 | **95.4%** |
| grandi (>200) | 76 | 75 | **98.7%** |
| **totale** | 865 | 630 | **72.8%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **92** in tutto, cioe' **5.8 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **0** su 16.

Dettaglio per caso: `reports/lesioni_per_caso/Dataset501_WMH__nnUNetTrainerWandb__nnUNetPlans__3d_fullres.json`.

## `Dataset511_WMH_T1` / `nnUNetTrainerWandb__nnUNetPlans__3d_fullres` — 16 casi (fold 0), 865 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 mm3) | 219 | 26 | **11.9%** |
| piccole (<=50) | 439 | 137 | **31.2%** |
| medie (<=200) | 131 | 96 | **73.3%** |
| grandi (>200) | 76 | 71 | **93.4%** |
| **totale** | 865 | 330 | **38.2%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **90** in tutto, cioe' **5.6 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **0** su 16.

Dettaglio per caso: `reports/lesioni_per_caso/Dataset511_WMH_T1__nnUNetTrainerWandb__nnUNetPlans__3d_fullres.json`.

## `Dataset512_WMH_T1_FLAIR` / `nnUNetTrainerWandb__nnUNetPlans__3d_fullres` — 16 casi (fold 0), 865 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 mm3) | 219 | 95 | **43.4%** |
| piccole (<=50) | 439 | 316 | **72.0%** |
| medie (<=200) | 131 | 128 | **97.7%** |
| grandi (>200) | 76 | 76 | **100.0%** |
| **totale** | 865 | 615 | **71.1%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **92** in tutto, cioe' **5.8 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **0** su 16.

Dettaglio per caso: `reports/lesioni_per_caso/Dataset512_WMH_T1_FLAIR__nnUNetTrainerWandb__nnUNetPlans__3d_fullres.json`.

