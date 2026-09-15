# Rilevamento per singola lesione

Prodotto da `scripts/metriche_per_lesione.py`, **non modificare a mano**.

Il Dice misura quanto bene si ricalca il contorno di cio' che si e' trovato, non
quante lesioni si sono perse. Qui una lesione di riferimento conta come **rilevata**
se almeno un voxel della predizione ci cade dentro: il criterio piu' permissivo, e
quindi quello che risponde a "l'ha vista?" invece che "l'ha delineata bene?".
Le fasce sono in millimetri cubi, misurati sulla griglia originale di ogni caso.

## `Dataset504_AINCP` / `nnUNetTrainerWandb_250ep_arresto50__nnUNetPlansDa502__3d_fullres` — 10 casi (5 fold), 71 lesioni annotate

| fascia di volume | lesioni | rilevate | tasso di rilevamento |
| --- | --- | --- | --- |
| minuscole (<=10 mm3) | 46 | 5 | **10.9%** |
| piccole (<=50) | 7 | 4 | **57.1%** |
| medie (<=200) | 6 | 6 | **100.0%** |
| grandi (>200) | 12 | 12 | **100.0%** |
| **totale** | 71 | 27 | **38.0%** |

Falsi positivi (componenti predette che non toccano alcuna lesione vera): **79** in tutto, cioe' **7.9 per caso**.

Casi in cui il modello non predice **nessuna** componente pur essendoci lesione annotata: **0** su 10.

Dettaglio per caso: `reports/lesioni_per_caso/Dataset504_AINCP__nnUNetTrainerWandb_250ep_arresto50__nnUNetPlansDa502__3d_fullres.json`.

