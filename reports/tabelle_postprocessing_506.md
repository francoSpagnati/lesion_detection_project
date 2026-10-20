# Z prima e dopo il postprocessing; che cosa toglie il volume minimo di 50 mm³

Prodotto da `scripts/tabelle_postprocessing_506.py`, **non modificare a mano**. Modello `nnUNetTrainerWandb_250ep_dropmod30_arresto50`. Griglia T1 nativa; lesione trovata se toccata da almeno un voxel; componente vera se tocca una lesione annotata, falso positivo se non ne tocca nessuna (componenti sotto 1 mm³ non contate). Cross-validation: fuori-fold dei 12 pazienti, dove il postprocessing è stato scelto. Test: 5 pazienti, solo descrittivo.

## Prima e dopo il postprocessing: Cross-validation (12 pazienti)

| passo | Dice medio | lesioni trovate | di cui < 50 mm³ | componenti predette | falsi positivi (per paziente) | precisione | F1 per lesione | errore sul % di cervello lesionato |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uscita di nnU-Net (soglia 0.5, nessun filtro) | 0.580 | 27/68 | 9/46 | 56 | 20 (1.7) | 0.64 | 0.491 | 0.48 punti |
| soglia 0.3, nessun filtro | 0.584 | 28/68 | 10/46 | 55 | 21 (1.8) | 0.62 | 0.494 | 0.40 punti |
| **soglia 0.3 + volume minimo 50 mm³ (Z)** | 0.584 | 27/68 | 10/46 | 19 | 1 (0.1) | 0.95 | 0.560 | 0.40 punti |

## Prima e dopo il postprocessing: Test (5 pazienti)

| passo | Dice medio | lesioni trovate | di cui < 50 mm³ | componenti predette | falsi positivi (per paziente) | precisione | F1 per lesione | errore sul % di cervello lesionato |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uscita di nnU-Net (soglia 0.5, nessun filtro) | 0.615 | 24/40 | 14/26 | 58 | 41 (8.2) | 0.29 | 0.394 | 0.83 punti |
| soglia 0.3, nessun filtro | 0.623 | 25/40 | 15/26 | 57 | 42 (8.4) | 0.26 | 0.370 | 0.66 punti |
| **soglia 0.3 + volume minimo 50 mm³ (Z)** | 0.622 | 21/40 | 13/26 | 14 | 4 (0.8) | 0.71 | 0.605 | 0.65 punti |

## Che cosa toglie il volume minimo di 50 mm³ (a soglia 0.3)

| | cross-validation (12) | test (5) |
| --- | --- | --- |
| componenti predette tolte (1–50 mm³) | 36 | 43 |
| di cui vere (toccano una lesione annotata) | 16 | 5 |
| di cui falsi positivi | 20 | 38 |
| falsi positivi tolti, sul totale prima del filtro | 20 su 21 | 38 su 42 |
| **lesioni annotate perse davvero** | **1** | **4** |
| volume delle lesioni perse | 346 mm³ | 4.6 mm³, 21 mm³, 124 mm³, 912 mm³ |

Una componente vera tolta non vuol dire una lesione persa: quasi sempre la stessa lesione è toccata anche da una componente più grande, che il filtro lascia. Persa davvero = la lesione era trovata senza filtro e non lo è più dopo.

### Componenti tolte per dimensione

| volume della componente | CV: vere | CV: falsi positivi | test: vere | test: falsi positivi |
| --- | --- | --- | --- | --- |
| 1–10 mm³ | 7 | 14 | 1 | 28 |
| 10–20 mm³ | 4 | 0 | 2 | 6 |
| 20–50 mm³ | 5 | 6 | 2 | 4 |
