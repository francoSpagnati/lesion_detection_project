# D prima e dopo il postprocessing; che cosa toglie il volume minimo di 20 mm³

Prodotto da `scripts/tabelle_postprocessing_506.py D`, **non modificare a mano**. Modello `nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_dropmod30_arresto50_dapre2c`. Griglia T1 nativa; lesione trovata se toccata da almeno un voxel; componente vera se tocca una lesione annotata, falso positivo se non ne tocca nessuna (componenti sotto 1 mm³ non contate). Cross-validation: fuori-fold dei 12 pazienti, dove il postprocessing è stato scelto. Test: 5 pazienti, solo descrittivo.

## Prima e dopo il postprocessing: Cross-validation (12 pazienti)

| passo | Dice medio | lesioni trovate | di cui < 50 mm³ | componenti predette | falsi positivi (per paziente) | precisione | F1 per lesione | errore sul % di cervello lesionato |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uscita di nnU-Net (soglia 0.5, nessun filtro) | 0.577 | 26/68 | 7/46 | 46 | 16 (1.3) | 0.65 | 0.482 | 0.50 punti |
| soglia 0.4, nessun filtro | 0.579 | 27/68 | 8/46 | 45 | 13 (1.1) | 0.71 | 0.510 | 0.45 punti |
| **soglia 0.4 + volume minimo 20 mm³ (D)** | 0.577 | 25/68 | 8/46 | 20 | 2 (0.2) | 0.90 | 0.522 | 0.45 punti |

## Prima e dopo il postprocessing: Test (5 pazienti)

| passo | Dice medio | lesioni trovate | di cui < 50 mm³ | componenti predette | falsi positivi (per paziente) | precisione | F1 per lesione | errore sul % di cervello lesionato |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uscita di nnU-Net (soglia 0.5, nessun filtro) | 0.671 | 22/40 | 10/26 | 46 | 21 (4.2) | 0.54 | 0.547 | 1.61 punti |
| soglia 0.4, nessun filtro | 0.669 | 24/40 | 12/26 | 49 | 25 (5.0) | 0.49 | 0.539 | 1.48 punti |
| **soglia 0.4 + volume minimo 20 mm³ (D)** | 0.672 | 24/40 | 12/26 | 24 | 5 (1.0) | 0.79 | 0.683 | 1.48 punti |

## Che cosa toglie il volume minimo di 20 mm³ (a soglia 0.4)

| | cross-validation (12) | test (5) |
| --- | --- | --- |
| componenti predette tolte (1–20 mm³) | 25 | 25 |
| di cui vere (toccano una lesione annotata) | 14 | 5 |
| di cui falsi positivi | 11 | 20 |
| falsi positivi tolti, sul totale prima del filtro | 11 su 13 | 20 su 25 |
| **lesioni annotate perse davvero** | **2** | **0** |
| volume delle lesioni perse | 346 mm³, 667 mm³ | — |

Una componente vera tolta non vuol dire una lesione persa: quasi sempre la stessa lesione è toccata anche da una componente più grande, che il filtro lascia. Persa davvero = la lesione era trovata senza filtro e non lo è più dopo.

### Componenti tolte per dimensione

| volume della componente | CV: vere | CV: falsi positivi | test: vere | test: falsi positivi |
| --- | --- | --- | --- | --- |
| 1–10 mm³ | 11 | 7 | 5 | 16 |
| 10–20 mm³ | 3 | 4 | 0 | 4 |
