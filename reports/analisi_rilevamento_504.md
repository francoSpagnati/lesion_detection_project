# Rilevamento su `Dataset504_AINCP` per fenotipo e per dimensione

Prodotto da `scripts/analisi_rilevamento_504.py`, **non modificare a mano**.

## Controlli della procedura

- ROI trasportate in T1 identiche a `labelsTr`: **3 casi su 10** (voxel diversi in totale: 26)
- ricampionamento identico al bersaglio salvato da nnU-Net: **10 casi su 10** (voxel diversi in totale: 0)

## 1. Quali lesioni arrivano alla rete

Il piano lavora a 1 mm isotropo. Una componente «scompare» se, ricampionata da sola con la funzione di nnU-Net sulla griglia del preprocessing, non lascia nemmeno un voxel.

| volume originale | lesioni | scompaiono a 1 mm | di cui gliosi / cistica / encefalomalacia |
| --- | --- | --- | --- |
| < 1 mm³ | 24 | 20 | 11 / 7 / 2 |
| 1 – 10 mm³ | 22 | 7 | 7 / 0 / 0 |
| 10 – 50 mm³ | 7 | 0 | 0 / 0 / 0 |
| > 50 mm³ | 18 | 0 | 0 / 0 / 0 |

Totale: **27 lesioni su 71** non esistono nel bersaglio di addestramento. Il bersaglio a 1 mm salvato da nnU-Net ha 50 componenti contro le 71 della griglia originale.

## 2. Rilevamento per fenotipo

| variante | gliosi (lesioni) | cistica (lesioni) | encefalomalacia (lesioni) | tutte | solo quelle che esistono a 1 mm |
| --- | --- | --- | --- | --- | --- |
| da zero | 21/54 | 5/13 | 1/4 | 27/71 | **24/44** |
| B zero, passo 1e-4 (FLAIR spenta) | 19/54 | 5/13 | 1/4 | 25/71 | **19/44** |
| B duplica, passo 1e-4 | 15/54 | 4/13 | 1/4 | 20/71 | **19/44** |
| B passo differenziato (FLAIR attiva) | 19/54 | 5/13 | 1/4 | 25/71 | **21/44** |

## 3. Copertura dei voxel per fenotipo

Frazione dei voxel di ciascun fenotipo coperta dalla predizione (sensibilita' a livello di voxel), sommata sui pazienti che hanno quel fenotipo. Per le lesioni grandi il rilevamento e' sempre 100% e non distingue niente; la copertura si'.

| variante | gliosi | cistica | encefalomalacia |
| --- | --- | --- | --- |
| da zero | 0.585 (10 pazienti) | 0.834 (4 pazienti) | 0.401 (4 pazienti) |
| B zero, passo 1e-4 (FLAIR spenta) | 0.472 (10 pazienti) | 0.933 (4 pazienti) | 0.809 (4 pazienti) |
| B duplica, passo 1e-4 | 0.499 (10 pazienti) | 0.917 (4 pazienti) | 0.559 (4 pazienti) |
| B passo differenziato (FLAIR attiva) | 0.525 (10 pazienti) | 0.933 (4 pazienti) | 0.775 (4 pazienti) |

Paziente per paziente, gliosi (ROI_1): FLAIR spenta contro FLAIR attiva.

| paziente | voxel di gliosi | coperti, FLAIR spenta | coperti, FLAIR attiva |
| --- | --- | --- | --- |
| FSM_032 | 53203 | 0.188 | 0.264 |
| FSM_039 | 7680 | 0.018 | 0.367 |
| FSM_040 | 20178 | 0.634 | 0.698 |
| FSM_041 | 10546 | 0.030 | 0.043 |
| FSM_045 | 14374 | 0.817 | 0.723 |
| FSM_046 | 2551 | 0.751 | 0.744 |
| FSM_052 | 26204 | 0.465 | 0.454 |
| FSM_053 | 3209 | 0.656 | 0.799 |
| FSM_054 | 36641 | 0.688 | 0.710 |
| FSM_056 | 29857 | 0.673 | 0.776 |

Migliora in 7 pazienti su 10, Wilcoxon p = 0.084.

## 4. Falsi positivi

| variante | falsi positivi | per paziente | volume mediano (mm³) | sotto 10 mm³ | sopra 200 mm³ |
| --- | --- | --- | --- | --- | --- |
| da zero | 79 | 7.9 | 0.9 | 63 | 2 |
| B zero, passo 1e-4 (FLAIR spenta) | 26 | 2.6 | 0.4 | 22 | 0 |
| B duplica, passo 1e-4 | 32 | 3.2 | 0.6 | 26 | 0 |
| B passo differenziato (FLAIR attiva) | 23 | 2.3 | 1.5 | 16 | 0 |

Appaiati contro «da zero», per paziente:

- B zero, passo 1e-4 (FLAIR spenta): meno falsi positivi in **8** pazienti su 10, di piu' in 1, Wilcoxon p = 0.043
- B duplica, passo 1e-4: meno falsi positivi in **6** pazienti su 10, di piu' in 2, Wilcoxon p = 0.062
- B passo differenziato (FLAIR attiva): meno falsi positivi in **8** pazienti su 10, di piu' in 2, Wilcoxon p = 0.033
