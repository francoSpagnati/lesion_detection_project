# Il volume minimo di 50 mm³ nel postprocessing di Z: che cosa toglie

Prodotto da `scripts/analisi_volume_minimo_506.py`, **non modificare a mano**. Soglia di probabilità 0.3 in tutte le righe; griglia T1 nativa; lesione trovata se toccata da almeno un voxel; falso positivo = componente predetta >= 1 mm³ che non tocca lesioni annotate.

## Cross-validation, fuori-fold (12 pazienti)

| filtro (soglia 0.3) | lesioni trovate | di cui < 50 mm³ | falsi positivi | F1 per lesione | Dice medio |
| --- | --- | --- | --- | --- | --- |
| volume >= 0 mm³ | 28/68 | 10/46 | 21 | 0.494 | 0.584 |
| volume >= 1 mm³ | 28/68 | 10/46 | 21 | 0.494 | 0.584 |
| volume >= 5 mm³ | 28/68 | 10/46 | 10 | 0.532 | 0.585 |
| volume >= 10 mm³ | 28/68 | 10/46 | 7 | 0.542 | 0.584 |
| volume >= 20 mm³ | 27/68 | 10/46 | 7 | 0.523 | 0.583 |
| **volume >= 50 mm³ (Z, scelto)** | 27/68 | 10/46 | 1 | 0.560 | 0.584 |
| isteresi: picco >= 0.5 | 28/68 | 10/46 | 15 | 0.513 | 0.584 |
| isteresi: picco >= 0.6 | 28/68 | 10/46 | 13 | 0.522 | 0.584 |
| isteresi: picco >= 0.7 | 28/68 | 10/46 | 11 | 0.527 | 0.584 |
| isteresi: picco >= 0.8 | 28/68 | 10/46 | 9 | 0.530 | 0.584 |
| isteresi: picco >= 0.9 | 27/68 | 10/46 | 6 | 0.526 | 0.583 |

Lesioni annotate sotto 50 mm³: 46. Toccate dalla predizione a soglia 0.3, senza filtro: 10; di queste, trovate **solo** da una macchia sotto 50 mm³ (le uniche che il filtro può far perdere): **0**. Le altre stanno dentro una macchia predetta più grande, quasi sempre quella di una lesione grande vicina.

Componenti predette fra 1 e 50 mm³ (quelle che il filtro toglie): **36**, di cui **16 toccano una lesione vera** e 20 sono falsi positivi.

Lesioni vere perse passando da nessun filtro a 50 mm³: FSM_056 346 mm³.

| componente tolta | mm³ | probabilità massima | tocca una lesione |
| --- | --- | --- | --- |
| FSM_041 | 31.8 | 1.00 | sì |
| FSM_001 | 34.6 | 1.00 | no |
| FSM_056 | 24.0 | 1.00 | no |
| FSM_041 | 27.5 | 1.00 | sì |
| FSM_041 | 27.9 | 1.00 | sì |
| FSM_041 | 29.9 | 0.99 | no |
| FSM_041 | 20.1 | 0.95 | no |
| FSM_045 | 3.7 | 0.95 | no |
| FSM_041 | 11.3 | 0.94 | sì |
| FSM_001 | 20.3 | 0.93 | no |
| FSM_028 | 23.0 | 0.86 | sì |
| FSM_056 | 19.4 | 0.85 | sì |
| FSM_028 | 20.0 | 0.84 | sì |
| FSM_036 | 3.0 | 0.83 | no |
| FSM_028 | 8.0 | 0.82 | no |
| FSM_028 | 4.0 | 0.80 | sì |
| FSM_041 | 8.5 | 0.78 | sì |
| FSM_001 | 4.1 | 0.77 | no |
| FSM_036 | 5.0 | 0.75 | sì |
| FSM_028 | 11.0 | 0.75 | sì |
| FSM_039 | 3.5 | 0.74 | no |
| FSM_045 | 10.3 | 0.74 | sì |
| FSM_041 | 3.3 | 0.66 | sì |
| FSM_053 | 7.4 | 0.66 | no |
| FSM_028 | 40.0 | 0.63 | no |
| FSM_045 | 6.6 | 0.61 | sì |
| FSM_028 | 2.0 | 0.57 | no |
| FSM_035 | 3.0 | 0.56 | no |
| FSM_001 | 1.3 | 0.50 | no |
| FSM_028 | 5.0 | 0.45 | no |
| FSM_028 | 4.0 | 0.43 | sì |
| FSM_045 | 6.3 | 0.42 | sì |
| FSM_053 | 1.3 | 0.42 | no |
| FSM_045 | 3.7 | 0.39 | no |
| FSM_045 | 1.3 | 0.34 | no |
| FSM_028 | 2.0 | 0.31 | no |

## Test (5 pazienti): solo descrittivo, nessuna scelta fatta qui

| filtro (soglia 0.3) | lesioni trovate | di cui < 50 mm³ | falsi positivi | F1 per lesione | Dice medio |
| --- | --- | --- | --- | --- | --- |
| volume >= 0 mm³ | 25/40 | 15/26 | 42 | 0.370 | 0.623 |
| volume >= 1 mm³ | 25/40 | 15/26 | 42 | 0.370 | 0.623 |
| volume >= 5 mm³ | 25/40 | 15/26 | 18 | 0.526 | 0.624 |
| volume >= 10 mm³ | 24/40 | 14/26 | 14 | 0.545 | 0.623 |
| volume >= 20 mm³ | 22/40 | 13/26 | 8 | 0.574 | 0.623 |
| **volume >= 50 mm³ (Z, scelto)** | 21/40 | 13/26 | 4 | 0.605 | 0.622 |
| isteresi: picco >= 0.5 | 25/40 | 15/26 | 34 | 0.411 | 0.624 |
| isteresi: picco >= 0.6 | 25/40 | 15/26 | 27 | 0.455 | 0.624 |
| isteresi: picco >= 0.7 | 24/40 | 15/26 | 17 | 0.515 | 0.623 |
| isteresi: picco >= 0.8 | 23/40 | 14/26 | 14 | 0.524 | 0.623 |
| isteresi: picco >= 0.9 | 22/40 | 13/26 | 9 | 0.561 | 0.623 |

Lesioni annotate sotto 50 mm³: 26. Toccate dalla predizione a soglia 0.3, senza filtro: 15; di queste, trovate **solo** da una macchia sotto 50 mm³ (le uniche che il filtro può far perdere): **2**. Le altre stanno dentro una macchia predetta più grande, quasi sempre quella di una lesione grande vicina.

Lesioni vere del test perse passando da nessun filtro a 50 mm³: FSM_030 124 mm³, FSM_030 912 mm³, FSM_052 21 mm³, FSM_054 4.6 mm³.
