# Ventricoli di SynthSeg come postprocessing di Z (fuori-fold, 12 pazienti)

Prodotto da `scripts/synthseg_aincp.py valuta`, **non modificare a mano**. SynthSeg 1.0 sulla T1; regola fissata prima: si toglie una componente predetta con almeno il 50% dei voxel nei ventricoli. Il test non entra.

## SynthSeg e la ROI_3 del clinico, e quanta lesione cade nei ventricoli

| paziente | ventricoli SynthSeg | ROI_3 (ventricolo annotato) | ROI_3 dentro SynthSeg | lesione | lesione nei ventricoli |
| --- | --- | --- | --- | --- | --- |
| FSM_001 | 16 978 mm³ | 6 399 mm³ | 85% | 166 mm³ | 30 mm³ (17.8%) |
| FSM_028 | 12 423 mm³ | 13 308 mm³ | 43% | 157 108 mm³ | 5.0 mm³ (0.0%) |
| FSM_032 | 32 330 mm³ | 20 205 mm³ | 89% | 123 642 mm³ | 243 mm³ (0.2%) |
| FSM_033 | 9 125 mm³ | — | — | 667 mm³ | 0.4 mm³ (0.1%) |
| FSM_035 | 12 191 mm³ | 7 414 mm³ | 57% | 40 742 mm³ | 18 mm³ (0.0%) |
| FSM_036 | 8 793 mm³ | 1 045 mm³ | 99% | 108 965 mm³ | 61 mm³ (0.1%) |
| FSM_039 | 13 170 mm³ | — | — | 1 394 mm³ | 0.0 mm³ (0.0%) |
| FSM_040 | 16 395 mm³ | 11 872 mm³ | 48% | 55 593 mm³ | 0.0 mm³ (0.0%) |
| FSM_041 | 20 498 mm³ | 14 299 mm³ | 54% | 1 901 mm³ | 26 mm³ (1.4%) |
| FSM_045 | 7 927 mm³ | — | — | 9 525 mm³ | 0.0 mm³ (0.0%) |
| FSM_053 | 37 472 mm³ | 16 178 mm³ | 91% | 582 mm³ | 24 mm³ (4.1%) |
| FSM_056 | 16 371 mm³ | — | — | 12 442 mm³ | 0.6 mm³ (0.0%) |

## Effetto con il postprocessing congelato di Z (soglia 0.3, volume minimo 50 mm³)

| | lesioni trovate | falsi positivi | F1 lesioni | Dice medio | componenti tolte |
| --- | --- | --- | --- | --- | --- |
| senza la regola dei ventricoli | 27/68 | 1 | 0.560 | 0.584 | — |
| con la regola dei ventricoli | 27/68 | 1 | 0.560 | 0.584 | 0 |

Lesioni vere perse per la regola: 0.

## Effetto sulle predizioni senza postprocessing (soglia 0.5, nessun filtro)

| | lesioni trovate | falsi positivi | F1 lesioni | Dice medio | componenti tolte |
| --- | --- | --- | --- | --- | --- |
| senza la regola dei ventricoli | 27/68 | 20 | 0.491 | 0.580 | — |
| con la regola dei ventricoli | 27/68 | 20 | 0.491 | 0.580 | 0 |

Lesioni vere perse per la regola: 0.
