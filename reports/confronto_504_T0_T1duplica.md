# `Dataset504_AINCP`: nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50 contro nnUNetTrainerWandb_250ep_arresto50

Dieci pazienti in cinque fold: per ciascuno si usa la predizione del fold in cui era fuori dal training. Soglia minima 0 mm3 sulle lesioni.

| | nnUNetTrainerWandb_250ep_arresto50 | nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50 | differenza |
| --- | --- | --- | --- |
| Dice medio sui 10 pazienti | 0.5814 | 0.5960 | **+0.0146** |
| Dice mediano | 0.6646 | 0.6061 | -0.0585 |
| lesioni rilevate su 71 | 27 (38.0%) | 20 (28.2%) | **-7** |
| falsi positivi (per paziente) | 79 (7.9) | 32 (3.2) | -47 |

Dice: migliora in **5** pazienti su 10, peggiora in **5**, Wilcoxon p = 0.492.
Rilevamento: migliora in **0**, peggiora in **4**, Wilcoxon p = 0.125.

## Paziente per paziente

| paziente | fold | Dice nnUNetTrainerWandb_250ep_arresto50 | Dice nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50 | Δ | lesioni | rilevate prima | rilevate dopo | FP prima | FP dopo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FSM_052 | 4 | 0.7675 | 0.6555 | -0.1119 | 10 | 4 | 2 | 9 | 0 |
| FSM_053 | 2 | 0.6431 | 0.5794 | -0.0637 | 6 | 6 | 5 | 5 | 2 |
| FSM_056 | 2 | 0.6861 | 0.6328 | -0.0534 | 7 | 3 | 3 | 3 | 4 |
| FSM_054 | 1 | 0.8803 | 0.8728 | -0.0075 | 15 | 4 | 2 | 8 | 3 |
| FSM_041 | 3 | 0.0444 | 0.0380 | -0.0064 | 6 | 2 | 2 | 7 | 2 |
| FSM_032 | 0 | 0.8066 | 0.8674 | +0.0608 | 10 | 1 | 1 | 1 | 1 |
| FSM_040 | 3 | 0.8253 | 0.8897 | +0.0644 | 12 | 3 | 1 | 2 | 2 |
| FSM_039 | 0 | 0.3599 | 0.4292 | +0.0693 | 2 | 1 | 1 | 4 | 3 |
| FSM_045 | 4 | 0.3529 | 0.4314 | +0.0785 | 1 | 1 | 1 | 1 | 3 |
| FSM_046 | 1 | 0.4479 | 0.5640 | +0.1161 | 2 | 2 | 2 | 39 | 12 |

Ordinati dal paziente che peggiora di piu' a quello che migliora di piu'. Con dieci coppie il p di Wilcoxon non scende sotto 0.002 nemmeno se tutte vanno nello stesso verso: contano le coppie e le grandezze, il p e' una cautela.
