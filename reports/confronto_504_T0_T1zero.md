# `Dataset504_AINCP`: nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50 contro nnUNetTrainerWandb_250ep_arresto50

Dieci pazienti in cinque fold: per ciascuno si usa la predizione del fold in cui era fuori dal training. Soglia minima 0 mm3 sulle lesioni.

| | nnUNetTrainerWandb_250ep_arresto50 | nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50 | differenza |
| --- | --- | --- | --- |
| Dice medio sui 10 pazienti | 0.5814 | 0.5552 | **-0.0262** |
| Dice mediano | 0.6646 | 0.6001 | -0.0645 |
| lesioni rilevate su 71 | 27 (38.0%) | 25 (35.2%) | **-2** |
| falsi positivi (per paziente) | 79 (7.9) | 26 (2.6) | -53 |

Dice: migliora in **5** pazienti su 10, peggiora in **5**, Wilcoxon p = 0.922.
Rilevamento: migliora in **1**, peggiora in **4**, Wilcoxon p = 0.5.

## Paziente per paziente

| paziente | fold | Dice nnUNetTrainerWandb_250ep_arresto50 | Dice nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50 | Δ | lesioni | rilevate prima | rilevate dopo | FP prima | FP dopo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FSM_039 | 0 | 0.3599 | 0.0360 | -0.3239 | 2 | 1 | 1 | 4 | 0 |
| FSM_053 | 2 | 0.6431 | 0.4198 | -0.2233 | 6 | 6 | 6 | 5 | 10 |
| FSM_052 | 4 | 0.7675 | 0.5874 | -0.1801 | 10 | 4 | 2 | 9 | 2 |
| FSM_056 | 2 | 0.6861 | 0.6129 | -0.0733 | 7 | 3 | 2 | 3 | 0 |
| FSM_054 | 1 | 0.8803 | 0.8143 | -0.0661 | 15 | 4 | 7 | 8 | 3 |
| FSM_041 | 3 | 0.0444 | 0.0585 | +0.0141 | 6 | 2 | 1 | 7 | 0 |
| FSM_046 | 1 | 0.4479 | 0.5301 | +0.0822 | 2 | 2 | 2 | 39 | 10 |
| FSM_032 | 0 | 0.8066 | 0.8935 | +0.0870 | 10 | 1 | 1 | 1 | 0 |
| FSM_040 | 3 | 0.8253 | 0.9166 | +0.0913 | 12 | 3 | 2 | 2 | 0 |
| FSM_045 | 4 | 0.3529 | 0.6825 | +0.3296 | 1 | 1 | 1 | 1 | 1 |

Ordinati dal paziente che peggiora di piu' a quello che migliora di piu'. Con dieci coppie il p di Wilcoxon non scende sotto 0.002 nemmeno se tutte vanno nello stesso verso: contano le coppie e le grandezze, il p e' una cautela.
