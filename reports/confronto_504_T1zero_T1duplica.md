# `Dataset504_AINCP`: nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50 contro nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50

Dieci pazienti in cinque fold: per ciascuno si usa la predizione del fold in cui era fuori dal training. Soglia minima 0 mm3 sulle lesioni.

| | nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50 | nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50 | differenza |
| --- | --- | --- | --- |
| Dice medio sui 10 pazienti | 0.5552 | 0.5960 | **+0.0409** |
| Dice mediano | 0.6001 | 0.6061 | +0.0060 |
| lesioni rilevate su 71 | 25 (35.2%) | 20 (28.2%) | **-5** |
| falsi positivi (per paziente) | 26 (2.6) | 32 (3.2) | +6 |

Dice: migliora in **6** pazienti su 10, peggiora in **4**, Wilcoxon p = 0.375.
Rilevamento: migliora in **2**, peggiora in **3**, Wilcoxon p = 0.812.

## Paziente per paziente

| paziente | fold | Dice nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50 | Dice nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50 | Δ | lesioni | rilevate prima | rilevate dopo | FP prima | FP dopo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FSM_045 | 4 | 0.6825 | 0.4314 | -0.2511 | 1 | 1 | 1 | 1 | 3 |
| FSM_040 | 3 | 0.9166 | 0.8897 | -0.0269 | 12 | 2 | 1 | 0 | 2 |
| FSM_032 | 0 | 0.8935 | 0.8674 | -0.0261 | 10 | 1 | 1 | 0 | 1 |
| FSM_041 | 3 | 0.0585 | 0.0380 | -0.0205 | 6 | 1 | 2 | 0 | 2 |
| FSM_056 | 2 | 0.6129 | 0.6328 | +0.0199 | 7 | 2 | 3 | 0 | 4 |
| FSM_046 | 1 | 0.5301 | 0.5640 | +0.0339 | 2 | 2 | 2 | 10 | 12 |
| FSM_054 | 1 | 0.8143 | 0.8728 | +0.0586 | 15 | 7 | 2 | 3 | 3 |
| FSM_052 | 4 | 0.5874 | 0.6555 | +0.0681 | 10 | 2 | 2 | 2 | 0 |
| FSM_053 | 2 | 0.4198 | 0.5794 | +0.1596 | 6 | 6 | 5 | 10 | 2 |
| FSM_039 | 0 | 0.0360 | 0.4292 | +0.3932 | 2 | 1 | 1 | 0 | 3 |

Ordinati dal paziente che peggiora di piu' a quello che migliora di piu'. Con dieci coppie il p di Wilcoxon non scende sotto 0.002 nemmeno se tutte vanno nello stesso verso: contano le coppie e le grandezze, il p e' una cautela.
