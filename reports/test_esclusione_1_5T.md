# Dataset521_ATLAS_3T contro Dataset502_ATLAS

| | casi | Dice medio |
| --- | --- | --- |
| Dataset502_ATLAS, tutta la sua validation | 112 | 0.6103 |
| Dataset521_ATLAS_3T, tutta la sua validation | 103 | 0.6160 |

## Confronto appaiato sugli stessi 103 casi

| | Dice medio |
| --- | --- |
| Dataset502_ATLAS | 0.6066 |
| Dataset521_ATLAS_3T | 0.6160 |
| **differenza** | **+0.0093** |

Migliora in **48** casi, peggiora in **51**, pari in 4.
Wilcoxon appaiato: **p = 0.6677**.

I 9 casi presenti solo in Dataset502_ATLAS avevano Dice **0.6528** (mediana 0.7930).

## I cinque casi che peggiorano di piu'

| caso | prima | dopo | delta |
| --- | --- | --- | --- |
| sub-r010s003 | 0.6441 | 0.4195 | -0.2246 |
| sub-r009s011 | 0.6160 | 0.4086 | -0.2074 |
| sub-r009s100 | 0.2263 | 0.0347 | -0.1916 |
| sub-r009s039 | 0.5846 | 0.4762 | -0.1084 |
| sub-r009s096 | 0.5821 | 0.4827 | -0.0994 |
