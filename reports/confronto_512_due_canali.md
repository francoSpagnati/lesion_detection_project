# Dataset512_WMH_T1_FLAIR contro Dataset501_WMH

| | casi | Dice medio |
| --- | --- | --- |
| Dataset501_WMH, tutta la sua validation | 16 | 0.7739 |
| Dataset512_WMH_T1_FLAIR, tutta la sua validation | 16 | 0.7535 |

## Confronto appaiato sugli stessi 16 casi

| | Dice medio |
| --- | --- |
| Dataset501_WMH | 0.7739 |
| Dataset512_WMH_T1_FLAIR | 0.7535 |
| **differenza** | **-0.0205** |

Migliora in **5** casi, peggiora in **11**, pari in 0.
Wilcoxon appaiato: **p = 0.02496**.

## I cinque casi che peggiorano di piu'

| caso | prima | dopo | delta |
| --- | --- | --- | --- |
| Utrecht_33 | 0.6970 | 0.5552 | -0.1418 |
| Amsterdam_GE1T5_156 | 0.7500 | 0.6739 | -0.0761 |
| Amsterdam_GE3T_123 | 0.4955 | 0.4264 | -0.0691 |
| Singapore_58 | 0.6762 | 0.6541 | -0.0221 |
| Singapore_56 | 0.8732 | 0.8639 | -0.0093 |
