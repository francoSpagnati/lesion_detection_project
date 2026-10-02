# Lesioni perse dallo skull stripping

Frazione di voxel di lesione annotata che cade **fuori** dalla maschera
di SynthStrip. Decide se la maschera puo' essere usata per ritagliare le
immagini o solo come canale/`ignore` label.

| dataset | casi | frazione media | mediana | massima | casi >1% | casi >5% |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset501_WMH | 79 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| Dataset502_ATLAS | 556 | 0.0012 | 0.0000 | 0.0649 | 18 | 3 |

## I dieci casi che perderebbero di piu'

| caso | dataset | lesione (mm³) | fuori (mm³) | frazione |
| --- | --- | --- | --- | --- |
| sub-r038s061 | Dataset502_ATLAS | 196 000 | 12 720 | 0.0649 |
| sub-r028s014 | Dataset502_ATLAS | 8 600 | 470 | 0.0546 |
| sub-r040s044 | Dataset502_ATLAS | 275 300 | 14 370 | 0.0522 |
| sub-r042s018 | Dataset502_ATLAS | 224 100 | 10 360 | 0.0462 |
| sub-r040s030 | Dataset502_ATLAS | 359 800 | 13 040 | 0.0362 |
| sub-r001s020 | Dataset502_ATLAS | 78 400 | 2 740 | 0.0350 |
| sub-r034s003 | Dataset502_ATLAS | 45 100 | 1 490 | 0.0331 |
| sub-r040s067 | Dataset502_ATLAS | 221 000 | 4 560 | 0.0206 |
| sub-r040s086 | Dataset502_ATLAS | 297 100 | 5 630 | 0.0190 |
| sub-r040s059 | Dataset502_ATLAS | 80 600 | 1 500 | 0.0186 |
