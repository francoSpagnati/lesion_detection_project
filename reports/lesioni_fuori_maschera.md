# Lesioni perse dallo skull stripping

Frazione di voxel di lesione annotata che cade **fuori** dalla maschera
di SynthStrip. Decide se la maschera puo' essere usata per ritagliare le
immagini o solo come canale/`ignore` label.

| dataset | casi | frazione media | mediana | massima | casi >1% | casi >5% |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset501_WMH | 79 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| Dataset502_ATLAS | 556 | 0.0012 | 0.0000 | 0.0649 | 18 | 3 |

## I dieci casi che perderebbero di piu'

| caso | dataset | lesione (ml) | fuori (ml) | frazione |
| --- | --- | --- | --- | --- |
| sub-r038s061 | Dataset502_ATLAS | 196.0 | 12.72 | 0.0649 |
| sub-r028s014 | Dataset502_ATLAS | 8.6 | 0.47 | 0.0546 |
| sub-r040s044 | Dataset502_ATLAS | 275.3 | 14.37 | 0.0522 |
| sub-r042s018 | Dataset502_ATLAS | 224.1 | 10.36 | 0.0462 |
| sub-r040s030 | Dataset502_ATLAS | 359.8 | 13.04 | 0.0362 |
| sub-r001s020 | Dataset502_ATLAS | 78.4 | 2.74 | 0.0350 |
| sub-r034s003 | Dataset502_ATLAS | 45.1 | 1.49 | 0.0331 |
| sub-r040s067 | Dataset502_ATLAS | 221.0 | 4.56 | 0.0206 |
| sub-r040s086 | Dataset502_ATLAS | 297.1 | 5.63 | 0.0190 |
| sub-r040s059 | Dataset502_ATLAS | 80.6 | 1.50 | 0.0186 |
