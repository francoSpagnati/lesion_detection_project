# Lesioni perse dallo skull stripping

Frazione di voxel di lesione annotata che cade **fuori** dalla maschera
di SynthStrip. Decide se la maschera puo' essere usata per ritagliare le
immagini o solo come canale/`ignore` label.

| dataset | casi | frazione media | mediana | massima | casi >1% | casi >5% |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset501_WMH | 79 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| Dataset502_ATLAS | 100 | 0.0007 | 0.0000 | 0.0186 | 3 | 0 |

## I dieci casi che perderebbero di piu'

| caso | dataset | lesione (ml) | fuori (ml) | frazione |
| --- | --- | --- | --- | --- |
| sub-r040s059 | Dataset502_ATLAS | 80.6 | 1.50 | 0.0186 |
| sub-r031s034 | Dataset502_ATLAS | 147.5 | 2.24 | 0.0152 |
| sub-r040s037 | Dataset502_ATLAS | 374.0 | 4.17 | 0.0111 |
| sub-r042s008 | Dataset502_ATLAS | 169.7 | 1.35 | 0.0079 |
| sub-r018s011 | Dataset502_ATLAS | 108.1 | 0.54 | 0.0050 |
| sub-r038s028 | Dataset502_ATLAS | 51.9 | 0.14 | 0.0027 |
| sub-r023s008 | Dataset502_ATLAS | 204.5 | 0.48 | 0.0023 |
| sub-r038s084 | Dataset502_ATLAS | 41.9 | 0.07 | 0.0017 |
| sub-r040s076 | Dataset502_ATLAS | 194.6 | 0.25 | 0.0013 |
| sub-r038s082 | Dataset502_ATLAS | 2.6 | 0.00 | 0.0008 |
