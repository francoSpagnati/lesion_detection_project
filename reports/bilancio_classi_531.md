# Bilancio delle classi in Dataset531_UNION_D

Misurato sulle segmentazioni preprocessate a 1 mm, cioe' su cio' che la rete vede davvero.

| sorgente | casi | foreground mediano per caso (voxel) | frazione del volume |
| --- | --- | --- | --- |
| wmh | 79 | 9,384 | 0.2054% |
| atlas | 556 | 5,232 | 0.3761% |
| dbb | 207 | 1,483,524 | 22.0594% |

**Un caso DBB porta 266 volte il foreground di un caso lesionale** (mediane: 1,483,524 contro 5,568 voxel).

## Composizione del foreground totale

| classe | voxel | quota del foreground |
| --- | --- | --- |
| 1 — lesione | 19,411,530 | 5.87% |
| 2 — csf | 49,469,163 | 14.95% |
| 3 — grigia | 127,708,714 | 38.61% |
| 4 — bianca | 86,855,819 | 26.26% |
| 5 — grigia profonda | 9,230,969 | 2.79% |
| 6 — tronco | 4,274,091 | 1.29% |
| 7 — cervelletto | 33,840,480 | 10.23% |

La lesione e' il **5.87%** di tutto il foreground annotato. nnU-Net forza il 33% delle patch a contenere foreground, ma non bilancia fra le classi: quelle patch andranno quindi quasi tutte su tessuto DBB. E' il motivo per cui il risultato di D va letto per classe e per sorgente, mai come un Dice medio unico.
