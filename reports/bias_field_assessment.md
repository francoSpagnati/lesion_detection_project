# Fase 0 - bias di campo residuo

Generato da `scripts/measure_bias_field.py`. Nessun dato e' stato modificato.

Il campo di bias e' stimato con N4 (ANTs) e misurato dentro la testa. `p99/p1` e' il
rapporto fra il valore alto e quello basso del campo: 1.00 = campo piatto, nessun bias.

La misura e' ripetuta erodendo la maschera di un margine **fisico** crescente. Il valore
"bordo" usa tutta la maschera, il valore "interno" solo la parte piu' profonda: se il
secondo e' molto piu' basso del primo, l'ampiezza veniva dal contorno (scalpo, grasso)
e non da un bias interno al cervello.

Il confronto e' **appaiato**: la tabella segue i soli casi che arrivano al margine piu'
profondo, misurati a tutte le profondita'. Aggregare il "margine massimo raggiunto" caso
per caso falserebbe il confronto, perche' le teste piccole si fermano prima e mostrano
quindi rapporti sistematicamente piu' alti.

| Dataset | Casi appaiati | Margine max | p99/p1 bordo | p99/p1 interno | max interno | Giudizio | Azione |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WMH | 7 | 12 mm | 1.678 | 1.119 | 1.288 | moderato | N4 opzionale, solo come variante misurata contro la baseline |
| ATLAS | 6 | 12 mm | 1.594 | 1.120 | 1.520 | moderato | N4 opzionale, solo come variante misurata contro la baseline |
| DBB | 10 | 12 mm | 1.172 | 1.095 | 1.439 | trascurabile | nessuna correzione N4: campo piatto all'interno |

## Profilo di erosione appaiato (mediana sugli stessi casi)

| Dataset | 0 mm | 3 mm | 6 mm | 12 mm |
| --- | --- | --- | --- | --- |
| WMH | 1.678 | 1.550 | 1.163 | 1.119 |
| ATLAS | 1.594 | 1.548 | 1.448 | 1.120 |
| DBB | 1.172 | 1.148 | 1.124 | 1.095 |

## Limite della misura

La maschera non ha la stessa natura nei tre dataset, e questo va tenuto presente prima di
confrontare i numeri a profondita' bassa:

| Dataset | Maschera | Volume mediano |
| --- | --- | --- |
| WMH | testa (soglia di intensita) | 1 573 000 mm³ |
| ATLAS | testa (soglia di intensita) | 2 770 000 mm³ |
| DBB | cervello (fornita dal dataset) | 1 487 000 mm³ |

Dove la maschera e' la **testa** (WMH, ATLAS) i primi millimetri di erosione cadono ancora
su cranio, scalpo e grasso sottocutaneo, dove N4 estrapola male: il rapporto misurato li'
sovrastima il bias interno al cervello. Dove la maschera e' gia' il **cervello** (DBB) la
misura parte pulita. Il confronto fra dataset e' quindi affidabile solo al margine piu'
profondo. Una misura definitiva per WMH e ATLAS richiederebbe prima lo skull stripping.

## Note per dataset

- **WMH**: file gia' corretti dagli organizzatori (_BiasCorr)
- **ATLAS**: T1w in spazio MNI, preprocessing degli autori non documentato come N4
- **DBB**: T1 pediatrici in spazio MNI, maschera cerebrale fornita dal dataset
