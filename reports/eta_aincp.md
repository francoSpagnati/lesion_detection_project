# Età dei pazienti AInCP (17)

Prodotto da `scripts/eta_aincp.py`, **non modificare a mano**. Fonte: campo DICOM «Patient's Age» (anni compiuti) nei fogli di metadati dei due file in `docs/`.

Età: da 3 a 15 anni; mediana 7; media ± deviazione standard **7.7 ± 3.7** anni (deviazione standard campionaria, n − 1; con n al denominatore 3.5). Sotto i 5 anni (strettamente): **4** (FSM_028, FSM_032, FSM_035, FSM_036). A esattamente 5 anni: **2** (FSM_030, FSM_033). Oltre i 5 anni: **11**.

| paziente | età (anni) | fonte (file, foglio, valore) | sotto 5 anni |
| --- | --- | --- | --- |
| FSM_001 | 7 | v2 FSM «007Y»; v2 SITE & MRFIELD «007Y»; v2 Sheet1 «007Y» | no |
| FSM_028 | 4 | v2 Sheet1 «004Y» | sì |
| FSM_030 | 5 | v2 Sheet1 «005Y» | 5 anni compiuti |
| FSM_032 | 3 | v1 FSM «003Y»; v1 SITE & MRFIELD «003Y» | sì |
| FSM_033 | 5 | v2 Sheet1 «005Y» | 5 anni compiuti |
| FSM_035 | 4 | v2 Sheet1 «004Y» | sì |
| FSM_036 | 3 | v2 Sheet1 «003Y» | sì |
| FSM_039 | 6 | v1 FSM «006Y»; v1 SITE & MRFIELD «006Y» | no |
| FSM_040 | 9 | v1 FSM «009Y»; v1 SITE & MRFIELD «009Y» | no |
| FSM_041 | 10 | v1 FSM «010Y»; v1 SITE & MRFIELD «010Y» | no |
| FSM_045 | 6 | v1 FSM «006Y»; v1 SITE & MRFIELD «006Y» | no |
| FSM_046 | 12 | v1 FSM «012Y»; v1 SITE & MRFIELD «012Y» | no |
| FSM_049 | 12 | v2 Sheet1 «012Y» | no |
| FSM_052 | 12 | v1 FSM «012Y» | no |
| FSM_053 | 8 | v1 FSM «008Y» | no |
| FSM_054 | 15 | v1 FSM «015Y» | no |
| FSM_056 | 10 | v1 FSM «010Y» | no |

Controlli: nessun problema.
