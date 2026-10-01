# Split di `Dataset506_AINCP_FINALE`

Prodotto da `scripts/pipeline_aincp.py dataset`, **non modificare a mano**. Fissato prima di qualunque training; il test si usa una volta sola.

| paziente | insieme | strato | centro | campo | FLAIR | fenotipi | lesione (ml) | % cervello |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FSM_001 | addestr. | altri centri | Az. Osp. Niguarda MI | 1.5 T | si | si | 0.17 | 0.01 |
| FSM_028 | addestr. | Stella Maris 1.5 T | Stella Maris | 1.5 T | si | si | 157.11 | 12.12 |
| FSM_030 | **test** | Stella Maris 1.5 T | Stella Maris | 1.5 T | si | si | 4.88 | 0.45 |
| FSM_032 | addestr. | Stella Maris 3 T | Stella Maris | 3 T | si | si | 123.64 | 9.17 |
| FSM_033 | addestr. | Stella Maris 1.5 T | Stella Maris | 1.5 T | si | si | 0.67 | 0.05 |
| FSM_035 | addestr. | Stella Maris 1.5 T | Stella Maris | 1.5 T | si | si | 40.74 | 3.97 |
| FSM_036 | addestr. | Stella Maris 1.5 T | Stella Maris | 1.5 T | si | si | 108.97 | 9.38 |
| FSM_039 | addestr. | Stella Maris 3 T | Stella Maris | 3 T | si | si | 1.39 | 0.09 |
| FSM_040 | addestr. | Stella Maris 3 T | Stella Maris | 3 T | si | si | 55.59 | 4.04 |
| FSM_041 | addestr. | Stella Maris 3 T | Stella Maris | 3 T | si | si | 1.90 | 0.14 |
| FSM_045 | addestr. | Stella Maris 3 T | Stella Maris | 3 T | si | si | 9.53 | 0.86 |
| FSM_046 | **test** | Stella Maris 3 T | Stella Maris | 3 T | si | si | 0.45 | 0.03 |
| FSM_049 | **test** | altri centri | IRCCS Burlo Garofolo | 1.5 T | si | si | 196.85 | 16.21 |
| FSM_052 | **test** | Stella Maris 3 T | Stella Maris | 3 T | si | si | 4.78 | 0.36 |
| FSM_053 | addestr. | Stella Maris 3 T | Stella Maris | 3 T | si | si | 0.58 | 0.04 |
| FSM_054 | **test** | Stella Maris 3 T | Stella Maris | 3 T | si | si | 33.66 | 2.52 |
| FSM_056 | addestr. | Stella Maris 3 T | Stella Maris | 3 T | si | si | 12.44 | 0.88 |

Fold 0, validazione: FSM_032, FSM_033, FSM_040

Fold 1, validazione: FSM_001, FSM_035, FSM_056

Fold 2, validazione: FSM_036, FSM_045

Fold 3, validazione: FSM_028, FSM_041

Fold 4, validazione: FSM_039, FSM_053
