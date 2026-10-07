# Lesioni annotate e bersaglio a 1 mm nel template pediatrico (17 pazienti)

Prodotto da `scripts/sopravvivenza_1mm_506.py`, **non modificare a mano**. Componenti a 26 vicini nella griglia della FLAIR originale, residui <= 1 mm³ esclusi, portate una per una nel template con la catena della pipeline (lineare, soglia 0.5).

| fenotipo | lesioni | scompaiono | sotto 10 mm³ | di cui scompaiono |
| --- | --- | --- | --- | --- |
| gliosis | 77 | 3 | 41 | 3 |
| cystic lesion | 11 | 1 | 4 | 1 |
| encephalomalacia | 2 | 0 | 0 | 0 |
| **totale** | 90 | 4 | | |

Volume massimo di una lesione persa: 3.0 mm³.

Controllo: voxel in cui l'unione delle componenti differisce dal bersaglio salvato, in % del bersaglio: FSM_001 0.00, FSM_028 0.00, FSM_030 0.00, FSM_032 0.00, FSM_033 0.00, FSM_035 0.00, FSM_036 0.00, FSM_039 0.00, FSM_040 0.00, FSM_041 0.00, FSM_045 0.00, FSM_046 0.00, FSM_049 0.00, FSM_052 0.00, FSM_053 0.00, FSM_054 0.00, FSM_056 0.00.
