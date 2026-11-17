# Due catene di misura per 505: da dove viene lo scarto

Prodotto da `scripts/scarto_misura_505.py`, **non modificare a mano**. 10 pazienti fuori-fold, nessun postprocessing (soglia 0.5), griglia T1 nativa, connettività 26, lesioni vere e falsi positivi contati sopra 1.001 mm³, lesione trovata se tocca almeno un voxel predetto.

- **maschera**: maschera binaria di nnU-Net riportata nel nativo (interpolazione lineare, taglio a 0.5; `porta_505_nativo.py`), come in `confronto_505_504.md`;
- **probabilità**: probabilità riportata nel nativo, taglio a 0.5 dopo (`postprocessing_aincp.py`);
- **gt504**: `gt_segmentations` di Dataset504 (prima pipeline);
- **pulita**: `lesione_pulita_inT1` (etichetta ripulita dai residui, `pipeline_aincp.py`), il riferimento di tutte le valutazioni di 506.

| modello | trasporto | riferimento | Dice medio | lesioni trovate | falsi positivi |
| --- | --- | --- | --- | --- | --- |
| Template, passo differenziato (505 A) | maschera | gt504 (= confronto_505_504) | 0.563 | 23/47 | 15 |
|  | maschera | pulita | 0.568 | 25/47 | 15 |
|  | probabilita | gt504 | 0.565 | 23/47 | 12 |
|  | probabilita | pulita (= postprocessing, default) | 0.569 | 25/47 | 12 |
| Template da Pretrain-Seq (505 B, da 542) | maschera | gt504 (= confronto_505_504) | 0.600 | 25/47 | 32 |
|  | maschera | pulita | 0.606 | 27/47 | 32 |
|  | probabilita | gt504 | 0.601 | 24/47 | 34 |
|  | probabilita | pulita (= postprocessing, default) | 0.607 | 26/47 | 35 |
| Template da Pretrain-Joint (505 D, da 543) | maschera | gt504 (= confronto_505_504) | 0.587 | 25/47 | 25 |
|  | maschera | pulita | 0.591 | 28/47 | 25 |
|  | probabilita | gt504 | 0.590 | 24/47 | 22 |
|  | probabilita | pulita (= postprocessing, default) | 0.595 | 27/47 | 22 |

Volume in cui i due riferimenti non coincidono (differenza simmetrica), per paziente: FSM_032 2 228 mm³, FSM_039 100 mm³, FSM_040 1 127 mm³, FSM_041 306 mm³, FSM_045 378 mm³, FSM_046 60 mm³, FSM_052 456 mm³, FSM_053 76 mm³, FSM_054 850 mm³, FSM_056 563 mm³.

- Template, passo differenziato (505 A): falsi positivi 15 / 15 con la maschera, 12 / 12 con la probabilità (gt504 / pulita); lesioni trovate 23 → 25 con la maschera e 23 → 25 con la probabilità passando da gt504 all'etichetta pulita.
- Template da Pretrain-Seq (505 B, da 542): falsi positivi 32 / 32 con la maschera, 34 / 35 con la probabilità (gt504 / pulita); lesioni trovate 25 → 27 con la maschera e 24 → 26 con la probabilità passando da gt504 all'etichetta pulita.
- Template da Pretrain-Joint (505 D, da 543): falsi positivi 25 / 25 con la maschera, 22 / 22 con la probabilità (gt504 / pulita); lesioni trovate 25 → 28 con la maschera e 24 → 27 con la probabilità passando da gt504 all'etichetta pulita.

Le due tabelle della tesi usano catene diverse: i loro «prima del postprocessing» non vanno confrontati fra loro.
