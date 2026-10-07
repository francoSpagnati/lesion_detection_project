# Lesioni annotate: 90 nella griglia FLAIR, 108 nella griglia T1 nativa

Prodotto da `scripts/conta_lesioni_griglie.py`, **non modificare a mano**. Componenti a 26 vicini; nella FLAIR residui <= 1 mm³ tolti (come la pipeline), nella T1 contano le componenti >= 1 mm³ (come le metriche).

- componenti nella griglia FLAIR: **90**
- spezzate dal passaggio alla T1: **15**, che diventano 48 pezzi (**+33**); volumi 50–196 922 mm³
- senza nessun pezzo >= 1 mm³ nella T1: **13** (**−13**), tutte fra 1.0 e 10.0 mm³
- pezzi di lesioni diverse che nella T1 si toccano e si fondono: **−2**
- totale: 90 − 13 + 33 − 2 = **108** (nella T1 ci sono altre 45 componenti sotto 1 mm³, non contate)

| paziente | FLAIR | T1 (>= 1 mm³) | spezzate | senza pezzi >= 1 mm³ |
| --- | --- | --- | --- | --- |
| FSM_001 | 2 | 2 | 0 | 0 |
| FSM_028 | 6 | 9 | 2 | 2 |
| FSM_030 | 9 | 11 | 2 | 0 |
| FSM_032 | 3 | 3 | 0 | 0 |
| FSM_033 | 1 | 1 | 0 | 0 |
| FSM_035 | 5 | 7 | 2 | 1 |
| FSM_036 | 13 | 21 | 3 | 2 |
| FSM_039 | 2 | 2 | 0 | 0 |
| FSM_040 | 9 | 6 | 1 | 4 |
| FSM_041 | 5 | 5 | 0 | 0 |
| FSM_045 | 1 | 1 | 0 | 0 |
| FSM_046 | 2 | 3 | 1 | 0 |
| FSM_049 | 11 | 10 | 1 | 3 |
| FSM_052 | 5 | 8 | 1 | 0 |
| FSM_053 | 5 | 6 | 1 | 0 |
| FSM_054 | 6 | 8 | 1 | 1 |
| FSM_056 | 5 | 5 | 0 | 0 |
