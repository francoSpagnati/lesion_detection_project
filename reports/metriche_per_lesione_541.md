# Rilevamento per lesione nel test di pretraining (541), 35 casi ATLAS a 1.5 T

Prodotto da `scripts/metriche_per_lesione_541.py`, **non modificare a mano**. Rilevata = almeno un voxel predetto dentro la lesione; falso positivo = componente predetta che non tocca lesioni vere.

| braccio | checkpoint | lesioni trovate | falsi positivi | per caso |
| --- | --- | --- | --- | --- |
| `521` senza adattamento | — | 44/61 | 28 | 0.8 |
| da zero | ultima epoca | 30/61 | 19 | 0.5 |
| da zero | checkpoint_best | 29/61 | 24 | 0.7 |
| fine-tuning, passo 1e-2 | ultima epoca | 37/61 | 13 | 0.4 |
| fine-tuning, passo 1e-2 | checkpoint_best | 47/61 | 32 | 0.9 |
| fine-tuning, passo 1e-3 | ultima epoca | 39/61 | 18 | 0.5 |
| fine-tuning, passo 1e-3 | checkpoint_best | 43/61 | 31 | 0.9 |
| fine-tuning, passo 1e-4 | ultima epoca | 45/61 | 39 | 1.1 |
| fine-tuning, passo 1e-4 | checkpoint_best | 46/61 | 36 | 1.0 |
