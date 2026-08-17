# Notebook

| Notebook | Scopo |
| --- | --- |
| `dataset_preparation.ipynb` | **Preparazione di WMH e ATLAS**: da dati grezzi a dataset nnU-Net v2, con analisi della risoluzione, QC delle label e split stratificati riproducibili. Genera `reports/dataset_preparation.*`. |
| `dataset_comparison_exploration.ipynb` | Confronto esplorativo read-only WMH vs ATLAS: geometria, risoluzione, overlay. |
| `wmh_validation.ipynb` | Verifica rapida del dataset WMH e dei percorsi nnU-Net. |

DBB non ha un notebook: è stato analizzato e convertito da script
(`scripts/analyze_dbb.py`, `scripts/convert_dbb_to_nnunet.py`), vedi
[`docs/cosa-e-stato-fatto.md`](../docs/cosa-e-stato-fatto.md).
