# Notebook

| Notebook | Scopo |
| --- | --- |
| `dataset_preparation.ipynb` | **Preparazione di WMH e ATLAS**: da dati grezzi a dataset nnU-Net v2, con analisi della risoluzione, QC delle label e split stratificati riproducibili. Genera `reports/dataset_preparation.*`. |
| `dataset_comparison_exploration.ipynb` | Confronto esplorativo read-only WMH vs ATLAS: geometria, risoluzione, overlay. |
| `preprocessing_501_502_503.ipynb` | Planning e preprocessing nnU-Net (`nnUNetv2_plan_and_preprocess`) di Dataset501, 502 e 503, prima del primo training. |
| `wmh_validation.ipynb` | Verifica rapida del dataset WMH e dei percorsi nnU-Net. |

`dataset_comparison_exploration.ipynb` è pubblicato senza output: per riavere
tabelle e figure va eseguito.

`aincp_primo_lotto.ipynb` esiste in locale ma non è su GitHub, perché contiene
immagini di pazienti. Si rigenera con
[`scripts/costruisci_notebook_aincp.py`](../scripts/costruisci_notebook_aincp.py).

DBB non ha un notebook: è stato analizzato e convertito da script
(`scripts/analyze_dbb.py`, `scripts/convert_dbb_to_nnunet.py`), vedi
[`docs/cosa-e-stato-fatto.md`](../docs/cosa-e-stato-fatto.md).
