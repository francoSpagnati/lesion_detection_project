# Prompt di avvio per una nuova sessione

Copia il blocco qui sotto come **primo messaggio** di una chat nuova. Serve a far
partire la sessione dai documenti invece di far riesplorare i dati da zero.

Quando lo stato del progetto avanza, aggiorna la sezione "Cosa fare adesso": è la
sola parte che invecchia.

---

```text
Progetto: /home/ccantini/lesion_detection_project — segmentazione di lesioni
cerebrali con nnU-Net v2, tesi su paralisi cerebrale infantile.

PRIMA DI QUALSIASI COSA, leggi in quest'ordine:
  1. docs/obiettivo-e-piano.md    (a cosa serve il progetto e cosa va fatto)
  2. docs/README.md               (indice + tabella di stato)
  3. docs/cosa-e-stato-fatto.md   (quale script ha fatto cosa)
Poi fermati e dimmi cosa hai capito dello stato attuale, in poche righe, prima di
agire. Leggi docs/dataset-e-decisioni.md e docs/training.md solo quando ti servono.

Non riesplorare i dataset: sono già misurati e i numeri stanno in reports/
(rigenerabili, non modificarli a mano). Non rilanciare gli script di analisi se
non è cambiato niente. Il progetto è un repo git: usa `git log` e `git show` per
la storia invece di rileggere tutto.

CONTESTO SU DI ME
Non ho background in risonanza magnetica e neuroimaging: spiegami i concetti di
dominio, non darli per scontati. Scrivimi in italiano.

VINCOLI OPERATIVI
- /data01 è in SOLA LETTURA: i dati sorgente non si modificano mai.
- `source scripts/env.sh` prima di ogni comando nnU-Net (imposta anche
  nnUNet_extTrainer, senza cui i nostri trainer non vengono trovati).
- Le 4 A100 sono condivise: controlla `nvidia-smi` e imposta CUDA_VISIBLE_DEVICES
  sulla scheda libera. Non fidarti di quale era libera l'ultima volta.
- splits_final.json NON si rigenera: se cambia, i confronti fra run perdono
  validità.
- .env contiene WANDB_API_KEY: non citarlo, non stamparlo, non committarlo.

COME VOGLIO IL CODICE
- Script in scripts/, output rigenerabile in reports/, commenti in italiano nello
  stile di quelli già presenti.
- Prima di implementare qualcosa, controlla se nnU-Net lo fa già: è installato in
  ~/.venv (nnunetv2 2.8.1) e il sorgente è leggibile. In questo progetto ho già
  perso tempo reimplementando un logger W&B che nnU-Net 2.8 aveva integrato.
- Misura prima di costruire una pipeline. La correzione N4 è stata scartata proprio
  così: mezza giornata di misura ha evitato di costruire un preprocessing inutile.
- Un esperimento cambia UNA variabile per volta, e le varianti si fanno come
  dataset nuovi con un nuovo ID, non sovrascrivendo l'originale.
- Verifica prima di affermare. Se un numero non l'hai calcolato in questa sessione
  o non è in un report, dillo invece di stimarlo.
- Documentazione: consolida nei file esistenti, non aggiungerne di nuovi. Compatta.

COSA FARE ADESSO
Preprocessing e prime baseline:
  nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
  nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity
  nnUNetv2_plan_and_preprocess -d 503 --verify_dataset_integrity
poi fold 0 di 3d_fullres su 501, 502 e 503 con -tr nnUNetTrainerWandb, e confronto
della target spacing effettiva con quella prevista (docs/training.md, Step 3).
Prima di lanciare, controlla lo spazio su /home e la GPU libera.
```

---

## Perché è fatto così

Quattro scelte deliberate, che valgono anche se lo riscrivi:

- **Chiede di leggere e poi fermarsi.** Senza quel vincolo una sessione nuova
  comincia a esplorare i dati e spende decine di migliaia di token per riscoprire
  cose già scritte.
- **Vieta esplicitamente la riesplorazione** e indica dove sono già i numeri. È la
  singola riga che fa risparmiare più token.
- **Cita gli errori già commessi** (il logger W&B reimplementato) invece di dare
  regole astratte: un esempio concreto è più efficace di "non reinventare la ruota".
- **Separa ciò che invecchia** — la sezione finale — da ciò che resta valido, così
  aggiornarlo costa una modifica sola.
