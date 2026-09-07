# Progetto: rilevamento di lesioni cerebrali nella paralisi cerebrale infantile

Tesi magistrale. Segmentazione automatica di lesioni con nnU-Net v2. Il bersaglio finale è un
dataset privato di ~20 pazienti AInCP / Stella Maris: i dataset pubblici servono a costruire i
**pesi di partenza**, non sono il compito.

## Vincoli operativi — non negoziabili

- **`/data01` è in sola lettura.** I dati sorgente non si modificano mai. Tutto l'output su `/home`,
  che è oltre il 90% pieno: `df -h /home` prima di ogni training.
- **`source scripts/env.sh`** prima di ogni comando nnU-Net. Imposta anche `nnUNet_extTrainer`,
  senza cui i trainer del progetto non vengono trovati.
- **Le 4 A100 sono condivise con altri utenti.** `nvidia-smi` prima di ogni lancio, e
  `CUDA_VISIBLE_DEVICES` sulla scheda libera. Se sono tutte occupate non si lancia: si aspetta, o
  si usa `-device cpu` per i lavori brevi.
- **I training si lanciano con `setsid nohup ... &`.** `nohup` da solo non protegge dalla
  terminazione del gruppo di processi: un training è già stato perso all'epoca 189 così.
- **Il progresso non si legge in `reports/logs/`**, che è stdout bufferizzato e resta indietro. Si
  legge in `nnUNet/nnUNet_results/Dataset<NNN>_*/*/fold_0/training_log_*.txt`.
- **`splits_final.json` non si rigenera mai.** Se cambia, tutti i confronti fra run perdono validità.
- **Il test set non si guarda.** Non è mai stato toccato in nessun dataset, e va usato una volta
  sola alla fine, sulla configurazione già decisa.
- **`.env` contiene `WANDB_API_KEY`**: non citarlo, non stamparlo, non committarlo.
- **Non ci sono credenziali GitHub su questa macchina.** I commit restano locali; il push lo fa
  l'utente. Riportare quanti commit restano da pushare a fine blocco di lavoro.

## Come si lavora qui

- **Verifica prima di affermare.** Se un numero non è stato calcolato in questa sessione o non sta
  in un file sotto `reports/`, dirlo invece di stimarlo.
- **Misura prima di costruire.** La correzione N4 è stata scartata così, e lo skull stripping pure:
  mezza giornata di misura ha evitato settimane di pipeline inutile.
- **Misura sull'intero dataset, non su un campione.** La perdita di lesione da skull stripping
  sembrava nulla su 100 casi; su tutti e 635 aveva tre casi sopra il 5%.
- **Un esperimento cambia una variabile per volta.** Le varianti sono dataset nuovi con un ID nuovo,
  mai sovrascrivendo l'originale. Convenzione: 50x principali, 51x/52x varianti, 53x unioni,
  54x esperimenti di pretraining, 599 usa e getta.
- **I confronti si fanno appaiati caso per caso** (`scripts/confronto_appaiato.py`), mai fra due
  medie complessive: se due modelli non hanno la stessa validation, la differenza può venire dalla
  composizione del campione.
- **Il Dice da solo non basta.** Su questo compito dipende quasi solo dal volume della lesione, e
  non dice nulla su quante lesioni si sono perse. Va sempre affiancato da
  `scripts/metriche_per_lesione.py`.
- **Prima di implementare, controllare se nnU-Net lo fa già.** È in `~/.venv` (nnunetv2 2.8.1) e il
  sorgente è leggibile. In questo progetto si è già perso tempo reimplementando un logger W&B che
  nnU-Net 2.8 aveva integrato.
- **Codice**: script in `scripts/`, output rigenerabile in `reports/`, commenti in italiano che
  spiegano il *perché* e non il *cosa*, nello stile di quelli già presenti.
- **Documentazione**: consolidare nei file esistenti in `docs/`, non aggiungerne di nuovi, tenerla
  compatta. Ogni nuovo training va aggiunto a `docs/modelli-ed-esperimenti.md`.

## Comunicazione

L'utente **non ha background in risonanza magnetica e neuroimaging**: i concetti di dominio vanno
spiegati, non dati per scontati. Scrivere sempre in italiano. Se un risultato contraddice qualcosa
detto in precedenza, dirlo esplicitamente invece di lasciarlo passare.
