# Obiettivo del progetto e piano di lavoro

Questo è il documento da leggere per primo: dice **a cosa serve il progetto**, quali dati abbiamo e
perché, e **cosa va fatto adesso**. Lo stato aggiornato è in [README.md](README.md#stato).

## L'obiettivo vero

Tesi: **rilevamento automatico di lesioni cerebrali in bambini con paralisi cerebrale**, su immagini di
risonanza magnetica. Il bersaglio comprende lesioni della sostanza bianca, malformazioni, cavità e altre
alterazioni strutturali.

Il problema che il progetto risolve: gli strumenti di analisi convenzionali si appoggiano a **template
cerebrali standard** e falliscono quando l'anatomia è fortemente alterata — cioè esattamente nei pazienti
che interessano.

La pipeline prevista ha **due stadi**:

1. **Individuazione delle lesioni** (dove siamo adesso). Segmentazione automatica delle alterazioni.
2. **Gemello sano virtuale.** Si applicano algoritmi di *inpainting* per rigenerare il tessuto lesionato
   e ottenere un'immagine sintetica priva di patologia. Su quel cervello virtuale — che ha anatomia
   regolare, quindi gli strumenti standard funzionano — si eseguono segmentazione e parcellazione
   anatomica, e le label ottenute vengono **riportate sull'immagine originale**. Così si ottengono misure
   quantitative affidabili anche su anatomie complesse.

Modello di riferimento: **nnU-Net 3D con deep supervision**. Ingressi previsti a regime: **T1 e FLAIR**.

## Il dato che decide tutto: il bersaglio finale ha ~10 casi

Il compito finale non è nessuno dei dataset pubblici. È il dataset privato **AInCP / Stella Maris**,
segmentato a mano dalla Dr.ssa Francy Cruz: circa **10 pazienti**, classificati con punteggio MRICS.
Non è ancora stato consegnato.

Da qui segue la conseguenza più importante per il piano: **con dieci casi il pretraining non è un
esperimento accessorio, è il meccanismo centrale del progetto.** Tutti i dataset pubblici servono a
costruire pesi da cui partire, non a produrre il modello finale.

## I quattro dataset e il ruolo di ciascuno

I tre pubblici non sono ridondanti: ciascuno copre una delle famiglie di lesione del bersaglio.

| | Compito | Popolazione | Canale | Ruolo nel progetto |
| --- | --- | --- | --- | --- |
| **WMH** | iperintensità della sostanza bianca | adulti, 4 scanner | FLAIR (+T1) | copre le lesioni della sostanza bianca; unico dataset con FLAIR |
| **ATLAS 2.0** | esiti di ictus | adulti, 33 coorti | T1 | copre cavità e lesioni focali; il più numeroso |
| **DBB** | 6 tessuti (non lesioni) | **pediatrica**, malformati | T1 | copre le malformazioni; anatomie alterate; parcellazione utile allo stadio 2 |
| **AInCP** (atteso) | lesioni da paralisi cerebrale | pediatrica | T1 + FLAIR | **il bersaglio** |

Dettaglio dei dataset e delle decisioni prese su ciascuno: [dataset-e-decisioni.md](dataset-e-decisioni.md).

DBB merita una nota: è l'unico costruito **come test di generalizzazione**. Si addestra su 207 cervelli
pediatrici sani e si valuta su 37 malformati (11 PFM, 10 MCD, 10 ACC, 6 HD), cioè su anatomie mai viste.
È la stessa domanda che conta per il bersaglio finale. Fornisce inoltre **maschere cerebrali per tutti i
955 soggetti**, quindi lì lo skull stripping è gratuito.

## Non sono tre versioni dello stesso problema

| | WMH | ATLAS | DBB |
| --- | --- | --- | --- |
| foreground mediano | 10.9 ml | 5.3 ml | 1479 ml |
| frazione del campo di vista | 0.0010 | 0.0006 | **0.2034** |

In WMH e ATLAS il bersaglio è **un voxel su mille**: individuazione di anomalie rare. In DBB è **un voxel
su cinque**: partizione del cervello in tessuti. Trecento volte di differenza significa loss diverse,
augmentation utili diverse, metriche che non si leggono allo stesso modo.

Da qui la conclusione: **un training congiunto con una testa condivisa non è praticabile.** Un voxel di
sostanza bianca sana è `3` in DBB e `0` in ATLAS; dandoli insieme alla stessa rete le si insegnano due
risposte opposte per lo stesso ingresso. Si potrebbe costruire un dataset unito a 8 classi usando la
`ignore` label di nnU-Net sulle regioni non annotate, ma resterebbe il divario adulti/pediatrico: molto
lavoro per un guadagno non dimostrato. **Si condivide il peso della rete, non l'etichetta.**

## Il vincolo T1 / FLAIR

nnU-Net pretende **tutti i canali per tutti i casi**. ATLAS e DBB non hanno FLAIR, quindi un modello a due
canali non può essere addestrato su di essi: resterebbero solo WMH e AInCP.

Ne servono quindi due filoni, e non è un compromesso ma la struttura corretta:

- **T1-only** per il pretraining ampio, perché è l'unica configurazione che copre tutti e quattro i
  dataset e che permette di trasferire pesi fra loro;
- **T1+FLAIR** per il modello finale, dove la FLAIR c'è (WMH e AInCP).

## Cosa permette davvero il transfer learning in nnU-Net

`nnUNetv2_train -pretrained_weights` trasferisce tutti i pesi **tranne i `seg_layers`**, le convoluzioni
finali che producono la mappa di segmentazione. Da qui due vincoli e una libertà:

- **stesso numero di canali di input.** Con la strategia T1-only sono tutti a 1 canale: soddisfatto.
- **architettura identica.** Nome e forma di ogni parametro non-`seg_layers` devono corrispondere.
  Poiché nnU-Net deriva l'architettura dal piano, i due dataset devono usare lo **stesso piano**, imposto
  con `nnUNetv2_move_plans_between_datasets`.
- **il numero di classi può differire.** Le teste non vengono trasferite, quindi 7 classi (DBB) → 2
  classi (ATLAS) è ammesso.

## Piano di lavoro, in ordine

**1. Baseline, un fold ciascuna.** `3d_fullres`, fold 0, su 501 (FLAIR), 502 (T1), 503 (T1). Senza numeri
di riferimento non c'è nulla contro cui misurare un guadagno. È il passo immediatamente successivo.

**2. Ablazione FLAIR: 511 contro 501.** Un solo training su un dataset piccolo (79 casi) e risponde
direttamente a "basta il T1?". Miglior rapporto fra informazione ottenuta e GPU spesa.

**3. Scelta della configurazione.** Su WMH la `2d` è una candidata seria e non una formalità: con fette
da 3 mm il contesto attraverso il piano è scarso. Si decide con
`nnUNetv2_find_best_configuration`, **sulla cross-validation, non sul test**.

**4. Transfer DBB → ATLAS.** L'unica coppia con una motivazione seria: stesso canale, stessa risoluzione,
spazi compatibili. L'idea è che le label dense dei 6 tessuti insegnino all'encoder il contrasto tissutale
del T1 e l'anatomia, che il compito lesionale poi riusa.

```bash
# il piano deve essere quello del bersaglio (502), applicato al dataset di pretraining (503)
nnUNetv2_move_plans_between_datasets -s 502 -t 503 -sp nnUNetPlans -tp nnUNetPlansDa502
nnUNetv2_preprocess -d 503 -plans_name nnUNetPlansDa502 -c 3d_fullres
nnUNetv2_train 503 3d_fullres 0 -p nnUNetPlansDa502 -tr nnUNetTrainerWandb
nnUNetv2_train 502 3d_fullres 0 -tr nnUNetTrainerWandb \
    -pretrained_weights "$nnUNet_results/Dataset503_DBB/nnUNetTrainerWandb__nnUNetPlansDa502__3d_fullres/fold_0/checkpoint_final.pth"
```

Il confronto va fatto contro la baseline 502 **sugli stessi fold**, altrimenti la differenza non è
attribuibile al pretraining.

**5. All'arrivo di AInCP:** fine-tuning dai pesi migliori, con validazione incrociata su così pochi casi
e metriche riportate per singolo paziente, non solo in media.

**6. Stadio 2 (inpainting e gemello sano):** interamente da fare. Va dopo, perché senza una maschera di
lesione affidabile non c'è niente da inpaintare. SynthSeg è il riferimento indicato per la parcellazione
sul cervello virtuale.

## Cosa non conviene fare

- **Training congiunto a testa condivisa** sui tre dataset: semantiche di label incompatibili.
- **La catena DBB → ATLAS → WMH.** Ogni fine-tuning successivo butta la testa di segmentazione e
  allontana dal dominio: WMH è l'unico FLAIR, anisotropo a 3 mm, in spazio nativo. Il terzo anello
  aggiunge divario, non informazione.
- **Usare i 711 soggetti DBB senza immagine.** Hanno label pubbliche ma le immagini di C-MIND e NIH-PED
  sono ad accesso ristretto: label senza volume non servono.

## Scostamenti dal preprocessing proposto nell'abstract

L'abstract propone una pipeline a mano (reorientation, resampling 1 mm, normalizzazione, padding/cropping
a dimensione fissa con salvataggio dei parametri per ripristinare). **nnU-Net la sussume**: fa
riorientamento (se richiesto), resampling e normalizzazione internamente, e rende il padding superfluo
perché lavora a patch con sliding window. Tre punti però restano aperti e vanno decisi:

| Punto dell'abstract | Stato reale |
| --- | --- |
| Riorientamento a RAS | attivo **solo su DBB** (`NibabelIOWithReorient`), dove serviva per il mix LAS/RAS. Se si fa transfer fra dataset con orientamenti diversi va esteso. |
| 1 mm isotropo per tutti | nnU-Net sceglie `1.2 × 0.98 × 3.0` mm per WMH, non 1 mm. Per un modello condiviso serve forzare una griglia comune con `-overwrite_target_spacing`, pagando l'interpolazione di fette da 3 mm. |
| N4 bias correction | **già applicata a monte** dagli organizzatori, e verificata da noi: campo piatto dentro il cervello. Vedi [dataset-e-decisioni.md](dataset-e-decisioni.md#niente-n4-la-misura-lo-esclude). |
| Random Bias Field in augmentation | **l'unica voce non coperta** da nnU-Net. Le altre (flip, affine, rumore, blur, gamma, bassa risoluzione) ci sono già. Se serve, va aggiunta nel trainer. |
