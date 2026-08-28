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

## Ordine di lavoro deciso il 28 agosto

**La variante D — un solo modello su tutti e tre i dataset a 1 mm — è l'obiettivo prioritario.** A, B e C
restano come alternative se questa strada non regge. I tre test che la precedono non sono esperimenti
indipendenti: ciascuno fissa un parametro con cui D verrà costruita.

| | test | cosa decide di D | stato |
| --- | --- | --- | --- |
| 1 | solo 3 T contro tutto (su ATLAS) | **quali casi** mettere in D | **in corso**, `Dataset521_ATLAS_3T` |
| 2 | skull stripping: funziona? | **se** applicarlo a D, e sblocca C | da fare |
| 3 | modello a due canali T1+FLAIR | **quanto costa** la strada D | da fare |
| 4 | **variante D** | — | obiettivo |

### D è necessariamente a un solo canale

Vincolo da tenere presente fin d'ora, perché non è aggirabile: nnU-Net pretende **tutti i canali per tutti
i casi**, e ATLAS e DBB non hanno la FLAIR. Quindi in D **WMH deve entrare con il canale T1**, cioè con i
dati di `Dataset511_WMH_T1` e non di `Dataset501_WMH`.

Il costo è già misurato: WMH in T1 rende **0.626** contro **0.774** con la FLAIR, e la perdita si
concentra sulle lesioni piccole (−0.24 contro −0.05 su quelle grandi). D parte quindi con uno svantaggio
noto sulla componente WMH, e questo va tenuto presente leggendo il suo risultato: un Dice più basso lì non
significa che l'unione abbia fallito.

Da qui il senso del test 3: **non** è un parametro di D, che resta a un canale comunque. È il confronto
fra due strategie alternative — un modello unico su tutto in T1, contro un modello che usa la FLAIR dove
esiste. Misura quanto costa la strada scelta.

### Il test 1 va misurato sugli stessi casi

Togliendo i casi a 1.5 T cambia anche l'insieme su cui si valuta, e un Dice più alto potrebbe venire solo
dall'aver rimosso casi difficili dalla validation. Per questo i fold di `Dataset521_ATLAS_3T` sono
**copiati** da quelli di 502 con i casi a 1.5 T rimossi: la validation del fold 0 è esattamente il
sottoinsieme a 3 T di quella della baseline (103 casi dei 112), e il confronto si fa uno per uno. Il piano
è stato verificato identico a quello di 502 — spacing, patch, batch e architettura — quindi l'unica
variabile sono i 45 casi tolti dal training.

Una cautela per leggere l'esito: il training set cala del 9% (556 → 511 casi). Un eventuale peggioramento
potrebbe venire dai dati in meno anziché dal dominio, e le due cause non sono separabili con questo solo
esperimento.

### Skull stripping: SynthStrip non è installabile qui

Verificato il 28 agosto: Docker è presente ma l'utente non è nel gruppo, i pesi su
`surfer.nmr.mgh.harvard.edu` rispondono `403` e lo script su GitHub `404`. **HD-BET** è invece su PyPI,
viene dallo stesso gruppo di nnU-Net ed è addestrato anche su cervelli patologici.

Non serve scegliere a priori: DBB fornisce **244 maschere cerebrali di riferimento**, 37 delle quali su
cervelli malformati, quindi lo strumento si sceglie **misurandolo**. Se SynthStrip diventa disponibile
(licenza accettata a mano, o accesso a Docker), si confronta con HD-BET sullo stesso banco di prova.

## Piano rivisto dopo l'incontro del 27 agosto

Sette punti emersi dai relatori. Due sono già stati verificati e hanno cambiato risposta; gli altri
entrano nel piano nell'ordine seguente.

| # | Richiesta | Stato |
| --- | --- | --- |
| 1 | Immagini di confronto predizione / ground truth per ogni training | **da fare per primo**: costa poca GPU e serve a tutto il resto |
| 2 | Chiudere il confronto sui canali con un modello **T1+FLAIR** | da fare: è la lacuna dell'ablazione a due punti |
| 3 | Studiare le maschere di lesione di DBB | **verificato: non esistono** (vedi decisioni) |
| 4 | Escludere i dati a 1.5 T | **la misura non lo supporta**: da decidere con un esperimento, non a priori |
| 5 | Skull stripping con SynthStrip, dataset separato, output a 1 mm | pianificato, **da validare su DBB prima di applicarlo altrove** |
| 6 | Dataset unificato a 1 mm e modello unico | pianificato in **quattro varianti**: A (WMH+ATLAS) e D (**tutti e tre**) subito, C (tutti e tre con `ignore`) dopo SynthStrip |
| 7 | Prova su 2 pazienti AInCP a 3 T | in attesa dei dati |

### Il dataset unificato: quattro varianti

Richiesta esplicita: **mettere tutti i dati insieme**, uniformati a 1 mm — comprese le fette da 3 mm di
WMH, che vanno interpolate — e addestrare un modello unico.

L'unificazione **geometrica** vale per tutte le varianti e si ottiene con un piano alternativo generato da
`nnUNetv2_plan_experiment -overwrite_target_spacing 1 1 1`, senza toccare il piano di default, così i due
esperimenti restano entrambi riproducibili. Il costo è dichiarato e accettato: su WMH due fette su tre
vengono interpolate e **informazione non se ne aggiunge**. Ma la griglia comune è il punto — è ciò che
permette a un solo modello di vedere tutti i dati e ai pesi di essere trasferiti fra dataset.

Ogni variante è un dataset nuovo con un ID proprio, e i suoi **fold si compongono da quelli esistenti**:
il fold *k* dell'unione è l'unione dei fold *k* dei dataset di origine. Così un caso che era in validation
nel fold 0 di ATLAS resta in validation nel fold 0 dell'unione, e il confronto con la baseline è appaiato
caso per caso invece che fra due medie.

| | contenuto | classi | serve SynthStrip? |
| --- | --- | --- | --- |
| **A** | WMH + ATLAS | 1: lesione | no |
| **B** | A + AInCP, quando arriva | 1: lesione | no |
| **D** | **tutti e tre** | 7: lesione + 6 tessuti | **no — fattibile subito** |
| **C** | tutti e tre, con `ignore` | 7: lesione + 6 tessuti | sì |

**A — WMH + ATLAS, bersaglio "lesione".** Le label vogliono dire la stessa cosa in entrambi. Risponde
alla domanda "un modello unico regge su dati eterogenei?" mettendo insieme FLAIR e T1, adulti da 37 fra
siti e coorti, risoluzioni diverse. È la più economica e la più pulita.

**D — training unico su tutti e tre, senza `ignore`.** Un solo dataset a 1 mm con sette classi: `1` la
lesione da WMH e ATLAS, `2..7` i tessuti da DBB. Dove un dataset non annota una classe, quei voxel restano
`0`. È realizzabile **subito**, senza dipendere da altro, ed è la versione più diretta della richiesta.

Il conflitto da tenere presente per interpretarne l'esito: il tessuto cerebrale sano vale `0` in WMH e
ATLAS e `2..7` in DBB, quindi la rete riceve segnali opposti sullo stesso ingresso, e DBB ha duecento
volte più foreground per caso. È il motivo per cui esiste anche la variante C. Ma l'esito **si misura**,
non si assume: il confronto con le baseline sugli stessi fold dirà se il modello unico perde, tiene o
guadagna, e su quali classi. Se perde, avremo quantificato il costo del conflitto invece di ipotizzarlo —
ed è un risultato riportabile, non un esperimento sprecato.

**C — tutti e tre, con `ignore`.** Come D, ma nei voxel cerebrali non lesionati di WMH e ATLAS l'etichetta
diventa `ignore` invece di `0`: alla rete si dice "non pronunciarti" invece di "qui non c'è nulla",
togliendo il conflitto alla radice.

| origine | cosa dice la label | cosa diventa `ignore` |
| --- | --- | --- |
| DBB | i 6 tessuti | nulla: i voxel cerebrali sono tutti annotati |
| WMH, ATLAS | la lesione | **tutto il resto del cervello**, di cui non si conosce il tessuto |

**Perché non basta binarizzare DBB.** La strada più diretta sembrerebbe fondere i sei tessuti in un'unica
classe e addestrare tutto su maschere binarie. **Misurato: la binarizzazione della parcellazione coincide
con la maschera cerebrale**, Dice `1.0000` su 12 casi campionati e `0.9999` sugli altri tre — i sei
tessuti coprono l'encefalo, quindi la loro unione *è* il cervello. Ne seguirebbe che `1` significa
"lesione" in WMH e ATLAS e "cervello" in DBB, cioè lo stesso valore per due cose opposte. Con classi
distinte (D o C) il problema non si pone.

**Quello che rende C possibile è lo skull stripping**, e qui le due richieste dell'incontro si
incastrano: per marcare `ignore` il tessuto cerebrale non lesionato di WMH e ATLAS serve sapere dove
finisce il cervello, cioè esattamente la maschera che SynthStrip produce. Su DBB la maschera c'è già.

**Ordine consigliato:** A e D si possono fare subito e in qualunque ordine; C dopo SynthStrip. Fare A
prima di D non è obbligatorio, ma dà un termine di paragone pulito per capire se un eventuale calo di D
viene dall'unione dei dati o dal conflitto fra le etichette.

### SynthStrip va validato su DBB prima di usarlo

DBB fornisce **244 maschere cerebrali di riferimento**, di cui 37 su cervelli malformati (ACC, HD, MCD,
PFM), e la binarizzazione della parcellazione le riproduce esattamente. Sono un banco di prova gratuito:
prima di applicare SynthStrip a WMH e ATLAS — dove nessun riferimento esiste — lo si fa girare su DBB e
si misura il Dice contro le maschere vere, riportandolo **per patologia**.

La domanda a cui risponde è quella che conta: SynthStrip regge sulle anatomie alterate, che sono il caso
in cui BET fallisce e che è la ragione per cui BET è stato escluso? Se cede proprio sui malformati, lo si
scopre prima di costruirci sopra la variante C e i dataset skull-stripped.

### Perché l'esclusione dei 1.5 T va misurata e non assunta

Su ATLAS i casi a 1.5 T rendono leggermente **meglio** degli altri (0.653 contro 0.607 nel fold 0).
L'argomento per escluderli non è quindi che vadano male di per sé, ma che aggiungano variabilità di
dominio dannosa per il modello nel complesso: una tesi legittima, che però si verifica **addestrando
senza quei casi** e confrontando sugli stessi fold. Come sempre nel progetto, la variante si costruisce
come dataset separato con un nuovo ID, senza cancellare dati.

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
