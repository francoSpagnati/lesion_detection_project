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
Il primo lotto di 10 è arrivato l'11 settembre 2026 (vedi «Il primo lotto AInCP» più sotto).

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

| | test | cosa decide di D | stato al 30 agosto |
| --- | --- | --- | --- |
| 1 | solo 3 T contro tutto (su ATLAS) | **quali casi** mettere in D | **chiuso: non escludere**, +0.009 con p=0.67 |
| 2 | skull stripping: funziona? | **se** applicarlo a D, e sblocca C | **chiuso: nessun effetto**, +0.004 (p=0.86) su WMH e −0.004 (p=0.58) su ATLAS |
| 3 | modello a due canali T1+FLAIR | **quanto costa** la strada D | **chiuso: il T1 non aggiunge**, −0.0205 con p=0.025 |
| 4 | **variante D** | — | **training in corso** dal 31 agosto |
| 5 | **variante D skull-stripped** | se lo skull stripping aiuta anche sull'unione | dopo D, per confronto diretto |

Restano sulla GPU `Dataset522_WMH_SS` e `Dataset523_ATLAS_SS`, in sequenza, circa 7 ore ciascuno.

**Cosa ha detto il test 3.** Dando alla rete FLAIR *e* T1 insieme il Dice scende da 0.774 a 0.753,
peggiorando in 11 casi su 16 (Wilcoxon p=0.025). Il secondo canale non viene semplicemente ignorato:
costa qualcosa, presumibilmente perche' la rete deve spendere capacita' su un'immagine in cui la
lesione e' molto meno visibile, e perche' la normalizzazione e l'augmentation ora lavorano su due
distribuzioni invece che una. Con soli 79 casi di training questo margine e' plausibile.

Per D il numero da tenere resta pero' quello di 511 (**0.626**), non questo: D fa entrare WMH **con il
solo T1**, non con due canali. Il test 3 serve a escludere l'alternativa — non si puo' salvare la FLAIR
mettendola come canale in piu' e riempiendola di zeri altrove — e quella strada e' ora chiusa con un
numero invece che con un'intuizione.

### Vincolo aggiornato: AInCP avrà la FLAIR, e sarà tutto a 3 T

Confermato dall'utente il 31 agosto, e cambia il quadro a monte di tutto il resto.

**AInCP avrà T1 *e* FLAIR.** Quindi il modello finale sarà a due canali. E siccome nnU-Net trasferisce
i pesi solo fra architetture identiche — copia tutto tranne i `seg_layers` e verifica il resto con un
assert — **anche il pretraining deve essere a due canali**. Un pretraining solo-T1 su ATLAS e DBB non è
direttamente riusabile.

Questo riabilita il `512`. Il suo risultato (`0.753` contro `0.774` della sola FLAIR) diceva che *se hai
già la FLAIR* aggiungere il T1 non aiuta; non diceva che due canali siano una cattiva architettura. Per
AInCP il confronto rilevante è un altro, ed è netto: **due canali `0.753` contro solo T1 `0.626`**. Il
`512` non è più un vicolo cieco ma il modello di riferimento per il pretraining.

Il problema che ne nasce è di dati: dei nostri tre dataset **solo WMH ha entrambe le sequenze**, cioè 79
casi di sviluppo. Servono sorgenti pubbliche con T1 e FLAIR appaiati — vedi la sezione seguente.

**AInCP contiene anche dati a 1.5 T — chiuso: i casi a basso campo restano.** Il relatore ha chiarito il
5 settembre che la coorte AInCP non è tutta a 3 T come si era creduto. Cade quindi l'unico argomento
rimasto a favore dell'esclusione, che era la coerenza di dominio: la misura già diceva che escluderli non
cambia nulla (+0.0093 con p=0.67 su ATLAS, e il modello solo-3 T segmenta comunque bene i 1.5 T mai
visti). **Decisione: si tengono, in tutti i dataset.** `Dataset521_ATLAS_3T` resta come sorgente di
pretraining per l'esperimento 541, non come configurazione da adottare.

Nota utile: quei 45 casi si sono rivelati preziosi per una ragione imprevista. Sono l'unico gruppo che
**nessun** modello di pretraining ha mai visto, e per questo sono diventati la coorte di simulazione del
test 541. Escluderli all'inizio avrebbe reso quella verifica impossibile.

### Dataset pubblici con T1 e FLAIR appaiati, da valutare

Cercati il 31 agosto. Nessuno è ancora stato scaricato: la scelta va fatta prima, perché ogni download
costa spazio su `/home` e tempo.

| Dataset | Casi | Sequenze | Perché | Accesso |
| --- | --- | --- | --- | --- |
| **MSLesSeg** | ~75 pazienti, 115 serie | T1, T2, FLAIR | Le lesioni della sclerosi multipla sono iperintensità periventricolari della bianca: è il fenotipo pubblico **più simile alla leucomalacia periventricolare**, che è il reperto più frequente nella paralisi cerebrale | CC BY 4.0, richiesta via [iplab.dmi.unict.it](https://iplab.dmi.unict.it/mfs/ms-les-seg/). Già skull-strippato e registrato su MNI-152 |
| **MSSEG-1 / Shifts** | decine | T1, FLAIR | Altre coorti di sclerosi multipla, multicentriche | registrazione alla challenge |
| **ISLES 2015 SISS** | 28 training | T1, T2, DWI, FLAIR | Ictus sub-acuto: è l'altro fenotipo, quello cavitato, ma **con la FLAIR** | scaricabile da Zenodo |
| **BraTS** | 1251 (2021) – 2877 (2025) | T1, T1ce, T2, FLAIR | L'unico grande abbastanza da fare da pretraining generico a due canali | registrazione, dimensioni notevoli |

Nota che rende tutto più semplice: MSLesSeg arriva già skull-strippato mentre i nostri dati non lo sono.
Prima dei test `522`/`523` sarebbe stato un problema di coerenza; ora **sappiamo per misura** che lo
skull stripping non sposta il risultato, quindi si possono mescolare sorgenti strippate e non.

### Quando serve uscire da nnU-Net e quando no

Punto di confusione da tenere chiaro, perché riguarda due cose diverse.

**Un modello a due canali non richiede nulla di speciale.** nnU-Net gestisce N canali da sempre:
`Dataset512_WMH_T1_FLAIR` è stato costruito, pianificato, addestrato e valutato senza una riga di codice
nostro. Anche il fine-tuning da un modello a due canali a un altro a due canali è supportato
(`nnUNetv2_train ... -pretrained_weights`), perché le architetture combaciano.

**Serve codice nostro solo per cambiare il numero di canali fra pretraining e fine-tuning**, cioè solo
per la proposta B.

| percorso | serve codice nostro? |
| --- | --- |
| addestrare a 2 canali su WMH, poi fine-tuning a 2 canali su AInCP | **no** |
| addestrare a 1 canale su ATLAS/DBB, poi fine-tuning a 1 canale | **no** |
| addestrare a 1 canale, poi **espandere a 2** e fare fine-tuning | **sì**, un trainer che sovrascrive il caricamento dei pesi |

**Quali dataset hanno davvero la FLAIR.** Verificato sui file, non sulla documentazione:

| dataset | file disponibili | FLAIR? |
| --- | --- | --- |
| WMH | `*_FLAIR_BiasCorr`, `*_T1w_BiasCorr_2FLAIR_Aligned`, `*_wmh` | **sì**, già co-registrata |
| ATLAS | 955 `*_T1w`, 655 `*_desc-T1lesion_mask` | **no**, solo T1 |
| DBB | `anat-t1w`, `mask`, `parcellation-volume` | **no**, solo T1 |

Quindi «aggiungere la FLAIR ai dataset che abbiamo» si può fare **solo su WMH**, ed è già fatto: è il
`512`. Per avere altra FLAIR servono dataset nuovi (proposta A) o l'espansione dei canali (proposta B).

### Proposta B: pretraining a un canale, poi espansione a due

Alternativa all'aggiunta di dataset, e **complementare** ad essa: pre-addestrare a un solo canale sugli
842 casi che già abbiamo, e poi espandere il primo strato convoluzionale da uno a due canali prima del
fine-tuning sui dati che hanno la FLAIR. Usa anche i 763 casi solo-T1 che un modello a due canali non
potrebbe mai vedere.

**Perché nnU-Net non lo fa da solo.** `load_pretrained_weights` copia tutto tranne i `seg_layers` e
verifica con un `assert` che ogni altro parametro abbia forma identica. Il primo strato passa da
`[32, 1, 3, 3, 3]` a `[32, 2, 3, 3, 3]`: l'assert fallisce.

**Non serve però abbandonare nnU-Net, basta sostituire quella funzione.** Il progetto ha già un trainer
personalizzato (`scripts/nnunet_trainers/nnUNetTrainerWandb.py`), quindi il meccanismo esiste. Tre
livelli, in ordine di costo:

| livello | cosa si scrive | cosa consente |
| --- | --- | --- |
| 1, minimo | un trainer che sovrascrive il solo caricamento dei pesi | l'espansione da 1 a 2 canali; tutto il resto resta di nnU-Net |
| 2, intermedio | + il controllo del fine-tuning | lr ridotto, encoder congelato nelle prime epoche |
| 3, massimo | ciclo di training proprio | solo per ciò che nnU-Net non prevede, in particolare il **modality dropout** (oscurare a caso la FLAIR, così il modello resta usabile su pazienti che non ce l'hanno) |

**Come inizializzare il canale nuovo**, l'unica decisione tecnica del livello 1. Due opzioni standard, da
provare entrambe: il canale FLAIR **a zero** (il modello parte identico a quello a un canale e impara a
usare la FLAIR strada facendo, opzione conservativa), oppure il filtro **duplicato e dimezzato** (parte
trattando le due sequenze come intercambiabili, sfrutta subito la FLAIR ma da un punto meno affidabile).

**Struttura a tre stadi:**

| stadio | dati | casi | canali | cosa impara |
| --- | --- | --- | --- | --- |
| 1 | ATLAS + WMH in T1 | **635** | 1 | lesioni cavitate |
| 2 | WMH con T1 e FLAIR | 79 | 2 | a usare la FLAIR |
| 3 | AInCP | ~20 | 2 | il dominio pediatrico |

**Correzione allo stadio 1: 635 casi, non 842.** I 207 pediatrici di DBB erano contati quando il
piano prevedeva ancora il modello a 7 classi. Abbandonato il multiclasse (decisione del 7 settembre),
un pretraining sul **solo compito lesionale** non può usarli: non contengono maschere di lesione.
Restano utilizzabili solo come pretraining su compito diverso, che è la strada non ancora misurata.

**Livello 1 realizzato e verificato (7 settembre).**
`scripts/nnunet_trainers/nnUNetTrainerEspansioneCanali.py` sostituisce la sola
`load_pretrained_weights`, agganciandola in `__init__` così vale solo per questo trainer. Quattro
classi: `_zero`, `_duplica` e le due varianti `_lr1e3` — sono classi distinte e non un parametro
perché nnU-Net ricava dal nome del trainer la cartella dei risultati, e una strategia scelta a
runtime sovrascriverebbe l'altra.

Due cose scoperte scrivendolo, entrambe capaci di produrre in silenzio un esperimento sbagliato:

- **i canali vanno appaiati per nome, non per indice.** In `Dataset512` il canale 0 è la **FLAIR**;
  il filtro pre-addestrato è un filtro T1 e va nella posizione 1. I nomi del modello di partenza si
  leggono dal checkpoint, in `init_args['dataset_json']['channel_names']`.
- **il primo strato compare sotto quattro chiavi**, non una: `ConvDropoutNormReLU` tiene sia
  `conv.weight` sia `all_modules.0.weight`, e il decoder referenzia l'encoder. Vanno espanse tutte,
  o `load_state_dict` riceve due valori in conflitto per lo stesso parametro.

`scripts/verifica_espansione_canali.py` prova tutto su CPU sui pesi veri di 502: 14/14, e la verifica
che conta è l'equivalenza funzionale — a parità di T1 la rete espansa con la strategia `zero` dà la
stessa uscita del modello a un canale, **scarto massimo 0.00e+00**, qualunque cosa ci sia nel canale
FLAIR.

**Prerequisito rimasto prima di lanciare lo stadio 2:** i piani di 502 (1 mm, patch 128³) e di 512
(3.0×0.98×1.2, patch 64×224×128) non coincidono, e il trasferimento funziona solo fra architetture
identiche a meno dei canali. Va imposto il piano dello stadio 1 al dataset a due canali con
`nnUNetv2_move_plans_between_datasets`.

**Il rischio.** Allo stadio 2 il canale FLAIR parte da nulla e ci sono solo 79 casi per insegnare al
modello a servirsene. Se non bastano resta un modello a T1 con un canale inerte, cioè si torna al punto
di partenza dopo tre training. È esattamente il rischio che la proposta A elimina: **A fornisce i dati a
due canali che rendono affidabile lo stadio 2 di B**, e B permette di sfruttare i casi solo-T1 che A non
può usare. Se se ne sceglie una sola: A è la più sicura, B la più economica.

### Decisione del 7 settembre: si resta sul solo compito lesionale

Il relatore ha osservato che «non ha molto senso lavorare su due codomini diversi», e i dati gli danno
ragione in modo più forte di quanto ci si aspettasse: il modello unico **non produce mai entrambe le
cose**, ha imparato a riconoscere la sorgente dell'immagine (149 casi su 151). La rete ha rifiutato da
sola l'unione dei due compiti.

**Le varianti multiclasse (D e C con `ignore`) sono quindi abbandonate.** Restano tre usi legittimi di
DBB, in ordine di solidità:

1. **banco di prova per gli strumenti** — già usato così: le sue 244 maschere cerebrali sono l'unico modo
   che avevamo per verificare SynthStrip sulle anatomie malformate;
2. **il secondo stadio della pipeline** — la parcellazione anatomica del gemello sano è esattamente il
   compito per cui DBB è nato;
3. **pretraining su compito diverso** — il più incerto, e non ancora misurato. Richiede prima di
   spostare il piano di 502 su DBB e riaddestrarlo, perché le architetture non sono compatibili per il
   trasferimento dei pesi (verificato: le convoluzioni trasposte hanno dimensione pari allo stride).

### La ricetta di fine-tuning per AInCP, misurata

Dal test 541, e da usare sui pazienti veri:

| parametro | valore | perché |
| --- | --- | --- |
| passo di apprendimento iniziale | **`1e-4`** | col default `1e-2` il fine-tuning *peggiora* il modello di partenza (0.564 contro 0.677) |
| epoche | **basso**, molto meno di 250 | il minimo della perdita di validation cade fra l'epoca 69 e la 128 |
| checkpoint | `checkpoint_best`, non l'ultima epoca | nnU-Net non fa early stopping ma il checkpoint migliore lo salva |
| termine di paragone obbligatorio | il modello **non adattato** | può battere quello adattato, ed è successo |

### Loss per le lesioni piccole: cosa provare, in ordine

Richiesta del relatore. Prima però il problema è stato **quantificato** con le metriche di rilevamento per
lesione (`scripts/metriche_per_lesione.py`): su ATLAS con il solo T1 il modello trova l'**8.7%** delle
lesioni sotto i 10 mm³ (un voxel di ATLAS vale 1 mm³). Il problema esiste ed è enorme.

La letteratura che testa esattamente questo su lesioni cerebrali piccole con nnU-Net
([arXiv 2604.08015](https://arxiv.org/html/2604.08015v2)) trova che **Tversky fa peggio del default**
(0.7706 contro 0.7796) e Focal Tversky è sostanzialmente pari (0.7802) — *in Dice*. Ma osserva che il
default è il peggiore sulle metriche di lesione mancata. Il guadagno non sta nel Dice, sta nel non
perdere lesioni. Da cui: **cambiare la loss senza cambiare la metrica non mostrerebbe nulla.**

Ordine proposto, dal più economico:

1. **`batch_dice: true`** — una riga nel JSON del piano, nessun codice. Attualmente è `false` su tutti i
   nostri `3d_fullres`. È pensato esattamente per annotazioni piccole e sparse.
2. **`nnUNetTrainerDiceTopK10Loss`** — già in nnU-Net, concentra l'addestramento sui voxel difficili.
3. **Focal Tversky** — va implementata, poche decine di righe in `compound_losses.py`.

Ognuna è un training da confrontare in modo appaiato guardando **sia il Dice sia il tasso di rilevamento
per fascia di volume**.

### Proposta C: lesioni sintetiche sui cervelli pediatrici sani di DBB

Affronta un problema diverso dalle altre due: **tutti i nostri dati lesionali sono di adulti**, e l'unico
dataset pediatrico che abbiamo non ha annotazioni di lesione. L'idea è **inserire lesioni artificiali nei
207 cervelli sani di DBB**, ottenendo un riferimento esatto per costruzione. È l'unica delle tre proposte
che **non richiede dati esterni** ed è lavoro di sola CPU.

**Perché non ricavarle invece dai 37 patologici con la modellazione normativa** (costruire dai sani la
distribuzione di ogni voxel e chiamare lesione ciò che devia). Tre ragioni, la prima decisiva:

1. **Fenotipo sbagliato.** I 37 patologici sono ACC, MCD, PFM e idrocefalo: *malformazioni*, strutture che
   non si sono formate o si sono formate male, non danni acquisiti da ipossia. L'agenesia del corpo
   calloso è una struttura *assente*, non *lesa*. Si etichetterebbe "qui differisce dal normale", che è
   un altro concetto.
2. **Circolare.** Serve allineare il cervello patologico a uno spazio comune, ed è esattamente ciò che si
   degrada sull'anatomia distorta — il problema per cui DBB esiste.
3. **Non verificabile**: nessun riferimento con cui controllare le etichette prodotte.

E sarebbero 37 casi contro 207.

**Perché DBB e non un dataset sano qualsiasi.** Ha la parcellazione dei tessuti, quindi sappiamo dove
sono bianca e ventricoli in ogni soggetto e possiamo mettere le lesioni **dove clinicamente stanno** —
periventricolari nella bianca, cavità nel territorio corticale-sottocorticale — invece che a caso.

**Come si costruisce.**

| passo | operazione | su cosa si basa |
| --- | --- | --- |
| 1 | trapianto di una maschera di lesione reale da ATLAS o WMH | la parcellazione DBB determina le posizioni ammissibili |
| 2 | riempimento con il contrasto del fenotipo da riprodurre | i rapporti **già misurati**: 0.62 cavitate, 1.07 non cavitate (vedi `reports/failure_analysis_502.md`) |
| 3 | dilatazione della maschera per un anello di transizione sfumato | è il passo decisivo: senza, la rete impara il bordo artificiale |

**Il rischio e il suo controllo.** La rete può imparare l'artefatto di sintesi invece della lesione. Il
test è netto: **si addestra sui sintetici e si valuta su ATLAS e WMH reali.** Se il Dice sui reali
migliora, la sintesi ha insegnato qualcosa di vero; altrimenti no.

**Ricaduta sullo stadio 2.** L'inpainting del gemello sano ha bisogno di coppie (con lesione, senza
lesione) della stessa testa. **Nessun dataset reale può fornirle.** La sintesi le produce per
costruzione, con il bersaglio dell'inpainting già noto. Un solo lavoro alimenta entrambi gli stadi.

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

### Skull stripping: risolto, SynthStrip è in uso

L'installazione era bloccata (Docker senza permessi, pesi `403`, script `404`) finché il 28 agosto i pesi
non sono stati scaricati a mano e lo script trovato col suo nome vero, `synthstrip` e non
`mri_synthstrip`. **HD-BET**, l'alternativa su PyPI, non è più necessaria: SynthStrip è stato misurato sul
banco di prova di DBB e ha superato la soglia che contava (0.957 sui cervelli malformati). I dettagli e la
cautela sugli idrocefali sono in [dataset-e-decisioni.md](dataset-e-decisioni.md).

## Piano rivisto dopo l'incontro del 27 agosto

Sette punti emersi dai relatori. Due sono già stati verificati e hanno cambiato risposta; gli altri
entrano nel piano nell'ordine seguente.

| # | Richiesta | Stato |
| --- | --- | --- |
| 1 | Immagini di confronto predizione / ground truth per ogni training | **fatto**: `scripts/plot_predictions.py`, figure per 501, 502, 503 e 511 |
| 2 | Chiudere il confronto sui canali con un modello **T1+FLAIR** | **fatto e chiuso**: `0.753` contro `0.774`, il T1 in piu' peggiora (p=0.025) |
| 3 | Studiare le maschere di lesione di DBB | **verificato: non esistono** (vedi decisioni) |
| 4 | Escludere i dati a 1.5 T | **misurato e scartato**: +0.0093 appaiato, p=0.67. I 1.5 T restano dentro |
| 5 | Skull stripping con SynthStrip, dataset separato, output a 1 mm | strumento validato, dataset costruiti, **training `522` e `523` in corso**; il "tutto a 1 mm" si fa al planning, non duplicando i dati |
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
ATLAS e `2..7` in DBB, quindi la rete riceve segnali opposti sullo stesso ingresso. Lo sbilanciamento
è ora **misurato** (`scripts/bilancio_classi_union.py`, `reports/bilancio_classi_531.md`): un caso DBB
porta **266 volte** il foreground di un caso lesionale, e la lesione è il **5.9%** del foreground
totale. È il motivo per cui esiste anche la variante C. Ma l'esito **si misura**,
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

**1. Baseline, un fold ciascuna — FATTO.** `3d_fullres`, fold 0: 501 `0.774`, 502 `0.610`, 503 `0.971`.
Sono i numeri contro cui si misura ogni guadagno successivo.

**2. Ablazione FLAIR: 511 contro 501 — FATTO.** T1 `0.626` contro FLAIR `0.774`, peggiore in 16 casi su
16, Wilcoxon p=0.00003. La perdita si concentra sulle lesioni piccole.

**2-bis. Due canali: 512 contro 501 — FATTO.** FLAIR+T1 `0.753` contro FLAIR sola `0.774`, peggiore in
11 casi su 16, Wilcoxon p=0.025. Aggiungere il T1 alla FLAIR non aiuta.

**2-ter. Skull stripping: 522 e 523 — FATTO, nessun effetto.** +0.0045 con p=0.86 su WMH e −0.0037 con
p=0.58 su ATLAS. Non si adotta per default; resta lo strumento che rende possibile la variante C.
Dettaglio originale dell'esperimento: `522` (WMH skull-stripped) contro 501 e `523` (ATLAS
skull-stripped) contro 502, in sequenza sulla GPU 0 dal 30 agosto. Il confronto si fa con
`scripts/confronto_appaiato.py`, che interseca i casi invece di confrontare due medie. Va letto
ricordando che su questi due dataset nnU-Net ha cambiato anche patch e batch: il confronto misura la
pipeline, non la sola rimozione del cranio.

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
e metriche riportate per singolo paziente, non solo in media. Il lotto è arrivato: il piano concreto
è nella sezione «Il primo lotto AInCP».

**6. Stadio 2 (inpainting e gemello sano):** interamente da fare. Va dopo, perché senza una maschera di
lesione affidabile non c'è niente da inpaintare. SynthSeg è il riferimento indicato per la parcellazione
sul cervello virtuale.

## Il primo lotto AInCP: dieci pazienti, misurati e preprocessati (11 settembre)

Archivio `dataset_FSM_v1.7z`, 10 pazienti con T1 e FLAIR **originali** (defacciate) e le ROI tracciate
sulla FLAIR. La cartella `COREGISTERED` non è coregistrata: il responsabile lo ha scritto e il notebook
lo verifica, T1 e FLAIR non condividono la griglia in nessun soggetto. Tutto il resto del vecchio
archivio è stato cancellato. Analisi in `notebooks/aincp_primo_lotto.ipynb`, generato da
`scripts/costruisci_notebook_aincp.py`.

**Annotazioni.** Quattro ROI: `ROI_1` gliosi (iperintensa in FLAIR), `ROI_2` cistica, `ROI_4`
encefalomalacia, `ROI_3` **ventricolo** dell'emisfero colpito — annotato ma escluso dal bersaglio.
La maschera è `ROI_1 ∪ ROI_2 ∪ ROI_4`. La legenda ha tre versioni; i rapporti d'intensità sulla FLAIR
(gliosi 2.5–4.4, cistica 0.10–0.41, encefalomalacia 0.46–1.08) dicono che tutti e dieci seguono
l'ultima. Sovrapposizioni fra ROI trascurabili (max 35 voxel lesione∩ventricolo).

**Dimensioni.** Volume di lesione mediana 7.2 ml (0.5–123.5), fra ATLAS 5.3 e WMH 10.9. Le componenti
connesse sono **65** con la connettività a 26 vicini, quella di `metriche_per_lesione.py` (con quella a
6, il default di scipy, sarebbero 158: una lesione sottile e obliqua si spezza). Di queste, 42 (65%)
stanno sotto 10 mm³ e valgono lo 0.03% del volume.

**Nessuna soglia, per ora** (decisione del 14 settembre): `figura_lesioni_piccole.py` misura che le
componenti minuscole di gliosi, anche da un solo voxel, hanno intensità FLAIR 2.3–4.4 volte il tessuto
sano, cioè il segnale di una lesione vera e non di rumore. Solo 3 componenti su 65 cadono dove
l'immagine ha il livello dell'aria, fuori dal cervello. La soglia resta una domanda per il clinico, con
le due figure in mano.

**Preprocessing** (`scripts/preprocess_aincp.py`, ANTsPy, ~2 min/soggetto, output in
`data/processed/AINCP/`, versionati solo i `manifesto.json`): N4 → FLAIR→T1 rigida (maschere in
nearest neighbour) → T1→MNI152NLin2009aSym SyN → maschera binaria. Trasformazioni salvate.

Due misure che cambiano la decisione sullo spazio di addestramento:

- **FLAIR→T1 conserva il volume ma frammenta**: la FLAIR ha fette da 0.8 mm e la T1 da 1.0, e il
  nearest neighbour scarta una fetta ogni cinque. I tre soggetti con FLAIR già a 1.0 mm (039, 054,
  056) non cambiano numero di componenti. Colpisce quasi solo le componenti sotto soglia.
- **MNI ingrandisce le lesioni**: +51% mediano, fino a +106%. Non è la risoluzione (0% ricampionando
  a 1 mm senza registrare) né la deformazione non lineare (la sola affine fa peggio). È il
  determinante della matrice affine — fattore di volume 1.6–2.5 — perché il template è adulto e i
  pazienti sono bambini; prevede l'aumento misurato entro 4 punti su tutti e dieci
  (`reports/aincp_scala_mni.json`, `scripts/figura_registrazione_aincp.py`). La registrazione in sé
  è buona (figura `registrazione_mni.png`).

Conseguenze: i volumi e le metriche vanno calcolati **in spazio nativo**; nnU-Net normalizza la
spaziatura, non lo spazio anatomico, quindi i pesi di ATLAS non richiedono MNI. **Proposta: addestrare
nello spazio T1 nativo con la FLAIR coregistrata**, MNI solo per analisi di gruppo. Da discutere con il
responsabile, perché il docx prevedeva MNI.

**Piano di addestramento proposto** (non lanciato; GPU occupate; prima si decide lo spazio):
input 2 canali `{0: FLAIR, 1: T1}` come Dataset512; 5 fold per paziente con 039/054/056 su fold
diversi; nessun test set (i lotti futuri lo saranno); confronti appaiati in Dice e rilevamento
sopra soglia in spazio nativo.

| | cosa | trainer | domanda |
| --- | --- | --- | --- |
| T0 | da zero, solo AInCP | `nnUNetTrainerWandb_250ep_arresto50` | riferimento da battere |
| T1 | ATLAS 502 → espansione a 2 canali, FLAIR a zero → fine-tuning | `nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50` | **Proposta B** |
| T1' | come T1 con il filtro duplicato | `..._duplica_250ep_lr1e4_arresto50` | l'inizializzazione conta? |
| T2 | WMH 512 (già 2 canali) → fine-tuning diretto | da scrivere | conta più la quantità o la modalità? |

Il 13 settembre il responsabile ha scelto la Proposta B. `Dataset504_AINCP` è costruito e
preprocessato (`scripts/make_aincp_dataset.py`), la coda è `scripts/run_aincp_504.sh <fold>`; tutto
è pronto e fermo solo per le GPU. Tre strumenti scritti per il training vero:
`metriche_per_lesione.py` con fasce in mm³, soglia simmetrica e fold uniti; l'arresto sulla perdita
di validazione nel trainer (`pazienza_val`, 50 epoche); `confronto_fold.py` per il confronto
appaiato sulle dieci predizioni fuori-fold. Dettagli in `docs/modelli-ed-esperimenti.md`.

**Esito del primo giro (15 settembre)**: T0 0.581, T1 0.555, T1' 0.596, nessuna differenza appaiata;
il canale FLAIR di T1 è rimasto inerte (passo `1e-4` troppo basso per un canale nuovo). La FLAIR va
fatta imparare: **prossimo esperimento il passo differenziato** (primo strato a `1e-2`, resto a
`1e-4`), poi lo stadio 2 su WMH se non basta. Soglia: nessuna, in attesa del clinico. Modality
dropout: dopo, come variabile a sé. Dettaglio in `docs/modelli-ed-esperimenti.md`.

### Decisioni del 22 settembre, dopo il secondo giro e l'analisi per fenotipo

**Il modello a due canali è una premessa, non una variabile.** Su WMH la FLAIR da sola batte
FLAIR+T1 (0.774 contro 0.754, p = 0.025), e un modello solo-FLAIR su AInCP avrebbe probabilmente un
Dice migliore. La proposta è stata respinta dall'utente, e a ragione: T1 e FLAIR vedono lesioni
*diverse* — la gliosi è iperintensa in FLAIR, le cavità (cistica, encefalomalacia) sono ipointense in
T1 — e su WMH le lesioni sono definite sulla FLAIR per costruzione, quindi quel confronto non misura
la domanda della tesi. Il valore di ciascun canale si misura **per fenotipo**, non sul Dice
complessivo. Il responsabile vuole inoltre un modello a due canali utilizzabile anche con una sola
sequenza.

**La misura per fenotipo sostiene la premessa** (`scripts/analisi_rilevamento_504.py`,
`reports/analisi_rilevamento_504.md`). Copertura dei voxel di ciascun fenotipo:

| | gliosi (FLAIR) | cistica (T1) | encefalomalacia (T1) |
| --- | --- | --- | --- |
| da zero | **0.585** | 0.834 | 0.401 |
| Proposta B, FLAIR spenta (passo 1e-4) | 0.472 | **0.933** | **0.809** |
| Proposta B, FLAIR attiva (passo differenziato) | 0.525 | **0.933** | 0.775 |

Il pre-addestramento su ATLAS porta l'encefalomalacia da 0.02–0.48 a 0.61–0.87 **in tutti e quattro i
pazienti che la hanno**: le lesioni di ATLAS sono ictus cronici, cioè cavità, e il sapere trasferito è
quello. La gliosi la porta la FLAIR: con il passo differenziato e il freno corretto (23 settembre) la sua
copertura supera quella del modello con la FLAIR spenta in 9 pazienti su 10 (p = 0.006) e arriva al
livello del modello da zero (4 su 10, p = 0.49). Ogni sorgente porta il suo fenotipo. *(Il 22
settembre qui si leggeva che sulla gliosi la Proposta B restava sotto il modello da zero: era un
effetto della media pesata sul paziente con più gliosi, e paziente per paziente non regge.)*

**Conseguenza: lo stadio 2 su WMH diventa il prossimo esperimento principale**, non più un ripiego.
WMH ha 79 casi a due canali con iperintensità della sostanza bianca in FLAIR, il fenotipo più vicino
alla gliosi: può essere per la gliosi quello che ATLAS è stato per le cavità, cioè portarla oltre
quello che otto pazienti insegnano. È anche la via di espansione ad altri dataset già prevista nel
piano, e resta a due canali.

**Ricetta di adattamento decisa il 23 settembre: passo differenziato con il freno corretto**
(primo strato 1e-2, resto 1e-4, arresto sulla perdita di validazione). Raggiunge il modello da zero
sulla gliosi e i pre-addestrati sulle cavità, con meno falsi positivi di tutte (10 pazienti su 10,
p = 0.002). Il passo 1e-3 uniforme ha lo stesso Dice ma non mostra di usare la FLAIR. Dettaglio in
`docs/modelli-ed-esperimenti.md`.

**Che cosa si può affermare e che cosa no.** Con dieci pazienti nessuna variante si distingue dalle
altre nel Dice (passo differenziato contro da zero: +0.039, IC 95% [−0.036, +0.110], p = 0.275) né
nel numero di lesioni rilevate (27 contro 25 su 71, p = 0.75). Si distinguono i **falsi positivi**:
7.9 per paziente da zero, 2.3–3.2 con il pre-addestramento, in meno in 8 pazienti su 10 (p = 0.033
e 0.043, non corretti per confronti multipli ma concordi in tutte e tre le varianti pre-addestrate).
L'argomento della tesi va costruito su tre evidenze più solide del Dice medio: i pesi (la norma del
filtro FLAIR), la copertura per fenotipo, i falsi positivi. I Dice medi si riportano sempre con il
loro intervallo.

**Un limite strutturale: 27 lesioni su 71 non arrivano alla rete.** Il piano `nnUNetPlansDa502`
lavora a 1 mm isotropo, quello di 502, perché il trasferimento dei pesi richiede la stessa
architettura. Ricampionando l'annotazione come fa nnU-Net (procedura verificata: riproduce voxel per
voxel il bersaglio salvato), spariscono 20 delle 24 componenti sotto 1 mm³ e 7 delle 22 fra 1 e
10 mm³. Il bersaglio di addestramento ha 50 componenti, non 71. Il tetto del rilevamento è quindi
44 lesioni su 71, e la domanda al clinico sulle lesioni minuscole cambia natura: anche se sono lesioni
da trovare, questa configurazione non può impararle. Una strada c'è — un piano con la stessa
architettura a spaziatura più fine, a cui i pesi restano trasferibili — ma costa campo visivo e va
decisa solo dopo la risposta del clinico. Anche fra le piccole che esistono a 1 mm il modello ne
rileva poche (4–6 su 26): la risoluzione non è l'unica causa.

**Per l'incontro del 23 settembre, sulle lesioni piccole.** Sono due popolazioni. Fra 1 e 10 mm³
(20 componenti) il 95% è più brillante del tessuto sano (mediana 3.1 volte) e la griglia a 1 mm le
vede quasi tutte: la domanda è se siano lesioni da trovare. Sotto 1 mm³ (22 componenti, da uno a
cinque voxel originali) l'intensità mediana è 0.69 volte il tessuto sano, meno della metà sono più
brillanti, e 20 su 24 spariscono comunque a 1 mm: la domanda è se si possano ignorare. Se sì, la
soglia proposta è **1 mm³**, che coincide con un voxel della griglia di lavoro della rete e quindi
non è arbitraria. Più le tre componenti fuori dal cervello.

**Due fatti sui dati.** Il secondo lotto di dieci pazienti, atteso da agosto, non è arrivato: quando
arriva è il test set, e non si guarda prima della configurazione finale. In `/data01/fsm_mri_data/`
c'è un undicesimo paziente, `FSM_028`, preparato ad aprile da un altro utente (T1, FLAIR e maschera
binaria già a 1 mm in spazio T1, senza distinzione di fenotipo): non è usato, e prima di usarlo va
chiesto al responsabile chi l'ha segmentato e se la maschera include il ventricolo.

### Incontro del 23 settembre: decisioni e piano del training finale

**Spazio di lavoro: standardizzazione in MNI** (indicazione del responsabile). La pipeline di un
paziente nuovo sarà nativo → template → modello → nativo: si standardizza per il modello e si torna
nello spazio del paziente per volumi e metriche. L'obiezione dell'11 settembre (il template adulto
ingrandisce i bambini) è stata misurata contro l'argomento del responsabile, che conta la
*proporzione* di cervello lesionato (`scripts/confronto_template_aincp.py`, maschera del cervello
del paziente portata nel template con la stessa trasformazione della lesione):

| | ingrandimento del cervello | % lesionata: variazione relativa | allineamento al template (Dice) |
| --- | --- | --- | --- |
| MNI152NLin2009aSym (adulto, di ATLAS) | 1.62–2.48, mediana 1.77 | mediana −10% (−31 … +30%) | 0.82–0.93 |
| MNIPediatricAsym cohort-1 (4.5–18.5 anni) | 1.49–2.19, mediana 1.62 | mediana −11% (−38 … +40%) | **0.95–0.98** |

Il responsabile ha ragione sulla proporzione: il volume assoluto cresce fino a 2 volte, la
percentuale lesionata cambia di poco, e le variazioni grandi sono sulle lesioni sotto 1.5 ml, dove
pochi voxel pesano molto. Il template pediatrico **non** toglie l'ingrandimento (è anch'esso in
spazio MNI, con cervello da ~2 l) ma allinea molto meglio l'anatomia dei bambini, che è lo scopo
della standardizzazione. **Proposta, da confermare: MNIPediatricAsym per AInCP**; ATLAS e WMH
(adulti) nel template adulto. *(Correzione: una prima misura con SynthStrip applicato
all'immagine già in MNI dava +50–55% su FSM_032 e FSM_039; misurava la maschera, non la
registrazione, ed è stata sostituita.)* Per i pazienti nuovi, le età permetterebbero una coorte più
stretta del template (sei disponibili).

**Orientamento.** Le immagini appaiono «in verticale» nelle figure perché T1 e FLAIR sono acquisite
in fette sagittali. L'ordine degli assi è lo stesso di ATLAS (sinistra-destra, antero-posteriore,
supero-inferiore); in 6 pazienti su 10 il verso sinistra-destra è invertito (LAS contro RAS), cosa
che il ribaltamento casuale dell'augmentation copre già. In MNI tutti diventano RAS.

**Soglia: le componenti fino a 1 mm³ sono residui del tracciato**, per indicazione clinica. Si
tolgono dal bersaglio e si ignorano simmetricamente nelle metriche. Sulle annotazioni originali
sono 22, di cui 4 fuori dal cervello (SynthStrip sulla FLAIR originale; il vecchio criterio di
intensità ne dava 3). L'elenco per il clinico, con paziente, file ROI, fetta sagittale e
coordinate, è in `reports/residui_annotazione.md` (`scripts/elenco_residui_annotazione.py`). Con la
soglia restano **47 lesioni** sui dieci pazienti.

**Dati: 17 pazienti a 1.5 T in arrivo, nessun altro dopo.** Il totale sarà 27 (più FSM_028, se
utilizzabile). L'esperienza su ATLAS dice che i dati a 1.5 T non peggiorano i modelli.

**Split: 70% per il fine-tuning, 30% di test, consapevole del centro.** Centro e scanner equivalgono
a parametri di acquisizione diversi: lo split va stratificato per centro (e campo magnetico), e i
risultati riportati anche per centro. Con 27 pazienti: ~19 per l'addestramento (cinque fold per
l'arresto anticipato e le scelte) e ~8 di test, usati una volta sola alla fine. Serve l'elenco del
centro di ogni paziente. Nota da dichiarare: i dieci pazienti attuali hanno orientato la ricetta, e
quelli che finiranno nel test porteranno un piccolo ottimismo.

**Architettura: Proposta B confermata.**

**Il training finale**, con tutto ciò che è stato misurato:

| stadio | dati | spazio | cosa |
| --- | --- | --- | --- |
| 1 | ATLAS, 556 casi, T1 | MNI adulto | esistente: `Dataset502`, fold 0 |
| 2 | WMH, 79 casi, FLAIR + T1 | MNI adulto | `Dataset542_WMH_MNI`, in corso dal 23/9 (sotto) |
| 3 | AInCP, ~19 pazienti di addestramento | MNI pediatrico (da confermare) | cinque fold, insieme dei cinque modelli, test sugli ~8 |

Allo stadio 3: soglia 1 mm³ sulle annotazioni; passo di adattamento da decidere in
cross-validation sui soli pazienti di addestramento (la ricetta del 23/9 era pensata per un canale
FLAIR nuovo, che dopo lo stadio 2 non lo è più); modality dropout (p = 0.3) per l'uso con una sola
sequenza; arresto sulla perdita di validazione; metriche nello spazio nativo, per fenotipo, per
dimensione e per centro.

**Stadio 2, lanciato il 23 settembre** sulle GPU 1 e 2: WMH portato nel template di ATLAS
(`scripts/make_wmh_mni_dataset.py`; contrasto locale delle lesioni conservato al 90% in mediana,
sovrapposizioni controllate a occhio), piano e split di 512. Due varianti, una variabile fra loro:
passo differenziato contro 1e-3 uniforme, entrambe con modality dropout. Il rischio che le separa è
l'**oblio delle cavità**: WMH non ne ha, e riaddestrare tutta la rete a passo pieno potrebbe
cancellare ciò che ATLAS ha insegnato. Si misura valutando la rete sui casi di ATLAS con la FLAIR
vuota, oltre al Dice di validazione su WMH (512: 0.753).

**Domande aperte**: centro (e scanner) di ognuno dei 27 pazienti; età, per la coorte del template;
conferma del template pediatrico; FSM_028.

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
