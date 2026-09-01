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

**AInCP sarà tutto a 3 T.** L'utente preferisce quindi escludere i casi a 1.5 T dai dataset pubblici per
coerenza di dominio. La misura dice che l'esclusione **non cambia nulla in nessuna direzione**
(+0.0093 con p=0.67 su ATLAS, e il modello solo-3 T segmenta comunque bene i 1.5 T mai visti), quindi è
una scelta gratuita che si può fare per pulizia del disegno sperimentale — ma non va presentata come un
miglioramento. Applicata con coerenza tocca due dataset: `Dataset521_ATLAS_3T` esiste già (511 casi
invece di 556), e su WMH andrebbero tolti i **10 casi** `Amsterdam_GE1T5` (8 di sviluppo, 2 di test).

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
| 1 | ATLAS + DBB + WMH in T1 | 842 | 1 | anatomia e lesioni cavitate |
| 2 | WMH con T1 e FLAIR | 79 | 2 | a usare la FLAIR |
| 3 | AInCP | ~10 | 2 | il dominio pediatrico |

**Il rischio.** Allo stadio 2 il canale FLAIR parte da nulla e ci sono solo 79 casi per insegnare al
modello a servirsene. Se non bastano resta un modello a T1 con un canale inerte, cioè si torna al punto
di partenza dopo tre training. È esattamente il rischio che la proposta A elimina: **A fornisce i dati a
due canali che rendono affidabile lo stadio 2 di B**, e B permette di sfruttare i casi solo-T1 che A non
può usare. Se se ne sceglie una sola: A è la più sicura, B la più economica.

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
