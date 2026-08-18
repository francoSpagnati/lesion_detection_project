# Recap dello stato del lavoro

Documento di **comunicazione**, non documentazione tecnica: riassume per chi sta fuori dal progetto cosa
è stato analizzato, cosa è stato preparato, quali scelte sono state prese e con quale motivazione. I
dettagli tecnici restano negli altri documenti di `docs/`, i numeri nei report rigenerabili di `reports/`.

Aggiornato al 18 agosto 2026. **Sono disponibili i primi risultati di segmentazione** su due dei tre
dataset (sezione 8); il terzo è in corso.

---

## 1. In sintesi

Il lavoro si è svolto in due fasi. La prima, di **analisi dei dati e costruzione dell'infrastruttura
sperimentale**: tre dataset pubblici acquisiti e caratterizzati, convertiti nel formato di nnU-Net v2 con
split riproducibili, planning e preprocessing completati. La seconda, appena iniziata, di **misura**: sono
arrivati i primi risultati di riferimento su WMH e ATLAS, con un Dice di **0.77** e **0.61** in
cross-validation. Il terzo dataset è in addestramento.

Questi valori non sono il traguardo: sono il metro contro cui misurare tutto ciò che viene dopo. Senza di
essi non è possibile dire se il pretraining, che è il meccanismo centrale del progetto, porti un
guadagno.

La scelta metodologica di fondo, che spiega quasi tutte le decisioni prese: **misurare prima di
costruire**. Diversi passi di preprocessing dati per scontati sono stati sottoposti a verifica empirica
prima di implementarli, e in due casi la misura li ha esclusi, risparmiando lavoro e riducendo il rischio
di errori silenziosi.

## 2. Il vincolo che determina l'impostazione

Il bersaglio finale è il dataset privato AInCP / Stella Maris, segmentato manualmente dalla Dr.ssa Francy
Cruz: **circa 10 pazienti**, non ancora consegnato. Con dieci casi non è possibile addestrare una rete da
zero.

Ne segue che i dataset pubblici non servono a produrre il modello finale, ma a **costruire pesi di
partenza**: il pretraining non è un esperimento accessorio, è il meccanismo centrale del progetto. Tutta
l'organizzazione del lavoro discende da qui.

## 3. I dati acquisiti e il ruolo di ciascuno

Tre dataset pubblici, scelti perché ciascuno copre una famiglia di lesione del bersaglio, non perché
ridondanti fra loro.

| | Compito | Popolazione | Canale | Casi usabili | Copre |
| --- | --- | --- | --- | --- | --- |
| **WMH** | iperintensità sostanza bianca | adulti, 4 scanner | FLAIR (+T1) | 99 | lesioni sostanza bianca; unico con FLAIR |
| **ATLAS 2.0** | esiti di ictus | adulti, 33 coorti | T1 | 655 | cavità e lesioni focali; il più numeroso |
| **DBB** | 6 tessuti (non lesioni) | **pediatrica**, malformati | T1 | 244 | malformazioni; anatomie alterate |
| **AInCP** | lesioni da paralisi cerebrale | pediatrica | T1 + FLAIR | ~10 | **il bersaglio**, atteso |

DBB (Distorted Brain Benchmark, brainlife, CC-BY 4.0) merita una nota: è costruito **come test di
generalizzazione**. Si addestra su 207 cervelli pediatrici sani e si valuta su 37 malformati (11 PFM, 10
MCD, 10 ACC, 6 HD), cioè su anatomie mai viste — la stessa domanda che conta per il bersaglio finale.
Dei 955 soggetti pubblicati solo 244 sono utilizzabili: gli altri 711 hanno le label ma non le immagini,
trattenute dalle coorti ad accesso controllato C-MIND e NIH-PED.

## 4. Analisi svolte e cosa hanno rivelato

Tutte le analisi sono state fatte **in sola lettura** sui dati sorgente, che non sono mai stati
modificati, e producono report rigenerabili.

**Inventario e caratterizzazione geometrica.** Per ogni caso: spacing, shape, campo di vista,
orientamento, volume lesionale in mm³. Da qui è emerso che il numero di file è molto maggiore del numero
di casi utilizzabili, perché le cartelle sorgente contengono sottoprodotti (campi di bias, volumi non
allineati, matrici di trasformazione) che non vanno usati.

**Verifiche sulle label.** Tre problemi trovati misurando, tutti silenziosi — nessuno avrebbe fatto
fallire un training, tutti avrebbero prodotto un modello che impara la cosa sbagliata:

- in WMH il valore `2` non è una lesione WMH ma **altra patologia**, tipicamente infarti, presente in
  **34 casi su 99** con volumi fino a 33 ml, ed è escluso dalla valutazione nella challenge ufficiale;
- in ATLAS le due varianti di maschera fornite (`_mask` e `_mask_value2`) sono la **stessa
  segmentazione**, differiscono solo per il valore codificato;
- sempre in ATLAS, il caso `sub-r039s002` è codificato con valori `0` e `0.01`: binarizzando con `== 1`
  invece che `> 0`, i suoi 2810 voxel di lesione sparirebbero e il paziente entrerebbe nel training come
  soggetto sano. Un caso corrotto su 655 non sposta le metriche in modo visibile, ed è precisamente
  questo che lo renderebbe difficile da scoprire a valle.

**Eterogeneità dello spazio in DBB.** 213 casi su 244 sono in spazio MNI, ma 31 sono in spazio nativo e
12 di questi sono orientati RAS invece che LAS. Poiché nnU-Net legge gli array così come sono, mescolare
orientamenti significa dare alla rete volumi specchiati fra loro.

**Il confronto a tre ha mostrato che non sono tre versioni dello stesso problema:**

| | WMH | ATLAS | DBB |
| --- | --- | --- | --- |
| foreground mediano | 10.9 ml | 5.3 ml | 1479 ml |
| frazione del campo di vista | 0.0010 | 0.0006 | **0.2034** |

In WMH e ATLAS il bersaglio è un voxel su mille — individuazione di anomalie rare. In DBB è un voxel su
cinque — partizione del cervello in tessuti. Trecento volte di differenza implica loss diverse,
augmentation utili diverse e metriche che non si leggono allo stesso modo.

## 5. Preparazione dei dataset e scelte prese

**Split.** Per ciascun dataset: un test holdout tenuto fuori, e 5 fold di cross-validation sui restanti,
con seed fissato. La stratificazione è per **gruppo × terzile di volume lesionale** (gruppo = sito per
WMH, coorte per ATLAS). Motivo: con lesioni molto piccole il Dice è quasi rumore, e quel rumore si sposta
fra i fold — se un fold pesca quattro casi minuscoli e un altro nessuno, i due Dice differiscono per
composizione del fold, non per qualità del modello.

**Il test è tenuto fuori dal fingerprint.** nnU-Net analizza tutti i casi presenti nella cartella di
training per decidere il piano: se il test stesse lì, target spacing e patch size sarebbero in parte
derivate da dati che devono restare invisibili. Il test viene toccato una volta sola, alla fine, sulla
configurazione già scelta in cross-validation.

**Politica sulla label 2 di WMH:** esclusa (`2 → 0`), coerente con la valutazione ufficiale della
challenge. La documentazione iniziale del progetto prevedeva di fonderla con la lesione: sarebbe stato un
errore doppio, perché avrebbe insegnato al modello a segmentare gli infarti come iperintensità su un terzo
del dataset, e reso il Dice non confrontabile con la letteratura.

**Riorientamento su DBB soltanto**, dove lo spazio era mescolato, e realizzato al volo in lettura senza
riscrivere i dati su disco. Su WMH e ATLAS l'orientamento è uniforme e non serve.

**I dati non vengono duplicati:** le immagini sono referenziate con symlink ai file originali; solo le
label vengono riscritte, per allinearne esattamente la geometria a quella dell'immagine. I quattro
dataset preparati occupano decine di MB invece dei ~14 GB delle copie.

**Ogni variante è un dataset nuovo, non una sovrascrittura.** Per esempio esiste `Dataset511_WMH_T1`,
identico a WMH in tutto — stessi casi, stesse label, **stessi fold** — tranne il canale, che è T1 invece
di FLAIR. Poiché cambia una sola variabile, la differenza di Dice fra i due misura esattamente il
contributo della FLAIR.

## 6. Preprocessing: cosa fa nnU-Net e cosa abbiamo verificato

L'impostazione originariamente prevista comprendeva una pipeline manuale: riorientamento, resampling a 1
mm, normalizzazione, padding a dimensione fissa con salvataggio dei parametri per il ripristino.
**nnU-Net la sussume**: esegue internamente riorientamento (dove richiesto), resampling e normalizzazione,
e rende superfluo il padding perché lavora a patch con finestra scorrevole. Reimplementarla a monte
avrebbe prodotto una doppia interpolazione, che degrada proprio le lesioni piccole.

Planning e preprocessing sono stati completati sui tre dataset (24 GB, circa 3 minuti). La verifica che
conta è che la **risoluzione di destinazione scelta da nnU-Net coincida con quella calcolata da noi
misurando i dati**, ed è confermata su tutti e tre:

| | Risoluzione di destinazione | Patch 3D | Batch |
| --- | --- | --- | --- |
| WMH | `1.2 × 0.9766 × 3.0` mm | `64 × 224 × 128` | 2 |
| ATLAS | `1.0 × 1.0 × 1.0` mm | `128 × 128 × 128` | 2 |
| DBB | `1.0 × 1.0 × 1.0` mm | `128 × 160 × 112` | 2 |

Due esiti utili emersi dal planning:

- la configurazione a **bassa risoluzione è stata scartata** dal pianificatore su tutti e tre i dataset,
  perché produrrebbe volumi quasi identici a quelli a piena risoluzione. Restano quindi due candidati, 2D
  e 3D a piena risoluzione, e la cascata non è applicabile;
- su WMH la patch 3D copre **64 delle 79 fette**, cioè l'81% del volume. Questo indebolisce l'argomento a
  priori a favore della configurazione 2D, che con fette da 3 mm sembrava favorita dalla scarsità di
  contesto attraverso il piano: il contesto in realtà c'è quasi tutto. Il confronto va comunque misurato.

L'anisotropia di WMH resta il limite intrinseco del dataset: fette da 3 mm su tutti e 99 i casi, contro
circa 1 mm nel piano. **Non è correggibile**: l'informazione non è stata persa nel preprocessing, non è
mai stata acquisita, e interpolare crea voxel plausibili, non informazione.

## 7. Sui consigli ricevuti: skull stripping, BET, N4

Questa sezione risponde direttamente a cosa ne è stato degli strumenti suggeriti.

### Skull stripping con FSL BET: non applicato, con motivazione

**BET non è stato usato**, e la ragione è tecnica, non di comodità. BET fa crescere una superficie
deformabile guidata da soglie di intensità, assumendo una testa di forma regolare, ed è tarato sul T1.
Sono esattamente i due presupposti che cadono nel nostro caso: le immagini WMH sono **FLAIR**, e il
bersaglio finale sono **anatomie fortemente alterate** — cioè la situazione in cui un metodo basato su un
modello di forma regolare tende a fallire, ed è lo stesso motivo per cui gli strumenti basati su template
standard falliscono su questi pazienti, che è il problema che la tesi si propone di affrontare.

Va aggiunto che **nnU-Net non richiede lo skull stripping** per addestrare: ritaglia la regione non nulla
e normalizza per immagine. Quindi il passo non è un prerequisito, è un'opzione da giustificare.

Se servirà, i candidati individuati sono due, entrambi basati su reti addestrate e non su modelli di
forma:

- **HD-BET**, dello stesso gruppo che ha sviluppato nnU-Net, addestrato anche su cervelli patologici;
- **SynthStrip**, di FreeSurfer, il più robusto sulle sequenze non-T1 e quindi il candidato naturale per
  la FLAIR.

Su **DBB il problema non si pone**: il dataset fornisce già le maschere cerebrali per tutti i soggetti.

### Correzione del bias di campo N4: misurata ed esclusa

Era l'unico passo di preprocessing classico che avrebbe aggiunto qualcosa a quello che nnU-Net fa già,
quindi prima di implementarlo è stato misurato. Rapporto fra 99º e 1º percentile del campo stimato
(`1.00` significa campo perfettamente piatto), erodendo la maschera di margini fisici crescenti verso
l'interno, sugli stessi casi:

| Dataset | 0 mm | 3 mm | 6 mm | 12 mm |
| --- | --- | --- | --- | --- |
| WMH | 1.678 | 1.550 | 1.163 | **1.119** |
| ATLAS | 1.594 | 1.548 | 1.448 | **1.120** |
| DBB | 1.172 | 1.148 | 1.124 | **1.095** |

Tutti convergono a 1.10–1.12 dentro il cervello: **nessun bias interno apprezzabile**. I valori alti in
superficie sono artefatto di contorno — scalpo, grasso, aria — non segnale biologico. Su WMH c'era anche
una ragione a priori: i file sono già corretti a monte dagli organizzatori.

Conclusione: **N4 non si applica**, e mezza giornata di misura ha evitato di costruire e mantenere una
pipeline di correzione inutile.

**Limite dichiarato di questa misura, ed è qui che le due questioni si toccano:** su WMH e ATLAS il campo
è stato misurato dentro una maschera della *testa*, mentre su DBB dentro la *maschera cerebrale* fornita.
I numeri sono pienamente confrontabili solo al margine più profondo. **Una misura definitiva
richiederebbe prima lo skull stripping** — è la ragione più concreta per cui la questione potrebbe essere
riaperta.

### FreeSurfer

Rientra nel progetto in due punti, entrambi futuri: **SynthStrip** come candidato per lo skull stripping
(sopra), e **SynthSeg** come riferimento per la parcellazione anatomica nella seconda fase del progetto,
quella del "gemello sano virtuale", dove le label vanno prodotte su un'immagine con anatomia regolarizzata
e poi riportate sull'originale.

### Altri passi valutati e non adottati

| Non adottato | Motivo |
| --- | --- |
| Resampling manuale a 1 mm | Lo fa nnU-Net una volta sola; anticiparlo produce una doppia interpolazione che degrada le lesioni piccole. |
| Normalizzazione z-score a monte | nnU-Net la applica per immagine dopo il ritaglio, con statistiche del solo training. Farla prima significherebbe calcolarle anche sul test, cioè leakage. |
| Pulizia o riempimento delle maschere | La ground truth è il riferimento per definizione: "migliorarla" cambia il compito e rende i risultati incomparabili con la letteratura. |
| Esclusione dei casi con lesioni minuscole | Sono casi clinici legittimi e fra i più difficili; rimuoverli gonfia le metriche senza migliorare il modello. |
| Training congiunto sui tre dataset | Le semantiche delle label sono incompatibili: un voxel di sostanza bianca sana vale `3` in DBB e `0` in ATLAS. Si condivide il peso della rete, non l'etichetta. |

## 8. Primi risultati

Configurazione: rete 3D a piena risoluzione, **primo fold** di cross-validation, 1000 epoche, su una
singola GPU A100 (circa 6 ore per dataset). Il terzo dataset, DBB, è in addestramento.

I valori riportati sono il Dice calcolato sui **casi interi** ricostruiti con la finestra scorrevole, in
**cross-validation**. Il test holdout non è stato toccato e resterà intatto fino alla scelta definitiva
della configurazione.

| | casi in validation | Dice medio | mediana | Q1–Q3 | min–max |
| --- | --- | --- | --- | --- | --- |
| **WMH** (FLAIR) | 16 | **0.774** | 0.837 | 0.705 – 0.880 | 0.426 – 0.906 |
| **ATLAS** (T1) | 112 | **0.610** | 0.721 | 0.465 – 0.839 | 0.000 – 0.946 |

Il dettaglio completo, rigenerabile, è in `reports/baseline_validation.md`. Quattro osservazioni contano
più dei due numeri medi.

**1. La mediana è nettamente più alta della media** in entrambi i casi (+6 punti su WMH, +11 su ATLAS).
La distribuzione è asimmetrica: il modello lavora bene sulla maggioranza dei casi, e la media è tirata giù
da una minoranza di casi molto difficili. Riportare solo la media descrive male il comportamento reale.

**2. La dimensione della lesione domina il risultato.** Dividendo i casi a metà per volume del bersaglio:

| | metà con lesioni più piccole | metà con lesioni più grandi |
| --- | --- | --- |
| WMH | 0.667 | 0.881 |
| ATLAS | 0.509 | 0.712 |

Su ATLAS i casi sotto i 100 voxel di lesione hanno un Dice medio di **0.28**. Non è una sorpresa ed era
previsto in fase di analisi: su bersagli così piccoli il Dice è instabile per costruzione, perché sbagliare
dieci voxel su venti fa crollare il punteggio mentre sulla stessa lesione da ventimila non si nota. È la
ragione per cui la valutazione va affiancata da metriche per lesione.

**3. Su ATLAS una singola coorte spiega gran parte del divario.** La coorte `r009` pesa 19 casi su 112 e
ha lesioni cinque volte più piccole della media delle altre:

| | casi | volume lesionale mediano | Dice medio |
| --- | --- | --- | --- |
| coorte `r009` | 19 | 1398 voxel | **0.337** |
| tutte le altre | 93 | 7053 voxel | **0.666** |

Il Dice complessivo di ATLAS, 0.61, è quindi in buona parte un effetto di composizione del campione, non
una misura della capacità del modello. È coerente con la natura del dataset — 33 coorti da centri diversi,
con casistiche molto disomogenee — ed è il motivo per cui la valutazione è stata stratificata per coorte
fin dalla costruzione degli split.

**4. Otto casi su 112 hanno Dice esattamente zero**, e non sono solo i più piccoli: uno ha **9586 voxel**
di lesione vera e 217 voxel predetti. Con nessuna sovrapposizione, non si tratta di bordi imprecisi ma di
**errori di localizzazione**: il modello ha segmentato nel posto sbagliato. Sono una categoria di
fallimento qualitativamente diversa e vanno esaminati singolarmente, non mediati insieme al resto.

**Su WMH il Dice per sito** — la domanda interessante di quel dataset, che raccoglie quattro scanner
diversi — dà Singapore 0.818, Amsterdam_GE3T 0.790, Amsterdam_GE1T5 0.750 e Utrecht 0.715. Un divario c'è,
ma con 16 casi distribuiti su quattro siti in un solo fold **non è ancora interpretabile**: serviranno i
cinque fold prima di trarne qualcosa.

**Cautele sulla lettura di questi numeri.** È un fold solo su cinque, quindi la variabilità fra fold non è
ancora nota e nessuna differenza piccola sarebbe credibile. Il confronto con i valori pubblicati sui due
benchmark va fatto a cinque fold completi e con lo stesso protocollo di valutazione, non su questi.

## 9. Questioni aperte su cui un parere sarebbe utile

- **Skull stripping:** vale la pena introdurlo con HD-BET o SynthStrip, sapendo che nnU-Net non lo
  richiede e che il beneficio atteso è soprattutto una misura più pulita del bias di campo e una
  riduzione del dominio irrilevante?
- **Risoluzione comune fra dataset.** Per trasferire pesi fra dataset serve che condividano lo stesso
  piano, e quindi la stessa griglia. Forzare WMH sulla griglia isotropa a 1 mm significa interpolare
  fette da 3 mm; mantenerla anisotropa impedisce il trasferimento diretto da e verso gli altri due.
- **Disponibilità di AInCP**, che determina quando può iniziare la parte finale del lavoro, e se le
  annotazioni comprenderanno anche la FLAIR oltre al T1.
