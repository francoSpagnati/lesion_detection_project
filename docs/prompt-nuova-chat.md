# Prompt di avvio per una nuova sessione

Copia il blocco qui sotto come **primo messaggio** di una chat nuova. Serve a far
partire la sessione dai documenti invece di far riesplorare i dati da zero.

Quando lo stato avanza, aggiorna la sezione "DA DOVE RIPARTIRE": è la sola parte che
invecchia davvero.

---

```text
Progetto: /home/ccantini/lesion_detection_project — rilevamento automatico di lesioni
cerebrali con nnU-Net v2, tesi magistrale sulla paralisi cerebrale infantile.

PRIMA DI QUALSIASI COSA, leggi in quest'ordine:
  1. docs/obiettivo-e-piano.md        a cosa serve il progetto e cosa va fatto
  2. docs/README.md                   indice + tabella di stato
  3. docs/modelli-ed-esperimenti.md   il registro unico dei modelli: architettura,
                                      ricetta di training, risultati, cosa dicono
Poi fermati e dimmi in poche righe cosa hai capito dello stato attuale, prima di agire.
Leggi docs/cosa-e-stato-fatto.md, docs/dataset-e-decisioni.md, docs/training.md e
docs/primer-mri-e-segmentazione.md solo quando ti servono davvero.

Non riesplorare i dataset: sono gia' misurati e i numeri stanno in reports/, che sono
output di script e non vanno modificati a mano. Non rilanciare gli script di analisi se
non e' cambiato niente. E' un repo git: usa `git log` e `git show` per la storia invece
di rileggere tutto.

============================================================
CONTESTO SU DI ME E SU COME VOGLIO GLI AGGIORNAMENTI
============================================================
Non ho background in risonanza magnetica e neuroimaging: spiegami i concetti di dominio
invece di darli per scontati. Scrivimi sempre in italiano.

Quando mi aggiorni: dimmi cosa e' successo, cosa dice il numero e cosa ne segue per il
piano. Se un risultato contraddice qualcosa che mi avevi detto prima, dimmelo
esplicitamente invece di lasciarlo passare. Se una cosa non l'hai misurata, dillo.

Ho incontri periodici con i relatori. Per quelli produco artefatti (pagine HTML
pubblicate) con risultati, figure e proposte. Ne esistono gia' due, e vanno AGGIORNATI
passando il loro url, non ricreati da zero:
  - relazione tecnica per i relatori:
    https://claude.ai/code/artifact/272e1e84-49f2-42a0-b3ca-31ad222adab9
  - guida personale su cosa mostrare all'incontro:
    https://claude.ai/code/artifact/9e2b7510-9851-4142-b1e2-bc11aeb0ac3c
Le figure per gli artefatti stanno in reports/figures/ e si rigenerano tutte da script
(vedi la tabella in docs/README.md). I relatori hanno chiesto esplicitamente di poter
vedere le immagini, non solo le metriche.

============================================================
IL PROGETTO IN BREVE
============================================================
Il bersaglio finale NON e' nessuno dei dataset pubblici: sono le lesioni in bambini con
paralisi cerebrale, dal progetto AInCP / Stella Maris, un dataset privato di CIRCA DIECI
PAZIENTI segmentati a mano. Su dieci casi non si addestra da zero, quindi tutto il
lavoro sui dati pubblici serve a costruire i PESI DI PARTENZA migliori possibili.

Due cose confermate su AInCP, che vincolano le scelte a monte:
  - avra' T1 **e FLAIR**  -> il modello finale sara' a due canali, quindi anche il
    pretraining dovra' esserlo (nnU-Net trasferisce i pesi solo fra architetture
    identiche);
  - contiene **anche acquisizioni a 1.5 T**, non solo 3 T (chiarito il 5 settembre:
    una versione precedente dei documenti diceva il contrario);
  - arriva a scaglioni su redcap.unipi: 10 soggetti, poi altri 10. In tutto ~20.

La pipeline prevista ha due stadi: (1) trovare la lesione; (2) inpainting per generare un
"gemello sano" virtuale su cui fare la parcellazione anatomica, perche' gli strumenti
basati su template falliscono quando l'anatomia e' distorta.

I tre dataset pubblici: WMH (79+20 casi, FLAIR **e** T1 co-registrati), ATLAS (556+99,
solo T1), DBB/dHCP (207+37, solo T1, 6 tessuti e NESSUNA annotazione di lesione).

============================================================
VINCOLI OPERATIVI — NON NEGOZIABILI
============================================================
- /data01 e' in SOLA LETTURA: i dati sorgente non si modificano mai. Tutto l'output su
  /home, che e' oltre il 90% pieno: controlla `df -h /home` prima di ogni training.
- `source scripts/env.sh` prima di ogni comando nnU-Net (imposta anche
  nnUNet_extTrainer, senza cui i nostri trainer non vengono trovati).
- Le 4 A100 80GB sono CONDIVISE con altri utenti: controlla `nvidia-smi` e imposta
  CUDA_VISIBLE_DEVICES sulla scheda libera. Non fidarti di quale era libera prima.
- I training si lanciano con `setsid nohup ... &`: nohup da solo non protegge dalla
  terminazione del gruppo di processi. Un training perso all'epoca 189 e' costato caro.
- Il progresso NON si legge in reports/logs/: quello e' stdout bufferizzato e puo'
  restare indietro di mezz'ora. Si legge in
  nnUNet/nnUNet_results/Dataset<NNN>_*/*/fold_0/training_log_*.txt
- splits_final.json NON si rigenera MAI: se cambia, tutti i confronti fra run perdono
  validita'.
- Il TEST SET non e' mai stato guardato, in nessun dataset. Va usato una volta sola,
  alla fine, sulla configurazione gia' decisa.
- .env contiene WANDB_API_KEY: non citarlo, non stamparlo, non committarlo.
- Non ho credenziali GitHub su questa macchina: i commit restano locali e li pusho io.
  Dimmi quanti ce ne sono da pushare quando finisci un blocco di lavoro.

============================================================
COME LAVORARE
============================================================
- Script in scripts/, output rigenerabile in reports/, commenti in italiano nello stile
  di quelli gia' presenti (spiegano il PERCHE', non il cosa).
- Prima di implementare, controlla se nnU-Net lo fa gia': e' in ~/.venv (nnunetv2 2.8.1)
  e il sorgente e' leggibile. Ho gia' perso tempo reimplementando un logger W&B che
  nnU-Net 2.8 aveva integrato.
- Misura prima di costruire. La correzione N4 e' stata scartata cosi': mezza giornata di
  misura ha evitato un preprocessing inutile.
- Un esperimento cambia UNA variabile per volta. Le varianti sono dataset nuovi con un
  ID nuovo, mai sovrascrivendo l'originale. Convenzione: 50x principali, 51x/52x
  varianti, 53x unioni, 54x esperimenti di pretraining, 599 usa e getta.
- I confronti si fanno APPAIATI caso per caso (scripts/confronto_appaiato.py), mai fra
  due medie complessive: se due modelli non hanno la stessa validation, la differenza
  puo' venire dalla composizione del campione.
- Misura sull'intero dataset, non su un campione: la perdita di lesione da skull
  stripping sembrava nulla su 100 casi, e su tutti e 635 aveva tre casi sopra il 5%.
- Verifica prima di affermare. Se un numero non l'hai calcolato in questa sessione o non
  e' in un report, dillo invece di stimarlo.
- Documentazione: consolida nei file esistenti, non aggiungerne di nuovi, tienila
  compatta. Ogni nuovo training va aggiunto a docs/modelli-ed-esperimenti.md.

============================================================
COSA ABBIAMO IMPARATO — I RISULTATI CHE CONTANO
============================================================
Tutti Dice di validation, fold 0, 3d_fullres, 1000 epoche salvo dove indicato.

Baseline:  501 WMH/FLAIR 0.774 | 502 ATLAS/T1 0.610 | 503 DBB/6 tessuti 0.971
           (il valore di 503 non e' confrontabile: compito diverso, foreground denso)

1. IL T1 DA SOLO NON VEDE LE LESIONI PICCOLE E NON CAVITATE. E' il risultato che
   struttura la tesi, trovato per due vie indipendenti: l'ablazione FLAIR (511 = 0.626
   contro 0.774, peggiore in 16 casi su 16, p=0.00003, con perdita -0.242 sulle lesioni
   piccole contro -0.055 sulle grandi) e l'analisi dei 10 casi falliti di ATLAS
   (rapporto di intensita' lesione/tessuto 1.07 nei falliti contro 0.62 nei riusciti).
   Coincide con la distinzione clinica di Simarro et al. fra iperintensita' FLAIR
   (leucomalacia periventricolare) e ipointensita' T1 (cavita').
2. NON SI AGGIRA affiancando i canali: 512 (FLAIR+T1) = 0.753 contro 0.774, p=0.025.
3. ESCLUDERE I DATI A 1.5 T NON SERVE: 521 = +0.009 con p=0.67, e il modello solo-3T
   applicato ai 1.5 T mai visti fa 0.672, meglio dei suoi stessi casi a 3 T.
4. LO SKULL STRIPPING NON CAMBIA NULLA: 522 +0.0045 (p=0.86), 523 -0.0037 (p=0.58).
   SynthStrip pero' e' validato e affidabile (0.9657 su 244 maschere DBB, 0.9569 sui
   patologici), e serve per la variante con etichetta `ignore`.
5. LA VARIANTE D (tutti e tre i dataset insieme, 842 casi, 7 classi) non crolla ma non
   guadagna: lesione -0.007 su ATLAS (p=0.073). E ha risolto la contraddizione fra le
   annotazioni IMPARANDO A RICONOSCERE LA SORGENTE: sui casi adulti predice solo la
   lesione e zero tessuti, sui pediatrici solo i tessuti e zero lesione, su 149 casi
   su 151. E' un modello a due modalita', ed e' un rischio da dichiarare.
6. IL PRETRAINING SERVE, E MOLTO. Test 541 su 10 casi di target mai visti: da zero
   0.4507, con fine-tuning da 521 0.5642, +0.1135 con p=0.0001.
   MA il modello pre-addestrato applicato ZERO-SHOT fa 0.6772, cioe' MEGLIO del modello
   messo a punto (p=0.013). La ricetta di fine-tuning di default DANNEGGIA il modello:
   nnU-Net riparte con passo di apprendimento 0.01, che su dieci casi cancella le
   feature invece di raffinarle.
7. Il Dice su questi compiti dipende quasi solo dal VOLUME della lesione: va sempre
   letto con la distribuzione accanto, mai da solo.

============================================================
FONTI IMPORTANTI
============================================================
- Simarro et al. 2025, "Automatic brain quantification in children with unilateral
  cerebral palsy", Frontiers in Neuroscience (PMC11931148). E' il lavoro di riferimento
  piu' vicino al nostro, ed e' nella lista del responsabile della tesi. T1+FLAIR, 35
  bambini di training, modello di lesione addestrato DA ZERO con inizializzazione di He
  e senza riportare il Dice del modello di lesione. Definisce i due fenotipi di lesione
  che ritroviamo nei nostri dati.
- Amorosino et al. 2022, "DBB — A Distorted Brain Benchmark for Automatic Tissue
  Segmentation in Paediatric Patients", NeuroImage. Il paper del nostro terzo dataset:
  nasce perche' gli strumenti basati su template falliscono sull'anatomia distorta, che
  e' la stessa motivazione del nostro stadio 2.
- ATLAS v2.0: vincitore della challenge MICCAI 2022 a Dice 0.667 (MAPPING, ensemble +
  post-processing); lavori successivi rivendicano 0.71-0.73. Noi 0.610 con un solo fold,
  senza ensemble e senza post-processing.
- WMH 2017: i metodi migliori fra 39 stanno a 0.80-0.83. Noi 0.774, un fold.
- "Learning to segment anatomy and lesions from disparately labeled sources",
  arXiv 2503.18840 (Medical Image Analysis): affronta esattamente il problema della
  nostra variante D con due rami separati e una fusione. La strada alternativa se
  l'unione diretta non regge.
- SynthSeg (Billot et al. 2023) e SynthStrip (FreeSurfer), indicati dal responsabile.

============================================================
DA DOVE RIPARTIRE (aggiornato il 2 settembre 2026)
============================================================
IN CORSO sulla GPU 0 dalle 17:30 del 2 settembre: il seguito del test del pretraining.
Due bracci di fine-tuning con passo di apprendimento ridotto (1e-3 e 1e-4) partendo dai
pesi di 521, lanciati con scripts/run_test_finetuning_lr.sh. Chiudono verso le 22:30.
La domanda: un passo piu' prudente supera lo zero-shot (0.6772) invece di danneggiarlo?
L'analisi si fa con scripts/analisi_test_pretraining.py, che va esteso ai nuovi bracci.

PROSSIMI PASSI, in ordine di valore:
1. Leggere il risultato dei due bracci. Se nemmeno un passo ridotto batte lo zero-shot,
   provare a congelare l'encoder nelle prime epoche.
2. RIPETERE L'ESTRAZIONE DEI 10 CASI. E' il limite piu' serio del test 541: con dieci
   casi la varianza e' alta e ne abbiamo una sola estrazione. Due o tre estrazioni
   diverse renderebbero il risultato solido invece che indicativo.
3. Transfer da compito diverso (DBB -> lesione). ATTENZIONE: richiede prima di spostare
   il piano di 502 su DBB e riaddestrarlo (7 ore), perche' le convoluzioni trasposte del
   decoder hanno dimensione pari allo stride e l'ultimo stadio di 503 usa [2,2,1] invece
   di [2,2,2]: i pesi di 503 NON si caricano in una rete con il piano di 502. Verificato.
4. La variante D skull-stripped, per il confronto diretto con 531 che avevo chiesto.
5. Confronto 2d contro 3d_fullres in cross-validation.
6. I cinque fold sulla configurazione scelta, per il numero finale della tesi.

TRE PROPOSTE FERME, in attesa che io ne parli ai relatori. Non attuarle senza mio ok:
  A. aggiungere dataset pubblici con T1 e FLAIR appaiati (MSLesSeg per le iperintensita'
     periventricolari, ISLES 2015 SISS per le cavita'). Serve perche' dei nostri tre
     dataset solo WMH ha entrambe le sequenze, cioe' 79 casi.
  B. pre-addestrare a un canale sugli 842 casi e poi ESPANDERE il primo strato a due
     canali. Richiede un trainer che sovrascriva il caricamento dei pesi, perche'
     nnU-Net verifica con un assert che le forme combacino. Non serve abbandonare
     nnU-Net: e' una funzione sola.
  C. generare LESIONI SINTETICHE sui 207 cervelli pediatrici sani di DBB, usando i
     rapporti di intensita' gia' misurati (0.62 cavitate, 1.07 non cavitate) e la
     parcellazione DBB per collocarle dove stanno clinicamente. E' l'unica delle tre che
     non richiede dati esterni, ed e' l'unica che produce anche le coppie
     (con lesione, senza lesione) di cui lo stadio 2 ha bisogno.
Il dettaglio delle tre proposte e' in docs/obiettivo-e-piano.md e nell'artefatto.

Prima di lanciare qualsiasi training: `df -h /home` e `nvidia-smi`.
```

---

## Perché è fatto così

Scelte deliberate, che valgono anche se lo riscrivi:

- **Chiede di leggere e poi fermarsi.** Senza quel vincolo una sessione nuova comincia a
  esplorare i dati e spende decine di migliaia di token per riscoprire cose già scritte.
- **Vieta esplicitamente la riesplorazione** e indica dove sono già i numeri. È la
  singola riga che fa risparmiare più token.
- **Porta i risultati dentro il prompt**, non solo il rimando ai documenti: sono la cosa
  che serve più spesso, e averli sott'occhio evita alla sessione nuova di aprire tre file
  per rispondere a una domanda semplice.
- **Cita gli errori già commessi** (il logger W&B reimplementato, il training perso senza
  `setsid`, i pesi di DBB che non si caricano) invece di dare regole astratte: un esempio
  concreto è più efficace di "non reinventare la ruota".
- **Separa ciò che invecchia** — la sezione finale — da ciò che resta valido, così
  aggiornarlo costa una modifica sola.
