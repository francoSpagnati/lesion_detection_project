# Prompt di avvio per una nuova sessione

Copia il blocco qui sotto come **primo messaggio** di una chat nuova.

Da settembre esiste anche `CLAUDE.md` nella radice del progetto: i vincoli operativi e lo stile di
lavoro si caricano da soli a ogni sessione, quindi il prompt non deve più ripeterli. Serve solo per
ciò che `CLAUDE.md` non può contenere: **cosa leggere, in che ordine, com'è fatto il piano, e a che
punto siamo**.

Quando lo stato avanza, aggiorna la sezione "DA DOVE RIPARTIRE": è la sola parte che invecchia.

---

```text
Progetto: /home/ccantini/lesion_detection_project — rilevamento automatico di lesioni cerebrali
con nnU-Net v2, tesi magistrale sulla paralisi cerebrale infantile.

I vincoli operativi e lo stile di lavoro stanno in CLAUDE.md e li hai gia' caricati. Qui c'e' il
resto.

============================================================
COSA LEGGERE, IN QUEST'ORDINE
============================================================
  1. docs/obiettivo-e-piano.md        a cosa serve il progetto, le decisioni prese e perche'
  2. docs/README.md                   indice + tabella di stato con tutti i risultati
  3. docs/modelli-ed-esperimenti.md   il registro unico dei modelli: architettura, ricetta di
                                      training, risultati, e cosa dice ciascun risultato
Poi FERMATI e dimmi in poche righe cosa hai capito dello stato attuale, prima di agire.

Da leggere solo quando servono davvero:
  docs/cosa-e-stato-fatto.md          cronologia: quale script ha prodotto cosa
  docs/dataset-e-decisioni.md         com'e' fatto ogni dataset e i problemi trovati misurandolo
  docs/training.md                    la procedura: come si lancia, si legge, si valuta
  docs/primer-mri-e-segmentazione.md  concetti di neuroimaging, se servono a me

NON riesplorare i dataset: sono gia' misurati e i numeri stanno in reports/, che sono output di
script e non vanno modificati a mano. Non rilanciare gli script di analisi se non e' cambiato
niente. E' un repo git: usa `git log` e `git show` per la storia invece di rileggere tutto.

============================================================
COM'E' FATTO IL PROBLEMA
============================================================
Il bersaglio finale NON e' nessuno dei dataset pubblici: sono le lesioni in bambini con paralisi
cerebrale, dal progetto AInCP / Stella Maris. Su una ventina di pazienti non si addestra da zero,
quindi tutto il lavoro sui dati pubblici serve a costruire i PESI DI PARTENZA migliori possibili.

La pipeline prevista ha due stadi: (1) trovare la lesione; (2) inpainting per generare un "gemello
sano" virtuale su cui fare la parcellazione anatomica, perche' gli strumenti basati su template
falliscono quando l'anatomia e' distorta.

I DATI PUBBLICI, tutti gia' convertiti in formato nnU-Net sotto data/processed/nnUNet_raw/:

| dataset | casi (sviluppo + test) | sequenze | annotazione |
| WMH      | 79 + 20   | FLAIR **e** T1 co-registrati | lesione (iperintensita' della bianca) |
| ATLAS    | 556 + 99  | solo T1                      | lesione (ictus) |
| DBB/dHCP | 207 + 37  | solo T1                      | 6 tessuti, NESSUNA lesione |

Verificato sui file: ATLAS e DBB la FLAIR non ce l'hanno proprio. "Aggiungere la FLAIR ai dataset
che abbiamo" si puo' fare solo su WMH, ed e' gia' fatto (Dataset512).

I DATI PRIVATI (AInCP), in arrivo:
  - avranno T1 **e FLAIR** -> il modello finale sara' a due canali;
  - contengono **anche acquisizioni a 1.5 T**, non solo 3 T;
  - arrivano a scaglioni su redcap.unipi: 10 soggetti, poi altri 10. In tutto ~20.
  - da chiarire con il relatore: se le maschere distinguono i due fenotipi (iperintensita' FLAIR e
    cavita') o sono una classe sola; se T1 e FLAIR sono gia' co-registrati; come suddividere i 20
    soggetti fra addestramento e valutazione (da decidere PRIMA di guardare i dati).

============================================================
COSA ABBIAMO IMPARATO — I RISULTATI CHE CONTANO
============================================================
Dice di validation, fold 0, 3d_fullres, 1000 epoche salvo dove indicato.
Baseline: 501 WMH/FLAIR 0.774 | 502 ATLAS/T1 0.610 | 503 DBB/6 tessuti 0.971 (non confrontabile)

1. IL T1 DA SOLO NON VEDE LE LESIONI PICCOLE. E' il risultato che struttura la tesi, trovato per
   TRE vie indipendenti: l'ablazione FLAIR (511 = 0.626 contro 0.774, peggiore in 16 casi su 16,
   p=0.00003); l'analisi dei 10 casi falliti di ATLAS (rapporto di intensita' lesione/tessuto 1.07
   nei falliti contro 0.62 nei riusciti); e le metriche di RILEVAMENTO PER LESIONE, che sono la
   prova piu' netta:
       sulle stesse 865 lesioni di WMH, quota rilevata sotto i 10 voxel:
       FLAIR 61.9%  |  FLAIR+T1 59.1%  |  solo T1 20.9%
       su ATLAS con solo T1: 8.7% delle lesioni sotto i 10 voxel
2. I DUE CANALI NON PEGGIORANO, come si era creduto. In Dice 512 fa 0.753 contro 0.774, ma sul
   rilevamento e' equivalente alla sola FLAIR (71.1% contro 72.8%): non perde lesioni, perde
   qualche punto di contorno.
3. IL PRETRAINING VALE +0.23 DI DICE, ma solo con il passo di apprendimento giusto. Test 541 su 10
   casi di target mai visti: da zero 0.4507, fine-tuning col default (1e-2) 0.5642, con 1e-3
   0.6419, modello NON adattato 0.6772, con 1e-4 **0.6851**. Col passo di default il fine-tuning
   PEGGIORA il modello di partenza.
4. NIENTE PREPROCESSING AIUTA. Escludere i 1.5 T: +0.009, p=0.67. Skull stripping: +0.0045
   (p=0.86) su WMH e -0.0037 (p=0.58) su ATLAS. SynthStrip e' pero' validato e affidabile
   (0.9657 su 244 maschere DBB, 0.9569 sui patologici).
5. IL MULTICLASSE TESSUTI+LESIONE E' STATO ABBANDONATO, d'accordo col relatore. La variante D
   (842 casi, 7 classi) non crollava ma non guadagnava, e la rete aveva imparato a RICONOSCERE LA
   SORGENTE: sui casi adulti predice solo la lesione e zero tessuti, sui pediatrici il contrario,
   su 149 casi su 151. Ha rifiutato da sola l'unione dei due codomini.
6. nnU-Net NON FA EARLY STOPPING (verificato nel sorgente), e il suo `checkpoint_best` NON
   protegge dall'overfitting perche' e' selezionato sulla pseudo-dice mediata e non sulla perdita
   di validation: misurato, da' 0.4491 contro 0.4507. Un criterio di arresto vero va scritto.
7. L'OVERFITTING E' MASSICCIO SUI POCHI CASI. Divario fra dati visti e non visti: +0.534 da zero,
   poi +0.398, +0.275, +0.179 al ridursi del passo. Stessa progressione dei Dice.

RICETTA DI FINE-TUNING PER AInCP, misurata e da usare sui pazienti veri:
  passo di apprendimento 1e-4 (non il default) | poche epoche, il minimo cade fra la 69 e la 128 |
  misurare sempre il modello non adattato come termine di paragone.

============================================================
FONTI IMPORTANTI DA CONOSCERE
============================================================
- Simarro et al. 2025, "Automatic brain quantification in children with unilateral cerebral palsy",
  Frontiers in Neuroscience (PMC11931148). Il lavoro di riferimento piu' vicino al nostro, indicato
  dal responsabile della tesi. T1+FLAIR, 35 bambini, modello di lesione addestrato DA ZERO con
  inizializzazione di He e senza riportare il Dice del modello di lesione. Definisce i due fenotipi
  (iperintensita' FLAIR = leucomalacia periventricolare, ipointensita' T1 = cavita') che ritroviamo
  nei nostri dati.
- Amorosino et al. 2022, "DBB — A Distorted Brain Benchmark", NeuroImage. Il paper del nostro terzo
  dataset: nasce perche' gli strumenti basati su template falliscono sull'anatomia distorta, che e'
  la stessa motivazione del nostro stadio 2.
- arXiv 2604.08015, "Component-Adaptive and Lesion-Level Supervision": confronta con nnU-Net le
  loss per lesioni cerebrali piccole. Tversky 0.7706 PEGGIO del default DiceCE 0.7796, Focal
  Tversky 0.7802 pari. Ma il default e' il peggiore sulle metriche di lesione mancata: il guadagno
  non sta nel Dice, sta nel non perdere lesioni.
- arXiv 2503.18840 (Medical Image Analysis), "Learning to segment anatomy and lesions from
  disparately labeled sources": la strada principiata per unire sorgenti annotate diversamente.
- ATLAS v2.0: vincitore MICCAI 2022 a 0.667 (ensemble + post-processing), lavori successivi
  0.71-0.73. Noi 0.610 con un fold solo e senza ensemble.
- WMH 2017: i metodi migliori fra 39 stanno a 0.80-0.83. Noi 0.774, un fold.
- SynthSeg (Billot et al. 2023) e SynthStrip (FreeSurfer), indicati dal responsabile.

============================================================
SKILL DI CLAUDE DA USARE
============================================================
Sono gia' disponibili, si invocano con /nome. Non vanno scaricate.

DA USARE SEMPRE, quando ricorre il caso:
  /artifact-design   OBBLIGATORIA prima di scrivere qualunque artefatto. In questo progetto gli
                     artefatti sono il canale con cui parlo ai relatori, quindi ricorre spesso.
  /dataviz           OBBLIGATORIA prima di scrivere il codice di QUALSIASI grafico, comprese le
                     figure matplotlib che finiscono in reports/figures/.
  /handoff           quando il contesto si sta esaurendo. E' il modo corretto di passare il lavoro
                     a una sessione nuova senza perdere niente: meglio che riassumere a mano.

UTILI QUANDO SERVONO:
  /code-review       revisione degli script prima di fidarsi dei numeri che producono. Su questo
                     progetto un bug in uno script di analisi si traduce in una conclusione
                     sbagliata riportata ai relatori, quindi vale la pena.
  /simplify          pulizia del codice gia' scritto, senza cercare bug.
  /artifact-diagramming  se un artefatto ha bisogno di uno schema, non solo di grafici.
  /security-review   non pertinente qui, ma esiste.

PER RIDURRE IL CONSUMO DI TOKEN:
  /fewer-permission-prompts   analizza le sessioni passate e scrive una allowlist in
                              .claude/settings.json per i comandi di sola lettura che uso di
                              continuo (ls, cat, git log, nvidia-smi, df). Toglie interruzioni e
                              giri a vuoto. Conviene lanciarla una volta.
  /init                       gia' fatto: CLAUDE.md esiste. Da rilanciare solo se cambia la
                              struttura del progetto.

ABITUDINI CHE FANNO RISPARMIARE PIU' DELLE SKILL:
  - leggere i tre documenti indicati sopra e fermarsi, invece di esplorare;
  - usare `grep -n` e `sed -n 'X,Yp'` per leggere porzioni di file, non file interi;
  - fidarsi dei report in reports/ invece di ricalcolare;
  - non rileggere un file appena modificato: se Edit non ha dato errore, la modifica c'e'.

============================================================
DA DOVE RIPARTIRE (aggiornato il 7 settembre 2026)
============================================================
IN ESECUZIONE: rivalutazione dei 4 bracci di 541 con `checkpoint_best` invece dell'ultima epoca,
su CPU perche' le GPU sono tutte al 100% per un lavoro di altri utenti
(`scripts/run_val_best.sh`, ~2h30 per braccio). Il primo braccio ha gia' risposto: 0.4491 contro
0.4507, cioe' non recupera nulla, per il motivo spiegato al punto 6 sopra. Gli altri tre servono
solo a confermare. Non rilanciare: lo script salta i bracci gia' fatti.

TUTTE E QUATTRO LE GPU SONO OCCUPATE da altri utenti. Controllare prima di pianificare training.

PROSSIMI PASSI, in ordine:
1. `batch_dice: true` nel JSON del piano di 502 e riaddestrare. E' UNA RIGA, nessun codice, ed e'
   pensato esattamente per annotazioni piccole e sparse; nei nostri 3d_fullres e' `false`. Miglior
   rapporto fra costo e probabilita' di riuscita fra tutto cio' che resta.
2. Ripetere l'estrazione dei 10 casi del test 541 due o tre volte con semi diversi. E' il limite
   piu' serio di quel risultato: con dieci casi la varianza e' alta e ne abbiamo una sola.
3. Proposta B, la preferita dal relatore: pre-addestrare a un canale sugli 842 casi e poi
   ESPANDERE il primo strato a due canali per usare la FLAIR. Serve un trainer che sovrascriva il
   caricamento dei pesi, perche' nnU-Net verifica con un assert che le forme combacino. Non serve
   abbandonare nnU-Net: e' una funzione sola. Dettaglio in docs/obiettivo-e-piano.md.
4. Un criterio di arresto vero sulla perdita di validation, nel nostro trainer.
5. Loss per le lesioni piccole, dopo batch_dice: `nnUNetTrainerDiceTopK10Loss` (gia' in nnU-Net),
   poi Focal Tversky (da implementare). Ogni confronto va guardato SIA in Dice SIA in tasso di
   rilevamento per fascia di volume, altrimenti non si vede niente.
6. Proposta C, accettata dal relatore come strada percorribile: lesioni sintetiche sui 207 cervelli
   pediatrici sani di DBB, usando i rapporti di intensita' gia' misurati (0.62 cavitate, 1.07 non
   cavitate) e la parcellazione DBB per collocarle dove stanno clinicamente. Da validare
   addestrando sul sintetico e verificando su ATLAS e WMH reali.
7. Alla fine: confronto 2d contro 3d_fullres, e i cinque fold sulla configurazione scelta.

QUANDO ARRIVANO I PAZIENTI AInCP, e' la priorita' assoluta e cambia tutto. Applicare la ricetta di
fine-tuning misurata (passo 1e-4, poche epoche, sempre il modello non adattato come paragone).

ARTEFATTI ESISTENTI, da AGGIORNARE passando il loro url e non ricreare:
  relazione tecnica per i relatori:
    https://claude.ai/code/artifact/272e1e84-49f2-42a0-b3ca-31ad222adab9
  risposte ai chiarimenti del relatore:
    https://claude.ai/code/artifact/2ab3bbc3-27b4-41d6-8b11-0380446c3e86
  guida personale su cosa mostrare all'incontro:
    https://claude.ai/code/artifact/9e2b7510-9851-4142-b1e2-bc11aeb0ac3c
Le figure stanno in reports/figures/ e si rigenerano tutte da script: la tabella e' in
docs/README.md. I relatori hanno chiesto esplicitamente di poter vedere le immagini.
```

---

## Perché è fatto così

- **I vincoli operativi non sono più nel prompt** ma in `CLAUDE.md`, che si carica da solo. Il
  prompt resta più corto e non c'è il rischio che una sessione parta senza le regole perché
  l'utente ha incollato una versione vecchia.
- **Chiede di leggere e poi fermarsi.** Senza quel vincolo una sessione nuova comincia a esplorare
  i dati e spende decine di migliaia di token per riscoprire cose già scritte.
- **Porta i risultati dentro il prompt**, non solo il rimando ai documenti: sono la cosa che serve
  più spesso, e averli sott'occhio evita di aprire tre file per rispondere a una domanda semplice.
- **Cita gli errori già commessi** invece di dare regole astratte: il logger W&B reimplementato, il
  training perso senza `setsid`, i pesi di DBB che non si caricano, `checkpoint_best` che non
  protegge da niente.
- **Separa ciò che invecchia** — l'ultima sezione — da ciò che resta valido.
