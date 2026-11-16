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
cerebrale, dal progetto AInCP / Stella Maris. Su 17 pazienti non si addestra da zero,
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

I DATI PRIVATI (AInCP), tutti arrivati: 17 pazienti, e non ne arriveranno altri.
  - lotto 1 (11/9): 10 pazienti Stella Maris a 3 T, data/raw/AINCP/dataset_FSM_v1;
  - lotto 2 (1/10): 7 pazienti a 1.5 T (5 Stella Maris, FSM_001 Niguarda, FSM_049 Burlo Garofolo),
    data/raw/AINCP/dataset_FSM_v2. Foglio con centro, campo ed eta': docs/AINCP_SERIES_TYPE_dataset_FSM_v2.xlsx (non incluso nel repository: metadati dei pazienti);
  - T1 e FLAIR originali, ROI per fenotipo disegnate sulla FLAIR: ROI_1 gliosi, ROI_2 cistica,
    ROI_3 ventricolo (ESCLUSO dall'etichetta), ROI_4 encefalomalacia;
  - si usano SOLO i lotti consegnati da AInCP (la copia di FSM_028 in /data01/fsm_mri_data no).

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

SUI PAZIENTI (dettaglio e numeri in docs/modelli-ed-esperimenti.md, «Quarta parte»):
8. L'ESPANSIONE DEL PRIMO STRATO DA 1 A 2 CANALI FUNZIONA, ma il canale FLAIR nuovo impara solo con
   il passo differenziato (FLAIR 1e-2, resto 1e-4); a 1e-4 uniforme resta spento.
9. IL PRETRAINING PORTA LE CAVITA' E TOGLIE I FALSI POSITIVI; il Dice medio non distingue le ricette
   con dieci pazienti. Le misure che contano: copertura per fenotipo, lesioni rilevate, falsi positivi.
10. I PRETRAINING IN SEQUENZA DIMENTICANO (WMH dopo ATLAS: ATLAS da 0.610 a 0.217); INSIEME NO
    (Dataset543, ATLAS con FLAIR vuota + WMH, modality dropout).
11. SPAZIO DI LAVORO: template pediatrico MNIPediatricAsym (stessa griglia di ATLAS), ritorno al nativo
    per le metriche. Componenti delle annotazioni fino a 1 mm3 = residui, tolte.
12. IL POSTPROCESSING (soglia e volume minimo scelti sui fuori-fold) toglie 3/4 dei falsi positivi.

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
DA DOVE RIPARTIRE (aggiornato il 2 ottobre 2026)
============================================================
FATTO: il MODELLO FINALE su Dataset506_AINCP_FINALE (17 pazienti nel template; 12 di addestramento a
5 fold, 5 di test). 15 run: ricette A (da 502, passo differenziato), D (da 543 a 1000 epoche), Z (da
zero), x 5 fold, tutti fermati dall'arresto. Scelta prima del test, per regola: Z. TEST FATTO il 2/10
(una volta, non si ripete): reports/valutazione_test_Dataset506_*.md. Le tre ricette sono
indistinguibili in cross-validation e sul test; Z sul test: Dice 0.622, mediana 0.770, F1 0.605, con
un falso positivo di circa 9 000 mm³ nel ventricolo di FSM_046 che A e D non fanno. Tutto in
docs/modelli-ed-esperimenti.md, sezione 506. Relazione tecnica sul modello finale: artefatto MPQTHXQvdyoKf9JCE8NHak (vedi sotto).

PROSSIMI PASSI, in ordine:
1. Ablazione dei canali in inferenza sul modello scelto (FLAIR a zero, T1 a zero), per fenotipo.
2. Eventuale maschera dei ventricoli (SynthSeg) nel postprocessing: va scelta sui FUORI-FOLD dei 12,
   mai sul test, e riportata come analisi successiva al test.
3. Correzioni delle annotazioni dal clinico (reports/residui_annotazione.md); FSM_033 da far vedere.
4. Aggiornare l'artefatto principale con il modello finale; stadio 2 della tesi (inpainting).
FSM_030: si usa ROI_1.nii, non la copia (decisione dell'utente, 1/10).

ARTEFATTI ESISTENTI, da AGGIORNARE passando il loro url e non ricreare:
  IL PRINCIPALE, pesi di partenza e risultati sui pazienti (versione 7, 23/9; da aggiornare con il
  modello finale):
    https://claude.ai/artifact/BhCwc5KKbwoQ5cVBiN25hA
  relazione tecnica per i relatori:
    https://claude.ai/code/artifact/272e1e84-49f2-42a0-b3ca-31ad222adab9
  risposte ai chiarimenti del relatore:
    https://claude.ai/code/artifact/2ab3bbc3-27b4-41d6-8b11-0380446c3e86
  guida personale su cosa mostrare all'incontro:
    https://claude.ai/code/artifact/9e2b7510-9851-4142-b1e2-bc11aeb0ac3c
  relazione tecnica sul MODELLO FINALE (506, 2/10: architettura, addestramento, overfitting,
  postprocessing, cross-validation e test):
    https://claude.ai/artifact/MPQTHXQvdyoKf9JCE8NHak
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
