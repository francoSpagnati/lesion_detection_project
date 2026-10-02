# Segmentazione di lesioni cerebrali — il percorso in breve

Rilevamento automatico di lesioni in bambini con **paralisi cerebrale**, con nnU-Net v2.
Stato al 1 ottobre 2026. Questo è il documento di **comunicazione**: racconta il percorso a chi sta
fuori dal progetto. I numeri vengono dal registro [modelli-ed-esperimenti.md](modelli-ed-esperimenti.md),
dove ognuno ha il suo script e il suo report.

---

## Il problema e il vincolo che decide tutto

Il bersaglio sono le lesioni cerebrali dei bambini del progetto **AInCP / Stella Maris**, segmentate a
mano su due sequenze di risonanza: la **T1**, dove le cavità (tessuto perso, pieno di liquido) appaiono
scure, e la **FLAIR**, dove la gliosi (tessuto cicatriziale) appare chiara. I pazienti annotati sono
**17**. Con così pochi casi una rete non si addestra da zero: i dataset pubblici servono a costruire i
**pesi di partenza**, e il modello finale è un adattamento di quei pesi ai pazienti.

| | che cosa contiene | sequenze | casi | ruolo |
| --- | --- | --- | --- | --- |
| **ATLAS 2.0** | esiti di ictus, adulti | solo T1 | 655 | insegna le **cavità**; il più numeroso |
| **WMH** | iperintensità della sostanza bianca, adulti | FLAIR + T1 | 99 | insegna la **FLAIR** |
| **DBB** | sei tessuti in cervelli pediatrici malformati | solo T1 | 244 | provato e abbandonato (sotto) |
| **AInCP** | lesioni da paralisi cerebrale, 3–15 anni | FLAIR + T1 | **17** | **il bersaglio**: 10 Stella Maris a 3 T, 5 Stella Maris a 1.5 T, 2 da altri centri a 1.5 T |

## Fase 1 (agosto–settembre): che cosa insegnano i dati pubblici

- **Il T1 da solo non vede le lesioni piccole.** Su WMH la FLAIR vale 15 punti di Dice in più del T1
  (0.774 contro 0.626, meglio in 16 casi su 16), e la perdita sta quasi tutta sulle lesioni piccole.
  Su ATLAS i casi che il modello manca sono lesioni senza contrasto in T1. Quindi il modello finale
  **deve** usare anche la FLAIR — ma ATLAS, il dataset più grande, la FLAIR non ce l'ha.
- **Il preprocessing non è la leva.** Togliere i dati a 1.5 T, togliere il cranio (skull stripping),
  correggere il bias di campo: misurati, nessun effetto. nnU-Net di serie è già vicino al massimo.
- **Un modello unico su tutti i dataset non funziona** (531): la rete aveva imparato a riconoscere la
  sorgente — sui bambini predice tessuti, sugli adulti lesioni — invece di unire i due compiti.
- **Il pretraining vale molto, ma solo con il passo giusto** (541, simulazione con 10 casi): da zero
  0.451, adattando i pesi di ATLAS con passo 1e-4 0.685. Con il passo di default (0.01)
  l'adattamento distrugge ciò che i pesi sapevano.

## L'architettura scelta

**nnU-Net 3D** (`PlainConvUNet`): una U-Net a 6 livelli, da 32 a 320 filtri, circa 31 milioni di
parametri, deep supervision, loss Dice + entropia incrociata. L'architettura la sceglie nnU-Net dai
dati; il responsabile l'ha confermata all'incontro del 23 settembre. Sopra di essa, le scelte nostre:

1. **Due canali in ingresso, FLAIR e T1**, perché le due sequenze vedono fenotipi diversi.
2. **Espansione del primo strato da uno a due canali** (la «Proposta B» del relatore). I pesi di ATLAS
   hanno un solo canale; il primo strato della rete viene allargato: il filtro del T1 si copia, quello
   della FLAIR parte da zero, e tutto il resto della rete eredita i pesi così com'è. All'inizio la rete
   si comporta esattamente come il modello di ATLAS e impara la FLAIR adattandosi.
3. **Passo differenziato**: il filtro nuovo della FLAIR impara a passo pieno (0.01), tutto il resto,
   pre-addestrato, a passo piccolo (0.0001), per non dimenticare.
4. **Pretraining congiunto ATLAS + WMH**: una seconda strada per i pesi di partenza, in cui la rete
   impara le cavità da ATLAS e la FLAIR da WMH nello stesso addestramento.
5. **Modality dropout**: durante l'addestramento, in un esempio su tre una delle due sequenze viene
   spenta a caso. Il modello impara a funzionare anche con una sola sequenza, che è la richiesta del
   responsabile e il caso degli esami incompleti.
6. **Spazio standard**: ogni paziente viene registrato sul **template pediatrico MNI** (un cervello
   medio di bambini 4.5–18.5 anni), il modello lavora lì, e la predizione torna nello spazio del
   paziente con la trasformazione inversa. Indicazione del responsabile; misurato che non costa nulla.
7. **Arresto anticipato e postprocessing**: l'addestramento si ferma quando l'errore sui casi di
   validazione smette di scendere; alla fine si scelgono sui dati una soglia di probabilità e un
   volume minimo delle lesioni predette.

## Fase 2 (settembre): i primi dieci pazienti

- **Il Dice non distingue le varianti** con dieci pazienti: tutte fra 0.56 e 0.62, differenze dentro il
  rumore. Le misure che distinguono sono altre.
- **Il pretraining porta le cavità**: l'encefalomalacia è coperta per il 70–81% dai modelli
  pre-addestrati contro il 40% del modello da zero, in 4 pazienti su 4.
- **Il pretraining toglie i falsi positivi**: meno lesioni inventate del modello da zero in 10 pazienti
  su 10.
- **Pretraining in sequenza: dimentica.** Addestrare prima su ATLAS e poi su WMH faceva crollare la
  rete su ATLAS (Dice da 0.61 a 0.22): nnU-Net riaddestra da capo l'ultimo strato a ogni passaggio.
  Addestrando ATLAS e WMH **insieme** il problema sparisce (543: 0.606 su ATLAS, 0.723 su WMH).
- **Il postprocessing è la leva più forte trovata**: toglie tre quarti dei falsi positivi senza perdere
  lesioni (da 22 a 6 sui dieci pazienti).
- **La percentuale di cervello lesionato**, la grandezza clinica indicata dal responsabile, è stimata
  bene: correlazione per ranghi 0.90 con quella vera, scarto medio −0.12 punti.

## L'espansione da uno a due canali ha funzionato?

**Sì, con una condizione.** Il meccanismo funziona: la rete espansa accetta due canali, conserva ciò
che sapeva di ATLAS e impara la FLAIR. Ma **solo se il canale nuovo ha un passo abbastanza grande**:

| prova | risultato |
| --- | --- |
| filtro FLAIR a zero, passo piccolo ovunque | la FLAIR **resta spenta**: il suo filtro arriva al 5% della forza di quello del T1 |
| filtro FLAIR a zero, **passo differenziato** | la FLAIR impara: il filtro arriva al 65–272% di quello del T1, come in un modello nato a due canali |
| copertura della gliosi (la lesione visibile in FLAIR), FLAIR spenta contro attiva | migliora in 7 pazienti su 10 (p = 0.084: tendenza, non prova) |
| rete espansa nel pretraining congiunto, su WMH (compito di sola FLAIR) | **0.723**, contro 0.753 di una rete nata a due canali |

La prova definitiva sul modello finale sarà l'**ablazione in inferenza**: le stesse predizioni con la
FLAIR spenta e con il T1 spento, lette per fenotipo.

## Che cosa stiamo addestrando adesso (dal 1 ottobre)

Il **modello finale**, su `Dataset506`: 17 pazienti nel template, **12 per addestrare** (in 5 gruppi a
rotazione, ognuno a turno usato per la validazione) e **5 per il test**, scelti per centro di
acquisizione e messi da parte senza guardarli: tre Stella Maris a 3 T, uno a 1.5 T, uno di un altro
centro, con lesioni da 454 a 196 847 mm³. Tre ricette, che differiscono solo nei pesi di partenza:

| | parte da | perché |
| --- | --- | --- |
| **A** | ATLAS, primo strato espanso, passo differenziato | la ricetta scelta sui primi dieci |
| **D** | pretraining congiunto ATLAS + WMH | il pretraining più completo |
| **Z** | nessun pretraining | il riferimento: quanto vale il pretraining |

La ricetta con il miglior equilibrio fra lesioni trovate e lesioni inventate, misurato **solo sui 12**,
diventa il modello della tesi; poi si misura una volta sola sui 5 di test, con le metriche nello spazio
del paziente, per centro, per fenotipo e per dimensione.

## Che cosa resta

1. Scelta della ricetta e del postprocessing; test; ablazione dei canali.
2. Risultati e figure per la tesi.
3. Correzioni delle annotazioni da parte del clinico (46 piccole componenti spurie elencate). Per
   FSM_030, che aveva due ROI_1, si usa `ROI_1.nii`.
4. Il secondo stadio della tesi — l'inpainting per il «gemello sano» — dopo la segmentazione.

**Limiti da dichiarare**: 17 pazienti, 5 di test; i 3 pazienti di test a 3 T erano fra i dieci su cui
si è scelta la ricetta; le lesioni sotto 50 mm³ sono trovate solo in un caso su tre; la registrazione
sul template deforma localmente le lesioni vicine ai ventricoli (le misure si fanno comunque nello
spazio del paziente).
