# Segmentazione di lesioni cerebrali — panoramica del lavoro

Rilevamento automatico di lesioni in bambini con **paralisi cerebrale**, con nnU-Net v2.
Stato al 27 agosto 2026.

---

## Il vincolo che determina tutto

Il bersaglio finale è il dataset **AInCP / Stella Maris**: circa **10 pazienti** segmentati a mano, non
ancora consegnati. Con dieci casi non si addestra una rete da zero, quindi i dataset pubblici non servono
a produrre il modello finale ma a **costruire pesi di partenza**. Il pretraining non è un esperimento
accessorio: è il meccanismo centrale del progetto.

## I dati

| | Compito | Popolazione | Canale | Casi usabili | Perché è nel progetto |
| --- | --- | --- | --- | --- | --- |
| **WMH** | iperintensità sostanza bianca | adulti, 4 scanner | FLAIR (+T1) | 99 | lesioni della sostanza bianca; **unico con FLAIR** |
| **ATLAS 2.0** | esiti di ictus | adulti, 33 coorti | T1 | 655 | cavità e lesioni focali; il più numeroso |
| **DBB** | 6 tessuti (non lesioni) | **pediatrica**, malformati | T1 | 244 | malformazioni e anatomie alterate |
| **AInCP** | lesioni da paralisi cerebrale | pediatrica | T1 + FLAIR | ~10 | **il bersaglio**, atteso |

**Non sono tre versioni dello stesso problema.** Il foreground occupa **1 voxel su 1000** in WMH e ATLAS
(trovare anomalie rare) contro **1 su 5** in DBB (partizionare il cervello in tessuti). Tre ordini di
grandezza: un training congiunto a testa condivisa non è praticabile, perché lo stesso voxel di sostanza
bianca sana varrebbe `3` in DBB e `0` in ATLAS. **Si condivide il peso della rete, non l'etichetta.**

## Cosa è stato trovato analizzando i dati

Problemi tutti **silenziosi**: nessuno fa fallire un training, tutti producono un modello che impara la
cosa sbagliata.

- in **WMH** il valore `2` non è una lesione WMH ma **altra patologia** (infarti), in 34 casi su 99: nella
  challenge ufficiale è escluso dalla valutazione, e infatti lo escludiamo anche noi;
- in **ATLAS** un caso è codificato `0/0.01`: binarizzato in modo ingenuo, i suoi 2810 voxel di lesione
  sparirebbero e il paziente entrerebbe nel training come sano;
- in **DBB** 31 casi su 244 sono in spazio nativo e 12 orientati diversamente dagli altri: mescolarli
  significherebbe dare alla rete volumi specchiati fra loro.

## Preprocessing: leggero, e verificato

nnU-Net esegue internamente riorientamento, resampling e normalizzazione, e rende superfluo il padding
perché lavora a patch. Rifare questi passi a monte produrrebbe una **doppia interpolazione**, che degrada
proprio le lesioni piccole.

**La correzione di bias N4** era l'unico passo che avrebbe aggiunto qualcosa. È stata **misurata prima di
implementarla**: dentro il cervello il campo risulta piatto (rapporto 1.10–1.12 su tutti e tre i dataset),
quindi è stata scartata. **Nessuno skull stripping** per ora: nnU-Net non lo richiede, e BET assume una
testa di forma regolare ed è tarato sul T1 — i due presupposti che cadono con la FLAIR e con le anatomie
alterate. Se servirà: HD-BET o SynthStrip, addestrati anche su cervelli patologici.

## Impostazione del training

| | WMH | ATLAS | DBB |
| --- | --- | --- | --- |
| **risoluzione di destinazione** | `1.2 × 0.9766 × 3.0` mm | `1.0 × 1.0 × 1.0` mm | `1.0 × 1.0 × 1.0` mm |
| **configurazione** | 3D piena risoluzione | 3D piena risoluzione | 3D piena risoluzione |
| **patch** | `64 × 224 × 128` | `128 × 128 × 128` | `128 × 160 × 112` |
| **batch** | 2 | 2 | 2 |

Quattro scelte che vale la pena esplicitare:

- **la risoluzione non è imposta a 1 mm per tutti.** La deriva nnU-Net dalla mediana degli spacing di
  ciascun dataset, e l'abbiamo verificata contro i valori calcolati a mano sui dati. Su WMH le fette sono
  da **3 mm su tutti e 99 i casi**: portarle a 1 mm significherebbe inventare due fette su tre per
  interpolazione. nnU-Net applica invece il resampling separato lungo l'asse anisotropo;
- **modello 3D**, non 2D. La configurazione a bassa risoluzione è stata scartata dal planner su tutti e
  tre i dataset, quindi il confronto ancora aperto è 2D contro 3D ed è il prossimo passo;
- **patch locali, non immagine intera con padding.** In inferenza si usa una finestra scorrevole con
  sovrapposizione, che rende il padding a dimensione fissa superfluo. Su WMH la patch copre comunque
  **64 delle 79 fette**, l'81% del volume;
- **le patch sono bilanciate**: un terzo è forzato a contenere lesione. Con un bersaglio che occupa un
  voxel su mille, patch prese a caso sarebbero quasi tutte di solo tessuto sano e la rete imparerebbe a
  rispondere "niente".

Rete a 6 stadi (32 → 320 canali), **deep supervision** attiva, loss Dice + Cross Entropy, SGD con momentum
0.99, 1000 epoche. Ogni run è tracciato su **Weights & Biases** con la risoluzione, la patch, i canali e
la mappa delle label registrati nel config: due esperimenti sono confrontabili solo se quei valori
coincidono.

## Come valutiamo

- **test holdout tenuto fuori fin dall'inizio**, escluso anche dall'analisi che determina il piano, e da
  toccare una volta sola alla fine;
- **5 fold** di cross-validation, stratificati per **sito/coorte e per volume della lesione**: senza
  stratificare, il Dice cambierebbe a seconda di quali pazienti capitano in un fold;
- **split fissati una volta sola**: se cambiassero, i confronti fra esperimenti perderebbero validità;
- metriche lette **per sito** su WMH e **per classe** su DBB, mai solo in media.

## Primi risultati

Dice sui casi interi, primo fold di cross-validation.

| | Dice medio | mediana |
| --- | --- | --- |
| **WMH** (FLAIR) | **0.774** | 0.837 |
| **ATLAS** (T1) | **0.610** | 0.721 |
| **DBB** (T1, 6 tessuti) | **0.971** | 0.928 – 0.989 per classe |

DBB **non è confrontabile** con gli altri due: il compito è diverso e un Dice alto è il punto di partenza
atteso. Inoltre quel valore riguarda i 23 casi *sani*; la domanda per cui DBB è stato scelto — la tenuta
su anatomie mai viste — si misura sui 37 casi malformati del test, non ancora toccati.

Su entrambi i dataset lesionali **la mediana è più alta della media**: il modello lavora bene sulla
maggioranza dei casi, ed è una minoranza difficile a spostare il valore medio. Il fattore che domina è la
**dimensione della lesione**: su ATLAS i casi sotto i 100 voxel hanno Dice medio 0.28.

## Il risultato che orienta le scelte: quanto vale la FLAIR

Stesso dataset, stessi casi, stesse label, stessi fold, **stessa identica architettura**: cambia solo il
canale in ingresso.

| | Dice |
| --- | --- |
| WMH con **FLAIR** | **0.774** |
| WMH con **solo T1** | **0.626** |

La FLAIR vale **quasi 15 punti di Dice**, ed è migliore in **16 casi su 16** (test appaiato, p = 0.00003).
Ma il dato che conta è come si distribuisce la perdita:

| | perdita passando da FLAIR a T1 |
| --- | --- |
| lesioni **piccole** | **−0.242** |
| lesioni **grandi** | **−0.055** |

Il T1 da solo regge sulle lesioni estese e **crolla su quelle piccole**.

Lo stesso limite è emerso per una via indipendente esaminando i 10 casi di ATLAS che il modello non trova
affatto. Non è un errore di lateralità (verificato), non è un problema di soglia (le probabilità dentro
quelle lesioni sono praticamente nulle): quelle regioni **non hanno contrasto sul T1**. Rapporto fra
intensità della lesione e del tessuto circostante: **1.07 nei casi falliti contro 0.62 nei riusciti**,
dove il modello riesce quando la lesione è una cavità, quindi nerissima su T1.

## Cosa ne segue, e le domande aperte

**Un modello T1-only è strutturalmente limitato sulle lesioni piccole e non cavitate** — che nella paralisi
cerebrale sono una parte importante del bersaglio.

Da qui la questione da decidere: nnU-Net richiede lo **stesso numero di canali** per trasferire i pesi.
Il pretraining ampio può essere solo **T1-only**, perché è l'unica configurazione che copre tutti e quattro
i dataset; ma se AInCP avrà la FLAIR, un modello a due canali **non potrà ereditare quei pesi**. O si
rinuncia alla FLAIR per sfruttare il pretraining, o si rinuncia al pretraining per usare la FLAIR.

Restano aperte anche:

- **skull stripping**: introdurlo con HD-BET o SynthStrip, sapendo che nnU-Net non lo richiede?
- **risoluzione comune fra dataset**: per trasferire pesi serve la stessa griglia, ma portare WMH a 1 mm
  isotropo significa interpolare fette da 3 mm;
- **quando arriva AInCP**, e se le annotazioni copriranno anche la FLAIR oltre al T1.

## Prossimi passi

1. confronto 2D contro 3D, deciso in cross-validation;
2. **transfer DBB → ATLAS**: le label dense dei sei tessuti insegnano all'encoder il contrasto tissutale e
   l'anatomia, che il compito lesionale riusa. È la prima verifica vera che il pretraining funzioni;
3. all'arrivo di AInCP, fine-tuning dai pesi migliori con metriche riportate per singolo paziente;
4. seconda fase: inpainting e "gemello sano virtuale", da affrontare dopo, perché senza una maschera di
   lesione affidabile non c'è nulla da inpaintare.
