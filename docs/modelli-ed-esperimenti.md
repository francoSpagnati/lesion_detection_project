# Modelli ed esperimenti

Questo è il **registro unico** dei modelli addestrati: per ciascuno, quale dataset ha visto, com'è fatta
la rete, con che ricetta è stata addestrata, cosa è uscito e cosa ne abbiamo concluso. È il file da
aprire per rispondere a "che modelli abbiamo e cosa ci hanno detto".

Gli altri documenti fanno altro e non ripetono questo:
[training.md](training.md) è la **procedura** (come si lancia, dove si legge il progresso, quali comandi),
[cosa-e-stato-fatto.md](cosa-e-stato-fatto.md) è la **cronologia** di tutto il lavoro (anche non di
training), [dataset-e-decisioni.md](dataset-e-decisioni.md) descrive **i dati** e le scelte fatte su di
essi, [obiettivo-e-piano.md](obiettivo-e-piano.md) dice **cosa viene dopo**.

Tutti i numeri qui sono letti dai file prodotti dai run: `nnUNetPlans.json` per le architetture,
`training_log_*.txt` per i tempi, `validation/summary.json` per il Dice. Nessuno è stimato.

---

## Prima parte: cosa è identico in tutti i training

Questa sezione si legge **una volta sola**. Ogni esperimento più sotto elenca solo ciò che cambia
rispetto a qui: è il modo di leggere il registro senza ripetere trenta righe di iperparametri otto volte,
ed è anche la ragione per cui i confronti hanno senso.

### Il software

nnU-Net **v2.8.1** in `~/.venv`, PyTorch **2.6.0+cu124**, su A100 80GB. Il trainer è
`nnUNetTrainerWandb`, che vive in `scripts/nnunet_trainers/`: **non cambia nulla della logica di
training** rispetto al `nnUNetTrainer` di serie — stessa rete, stessa loss, stesso scheduler — e si
limita ad accendere il logger W&B integrato dando al run un nome sensato e registrando target spacing,
patch, canali e mappa delle label. Un run con questo trainer e uno con il default restano confrontabili.

### La rete

Tutti i modelli sono la stessa architettura, perché **è nnU-Net a sceglierla** a partire dal
fingerprint dei dati: non è una scelta nostra, ed è esattamente il motivo per cui usiamo nnU-Net come
baseline seria invece di progettare una rete a mano.

- `PlainConvUNet`: una U-Net convoluzionale classica, encoder–decoder con skip connection.
- **6 stadi**. Canali per stadio in 3D: `32, 64, 128, 256, 320, 320` (il tetto a 320 è di nnU-Net, serve
  a non far esplodere la memoria); in 2D si arriva a `512`.
- **2 blocchi convoluzionali per stadio**, sia in discesa che in risalita. Ogni blocco è
  convoluzione → `InstanceNorm` → `LeakyReLU`. La normalizzazione è *per istanza*, non batch norm:
  con batch da 2 una batch norm sarebbe rumore.
- **Deep supervision**: la loss non guarda solo l'uscita a piena risoluzione ma anche quelle
  intermedie del decoder, con peso che dimezza a ogni livello. Fa arrivare gradiente utile agli strati
  profondi fin dalle prime epoche.
- **Circa 31 milioni di parametri** (30.7–31.2 M a seconda del numero di canali e di classi).

Kernel e strides cambiano da dataset a dataset e sono riportati caso per caso: è lì che si vede la
differenza fra un volume isotropo a 1 mm e uno con fette da 3 mm.

### La ricetta di training

Sono i default di nnU-Net, non toccati. Elencarli serve perché finiscono in tesi e perché un confronto
è valido solo se sono gli stessi ovunque.

| | |
| --- | --- |
| ottimizzatore | SGD, momentum **0.99** con Nesterov, weight decay `3e-5` |
| learning rate | **0.01** iniziale, decadimento *poly* con esponente 0.9 fino a 0 |
| epoche | **1000**, ciascuna di **250 iterazioni** (nnU-Net definisce l'epoca così, non come un passaggio sui dati) |
| loss | **Dice + Cross Entropy** con peso 1 e 1; il Dice è calcolato per campione, non sul batch (`batch_dice=False`), e ignora il background |
| campionamento | **33% delle patch forzate a contenere foreground**: senza, su lesioni che occupano lo 0.1% del volume la rete vedrebbe quasi solo sfondo |
| augmentation | rotazioni, scaling, rumore gaussiano, sfocatura, luminosità, contrasto, simulazione di bassa risoluzione, gamma, mirroring |
| validation durante il training | 50 iterazioni per epoca, che producono la "pseudo dice" dei log: è una stima su patch, **non** il numero finale |
| validation finale | sliding window sui casi interi del fold, ed è **questo** il Dice riportato qui |
| checkpoint | `checkpoint_latest.pth` ogni 50 epoche, `checkpoint_best.pth` sulla pseudo dice, `checkpoint_final.pth` alla fine |

### Il protocollo sperimentale

Tre regole che rendono confrontabili i numeri di questo registro, e che sono state rispettate in tutti i
run elencati:

1. **Un fold solo, il fold 0.** Cinque fold per esperimento sarebbero settimane di GPU. Il fold 0 è
   sufficiente per *ordinare* le varianti; i cinque fold si faranno alla fine, sulla configurazione
   scelta.
2. **`splits_final.json` non si rigenera mai.** È scritto da noi e nnU-Net lo riusa. Se cambiasse, i
   confronti fra run precedenti e successive perderebbero validità.
3. **Si confrontano i casi, non le medie.** Due dataset derivati non hanno necessariamente la stessa
   validation: una differenza fra medie complessive può venire dalla composizione del campione invece che
   dal cambiamento. `scripts/confronto_appaiato.py` interseca i casi e applica un Wilcoxon appaiato,
   che non assume normalità — e sul Dice di lesioni piccole la normalità non vale.

E una regola sui dati: **il test set non è stato ancora guardato**, in nessun dataset. Tutti i numeri qui
sono di validation.

---

## Seconda parte: i modelli, uno per uno

### Quadro d'insieme

| Modello | Dataset visto | Canale | Classi | Sviluppo | Val fold 0 | Dice | Stato |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **501** | WMH | FLAIR | lesione | 79 | 16 | **0.7739** | baseline |
| **502** | ATLAS | T1 | lesione | 556 | 112 | **0.6103** | baseline |
| **503** | dHCP/DBB | T1 | 6 tessuti | 207 | 23 | **0.9712** | baseline |
| **511** | WMH | T1 | lesione | 79 | 16 | **0.6257** | ablazione: quanto vale la FLAIR |
| **512** | WMH | FLAIR + T1 | lesione | 79 | 16 | **0.7535** | ablazione: due canali aiutano? |
| **521** | ATLAS senza 1.5 T | T1 | lesione | 511 | 103 | **0.6160** | ablazione: escludere i 1.5 T? |
| **522** | WMH skull-stripped | FLAIR | lesione | 79 | 16 | *in corso* | ablazione: skull stripping |
| **523** | ATLAS skull-stripped | T1 | lesione | 556 | 112 | *in coda* | ablazione: skull stripping |

Tutti `3d_fullres`, fold 0, 1000 epoche. `599` è un dataset usa e getta da 16 casi per gli smoke test da
5 epoche: verifica che la catena giri, e il suo Dice (0.37) non significa nulla.

---

### 501 — WMH su FLAIR: la baseline migliore che abbiamo

**Perché.** È il dataset più facile e il più piccolo: 79 casi, un solo canale, un bersaglio che sulla
FLAIR è iperintenso e quindi visibile. Serve come **limite superiore pratico**: dice quanto rende questa
pipeline quando le condizioni sono buone.

**Cosa vede.** Lesioni della sostanza bianca (iperintensità di origine vascolare) su immagini FLAIR di
quattro siti — Amsterdam GE 1.5 T, Amsterdam GE 3 T, Singapore, Utrecht.

**Il piano.** Spacing target `3.0 × 0.9766 × 1.2` mm: nnU-Net **non** porta il dataset a isotropo, tiene
le fette spesse. Patch `64 × 224 × 128`, batch 2, 31.1 M parametri. Il primo stadio ha kernel `1×3×3` e
stride `1×2×2`: la rete lavora *nel piano* prima di scendere lungo l'asse a 3 mm, che è il modo giusto di
trattare un volume anisotropo. La patch copre 64 fette su 79, l'81% del volume.

**Risultato: Dice 0.7739** su 16 casi (mediana 0.837, Q1–Q3 0.705–0.880). 5h42 di training, 23.2 s/epoca.

**Cosa dice.** La mediana è 6 punti sopra la media: pochi casi difficili tirano giù il numero, sulla
maggioranza il modello lavora bene. Il volume del bersaglio domina tutto — metà dei casi con lesioni più
piccole sta a 0.667, l'altra metà a 0.881. Il Dice per sito (Singapore 0.818, Amsterdam GE3T 0.790,
Amsterdam GE1T5 0.750 su un caso solo, Utrecht 0.715) con 16 casi non è ancora interpretabile.

---

### 502 — ATLAS su T1: la baseline che conta davvero

**Perché.** È il dataset più vicino al problema della tesi: lesioni focali su T1, che è la sequenza che
avremo quasi certamente sui pazienti AInCP. Ed è il più grande, 556 casi di sviluppo.

**Cosa vede.** Lesioni da ictus post-acuto su T1 pesato, 3 T e 1.5 T, molte coorti diverse.

**Il piano.** Spacing `1 × 1 × 1` mm, patch **`128 × 128 × 128`** cubica, batch 2, 31.2 M parametri.
Kernel `3×3×3` e stride `2×2×2` ovunque: dato isotropo, rete simmetrica.

**Risultato: Dice 0.6103** su 112 casi (mediana 0.721, Q1–Q3 0.465–0.839). 6h43, 23.1 s/epoca.

**Cosa dice.** Sedici punti sotto WMH, ma buona parte del divario **non** è qualità del modello: è
composizione del campione. Metà con lesioni più piccole sta a 0.509 contro 0.712; i casi sotto 100 voxel
stanno a 0.28; la sola coorte `r009` (19 casi su 112, lesioni mediane da 1398 voxel contro 7053 delle
altre) sta a 0.337 contro 0.666.

Otto casi su 112 hanno **Dice 0**, e non sono solo i minuscoli: uno ha 9586 voxel di riferimento e 217
predetti. Esaminandoli uno per uno (`scripts/inspect_failures.py`) è emerso che non sono errori di bordo
né di soglia — le probabilità dentro quelle lesioni sono praticamente nulle — ma **mancanza di contrasto
sul T1**: rapporto fra intensità della lesione e del tessuto circostante 1.07 nei falliti contro 0.62 nei
riusciti. Il modello riesce quando la lesione è una cavità, quindi nerissima. **Non è un difetto della
pipeline, è un limite dell'informazione contenuta nel T1**, ed è il risultato singolo che condiziona di
più il resto del progetto.

---

### 503 — dHCP/DBB, sei tessuti: un compito diverso

**Perché.** Non è lesion detection: è la partizione del cervello in sei tessuti su neonati. Serve a due
cose — avere un dataset pediatrico con anatomie patologiche, e produrre pesi da trasferire, perché label
dense di tessuto insegnano all'encoder il contrasto tissutale del T1 e l'anatomia.

**Cosa vede.** 207 casi di sviluppo, T1 neonatale, sei classi: CSF, sostanza grigia, sostanza bianca,
grigia profonda, tronco encefalico, cervelletto.

**Il piano.** Spacing `1 × 1 × 1` mm, patch `128 × 160 × 112`, batch 2, 30.8 M parametri. L'ultimo stadio
ha stride `2×2×1`, cioè smette di ridurre l'asse più corto. 8h03, 27.4 s/epoca — il più lento, perché sei
classi e foreground denso.

**Risultato: Dice 0.9712.** Per classe: sostanza bianca 0.9892, grigia 0.9845, cervelletto 0.9806, tronco
encefalico 0.9732, grigia profonda 0.9718, **CSF 0.9278**.

**Cosa dice, e come non va letto.** Il numero **non è confrontabile** con quelli di lesione: qui il
foreground è un voxel su cinque invece di uno su mille, e il Dice è alto per costruzione. Il CSF è la
classe più difficile perché ha molta superficie per poco volume, quindi il Dice la penalizza; le classi
piccole ma compatte (grigia profonda, tronco) tengono bene. Soprattutto: **questa validation è sui 23
casi sani**. I 37 casi malformati stanno nel test e non sono stati toccati, ed è su quelli che il
benchmark ha senso.

---

### 511 — WMH sul solo T1: quanto vale la FLAIR

**Perché.** Domanda decisiva per il progetto, non curiosità. Per unire tutti i dataset in un modello solo
serve un canale comune, e l'unico che tutti hanno è il T1: WMH dovrebbe quindi entrare *rinunciando* alla
sua FLAIR. Questo esperimento **misura il prezzo** di quella rinuncia.

**Cosa cambia rispetto a 501.** Solo il canale: stessi 79 casi, stesse label, stessi fold. Prima di
lanciare è stato verificato che il piano fosse **identico** a quello di 501 — stessa spacing, stessa
patch `64×224×128`, stessi kernel e strides — e che gli `splits_final.json` fossero bit-identici. L'unica
voce diversa è `median_image_size_in_voxels`, statistica del fingerprint che non entra nella rete.

**Risultato: Dice 0.6257** contro 0.7739. 6h37.

**Cosa dice. La FLAIR vale 14.8 punti di Dice ed è migliore in 16 casi su 16** (Wilcoxon appaiato,
p = 0.00003). Ma il dettaglio conta più della media: la perdita è **−0.242 sulle lesioni sotto la
mediana** e **−0.055 su quelle sopra**. Il T1 da solo regge sulle lesioni estese e crolla su quelle
piccole — che è, per una via completamente indipendente, la stessa conclusione dei casi falliti di ATLAS.

---

### 512 — WMH con FLAIR e T1 insieme: no, non aiuta

**Perché.** L'ultima scappatoia possibile prima di rassegnarsi al T1: se due canali funzionassero, si
potrebbe pensare a un modello a due canali che usa la FLAIR dove c'è. Costruito da
`scripts/make_wmh_two_channel_dataset.py`, che **non si fida** della co-registrazione dichiarata dal
dataset e verifica caso per caso che forma e affine coincidano.

**Cosa cambia rispetto a 501.** Solo il canale in più. Il confronto è pulito: 512 eredita da 501 la
stessa patch `64×224×128` e lo stesso batch 2, quindi l'unica variabile è quella.

**Risultato: Dice 0.7535** contro 0.7739. Differenza appaiata **−0.0205**, peggiora in 11 casi su 16,
Wilcoxon **p = 0.025**. Il caso peggiore è `Utrecht_33`, da 0.697 a 0.555. 6h52.

**Cosa dice.** Il verso non sorprende, l'ampiezza sì: ci si poteva aspettare che la rete imparasse a
ignorare il canale meno informativo, e invece **tenerlo costa due punti**. Due spiegazioni plausibili che
questo esperimento non separa — con 79 casi di training la capacità spesa sul secondo canale è capacità
tolta al primo, e normalizzazione e augmentation ora lavorano su due distribuzioni invece che una.

**Conseguenza:** nel modello unificato WMH entra **con il solo T1**, e il prezzo da mettere in conto è
quello misurato da 511 (0.626 contro 0.774), non un ibrido.

---

### 521 — ATLAS senza i dati a 1.5 T: escluderli non serve

**Perché.** I dati a 1.5 T hanno meno rapporto segnale/rumore, e una tentazione ricorrente è buttarli via
per "pulire" il training. Su ATLAS sono 45 casi di sviluppo su 556 (coorti `r029`, `r039`, `r040`): non
pochi. La domanda va decisa con un numero, non con un'intuizione.

**Cosa cambia rispetto a 502.** 511 casi di sviluppo invece di 556. I fold sono **filtrati** da quelli di
502, mai rigenerati: chi resta, resta nello stesso fold. Piano identico a 502 (`128³`, batch 2).

**Risultato: Dice 0.6160.** Ma il confronto che conta è appaiato sui 103 casi a 3 T presenti in entrambe
le validation: **0.6160 contro 0.6066, +0.0093**, migliora in 48 casi e peggiora in 51, Wilcoxon
**p = 0.6677**. 7h13.

**Cosa dice.** Nessun guadagno. E c'è di più, dalla direzione opposta: applicando il modello 521 ai 45
casi a 1.5 T che **non ha mai visto** (`scripts/inferenza_incrociata_15T.py`, verificando prima che
nessuno di quei casi fosse finito nel suo training) si ottiene **0.6721** — meglio dello 0.6160 che
ottiene sui suoi stessi casi a 3 T. Nessun crollo fuori dominio: quelle lesioni sono semplicemente più
grandi e più facili.

Sui 9 casi mai visti da **nessuno** dei due modelli il confronto appaiato dà −0.0723 con p = 0.25, che
sembrerebbe un costo, ma non va letto così: nove casi sono pochissimi, vengono **tutti dalla coorte
`r040`** (quindi si sta misurando uno scanner, non "il 1.5 T"), e due terzi del divario vengono da un
caso solo — togliendo `sub-r040s059` resta −0.0270.

**Conseguenza: i casi a 1.5 T restano dentro.** Escluderli non fa guadagnare nulla e costerebbe 45 casi
annotati.

---

### 522 e 523 — skull stripping: in corso

**Perché.** Rimuovere cranio, cuoio capelluto e collo prima di dare l'immagine alla rete. Ci sono due
motivi concreti per aspettarsi un guadagno, ed è il secondo quello interessante:

1. la rete smette di spendere capacità su tessuto che non contiene mai il bersaglio;
2. **la normalizzazione cambia.** nnU-Net attiva `use_mask_for_norm` quando il ritaglio al non-zero
   scende sotto i 3/4 del volume originale. Su 501/502/503 era `False` (teste intere); su 522 e 523 è
   diventata **`True`**. Vuol dire che lo z-score viene calcolato **sui soli voxel di cervello** invece
   che su una distribuzione che comprende grasso sottocutaneo, diploe e aria.

**Lo strumento.** SynthStrip di FreeSurfer, validato prima dell'uso sulle 244 maschere cerebrali di
riferimento di DBB: Dice medio **0.9657**, e soprattutto **0.9605 sui cervelli malformati** contro 0.9673
sui sani — un calo di un punto, che è il motivo per cui ci si può fidare anche sui pazienti con paralisi
cerebrale. Il caso peggiore è l'idrocefalo (0.9267), dove il volume escluso è quasi tutto liquido nei
ventricoli dilatati; è una cautela da ricordare, perché la dilatazione ventricolare è comune nella
paralisi cerebrale.

**Il rischio misurato prima di procedere.** Ritagliare può tagliare via lesione. Misurato sull'**intero**
dataset (556 casi ATLAS + 79 WMH, non su un campione): WMH perde **0.0000**, ATLAS perde in media
**0.0012**, con 18 casi sopra l'1% e **3 sopra il 5%**. I casi peggiori sono infarti massivi (196–360 ml)
che arrivano alla superficie corticale. Accettabile, e nessuna dilatazione della maschera: 2 mm
dimezzerebbero la perdita ma riaggiungerebbero circa 160 ml di guscio non cerebrale per caso.

**Cosa cambia nel piano — e perché il confronto non sarà pulito.** Riducendo i volumi, nnU-Net ha
**ricalcolato patch e batch**:

| | 501 → 522 | 502 → 523 |
| --- | --- | --- |
| shape mediana | `79×240×132` → `48×173×121` | `189×233×197` → `161×185×148` |
| patch | `64×224×128` → **`48×192×128`** | `128×128×128` → **`128×160×112`** |
| batch | 2 → **3** | 2 → 2 |
| `use_mask_for_norm` | False → **True** | False → **True** |

Su 522 la patch ora copre il **100%** delle fette invece dell'81%. Questi due confronti misurano quindi
"la pipeline nnU-Net con e senza skull stripping" — che è la domanda pratica giusta — ma **non una sola
variabile**, e questo va detto ogni volta che si riporta il numero.

**Stato.** 522 avviato il 30 agosto alle 12:34 sulla GPU 0, 523 in coda dietro; circa 7 ore ciascuno.
DBB è deliberatamente escluso da questo esperimento: l'unione delle sue 6 classi **è** la maschera
cerebrale (Dice 0.9999), quindi strippare significherebbe regalare alla rete l'estensione esatta del
foreground.

---

## Terza parte: cosa abbiamo imparato, in cinque righe

1. **Il T1 non basta per le lesioni piccole.** Trovato due volte per vie indipendenti: l'ablazione FLAIR
   (−0.242 sulle piccole contro −0.055 sulle grandi) e i casi falliti di ATLAS (contrasto 1.07 contro
   0.62). È il vincolo strutturale del progetto, perché nella paralisi cerebrale le lesioni piccole e non
   cavitate sono una parte importante del bersaglio.
2. **Non si aggira affiancando i canali.** FLAIR+T1 rende meno della sola FLAIR.
3. **Non si buttano via dati per pulizia.** Escludere i 1.5 T non ha dato nulla, e il modello che non li
   ha mai visti li segmenta comunque bene.
4. **Il Dice medio, da solo, mente.** Su questi compiti dipende quasi solo dal volume del bersaglio: metà
   della differenza fra WMH e ATLAS è composizione del campione. Ogni numero va letto con la
   distribuzione accanto.
5. **Confrontare medie di validation diverse è un errore facile.** Da qui la scelta di fare tutti i
   confronti appaiati, caso per caso.

## Quarta parte: cosa non è ancora stato addestrato

| Modello | Domanda | Stato |
| --- | --- | --- |
| **D** — un solo modello sui tre dataset a 1 mm, 7 classi | tutti i dati insieme reggono? | **prossimo da costruire** |
| **D skull-stripped** | e con lo skull stripping? | dopo D, per confronto diretto |
| varianti A (WMH+ATLAS), B (A+AInCP), C (tutti con `ignore`) | alternative se D non regge | pianificate |
| transfer DBB → ATLAS | i tessuti insegnano l'anatomia al task lesionale? | pianificato |
| `2d` contro `3d_fullres` | quale configurazione, decisa in cross-validation | pianificato |
| i 5 fold della configurazione scelta | il numero finale | alla fine |
| fine-tuning su AInCP | l'obiettivo vero | in attesa dei dati |

Il dettaglio del piano, con le motivazioni e l'ordine, è in [obiettivo-e-piano.md](obiettivo-e-piano.md).
