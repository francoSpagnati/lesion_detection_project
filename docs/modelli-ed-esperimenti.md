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
| **522** | WMH skull-stripped | FLAIR | lesione | 79 | 16 | **0.7785** | ablazione: skull stripping, +0.0045 con p=0.86 |
| **523** | ATLAS skull-stripped | T1 | lesione | 556 | 112 | **0.6067** | ablazione: skull stripping, -0.0037 con p=0.58 |
| **531** | **tutti e tre**, a 1 mm | T1 | lesione + 6 tessuti | 842 | 151 | **0.603** lesione | variante D: perde poco su tutto, non guadagna nulla |
| **541** | 45 casi ATLAS a 1.5 T | T1 | lesione | 10 | 35 | **0.685** al meglio, **0.451** da zero | **il test del pretraining**: vale +0.23, ma solo con il passo di apprendimento giusto |

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

### 522 e 523 — skull stripping: nessun effetto

**Perché.** Rimuovere cranio, cuoio capelluto e collo prima di dare l'immagine alla rete. Ci sono due
motivi concreti per aspettarsi un guadagno, ed è il secondo quello interessante:

1. la rete smette di spendere capacità su tessuto che non contiene mai il bersaglio;
2. **la normalizzazione cambia.** nnU-Net attiva `use_mask_for_norm` quando il ritaglio al non-zero
   scende sotto i 3/4 del volume originale. Su 501/502/503 era `False` (teste intere); su 522 e 523 è
   diventata **`True`**. Vuol dire che lo z-score viene calcolato **sui soli voxel di cervello** invece
   che su una distribuzione che comprende grasso sottocutaneo, diploe e aria.

**Lo strumento.** SynthStrip di FreeSurfer, validato prima dell'uso sulle 244 maschere cerebrali di
riferimento di DBB: Dice medio **0.9657**, e soprattutto **0.9569 sull'insieme dei cervelli
patologici** contro 0.9673
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

**Risultato: nessun effetto, in nessuna delle due direzioni.**

| | senza stripping | con stripping | differenza appaiata | migliora / peggiora | Wilcoxon |
| --- | --- | --- | --- | --- | --- |
| WMH (`501` → `522`) | 0.7739 | **0.7785** | +0.0045 | 7 / 9 | p = 0.86 |
| ATLAS (`502` → `523`) | 0.6103 | **0.6067** | −0.0037 | 59 / 46 | p = 0.58 |

**Cosa dice, e un dettaglio che vale la pena guardare.** Su ATLAS *più casi migliorano che peggiorano*
(59 contro 46), eppure la media scende: il segno è deciso da una manciata di casi che **crollano**
(`sub-r050s005` da 0.603 a **0.000**, `sub-r019s010` da 0.779 a 0.367). Verificato subito il sospetto
ovvio — che la maschera avesse tagliato via la lesione — e **non è così**: quei casi perdono fra lo 0% e
l'1.7% dei voxel di lesione, mentre i tre casi che ne perdono davvero (fino al 6.5%) non sono fra i
peggiorati. Il crollo viene dal modello, non dai dati: lo stripping cambia anche la normalizzazione e la
patch, e su qualche caso quel cambiamento sposta la predizione oltre la soglia di localizzazione.

Il guadagno atteso dalla normalizzazione sui soli voxel di cervello, quindi, **non si è materializzato**;
e va ricordato che questi due confronti non isolavano una sola variabile (patch e batch sono cambiati
con essa), quindi il risultato è "questa pipeline non migliora con lo skull stripping", non "lo skull
stripping è inutile in assoluto".

**Conseguenza.** Nessun motivo di adottarlo per default: costa un passo di preprocessing e introduce una
modalità di fallimento nuova, senza comprare accuratezza. Resta però lo strumento che rende possibile la
variante C, dove la maschera cerebrale serve a marcare `ignore`, e verrà comunque provato sull'unione
(D skull-stripped) perché lì la domanda è diversa: con sette classi la normalizzazione conta di più.

DBB è deliberatamente escluso da questo esperimento: l'unione delle sue 6 classi **è** la maschera
cerebrale (Dice 0.9999), quindi strippare significherebbe regalare alla rete l'estensione esatta del
foreground.

### 531 — variante D: tutti e tre i dataset in un modello solo

**Perché.** È la richiesta centrale del progetto: **mettere tutti i dati insieme** e addestrare un
modello unico, su una griglia comune, così che i pesi valgano per tutti e possano poi essere trasferiti
sui pazienti AInCP. È la strada verso il dataset privato da ~10 casi, dove addestrare da zero non è
un'opzione.

**Cosa vede.** 842 casi di sviluppo: 79 da WMH (**con il T1**, non la FLAIR), 556 da ATLAS, 207 da DBB.
Un canale, sette classi: `1` la lesione da WMH e ATLAS, `2..7` i sei tessuti di DBB. Costruita da
`scripts/make_union_dataset.py`.

**Come è costruita, e perché così.**

- Le immagini e le label di lesione **non sono copiate ma collegate** ai dataset di origine: sono
  identiche, e un symlink rende impossibile che una copia divergga. Si riscrivono soltanto le 244 label
  di DBB, rimappate da `1..6` a `2..7` perché il valore `1` è ora occupato dalla lesione.
- I nomi dei casi sono prefissati `wmh_`, `atlas_`, `dbb_`. Serve a evitare collisioni silenziose
  (ATLAS e DBB usano entrambi nomi che iniziano per `sub-`) e a **rendere leggibile la provenienza nel
  `summary.json`**, che è l'unico modo di riportare poi il Dice separatamente per sorgente.
- I fold sono l'**unione dei fold k dei tre sorgenti**, non rigenerati. Un caso che era in validation
  nel fold 0 di ATLAS resta in validation nel fold 0 di D, quindi il confronto con le baseline è
  appaiato caso per caso. Fold 0: 691 train, 151 validation (444+184+63 e 112+23+16).

**Il piano.** Ed è una buona notizia: il planner di default sceglie da solo spacing `1 × 1 × 1` mm e
patch `128 × 128 × 128` con batch 2, cioè **esattamente il piano di 502**. Non è servito forzare la
risoluzione con `-overwrite_target_spacing`: con 763 casi su 842 già isotropi a 1 mm, la mediana ci
arriva da sola, e WMH viene interpolato dalle sue fette da 3 mm come previsto. Il confronto fra D e la
baseline ATLAS misura quindi l'unione dei dati e non un cambio di architettura. `3d_lowres` è stata
scartata dal planner, come sugli altri dataset. 19 GB di preprocessing.

**Il conflitto, ora misurato.** D contiene di proposito una contraddizione: il tessuto cerebrale sano
vale `0` in WMH e ATLAS e `2..7` in DBB, quindi la rete riceve segnali opposti sullo stesso ingresso.
Quanto pesi non era un numero ma un'intuizione; ora lo è (`scripts/bilancio_classi_union.py`):

| sorgente | casi | foreground mediano per caso | quota del volume |
| --- | --- | --- | --- |
| WMH | 79 | 9 384 voxel | 0.21% |
| ATLAS | 556 | 5 232 voxel | 0.38% |
| **DBB** | 207 | **1 483 524 voxel** | **22.06%** |

**Un caso DBB porta 266 volte il foreground di un caso lesionale**, e la lesione è solo il **5.87%** di
tutto il foreground annotato (contro il 38.6% della sostanza grigia e il 26.3% della bianca). nnU-Net
forza il 33% delle patch a contenere foreground ma **non bilancia fra le classi**: quelle patch
finiranno quasi tutte su tessuto DBB.

**Come andrà letto.** Non con un Dice medio unico, che qui sarebbe privo di significato: sette classi di
cui sei facili e densissime e una difficile e rarissima produrrebbero un numero alto e vuoto. Il
risultato va riportato **per classe** e **per sorgente**, e la domanda vera è una sola: *il Dice della
classe 1 sui casi ATLAS regge il confronto con lo 0.6103 di 502, sugli stessi casi di validation?* Se
perde, avremo quantificato il costo del conflitto invece di ipotizzarlo — che è un risultato
riportabile, ed è esattamente ciò che motiva la variante C, dove quei voxel diventano `ignore`.

**Risultato.** Concluso il 31 agosto alle 23:40, 7h06. nnU-Net stampa un Dice medio di **0.8966**, che
è però la media sulle sette classi ed è dominato dai sei tessuti densi: non descrive il compito. Il
numero va letto per classe e per sorgente (`scripts/analisi_union_d.py`, `reports/risultato_531.md`),
misurando ogni classe **sui soli casi che possono contenerla** e confrontandola con la propria baseline
sugli stessi casi.

| confronto | classe | casi | baseline | D | differenza | p |
| --- | --- | --- | --- | --- | --- | --- |
| contro `502` | lesione su ATLAS | 112 | 0.6103 | **0.6031** | −0.0072 | 0.073 |
| contro `511` | lesione su WMH | 16 | 0.6257 | **0.6032** | −0.0225 | 0.10 |
| contro `503` | csf | 23 | 0.9278 | 0.9034 | −0.0245 | 2e-07 |
| contro `503` | grigia | 23 | 0.9845 | 0.9741 | −0.0104 | 2e-07 |
| contro `503` | bianca | 23 | 0.9892 | 0.9811 | −0.0080 | 2e-07 |
| contro `503` | grigia profonda | 23 | 0.9718 | 0.9514 | −0.0205 | 2e-07 |
| contro `503` | tronco | 23 | 0.9732 | 0.9639 | −0.0093 | 8e-06 |
| contro `503` | cervelletto | 23 | 0.9806 | 0.9753 | −0.0054 | 2e-07 |

Il confronto per WMH è con **511 e non con 501**: dentro D, WMH entra con il solo T1, perché ATLAS e DBB
la FLAIR non ce l'hanno.

**Cosa dice, e non è quello che ci si aspettava.** Il conflitto fra le etichette — tessuto sano che vale
`0` in 635 casi su 842 e `2..7` negli altri 207 — **non ha fatto crollare niente**. Sulla lesione D
perde 0.007 su ATLAS (p=0.073) e 0.022 su WMH (p=0.10): in nessuno dei due casi la differenza è
significativa al 5%.

Ma non guadagna neppure, e il costo c'è: **perde su tutte e sette le classi**. Sui tessuti la perdita è
piccola (0.5–2.5 punti) e insieme *perfettamente sistematica* — in cinque classi su sei peggiorano
**23 casi su 23**, con p dell'ordine di 1e-07. È la quantificazione precisa di quanto costa il
conflitto: non un crollo, un pedaggio piccolo e pagato ovunque.

**Come leggerlo per il progetto.** D come *modello finale* è leggermente peggiore degli specialisti, e
non c'è ragione di preferirlo. Ma D non è nato per essere il modello finale: è un candidato **punto di
partenza**, e "peggiore come modello" non implica "peggiore come inizializzazione" — sono due domande
diverse, e la seconda si risponde solo con l'esperimento di fine-tuning simulato.

**Il risultato più interessante non è nella tabella dei Dice.** Guardando le predizioni è emerso che D
**non produce entrambe le cose su tutte le immagini**. Misurato su tutti i 151 casi di validation:

| sorgente | casi | voxel predetti come lesione | voxel predetti come tessuto | casi con voxel dell'altro tipo |
| --- | --- | --- | --- | --- |
| ATLAS | 112 | 0.35% | **0.00%** | 2 |
| WMH | 16 | 0.17% | **0.00%** | 0 |
| DBB | 23 | **0.00%** | 20.84% | 0 |

Su 149 casi su 151 il comportamento è **puro**: sulle immagini adulte D predice soltanto la lesione e
nessun tessuto, su quelle pediatriche soltanto i tessuti e nessuna lesione. La rete non ha mediato fra
le due annotazioni contraddittorie: **ha imparato a riconoscere da quale dataset viene l'immagine** e a
cambiare comportamento di conseguenza. È di fatto un modello con due modalità, selezionate dal dominio.

Questo spiega perché D non è crollata — non ha dovuto conciliare niente — ma **smonta l'argomento del
"modello unico che serve a entrambi gli stadi"**: su un paziente con paralisi cerebrale, che non
somiglia esattamente né a un adulto con ictus né a un neonato sano, non sappiamo quale delle due
modalità verrebbe scelta. È un rischio da dichiarare, non un vantaggio.

È anche la migliore motivazione empirica per la variante C: marcando `ignore` il tessuto non annotato di
WMH e ATLAS si toglie alla rete proprio l'informazione con cui costruisce questa scorciatoia. Figura:
`reports/figures/modello_unico_7_classi.png`.

### 541 — il test del pretraining: serve davvero, con dieci casi?

**Perché.** L'intero progetto poggia su una premessa mai misurata: che pre-addestrare sui dataset
pubblici serva quando il bersaglio è di circa dieci pazienti. Se fosse falsa lo si scoprirebbe il giorno
in cui arrivano i dati veri, cioè troppo tardi. Questo esperimento la verifica **adesso**, senza i
pazienti.

**Il disegno, e la parte che lo rende valido.** Servono casi di target che nessun modello di pretraining
abbia mai visto. I **45 casi di ATLAS acquisiti a 1.5 T** sono esattamente questo: `Dataset521_ATLAS_3T`
li esclude per costruzione, e lo script verifica prima di procedere che zero di essi compaia in un
qualsiasi fold di 521 o di 503. Non si tocca alcun test set: vengono dal pool di sviluppo di ATLAS.

| | |
| --- | --- |
| dataset | `Dataset541_SIM_AINCP`, 45 casi, costruito da `scripts/make_finetuning_sim_dataset.py` |
| fold 0 | **10 casi di training** (quanti saranno i pazienti AInCP), **35 di validation** |
| scelta dei 10 | stratificata per volume di lesione — mediana 13.7 ml contro 13.2 del gruppo di valutazione, con lo spettro coperto da 0.9 a 238 ml. Su questo compito il Dice dipende quasi solo dal volume, e dieci casi tutti piccoli o tutti grandi renderebbero il risultato illeggibile |
| piano | **copiato da 502** con `nnUNetv2_move_plans_between_datasets`, non ricalcolato: il trasferimento dei pesi funziona solo fra architetture identiche |
| budget | 250 epoche per entrambi i bracci, `nnUNetTrainerWandb_250epochs` |

**I due bracci** differiscono *solo* per l'inizializzazione dei pesi: da zero, contro fine-tuning da
`Dataset521_ATLAS_3T` (511 casi, stesso compito). Usano due identificativi di piano diversi
(`nnUNetPlans` e `nnUNetPlansFT`) solo perché nnU-Net deriva da quel nome la cartella dei risultati: i
due file sono identici e puntano agli stessi dati preprocessati.

**Un vincolo scoperto preparando l'esperimento, che vale anche per il transfer DBB → ATLAS.** Volevamo un
terzo braccio con i pesi di DBB, per misurare se il transfer *fra compiti diversi* funziona. **Non è
possibile senza rilavorare DBB**: le convoluzioni trasposte del decoder hanno dimensione pari allo
stride, e l'ultimo stadio di 503 usa `[2, 2, 1]` contro il `[2, 2, 2]` di 502. Le forme non combaciano e
il caricamento si interrompe. Verificato caricando davvero i checkpoint prima di impegnare la GPU:

| sorgente | i pesi si caricano in 541? |
| --- | --- |
| `Dataset521_ATLAS_3T` | **sì** |
| `Dataset531_UNION_D` | sì, ma **ha visto i 45 casi**: inutilizzabile qui |
| `Dataset503_DBB` | **no**, architettura incompatibile |

Per usare DBB come sorgente di pretraining bisognerà spostarci sopra il piano di 502 e riaddestrarlo:
sono 7 ore di GPU, ed è il prerequisito dell'esperimento «transfer DBB → ATLAS» già in programma.

**Limiti da dichiarare quando si riporterà il risultato.** Una sola estrazione dei 10 casi, e con dieci
casi la varianza è alta; una sola sorgente di pretraining; e i 45 casi di target vengono per 39 dalla
coorte `r040`, quindi è in larga parte un solo scanner.

**Risultato: il pretraining serve, e molto.** I due bracci, sugli stessi 35 casi di valutazione:

| braccio | Dice |
| --- | --- |
| addestrato da zero sui 10 casi | 0.4507 |
| fine-tuning da `521` sui 10 casi | **0.5642** |
| **differenza** | **+0.1135** |

Migliora in 24 casi su 35, peggiora in 5, Wilcoxon **p = 0.000124**. Il guadagno c'è su entrambe le
fasce di volume: +0.087 sotto la mediana e +0.142 sopra. In alcuni casi è la differenza fra vedere e non
vedere: `sub-r040s002` passa da **0.000 a 0.781**.

**Ma c'è un terzo braccio, ed è quello che cambia la conclusione pratica.** Il modello pre-addestrato
applicato **tale e quale**, senza vedere i 10 casi, sugli stessi 35 di valutazione:

| | Dice sui 35 casi |
| --- | --- |
| da zero sui 10 | 0.4507 |
| fine-tuning da `521` sui 10 | 0.5642 |
| **`521` senza alcun fine-tuning** | **0.6772** |
| differenza fine-tuning meno zero-shot | **−0.1130**, p = 0.013 |

**Il fine-tuning con la ricetta predefinita peggiora un modello già buono.** Batte lo zero-shot in
appena 13 casi su 35. La spiegazione più probabile è nel passo di apprendimento: nnU-Net riparte da
`0.01`, che è il valore di un addestramento completo, e su dieci casi cancella le feature
pre-addestrate invece di raffinarle.

Due conclusioni distinte, ed entrambe contano:

1. **La premessa del progetto regge**: senza pretraining, con dieci casi, si perde più di undici punti.
2. **La ricetta di fine-tuning va progettata, non ereditata.** E va sempre misurato lo zero-shot come
   termine di paragone: senza quel braccio avremmo concluso "il pretraining funziona" e adottato una
   procedura che peggiora il modello.

Una cautela sulla generalizzazione: qui il dominio di partenza e quello di arrivo sono molto vicini
— stesso dataset, stesso protocollo di annotazione, cambia il campo magnetico. Sui pazienti AInCP, che
sono pediatrici e con un'altra patologia, lo zero-shot renderà molto meno e il fine-tuning sarà
necessario. Il risultato non dice "non fare fine-tuning": dice **"non farlo con i parametri di
default, e misura sempre il punto di partenza"**.

Report: `reports/test_pretraining.md`.

**Il seguito, e la risposta.** Lo stesso fine-tuning con passo di apprendimento ridotto, due bracci a
`1e-3` e `1e-4` (`scripts/run_test_finetuning_lr.sh`, conclusi il 2 settembre). Il quadro completo, sugli
stessi 35 casi:

| configurazione | Dice |
| --- | --- |
| da zero sui 10 casi | 0.4507 |
| fine-tuning, passo `1e-2` (default nnU-Net) | 0.5642 |
| fine-tuning, passo `1e-3` | 0.6419 |
| `521` senza alcun adattamento | 0.6772 |
| **fine-tuning, passo `1e-4`** | **0.6851** |

**Il pretraining vale oltre 23 punti di Dice** rispetto all'addestramento da zero. Ma ci si arriva solo
riducendo il passo di **cento volte**: con il default il fine-tuning *peggiorava* il modello di partenza.
La progressione è monotona e coincide esattamente con quella dell'overfitting misurato sulle curve
(vedi `reports/curve_training.md`): divario fra dati visti e non visti +0.534 da zero, poi +0.398,
+0.275, +0.179. Meno overfitting, migliore il risultato — la stessa grandezza vista da due lati.

**Conseguenze operative per AInCP**, ed è la ricetta che useremo sui pazienti veri:
- passo di apprendimento **`1e-4`**, non il default;
- **budget di epoche basso**: il minimo della perdita di validation cade fra l'epoca 69 e la 128 su 250;
- **misurare sempre il modello non adattato** come termine di paragone, perché può batterlo;
- **`checkpoint_best` protegge dall'overfitting del fine-tuning, non da quello da zero.** La
  rivalutazione dei quattro bracci con `--val_best` (`scripts/run_val_best.sh`, conclusa l'8
  settembre; i risultati originali stanno in `validation_final/`, quelli con `checkpoint_best` in
  `validation/`) dice:

  | braccio | ultima epoca | `checkpoint_best` | differenza |
  | --- | --- | --- | --- |
  | da zero | 0.4507 | 0.4491 | −0.002 |
  | fine-tuning, passo `1e-2` | 0.5642 | **0.6763** | **+0.112** |
  | fine-tuning, passo `1e-3` | 0.6419 | 0.6624 | +0.021 |
  | fine-tuning, passo `1e-4` | 0.6851 | 0.6873 | +0.002 |

  **Questo corregge quanto scritto il 7 settembre**, quando era stato letto il solo braccio da
  zero («0.4491 contro 0.4507, cioè nulla») e se ne era concluso che il criterio di nnU-Net —
  la pseudo-Dice mediata esponenzialmente — traccia la grandezza sbagliata. Sul braccio da zero
  è vero perché il modello non diventa mai buono, e il migliore coincide con l'ultimo. Sul
  fine-tuning con il passo di default il checkpoint migliore **recupera undici punti** e riporta
  il modello quasi al livello del non adattato (0.6772): l'overfitting c'è, ma la pseudo-Dice lo
  vede. Con il passo `1e-4` il minimo e l'ultima epoca quasi coincidono, quindi la ricetta resta
  la stessa; ma il criterio di arresto è una protezione in più che costa niente, ed è stato
  scritto (vedi sotto).

---

## Terza parte: cosa abbiamo imparato

1. **Il T1 non basta per le lesioni piccole.** Trovato due volte per vie indipendenti: l'ablazione FLAIR
   (−0.242 sulle piccole contro −0.055 sulle grandi) e i casi falliti di ATLAS (contrasto 1.07 contro
   0.62). È il vincolo strutturale del progetto, perché nella paralisi cerebrale le lesioni piccole e non
   cavitate sono una parte importante del bersaglio.
2. **Non si aggira affiancando i canali.** FLAIR+T1 rende meno della sola FLAIR.
3. **Non si buttano via dati per pulizia.** Escludere i 1.5 T non ha dato nulla, e il modello che non li
   ha mai visti li segmenta comunque bene.
4. **Le scorciatoie di preprocessing non hanno pagato.** Escludere i dati a bassa qualita', affiancare
   un secondo canale, togliere il cranio: tre interventi ragionevoli, tre risultati nulli o negativi. La
   pipeline di default di nnU-Net e' gia' vicina al massimo che si ottiene senza cambiare i **dati** o il
   **compito** — che e' precisamente il motivo per cui la variante D prova a cambiare i dati.
5. **Il Dice medio, da solo, mente.** Su questi compiti dipende quasi solo dal volume del bersaglio: metà
   della differenza fra WMH e ATLAS è composizione del campione. Ogni numero va letto con la
   distribuzione accanto.
6. **Confrontare medie di validation diverse è un errore facile.** Da qui la scelta di fare tutti i
   confronti appaiati, caso per caso.

## Quarta parte: cosa non è ancora stato addestrato

### 504 — i dieci pazienti AInCP: T0, T1, T1' (conclusi il 15 settembre)

`Dataset504_AINCP` (`scripts/make_aincp_dataset.py`): spazio T1 nativo, due canali `{0: FLAIR,
1: T1}`, label ROI_1+ROI_2+ROI_4, piano `nnUNetPlansDa502` trasferito da 502 (1 mm, patch 128³) con
le tre correzioni per canale documentate nello script, split a 5 fold per paziente fissato a mano e
versionato. Coda: `scripts/run_aincp_504.sh <fold>`.

| sigla | trainer | pesi di partenza | variabile isolata |
| --- | --- | --- | --- |
| **T1** | `nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50` | 502, canale FLAIR a zero | — (è la Proposta B) |
| **T0** | `nnUNetTrainerWandb_250ep_arresto50` | nessuno | il pretraining |
| **T1'** | `nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50` | 502, filtro duplicato | l'inizializzazione del canale nuovo |

Tutti: 250 epoche, passo `1e-4` per i due di espansione e default per T0, e **arresto sulla perdita
di validazione** con pazienza 50 (`nnUNetTrainerWandb.pazienza_val`): a ogni miglioramento si salva
`checkpoint_val_best.pth`; dopo 50 epoche senza miglioramento il training si ferma;
`checkpoint_final.pth` — quello su cui nnU-Net esegue la validazione — riceve i pesi dell'epoca
migliore, e l'ultima epoca resta in `checkpoint_ultima_epoca.pth`. Restano quindi confrontabili tre
scelte del checkpoint: ultima epoca, `checkpoint_best` di nnU-Net (pseudo-Dice), minimo della
perdita di validazione. Verificato a CPU con epoche minuscole (`pazienza_val = 1`): si ferma, i file
esistono, `checkpoint_final` coincide con `checkpoint_val_best` e non con l'ultima epoca.

La valutazione: `scripts/metriche_per_lesione.py 504 --trainer <nome> --fold all --soglia-mm3 10`
(fasce in mm³, soglia simmetrica su riferimento e predizione, griglia originale) e
`scripts/confronto_fold.py 504 <base> <variante>`, che unisce i cinque fold in dieci predizioni
fuori-fold, una per paziente, e le confronta appaiate in Dice e in rilevamento. Con dieci coppie il
p di Wilcoxon non scende sotto 0.002 nemmeno se tutte concordano: contano le coppie e le grandezze.

**Risultato (15 settembre, 15 training, 14–15/9 su due GPU condivise).** Nessuna soglia sulle
lesioni. Report: `reports/confronto_504_*.md`, `reports/metriche_per_lesione_504_*.md`, figura
`reports/figures/aincp/predizioni_504.png` (`scripts/figura_predizioni_504.py`).

| variante | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | **Dice medio** | lesioni rilevate (71) | FP / paziente | epoca migliore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0, da zero | 0.583 | 0.664 | 0.665 | 0.435 | 0.560 | **0.581** | 27 (38.0%) | 7.9 | 75, 26, 92, 86, 80 |
| T1, espansione `zero` | 0.465 | 0.672 | 0.516 | 0.488 | 0.635 | **0.555** | 25 (35.2%) | 2.6 | 106, 126, 89, 28, 28 |
| T1', espansione `duplica` | 0.648 | 0.718 | 0.606 | 0.464 | 0.543 | **0.596** | 20 (28.2%) | 3.2 | 47, 199, 135, 57, 185 |

Appaiati sulle dieci predizioni fuori-fold: T1 contro T0 **−0.026** (migliora 5, peggiora 5,
p = 0.92); T1' contro T0 **+0.015** (5 e 5, p = 0.49); T1' contro T1 +0.041 (6 e 4, p = 0.38).
**Nessuna variante si distingue in Dice.** Quello che le distingue sono i falsi positivi: il modello
da zero ne produce 7.9 per paziente, i due pre-addestrati 2.6 e 3.2. Sulle lesioni sopra i 10 mm³
tutte e tre trovano quasi tutto (22, 17 e 18 su 25); sotto i 10 mm³ nessuna (5, 8 e 2 su 46).

**Perché la Proposta B non ha reso, misurato sui pesi.** Norma del filtro FLAIR contro quella del
filtro T1 nel primo strato, sui cinque fold:

| variante | ‖FLAIR‖ / ‖T1‖ | spostamento del T1 dalla partenza |
| --- | --- | --- |
| `zero` | 0.04 – 0.08 | 1.5 – 3.0 % |
| `duplica` | 1.2 – 1.4 | 15 – 29 % |
| da zero | 1.0 – 1.3 | — |

Con `zero` il canale FLAIR **è rimasto quasi a zero** e il T1 si è mosso del 2%: dopo 250 epoche a
passo `1e-4` su otto pazienti, il modello è ancora il 502 a solo T1, e la FLAIR — la sequenza su
cui le lesioni si vedono — non è mai entrata in gioco. È esattamente il rischio scritto nel piano
(«resta un modello a T1 con un canale inerte»). La ricetta `1e-4` era stata misurata su 541, dove
dominio e canale erano gli stessi del pre-addestramento; qui servono un canale nuovo e un dominio
nuovo, e quel passo non basta a impararli. `duplica` parte con la FLAIR già attiva e infatti la usa
(rapporto 1.2–1.4), ma trattandola come una copia del T1 non ne impara il contrasto proprio.

Questo **corregge l'aspettativa** scritta prima del training (che il pretraining valesse +23 punti
come su 541): su AInCP il modello da zero arriva a 0.58 perché le lesioni sono grandi (mediana
7.2 ml) e il Dice le premia; il vantaggio del pretraining, se c'è, va cercato nel rilevamento delle
piccole e nei falsi positivi, non nella media.

Il Dice per paziente dipende dalla forma della lesione più che dal modello: FSM_041 (gliosi
sottile ad anello attorno a una cavità) sta a 0.04–0.06 con tutte e tre; FSM_040 (cavità da 56 ml)
a 0.83–0.92.

### Il passo differenziato: la diagnosi era giusta (16 settembre)

Trainer `nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_arresto50`, cinque fold conclusi
fra il 15 e il 16 settembre. Il primo strato va in un gruppo dell'ottimizzatore a `1e-2`, tutto il
resto resta a `1e-4`, e uno scheduler per gruppo fa decadere ciascuno dal proprio passo. Il primo
strato è un solo tensore `[32, 2, 3, 3, 3]` con entrambi i canali: un hook sul gradiente frena la
fetta del T1 di un fattore 0.01, così il filtro pre-addestrato resta di fatto a `1e-4` e solo la
FLAIR impara a passo pieno. Il log dei run conferma che il meccanismo era attivo davvero e non
solo nella verifica a CPU.

| Variante | Dice medio | Coppie vinte contro `zero 1e-4` |
| --- | --- | --- |
| da zero (riferimento) | 0.581 | — |
| B `zero`, passo `1e-4` | 0.555 | — |
| B `duplica`, passo `1e-4` | 0.596 | — |
| **B `zero`, primo strato `1e-2`** | **0.621** | **7 / 10** |

Il numero che conta non è la media ma la diagnosi che la spiega. La norma del filtro FLAIR rispetto
a quella del T1, primo strato, fold per fold:

| Variante | rapporto FLAIR / T1 |
| --- | --- |
| B `zero`, passo `1e-4` | 0.044 – 0.077 |
| B `duplica` | 1.18 – 1.41 |
| B `zero`, primo strato `1e-2` | 0.65 – 2.72 |
| da zero | 0.98 – 1.31 |

Con il passo differenziato il canale FLAIR entra nello stesso regime di un modello addestrato da
zero, mentre a `1e-4` restava a un ventesimo. **La causa individuata dopo il primo giro era quella
giusta**: non mancavano i dati, mancava il passo sul canale nuovo. Il guadagno è +0.065 di Dice
sulla stessa variante a passo uniforme e +0.039 sul modello da zero, con 7 coppie su 10 in
miglioramento; su FSM_039 la differenza è +0.438, da 0.036 a 0.474. **Ma nessuna di queste
differenze è distinguibile dal rumore**: contro il modello da zero l'intervallo al 95% (bootstrap
sulle dieci differenze appaiate) è [−0.036, +0.110], Wilcoxon p = 0.275; la deviazione standard
del Dice fra pazienti è 0.24–0.31. Il rilevamento non si muove (25 lesioni contro 27, p = 0.75).
La prova è sui pesi, non sul Dice.

**Il confronto però non è pulito, per un difetto del freno.** L'hook scalava il gradiente, ma
`torch.optim.SGD` aggiunge il decadimento dei pesi *dopo* l'hook e lo moltiplica per il passo del
gruppo: la fetta del T1 ha quindi ricevuto un decadimento cento volte più forte del suo passo. Si
vede nei numeri: la norma del filtro T1 scende da 8.36 a 3.50–7.49, tanto più quanto più a lungo
gira il fold (223 epoche → 3.50; 66 epoche → 7.49), e il coseno con il filtro di partenza resta fra
0.996 e 0.9998. Direzione intatta, solo la scala ridotta: non è apprendimento, è decadimento. Il
canale T1 è stato quindi frenato come voluto *e* rimpicciolito come non voluto, e le due cose non
sono separate in questi run. Corretto il 22 settembre mettendo il peso del primo strato in un
gruppo con `weight_decay=0` e applicando il decadimento dentro il freno, dove viene scalato insieme
al gradiente; verificato numericamente che l'aggiornamento della fetta pre-addestrata vale ora
esattamente `1e-4 · (gradiente + wd · peso)`.

### Per fenotipo, per dimensione, e che cosa è significativo (22 settembre)

`scripts/analisi_rilevamento_504.py` → `reports/analisi_rilevamento_504.md`. Porta le singole ROI in
spazio T1 con la trasformazione salvata e misura, per ogni variante, rilevamento e copertura dei
voxel per fenotipo, falsi positivi appaiati, e quali lesioni sopravvivono al ricampionamento a 1 mm
del piano. Due controlli interni: le ROI trasportate ricostruiscono `labelsTr` a meno di 26 voxel in
tutto (arrotondamenti dell'header); il ricampionamento riproduce **esattamente** il bersaglio salvato
da nnU-Net in tutti e dieci i casi.

| | gliosi | cistica | encefalomalacia | falsi positivi / paziente |
| --- | --- | --- | --- | --- |
| da zero | **0.585** | 0.834 | 0.401 | 7.9 |
| B `zero` 1e-4 (FLAIR spenta) | 0.472 | **0.933** | **0.809** | 2.6 |
| B `duplica` 1e-4 | 0.499 | 0.917 | 0.559 | 3.2 |
| B passo differenziato (FLAIR attiva) | 0.525 | **0.933** | 0.775 | **2.3** |

Copertura dei voxel = frazione dei voxel di quel fenotipo toccata dalla predizione, sui pazienti che
lo hanno (gliosi 10, cistica 4, encefalomalacia 4).

- **Il pre-addestramento su ATLAS porta le cavità.** Encefalomalacia da 0.02–0.48 (da zero) a
  0.61–0.87 (pre-addestrati) in 4 pazienti su 4; FSM_045 da 0.018 a 0.612. Le lesioni di ATLAS
  sono ictus cronici, cioè cavità ipointense in T1.
- **La FLAIR porta la gliosi.** Accendere il canale FLAIR migliora la gliosi in 7 pazienti su 10
  (0.472 → 0.525, p = 0.084); ma il modello da zero, che la FLAIR la usa in pieno, resta il migliore
  su questo fenotipo. La variante `zero` a passo 1e-4, con il filtro FLAIR a 0.05 della norma del
  T1, è di fatto un modello a sola T1: l'esperimento naturale che misura il contributo della FLAIR
  senza addestrare un modello a un canale.
- **I falsi positivi sono l'unica differenza significativa**: meno in 8 pazienti su 10 con il
  passo differenziato (p = 0.033) e con `zero` (p = 0.043), 6 su 10 con `duplica` (p = 0.062). Da
  zero sono per lo più minuscoli (63 su 79 sotto 10 mm³, mediana 0.9 mm³).
- **27 lesioni su 71 non esistono nel bersaglio a 1 mm**: 20 delle 24 sotto 1 mm³ e 7 delle 22 fra 1
  e 10 mm³ (gliosi 18, cistica 7, encefalomalacia 2). Il tetto del rilevamento con questo piano è 44.
  Fra le 44 visibili il modello da zero ne trova 24, il passo differenziato 21; fra le 26 piccole
  visibili, 6 e 4.

**Prossimi esperimenti, in ordine, una variabile ciascuno:**

1. **In corso dal 22 settembre: il passo differenziato con il freno corretto**
   (`nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_arresto50`) e il **controllo a
   passo `1e-3` uniforme** (`nnUNetTrainerEspansioneCanali_zero_250ep_lr1e3_arresto50`), entrambi
   dai pesi di 502, cinque fold, in coda con `run_aincp_504.sh`. Il primo separa «la FLAIR ha imparato» da «il T1 si è
   ristretto»; il secondo dice se serviva un passo *differenziato* o solo un passo *più grande*.
   Si leggono con `analisi_rilevamento_504.py`, non solo con il Dice.
2. **Stadio 2 su WMH, ora il principale**: la rete espansa addestrata sui 79 casi a due canali di
   Dataset512 (piano `nnUNetPlansDa502` da trasferire, con le stesse tre correzioni per canale) e poi
   adattata su 504. L'analisi per fenotipo gli dà una ragione precisa: le iperintensità di WMH sono
   il fenotipo più vicino alla gliosi, e la gliosi è l'unico fenotipo su cui la Proposta B resta sotto
   il modello da zero. Il criterio di successo è la copertura della gliosi senza perdere le cavità.
3. **Modality dropout**, per l'uso a un solo canale: dopo, come variabile a sé. Il responsabile
   ha chiesto che il modello possa funzionare anche con una sola sequenza in ingresso. Riferimento
   classico per le modalità mancanti: HeMIS (Havaei et al., MICCAI 2016); alternativa recente con un
   canale «agnostico» addestrato su contrasti sintetici: Addison et al. 2025 (arXiv 2509.09290).

| Modello | Domanda | Stato |
| --- | --- | --- |
| **D skull-stripped** | lo skull stripping aiuta anche sull'unione? | da costruire, per confronto diretto con `531` |
| varianti A (WMH+ATLAS), B (A+AInCP), C (tutti con `ignore`) | alternative se D non regge | pianificate |
| transfer DBB → ATLAS | i tessuti insegnano l'anatomia al task lesionale? | pianificato |
| `2d` contro `3d_fullres` | quale configurazione, decisa in cross-validation | pianificato |
| i 5 fold della configurazione scelta | il numero finale | alla fine |
| fine-tuning su AInCP | l'obiettivo vero | in attesa dei dati |

Il dettaglio del piano, con le motivazioni e l'ordine, è in [obiettivo-e-piano.md](obiettivo-e-piano.md).
