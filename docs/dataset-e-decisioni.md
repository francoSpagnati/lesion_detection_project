# I dataset, i loro problemi e le decisioni prese

Cosa contiene ciascun dataset, quali problemi reali sono stati **trovati misurandoli** (non ipotizzati),
e cosa si è deciso di fare. I concetti di dominio sono in
[primer-mri-e-segmentazione.md](primer-mri-e-segmentazione.md); i numeri sono rigenerabili nei report
citati.

Una premessa sul metodo: quasi tutti i problemi qui sotto sono **silenziosi**. Non fanno crashare niente.
Producono un modello che addestra normalmente, mostra metriche plausibili e ha imparato la cosa sbagliata.
È il motivo per cui la preparazione è piena di `assert` invece di controlli facoltativi: l'unica difesa
contro un errore silenzioso è renderlo rumoroso.

## Dove stanno i dati

I dati originali sono su `/data01`, **in sola lettura**, e sono referenziati da `data/raw/` tramite
symlink. Tutto il derivato sta in `data/processed/`. `/home` è oltre il 90% di occupazione, con circa 250 GB liberi: è il vincolo
reale del progetto.

Convenzioni di materializzazione, valide per tutti i dataset:

- **immagini**: symlink ai file originali. Costo zero su disco, dati non modificati. I quattro dataset
  preparati occupano decine di MB invece dei ~14 GB delle copie.
- **label**: riscritte in `uint8` con l'**affine dell'immagine**, così le geometrie sono bit-identiche e
  nnU-Net non emette warning di mismatch. Eccezione: DBB, dove le parcellazioni hanno già valori `1..6`
  contigui e l'affine corretta, quindi sono symlink anche loro.

---

## WMH — `Dataset501_WMH` (FLAIR) e `Dataset511_WMH_T1` (T1)

99 casi appaiati su 4 siti/scanner, orientamento uniforme `LPS`. Sviluppo 79, test interno 20.

```
Utrecht_0/
  Utrecht_0_FLAIR_BiasCorr.nii.gz                  <- canale 0000 di Dataset501
  Utrecht_0_T1w_BiasCorr_2FLAIR_Aligned.nii.gz     <- canale 0000 di Dataset511, già su griglia FLAIR
  Utrecht_0_wmh.nii.gz                             <- label
```

Gli altri file (`_BiasField`, `_T1w_BiasCorr` non allineato, `.mat`) sono sottoprodotti e **non vanno
usati**. Solo `_T1w_BiasCorr_2FLAIR_Aligned` condivide la griglia della FLAIR: verificato su **99/99**.

### La label 2 non è una lesione WMH

I valori sono `0` background, `1` iperintensità, `2` **altra patologia** (tipicamente infarti), presente in
**34/99** casi con volumi fino a 32.9 ml. Nella challenge ufficiale quei voxel sono **esclusi dalla
valutazione**.

La documentazione iniziale del progetto diceva di fondere 1 e 2 in una lesione binaria. Sarebbe stato un
errore doppio: si insegna al modello a segmentare gli infarti come WMH su un terzo del dataset, e il Dice
non è più confrontabile con nessun numero pubblicato.

| Politica | Mappatura | Quando |
| --- | --- | --- |
| `exclude` (**default**) | `2 -> 0` | Coerente con la valutazione ufficiale. Il modello impara a ignorarli. |
| `ignore` | `2 -> ignore label` | Più rigoroso: nessuna loss su tessuto ambiguo. Con `exclude` diciamo "questo è tessuto normale", che è falso; con `ignore` diciamo "non ti sto dicendo cos'è". |
| `merge` | `2 -> 1` | **Da non usare.** Resta implementata solo per poterla confutare sperimentalmente. |

### Anisotropia: fette da 3 mm

Lo spacing lungo le fette è **3.0 mm su tutti e 99 i casi**, mentre nel piano è circa 1 mm. Per effetto di
volume parziale una lesione sottile viene mediata col tessuto sano e appare sbiadita, o sparisce — e le WMH
sono esattamente il tipo di lesione che ne soffre: molte, piccole, dai confini sfumati.

**Non è correggibile.** L'informazione non è stata persa nel preprocessing: non è mai stata acquisita.
Interpolare crea voxel plausibili, non informazione. Si può solo mitigare: non resamplare a mano, sfruttare
il resampling separato lungo z che nnU-Net applica da sé, e prendere sul serio la configurazione `2d`.

Una sottigliezza che è facile confondere: nnU-Net ha **due** decisioni distinte legate all'anisotropia. La
*target spacing* riceve un override solo se `spacing_peggiore > 3 × max(altri)`, cioè `3.0 > 3.6`, che è
**falso**. Il *metodo di resampling* usa invece `max/min = 3.07 > 3`, che è **vero**. Quindi: nessun
override della spacing, ma resampling separato lungo z sì.

### Quattro scanner: domain shift

| Sito | n | spacing (mm) | shape | anisotropia |
| --- | --- | --- | --- | --- |
| Amsterdam_GE1T5 | 10 | 1.30 x 1.211 x 3.0 | 128x256x103 | 2.48 |
| Amsterdam_GE3T | 49 | 1.20 x 0.977 x 3.0 | 132x256x83 | 3.07 |
| Singapore | 20 | 1.0 x 1.0 x 3.0 | 232x256x48 | 3.00 |
| Utrecht | 20 | 0.958 x 0.958 x 3.0 | 240x240x48 | 3.13 |

Cambia anche il campo magnetico (1.5 T contro 3 T) e quindi il "look" delle immagini. Due rischi:
Amsterdam_GE3T da solo è metà del dataset, quindi uno split casuale può produrre un fold di validation
quasi monosito e spacciare "bravo su Amsterdam" per un risultato generale; e se ogni sito compare sia in
training sia in validation, il modello può memorizzare le caratteristiche dello scanner.

Contromisura: split **stratificati per sito**, e **Dice riportato per sito**. Il passo naturale
successivo, se si vuole rispondere davvero alla domanda, è una valutazione *leave-one-site-out*.

---

## ATLAS 2.0 — `Dataset502_ATLAS` (T1)

655 casi appaiati su 33 coorti (`R001`…`R052`), più 300 immagini di `Testing/`. T1w in spazio MNI, struttura
BIDS, orientamento `RAS`, **una sola** risoluzione `1.0 × 1.0 × 1.0` mm e shape `197 × 233 × 189`.
Sviluppo 556, test interno 99.

- **Il `Testing/` ufficiale non ha label.** Sembra un test set pronto ma non lo è: le annotazioni sono
  trattenute dagli organizzatori. Sta in `imagesTs_official/`, fuori dal flusso nnU-Net; il test valutabile
  è ricavato dai 655 annotati. La separazione fisica è deliberata.
- **Le due varianti di maschera sono la stessa segmentazione.** `_mask` e `_mask_value2` hanno supporto
  binario **identico** e differiscono solo per il valore codificato. Era segnalata come decisione aperta e
  delicata: si è chiusa misurandola in pochi minuti. Si usa `_mask`.
- **Le maschere vanno binarizzate con `> 0`, non `== 1`.** Il caso `sub-r039s002` è codificato `0/0.01`:
  2810 voxel di lesione vera che un confronto con 1 azzererebbe, facendolo entrare nel training come
  "paziente senza lesione". Un caso corrotto su 655 non sposta le metriche in modo visibile — ed è proprio
  questo che rende il bug difficile da scoprire a valle.

Il principio generale: **mai fidarsi della codifica dichiarata delle label, verificarla.**

---

## DBB — `Dataset503_DBB` (T1, 6 tessuti)

Distorted Brain Benchmark, da brainlife (`60a14ca503bcad0ad27cada9`, DOI `10.25663/brainlife.pub.24`,
licenza **CC-BY 4.0**, va citato). Report: [`reports/dbb_analysis.md`](../reports/dbb_analysis.md).

**Non è un dataset di lesioni**: segmenta 6 classi di tessuto normale (`1` CSF, `2` sostanza grigia, `3`
sostanza bianca, `4` grigia profonda, `5` tronco encefalico, `6` cervelletto) su cervelli **pediatrici**.

**Solo 244 dei 955 soggetti sono usabili.** Gli altri 711 hanno le label ma **non l'immagine**: sono delle
coorti C-MIND e NIH-PED, ad accesso controllato, di cui brainlife può ridistribuire solo i derivati.

I 244 sono tutti EMEDEA-PED, già divisi dagli autori: 207 sani (184 train + 23 validation) e 37 patologici
di test (11 PFM, 10 MCD, 10 ACC, 6 HD). Design tipico da benchmark di generalizzazione: *si addestra sul
sano, si testa sul malformato*. Il dataset include **maschere cerebrali per tutti i 955 soggetti**.

### Il training set non è omogeneo

213 casi su 244 sono in spazio MNI (`182×218×182` a 1 mm), ma **31 sono in spazio nativo** con shape e
spacing diversi, e **12 di questi sono orientati RAS invece che LAS**. nnU-Net legge gli array così come
sono: mescolare orientamenti significa dare alla rete volumi specchiati fra loro. Tutti e 31 sono casi
*sani di training*; i 37 patologici di test sono conformi.

Due scelte conseguenti, scritte nel `dataset.json`:

- `overwrite_image_reader_writer: NibabelIOWithReorient` — riorienta al volo immagini e label in modo
  coerente, **senza riscrivere i dati su disco**;
- il **fold 0 riproduce esattamente lo split ufficiale** (184/23), così un run a fold singolo resta
  confrontabile con i risultati pubblicati. I fold 1-4 sono partizioni aggiuntive degli stessi 207 casi,
  stratificate sulla conformità allo spazio MNI.

---

## Target spacing: come nnU-Net decide

Prende la **mediana** degli spacing per asse e applica un override anisotropico (10º percentile sull'asse
peggiore) **solo se** valgono entrambe `spacing[peggiore] > 3 × max(altri)` e
`shape[peggiore] × 3 < min(altre shape)`.

| Dataset | Target spacing | Override | Resampling separato su z |
| --- | --- | --- | --- |
| WMH | `1.2 × 0.9766 × 3.0` mm | no | **sì** (anisotropia 3.07) |
| ATLAS | `1.0 × 1.0 × 1.0` mm | no | no |
| DBB | `1.0 × 1.0 × 1.0` mm | no | no |

Per confrontare risoluzioni diverse si genera un piano alternativo con
`nnUNetv2_plan_experiment -overwrite_target_spacing`, **senza** sovrascrivere il default: così entrambi
gli esperimenti restano riproducibili.

## Carico lesionale e split

Il volume va calcolato in mm³, non in voxel, altrimenti dataset con spacing diverso non sono confrontabili.
ATLAS va da 0.013 a 496 ml (mediana 5.29), WMH da 0.79 a 75 ml (mediana 10.93). Alla risoluzione target
**19 casi ATLAS su 655** scendono sotto i 100 voxel di lesione.

Su lesioni così piccole il Dice è quasi rumore: sbagliare 10 voxel su 20 fa crollare il punteggio, sulla
stessa lesione da 20.000 voxel non si nota. E quel rumore si sposta fra i fold: se un fold pesca quattro
casi minuscoli e un altro nessuno, i due Dice differiscono **per composizione, non per qualità del
modello**. Per questo la chiave di stratificazione è **`gruppo | terzile di volume lesionale`** (gruppo =
sito per WMH, coorte per ATLAS), con unione in `RARE` delle classi troppo piccole.

## Leakage: tre vie, due poco ovvie

1. **Il test dentro il fingerprint.** nnU-Net analizza tutti i casi in `imagesTr` per calcolare statistiche
   di intensità e decidere il piano. Se il test stesse lì, target spacing e patch size sarebbero in parte
   derivate da dati che dovrebbero essere invisibili.
   → Il test interno va in `imagesTs`/`labelsTs`, mai in `imagesTr`.
2. **Split casuale che spezza i gruppi.** Il modello si appoggia a caratteristiche dello scanner invece
   che dell'anatomia, e la validation non se ne accorge.
   → Stratificazione per sito/coorte, valutazione riportata per sito.
3. **Riuso del test per scegliere.** La più insidiosa: non è nei dati, è nel processo. Se guardi il test,
   cambi qualcosa e riguardi, il test è diventato un validation set.
   → Tutte le decisioni si prendono in cross-validation con `nnUNetv2_find_best_configuration`. Il test si
   tocca **una volta sola**, alla fine.

Un quarto rischio, procedurale: nnU-Net rigenera `splits_final.json` **se il file non esiste**. Cancellarlo
fra due esperimenti invaliderebbe i confronti, senza alcun errore visibile. Il file è scritto da noi e
**non va rigenerato**; all'avvio nnU-Net stampa le dimensioni, che vanno controllate (`63/16` WMH,
`445/111` ATLAS, `184/23` DBB fold 0).

Nessuna di queste vie richiede un errore di programmazione: sono tutte conseguenze di scorciatoie
ragionevoli.

## WMH ha solo 99 casi

La ground truth si disegna a mano fetta per fetta e costa tempo di esperto. Con un singolo split 80/20 il
test avrebbe 20 casi, e la differenza di Dice fra due modelli sarebbe dominata da quali pazienti sono
capitati nel test. Contromisure: **5 fold** invece di uno split singolo (la variabilità fra fold diventa
essa stessa un dato: se è alta, nessuna differenza piccola è credibile), stratificazione, un test holdout
comunque per un controllo di sanità finale, e il pretraining da dataset più grandi.

---

## Niente N4: la misura lo esclude

Il bias di campo è una variazione di intensità lenta, liscia e moltiplicativa dovuta alla sensibilità non
uniforme della bobina: lo stesso tessuto appare più luminoso al centro e più scuro ai bordi. **nnU-Net non
lo corregge**, quindi era l'unico passo di preprocessing classico che avrebbe aggiunto qualcosa. Prima di
costruire la pipeline, l'abbiamo misurato — vedi
[`reports/bias_field_assessment.md`](../reports/bias_field_assessment.md).

Rapporto `p99/p1` del campo stimato (`1.00` = piatto), erodendo la maschera di un margine **fisico**
crescente, **sugli stessi casi**:

| Dataset | 0 mm | 3 mm | 6 mm | 12 mm |
| --- | --- | --- | --- | --- |
| WMH | 1.678 | 1.550 | 1.163 | **1.119** |
| ATLAS | 1.594 | 1.548 | 1.448 | **1.120** |
| DBB | 1.172 | 1.148 | 1.124 | **1.095** |

Tutti convergono a `1.10 – 1.12` dentro il cervello: **nessun bias interno apprezzabile**. I valori alti al
bordo — fino a 1.68 — sono artefatto di contorno (scalpo, grasso, aria), non segnale. Su WMH c'era anche la
ragione a priori: i file sono già `_BiasCorr`.

Due cautele metodologiche che hanno ribaltato il risultato rispetto alla prima misura, e che vale la pena
conoscere perché sono generali:

- erodere di *N voxel* non è confrontabile fra dataset (WMH ha fette da 3 mm, ATLAS voxel da 1 mm: lo
  stesso raggio in voxel toglie tre volte più tessuto su WMH). Il margine va in **millimetri**;
- l'aggregazione va fatta **a parità di margine sugli stessi casi**: mediare il "margine massimo
  raggiunto" caso per caso premia le teste piccole, che si fermano prima e mostrano rapporti più alti.

**Limite dichiarato:** su WMH e ATLAS il campo è misurato dentro una maschera della *testa*, su DBB dentro
la *maschera cerebrale* fornita. I numeri sono confrontabili solo al margine più profondo. Una misura
definitiva richiederebbe prima lo skull stripping.

**Conseguenza:** N4 non si applica su nessuno dei tre dataset. Nessuna pipeline di correzione da
mantenere, nessun GB materializzato. La decisione si riapre solo se lo skull stripping renderà possibile
una misura più pulita.

## Cosa deliberatamente non facciamo

Decisioni prese, non dimenticanze. Vanno riaperte solo con una misura in mano.

| Non facciamo | Perché |
| --- | --- |
| **Resampling manuale** | Lo fa nnU-Net una volta sola. Anticiparlo produce una doppia interpolazione che degrada le lesioni piccole — proprio quelle già penalizzate dall'anisotropia. |
| **Z-score a mano** | nnU-Net lo applica per immagine dopo il crop, con statistiche calcolate sul solo training. Farlo prima significherebbe calcolarle anche sul test, cioè leakage. |
| **N4** | Misurato: campo piatto dentro il cervello su tutti e tre i dataset. |
| **FSL BET per lo skull stripping** | BET fa crescere una superficie deformabile guidata da soglie, assumendo una testa regolare, ed è tarato sul T1. Fallisce nei due casi che ci riguardano: FLAIR e anatomie alterate. Se servirà, i candidati sono **HD-BET** (stesso gruppo di nnU-Net, addestrato anche su cervelli patologici) o **SynthStrip** (FreeSurfer, il più robusto sul FLAIR). Su DBB non serve: la maschera c'è già. |
| **Riorientare WMH o ATLAS** | L'orientamento è uniforme dentro ciascuno e l'affine lo descrive. Su DBB invece era mescolato, e lì il riorientamento è attivo. |
| **Riempire buchi o pulire le maschere** | La ground truth è il riferimento per definizione. "Migliorarla" cambia il compito e rende i risultati incomparabili. |
| **Scartare i casi con lesioni minuscole** | Sono casi clinici legittimi e spesso i più difficili. Rimuoverli gonfia le metriche senza migliorare il modello. |

Il filo comune: **il preprocessing deve rendere i dati utilizzabili, non "migliori"**. Ogni trasformazione
in più è un'occasione in più di errore silenzioso o di risultati non confrontabili.

Una regola trasversale: ogni variante di preprocessing va costruita come **dataset separato con un nuovo
ID** (come è stato fatto per `Dataset511_WMH_T1`), non sovrascrivendo l'originale, va confrontata **sugli
stessi fold**, e va introdotta **una alla volta** — se due cambiano insieme e il Dice si muove, non sapremo
quale l'ha mosso.
