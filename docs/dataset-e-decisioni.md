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
licenza **CC-BY 4.0**, va citato). Articolo: Amorosino et al., *DBB — A Distorted Brain Benchmark for
Automatic Tissue Segmentation in Paediatric Patients*, NeuroImage 2022.
Report: [`reports/dbb_analysis.md`](../reports/dbb_analysis.md).

**A cosa serve secondo i suoi autori**, che è la chiave per capire come possiamo usarlo. Il benchmark
nasce perché gli strumenti basati su modelli e template — FreeSurfer, SPM e simili — **perdono
accuratezza quando l'anatomia è fortemente distorta** da lesioni gravi. Si addestra sui sani e si valuta
sui distorti, e il bersaglio è la robustezza della segmentazione **dei tessuti**, non il rilevamento
delle lesioni: *distorted* qualifica l'anatomia dei pazienti, non le annotazioni.

Vale la pena notare che **è esattamente la motivazione del secondo stadio della nostra pipeline**: il
"gemello sano" per inpainting esiste perché la parcellazione anatomica convenzionale fallisce quando
l'anatomia è alterata. DBB è quindi il benchmark del problema che lo stadio 2 affronta, non dello
stadio 1.

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

## AInCP — i pazienti (`Dataset504`, `505`, `506`)

Il bersaglio. **17 pazienti** in due lotti consegnati da AInCP ed estratti in `data/raw/AINCP/`
(primo lotto `dataset_FSM_v1`, 11/9, dall'archivio `.7z`; secondo `dataset_FSM_v2`, 1/10, da FileSender
GARR). Centro, campo ed età nel foglio `docs/AINCP_SERIES_TYPE_dataset_FSM_v2.xlsx`, foglio
`SITE & MRFIELD` (e `Sheet1` per il secondo lotto). **Si usano solo i lotti consegnati da AInCP**: la
copia di FSM_028 trovata in `/data01/fsm_mri_data/`, di provenienza ignota, è esclusa (FSM_028 è poi
arrivato regolarmente nel secondo lotto).

| strato | pazienti | età |
| --- | --- | --- |
| Stella Maris, 3 T | FSM_032, 039, 040, 041, 045, 046, 052, 053, 054, 056 | 3–15 anni |
| Stella Maris, 1.5 T | FSM_028, 030, 033, 035, 036 | 3–5 anni |
| altri centri, 1.5 T | FSM_001 (Niguarda), FSM_049 (Burlo Garofolo) | 7 e 12 anni |

**Formato.** Per ogni paziente `<sessione>/COREGISTERED/`: T1 e FLAIR originali, **non allineate**
fra loro, e una ROI per fenotipo **disegnata sulla FLAIR** (`ROI_type_definition.txt`, versione del
19 maggio 2026): `ROI_1` gliosi (iperintensità in FLAIR), `ROI_2` lesione cistica, `ROI_3` ventricolo
dell'emisfero leso, `ROI_4` encefalomalacia (perdita di tessuto non cistica). **L'etichetta è
`ROI_1 | ROI_2 | ROI_4`; il ventricolo resta fuori** per decisione del relatore. Non tutti i pazienti
hanno tutte le ROI.

**Problemi trovati e decisioni:**

- **Componenti fino a 1 mm³** (spesso un singolo voxel del tracciato): residui, per indicazione
  clinica (23/9); tolte dall'etichetta prima di ogni trasporto e ignorate nelle metriche. Elenco per il
  clinico, con paziente, file, fetta e coordinate: `reports/residui_annotazione.md` (46 componenti,
  12 fuori dal cervello).
- **Nomi irregolari**: FSM_052 ha `ROI_1 - Copia.nii` come unica ROI_1 (la ricerca usa una regex);
  FSM_030 ha **due ROI_1 diverse** (Dice 0.72 fra loro, differenze ai bordi): **si usa `ROI_1.nii`**,
  decisione dell'utente del 1/10 dopo il confronto in `scripts/figura_roi_fsm030.py`. FSM_049 contiene anche file già elaborati da altri, ignorati.
- **Età sotto i 5 anni** in 4 pazienti (FSM_032, 028, 035, 036), che il foglio segna fuori dal
  criterio d'inclusione dello studio clinico: inclusi (decisione dell'utente, 1/10), da dichiarare.
- **Spazio**: i pazienti si registrano sul template **MNIPediatricAsym cohort-1**, che ha la stessa
  griglia di ATLAS; le misure si riportano nel nativo. La proporzione di cervello lesionato si conserva
  (mediana −10%), ma localmente la registrazione può ingrandire una lesione più del cervello (FSM_030,
  FSM_033: 2.7 volte contro 1.5–1.9, lesioni periventricolari).

Pipeline e comandi: `scripts/pipeline_aincp.py` (procedura in `obiettivo-e-piano.md`).

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

## ATLAS: una parte dei casi non è risolvibile con il solo T1

Scoperto esaminando i casi falliti del primo training invece di mediarli con gli altri
(`scripts/inspect_failures.py` → [`reports/failure_analysis_502.md`](../reports/failure_analysis_502.md)).
Nel fold 0, **10 casi su 112 hanno Dice ≤ 0.05**. Tre ipotesi sono state testate e due escluse.

**Non è un errore di lateralità.** Il test dello specchio — confronto fra la distanza dei centroidi e la
distanza dal riflesso rispetto al piano sagittale mediano — marca **un solo caso** su 10 come predizione
controlaterale. Nessun problema sistematico di orientamento o di flip in augmentation.

**Non è un problema di soglia.** Le probabilità che il modello assegna ai voxel della lesione vera hanno
**mediana 0.0000** (massimo 0.03 nel caso peggiore). Il modello non è incerto: è convinto che non ci sia
nulla. Abbassare la soglia a 0.1 non recupera niente, il Dice resta a zero. Anche il numero di voxel
predetti è minimo: 217 contro 9586 di riferimento nel caso peggiore, cioè il modello **non segmenta**,
non "segmenta nel posto sbagliato".

**È il contrasto.** Rapporto fra intensità media dentro la lesione annotata e tessuto cerebrale attorno:

| | volume delle lesioni | rapporto lesione / cervello |
| --- | --- | --- |
| 10 casi falliti | 0.04 – 9.6 ml | **1.07** (mediana) |
| 6 casi migliori | 85 – 231 ml | **0.62** (mediana) |

Le due popolazioni sono separate su entrambe le dimensioni. Dove il modello riesce, la lesione è una
**cavità**: perdita di tessuto sostituita da liquido, quindi molto ipointensa su T1 e inequivocabile.
Dove fallisce, la regione annotata ha la stessa intensità del tessuto sano che la circonda — su T1 non
c'è niente da vedere. Sono lesioni senza cavitazione (gliosi, ischemia cronica lieve), il tipo che si
riconosce su FLAIR o T2 e non su T1.

**Conseguenze.** Nessuna correzione da fare alla pipeline: non è un bug. Ma tre implicazioni:

- il Dice medio di ATLAS (0.610) include una frazione di casi **intrinsecamente non risolvibili** con il
  canale disponibile: non è un margine recuperabile con più training o più epoche;
- è un argomento indipendente **a favore della FLAIR**, che l'ablazione 511 contro 501 misura
  direttamente;
- riguarda il bersaglio finale più di quanto sembri. Nella paralisi cerebrale molte lesioni della
  sostanza bianca non sono cavitate, quindi il limite osservato qui potrebbe presentarsi su AInCP —
  **da verificare quando i dati arriveranno**, non dimostrato.

**Limite della misura:** confronta intensità *medie*. Una lesione può essere riconoscibile per texture o
asimmetria senza differire in media, quindi la conferma vera richiede un'ispezione visiva dei casi.

## DBB non contiene maschere di lesione: verificato

Dopo l'incontro del 27 agosto è stato chiesto di ricontrollare se DBB fornisse annotazioni di lesione o
malformazione, dato che è pediatrico e patologico. **Verificato su tutti i 244 casi con immagine: non le
contiene.**

La verifica è esaustiva, non a campione: `mask` vale `1` e nient'altro in **244 casi su 244**, e
`parcellation-volume` contiene esattamente `1,2,3,4,5,6` in **244 su 244** — con gli stessi identici sei
valori nei 37 patologici e nei 207 sani. Lo conferma anche la descrizione ufficiale del progetto:
«benchmark to segment the brain **tissues** of distorted brains», dove *distorted* qualifica l'anatomia
dei pazienti, non l'esistenza di annotazioni di lesione.

Su brainlife la pubblicazione espone cinque tipi di dato, e sono tutti presenti in `data/raw/DBB/`:

| tipo | contenuto reale |
| --- | --- |
| `anat-t1w` | l'immagine T1 |
| `mask` | maschera **cerebrale binaria** (unico valore `1`), stesso volume della parcellazione |
| `parcellation-volume` | i 6 tessuti, valori `1..6` |
| `mask_raw`, `transform-nifti_linear` | solo per i 711 soggetti **senza immagine**, quindi inutilizzabili |

Il controllo decisivo: le parcellazioni dei casi **patologici** contengono esattamente gli stessi valori
`1..6` di quelle dei sani. Non esiste una classe aggiuntiva per la malformazione, e la `mask` è una
maschera di cervello, non di lesione. Nel benchmark **la patologia è implicita nell'anatomia**: si
addestra sul sano e si verifica se la parcellazione tiene su anatomie alterate.

**Una conseguenza che è facile non vedere:** DBB **non può essere usato come insieme di casi negativi**
in un training sulle lesioni. Darlo alla rete come "cervelli senza lesione" le insegnerebbe che le
anatomie malformate sono normali — cioè esattamente il contrario di ciò che serve sul bersaglio finale,
dove le anatomie alterate sono la norma e vanno esaminate, non liquidate.

**Conseguenza:** DBB resta utile per tre cose — pretraining dell'encoder sul contrasto tissutale del T1
pediatrico, test di generalizzazione anatomica, e parcellazione per lo stadio 2 — ma **non può fornire
esempi di lesione**, quindi non può entrare in un dataset unificato che abbia "lesione" come bersaglio.

## Campo magnetico: dove sono i dati a 1.5 T

Il campo non era stato censito. I metadati di ATLAS
(`20211112_ATLAS_2.0_SupplementaryInfo.csv`, prima riga di ogni coorte) lo dichiarano per tutte e 33 le
coorti; su WMH lo dice il nome del sito.

| | casi a 1.5 T | su totale |
| --- | --- | --- |
| WMH | 10 (`Amsterdam_GE1T5`) | 99 |
| ATLAS | 52 nei casi annotati (coorti `R029`, `R039`, `R040`) | 655 |
| DBB | non dichiarato nei dati disponibili | — |

**La misura preliminare non giustifica l'esclusione.** Nel fold 0 di ATLAS i 9 casi a 1.5 T presenti in
validation hanno Dice medio **0.653**, contro **0.607** dei 103 casi a 3 T: vanno leggermente *meglio*,
non peggio. E solo 1 dei 10 casi falliti è a 1.5 T.

Due precisazioni che tengono la questione aperta:

- con 9 casi in un solo fold il confronto è debole, e va rifatto sui 5 fold prima di concludere;
- l'argomento per escluderli non è necessariamente che vadano male **su di sé**, ma che introducano
  variabilità di dominio che danneggia il modello nel complesso. È una tesi diversa, e si verifica solo
  addestrando **senza** quei casi e confrontando il risultato sugli stessi fold — cioè come dataset
  separato con un nuovo ID, non cancellando dati.

Da notare, perché è una coincidenza che vale la pena tenere d'occhio: le coorti a 1.5 T sono anche quelle
dei due casi anomali già noti — `R039` è la coorte del caso codificato `0/0.01`, `R040` quella del caso
con 9586 voxel di lesione che il modello non vede affatto.

## SynthStrip: funziona, ma taglia il CSF nei ventricoli dilatati

Validato su **tutti i 244 casi di DBB** contro le maschere cerebrali di riferimento
(`scripts/run_synthstrip.py` → [`reports/synthstrip_dbb.md`](../reports/synthstrip_dbb.md)). Modello
principale `synthstrip.1.pt`, quello con CSF, che è l'unico confrontabile con il riferimento di DBB.

| gruppo | n | Dice medio | minimo |
| --- | --- | --- | --- |
| sani | 207 | 0.9673 | 0.9476 |
| ACC | 10 | 0.9675 | 0.9621 |
| MCD | 10 | 0.9605 | 0.9284 |
| PFM | 11 | 0.9604 | 0.9365 |
| **HD** (idrocefalo) | 6 | **0.9267** | **0.8791** |

**La risposta alla domanda posta è sì:** sui malformati SynthStrip perde solo **0.010** rispetto ai sani
(0.9569 contro 0.9673). Su ACC è indistinguibile dai sani. È il contrario di quanto ci si aspetta da BET,
ed è il motivo per cui era stato indicato come alternativa.

**Ma c'è un pattern sistematico che ci riguarda da vicino.** L'unico gruppo sotto 0.95 è l'idrocefalo, e
l'errore ha una direzione precisa: SynthStrip **esclude** volume che il riferimento include, e quel volume
è quasi tutto **liquido cerebrospinale**.

| caso | escluso da SynthStrip | di cui CSF |
| --- | --- | --- |
| `sub-0955` (HD) | 423 ml | **400 ml** |
| `sub-0924` (HD) | 353 ml | 209 ml |
| `sub-0958` (HD) | 251 ml | 132 ml |
| `sub-0922` (MCD) | 182 ml | 85 ml |

Nei cervelli normali il comportamento si inverte: SynthStrip è leggermente più generoso e include
100–160 ml di tessuto fuori dal cervello (dura madre e spazi subaracnoidei).

**Perché è un rischio per il bersaglio finale.** Nella paralisi cerebrale la dilatazione ventricolare è
frequente — esiti di emorragia intraventricolare, dilatazione *ex vacuo* nella leucomalacia
periventricolare. Se la maschera venisse usata per **ritagliare** le immagini, si rischierebbe di
rimuovere proprio le regioni periventricolari, dove stanno le lesioni tipiche di quella popolazione.

Due conseguenze operative:

- per l'uso come **`ignore` label** (variante C) il rischio è nullo: un voxel escluso per errore resta
  `0`, cioè il comportamento che si avrebbe senza skull stripping;
- per un **dataset skull-stripped** andava verificato **quanta parte delle lesioni annotate cadrebbe
  fuori dalla maschera**. Questa verifica è stata fatta, ed è il paragrafo seguente.

### Quante lesioni si perderebbero ritagliando: praticamente nessuna

`scripts/lesioni_fuori_maschera.py` applica SynthStrip a tutti i 79 casi WMH e a 100 casi ATLAS
campionati, e misura la frazione di voxel di lesione annotata che cade fuori dalla maschera
(`reports/lesioni_fuori_maschera.md`).

| dataset | casi | frazione media | massima | casi oltre 1% | casi oltre 5% |
| --- | --- | --- | --- | --- | --- |
| WMH (FLAIR) | 79 | 0.0000 | 0.0000 | 0 | 0 |
| ATLAS (T1) | 556 | 0.0012 | 0.0649 | 18 | 3 |

Su WMH la perdita è **esattamente zero in tutti e 79 i casi**. Su ATLAS la media è lo 0.12%, ma la coda
non è vuota: 18 casi su 556 perdono oltre l'1% e **tre superano il 5%**, con un massimo del 6.5%.

Sono quasi tutti infarti massivi (196–360 ml) che arrivano alla superficie corticale, dove il confine
stesso di "cervello" è ambiguo: i voxel persi stanno a **1–2 mm mediani dal bordo della maschera**,
cioè sono rifilatura di superficie, non asportazione. L'unica eccezione è `sub-r028s014`, una lesione
da 8.6 ml che ne perde il 5.5% — cioè 0.47 ml.

Una misura su 100 casi campionati dava media 0.0007 e nessun caso oltre il 5%: la coda esiste solo nel
campione completo, ed è il motivo per cui la misura è stata rifatta su tutti e 556.

**Quindi il dataset skull-stripped si può costruire.** Con una cautela che resta aperta: questa misura
è su adulti senza idrocefalo. Sui pazienti AInCP, dove la dilatazione ventricolare è attesa, va
ripetuta prima di fidarsi — è lì che il taglio del CSF descritto sopra potrebbe mordere.

## Cosa deliberatamente non facciamo

Decisioni prese, non dimenticanze. Vanno riaperte solo con una misura in mano.

| Non facciamo | Perché |
| --- | --- |
| **Resampling manuale** | Lo fa nnU-Net una volta sola. Anticiparlo produce una doppia interpolazione che degrada le lesioni piccole — proprio quelle già penalizzate dall'anisotropia. |
| **Z-score a mano** | nnU-Net lo applica per immagine dopo il crop, con statistiche calcolate sul solo training. Farlo prima significherebbe calcolarle anche sul test, cioè leakage. |
| **N4** | Misurato: campo piatto dentro il cervello su tutti e tre i dataset. |
| **FSL BET per lo skull stripping** | BET fa crescere una superficie deformabile guidata da soglie, assumendo una testa regolare, ed è tarato sul T1. Fallisce nei due casi che ci riguardano: FLAIR e anatomie alterate. Al suo posto si usa **SynthStrip**, misurato sulle 244 maschere di DBB e tenuto anche sui malformati (0.957); **HD-BET** non è più stato necessario. Su DBB lo strumento non serve comunque: la maschera c'è già. |
| **Riorientare WMH o ATLAS** | L'orientamento è uniforme dentro ciascuno e l'affine lo descrive. Su DBB invece era mescolato, e lì il riorientamento è attivo. |
| **Riempire buchi o pulire le maschere** | La ground truth è il riferimento per definizione. "Migliorarla" cambia il compito e rende i risultati incomparabili. |
| **Scartare i casi con lesioni minuscole** | Sono casi clinici legittimi e spesso i più difficili. Rimuoverli gonfia le metriche senza migliorare il modello. |

Il filo comune: **il preprocessing deve rendere i dati utilizzabili, non "migliori"**. Ogni trasformazione
in più è un'occasione in più di errore silenzioso o di risultati non confrontabili.

Una regola trasversale: ogni variante di preprocessing va costruita come **dataset separato con un nuovo
ID** (come è stato fatto per `Dataset511_WMH_T1`), non sovrascrivendo l'originale, va confrontata **sugli
stessi fold**, e va introdotta **una alla volta** — se due cambiano insieme e il Dice si muove, non sapremo
quale l'ha mosso.
