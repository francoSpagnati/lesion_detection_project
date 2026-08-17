# Primer: risonanza magnetica e segmentazione di lesioni

Documento introduttivo per chi arriva a questo progetto senza background in neuroimaging.
Non presuppone niente sulla risonanza magnetica. Serve a capire **perché** il preprocessing di questo
progetto fa quello che fa: quasi ogni scelta descritta in
[dataset-e-decisioni.md](dataset-e-decisioni.md) diventa ovvia una volta noto il contesto fisico,
e arbitraria senza.

---

## 1. Che cos'è un'immagine di risonanza magnetica

Una risonanza magnetica (MRI) non funziona come una radiografia o una TC. Non misura quanto un tessuto
"blocca" una radiazione. Misura invece come si comportano i nuclei di idrogeno dell'acqua nel corpo
quando vengono messi in un campo magnetico forte e poi "disturbati" con un impulso di radiofrequenza:
dopo l'impulso i nuclei tornano allo stato iniziale, e il modo in cui ci tornano dipende dal tessuto in
cui si trovano.

Da questo discende **la proprietà più importante da capire**, quella che spiega metà del preprocessing:

> I valori dei voxel di una MRI **non hanno un'unità di misura assoluta**.

In una TC il valore di un voxel è espresso in unità di Hounsfield: l'acqua vale 0, l'osso circa +1000,
sempre, su qualunque macchina. Un algoritmo può ragionare direttamente su quei numeri.

In una MRI no. Il valore `600` non significa niente di per sé: dipende dallo scanner, dalla taratura,
dalla sequenza usata, dal soggetto, e persino dalla posizione dentro la macchina. Due immagini dello
stesso cervello fatte da due scanner diversi possono avere intervalli di intensità completamente
diversi pur mostrando esattamente la stessa anatomia.

**Conseguenza pratica**: è impossibile addestrare una rete su intensità grezze e sperare che generalizzi.
Bisogna prima riportare ogni volume su una scala comune. È esattamente ciò che fa la
**normalizzazione z-score** (sottrai la media, dividi per la deviazione standard), che nnU-Net applica
di default alle modalità MR. Solo per la TC nnU-Net usa una strategia diversa basata sui percentili,
perché lì i numeri hanno un significato assoluto da preservare.

Questo è anche il motivo per cui, nel `dataset.json`, **il nome del canale conta**: scrivere `"FLAIR"` o
`"T1"` fa scegliere a nnU-Net la normalizzazione z-score, mentre scrivere `"CT"` ne farebbe scegliere
un'altra.

---

## 2. Modalità e sequenze: T1, T2, FLAIR

Cambiando i parametri dell'impulso si ottengono immagini con contrasti diversi **dello stesso cervello**.
Non sono immagini "migliori" o "peggiori": mettono in evidenza cose diverse.

| Sequenza | Come appare | A cosa serve |
| --- | --- | --- |
| **T1-pesata** (T1w) | Sostanza bianca chiara, sostanza grigia più scura, liquido nero | Anatomia, struttura, perdita di tessuto |
| **T2-pesata** (T2w) | Il liquido appare **bianco brillante** | Edema, infiammazione, accumuli di liquido |
| **FLAIR** | Come la T2, ma con il liquido cerebrospinale **soppresso** (reso nero) | Lesioni della sostanza bianca |

FLAIR sta per *FLuid Attenuated Inversion Recovery*, cioè "recupero con inversione ad attenuazione del
fluido". Il problema che risolve è questo: nelle immagini T2 le lesioni della sostanza bianca sono
brillanti, ma lo è anche il liquido cerebrospinale che riempie i ventricoli, e molte di quelle lesioni
stanno proprio **attorno ai ventricoli**. Su una T2 lesione e liquido si confondono. La FLAIR spegne il
segnale del liquido, e le lesioni restano gli unici oggetti brillanti su fondo scuro.

**Ecco perché in questo progetto**:

- Per **WMH** (iperintensità della sostanza bianca) il canale primario è la **FLAIR**: è la sequenza
  progettata apposta per rendere visibili quelle lesioni. Non è una preferenza, è la scelta fisicamente
  corretta.
- Per **ATLAS** (lesioni da ictus croniche) si usa la **T1w**: dopo mesi da un ictus il tessuto colpito
  è morto e riassorbito, lasciando una cavità piena di liquido. Su T1 appare come un buco nero ben
  delimitato, quindi la T1 basta.

### Più canali = più informazione

Una rete di segmentazione tratta le sequenze come **canali**, esattamente come i canali R, G, B di una
foto a colori. La differenza è che qui ogni canale è una fisica diversa, non un colore: dare alla rete
FLAIR + T1 insieme le permette di incrociare "questo punto è brillante sulla FLAIR" con "questo punto è
sostanza bianca sulla T1", e distinguere così una lesione vera da un artefatto.

Nel progetto questo è il parametro `WMH_USE_T1`: canale `0000` = FLAIR, canale opzionale `0001` = T1.
Perché due canali funzionino devono però essere **perfettamente sovrapposti** — vedere il punto 5.

---

## 3. Il volume 3D: voxel, shape, spacing

### Da pixel a voxel

Un'immagine normale è una griglia 2D di **pixel**. Una MRI è una griglia 3D di **voxel** (*volume
element*): la stessa idea con una dimensione in più. Il cervello viene acquisito a **fette** (*slice*)
impilate una sull'altra.

Un volume MRI è quindi, per il codice, semplicemente un array NumPy tridimensionale di numeri.

### Le tre grandezze da non confondere

| Grandezza | Cos'è | Unità | Esempio (Utrecht) |
| --- | --- | --- | --- |
| **shape** | quanti voxel per lato | numeri puri | `240 x 240 x 48` |
| **spacing** | quanto è grande **fisicamente** un voxel | millimetri | `0.958 x 0.958 x 3.0` |
| **FOV** | quanto spazio reale copre il volume (`shape x spacing`) | millimetri | `230 x 230 x 144` |

**Lo spacing è il concetto centrale di tutto questo progetto.**

Il motivo è che la shape da sola non dice assolutamente niente sul contenuto fisico. Un'analogia: due
foto della stessa stanza, una da 500x500 pixel e una da 5000x5000. Hanno shape diverse ma inquadrano la
stessa scena; quello che cambia è il **dettaglio per unità di spazio reale**. Nelle immagini normali
questa informazione non è memorizzata (un pixel non ha dimensione fisica); nelle immagini mediche sì, ed
è lo spacing.

Da qui discendono due regole che nel progetto sono ovunque:

1. **Un volume di lesione va calcolato in millimetri cubi, mai in voxel.** Contare 1000 voxel non
   significa niente finché non sai quanto è grande un voxel. In questo progetto un voxel WMH vale circa
   3.5 mm3 e uno ATLAS 1.0 mm3: gli stessi 1000 voxel sono 3.5 ml in un dataset e 1.0 ml nell'altro. Per
   questo il manifest registra `lesion_ml` e non solo il conteggio.
2. **Per confrontare o unire volumi diversi bisogna prima portarli allo stesso spacing.** Questa
   operazione si chiama **resampling**.

---

## 4. Anisotropia e volume parziale

### Voxel cubici e voxel a mattoncino

Se lo spacing è `1.0 x 1.0 x 1.0` mm il voxel è un **cubo**: si dice che l'immagine è **isotropa**
(uguale in tutte le direzioni). È il caso di ATLAS.

Se lo spacing è `1.0 x 1.0 x 3.0` mm il voxel è un **parallelepipedo** allungato: tre volte più lungo
nella direzione perpendicolare alle fette che nel piano della fetta. Si dice **anisotropo**. È il caso di
WMH, dove le fette sono spesse **3 mm** su tutti e quattro i siti.

Perché succede? Acquisire fette sottili richiede più tempo dentro lo scanner, e il tempo di scanner costa
e stanca il paziente. Fette da 3 mm sono un compromesso clinico standard: si tiene alta la risoluzione
nel piano (dove si guarda) e si risparmia sul numero di fette.

L'anisotropia si misura come `max(spacing) / min(spacing)`. Sopra 3 gli strumenti la trattano come un
caso speciale — vedere il punto 9.

### L'effetto di volume parziale

Questa è la conseguenza che conta davvero, ed è la ragione per cui il progetto insiste tanto sulle
lesioni piccole.

Ogni voxel restituisce **un solo numero**, che è la media del segnale di tutto ciò che sta dentro quel
voxel. Se un voxel è spesso 3 mm e contiene una lesione di 1 mm circondata da tessuto sano, il valore
registrato non è "lesione": è una **media** fra lesione e tessuto sano. La lesione appare sbiadita, con
un contrasto molto minore del reale, o sparisce del tutto.

Questo si chiama **effetto di volume parziale**, ed è informazione **persa in acquisizione**: nessun
algoritmo di preprocessing può recuperarla, perché non è mai stata registrata. È il limite fisico contro
cui si scontra la segmentazione di piccole lesioni su dati anisotropi.

È anche il motivo per cui, su WMH, la configurazione **2d** di nnU-Net è una candidata seria e non una
formalità: se le fette sono lontane 3 mm l'una dall'altra, il contesto tridimensionale che una rete 3D
può sfruttare è scarso, e guardare le fette una a una può essere altrettanto efficace.

---

## 5. Geometria: dove si trova un voxel nello spazio

### La matrice affine

Un array 3D di numeri, da solo, non dice dove si trovi il cervello nello spazio, in che verso sia
orientato, né quanto sia grande. Queste informazioni stanno nell'**affine**: una matrice 4x4, salvata
nell'header del file NIfTI, che converte gli **indici del voxel** in **coordinate fisiche in
millimetri**.

```
(i, j, k)  --[ affine ]-->  (x, y, z) in mm nel mondo reale
```

L'affine contiene tutto insieme: le dimensioni del voxel (lo spacing), la rotazione del volume e la sua
posizione. È il ponte fra "l'elemento `[100, 120, 30]` dell'array" e "un punto fisico dentro la testa del
paziente".

### Orientamento: cosa vogliono dire RAS e LPS

Quando si costruisce l'array, bisogna decidere in che verso crescono i tre assi. Le convenzioni si
scrivono con tre lettere, una per asse, che indicano **verso dove punta il verso positivo**:

- **R**ight / **L**eft — destra / sinistra
- **A**nterior / **P**osterior — davanti / dietro
- **S**uperior / **I**nferior — sopra / sotto

Quindi **RAS** significa: il primo asse cresce verso destra, il secondo verso il davanti, il terzo verso
l'alto. **LPS** significa che i primi due assi vanno nel verso opposto. Sono entrambe convenzioni valide
e diffuse (LPS è quella di DICOM e della radiologia, RAS quella di molti strumenti di neuroimaging).

In questo progetto WMH è tutto `LPS` e ATLAS è tutto `RAS`. **Non è un problema**, perché ogni dataset è
internamente coerente e viene trattato separatamente; l'affine descrive la convenzione, quindi il
software sa sempre interpretare i dati. Diventerebbe un problema solo se l'orientamento variasse
*dentro* lo stesso dataset, o se si volessero mescolare i due — ed è per questo che il notebook lo
verifica esplicitamente.

### Perché immagine e label devono avere lo stesso affine

La maschera di lesione (la label) è un volume della stessa forma dell'immagine, dove ogni voxel dice se
lì c'è lesione. Perché la corrispondenza sia valida, immagine e label devono descrivere **la stessa
griglia**: stessa shape e stesso affine. Se l'affine differisce anche di poco, il voxel `[100, 120, 30]`
dell'immagine e quello della label puntano a due posti fisici diversi, e la rete impara una
corrispondenza sfasata — silenziosamente, senza alcun errore.

Per questo il notebook di preparazione fa un `assert` sulla coerenza di shape e affine per ogni caso, e
riscrive le label **usando l'affine dell'immagine** invece del proprio: così le due geometrie sono
identiche bit per bit e nnU-Net non può nemmeno emettere un warning di mismatch.

### Registrazione e spazio standard

**Registrazione** (*registration*) è l'operazione che allinea due volumi, cioè calcola e applica la
trasformazione che porta l'uno sulla griglia dell'altro. Nel dataset WMH è già stata fatta: il file
`*_T1w_BiasCorr_2FLAIR_Aligned.nii.gz` è la T1 **già portata sulla griglia della FLAIR** (il `2FLAIR`
nel nome vuol dire "verso la FLAIR"). Solo quel file è utilizzabile come secondo canale; la T1 originale
`*_T1w_BiasCorr.nii.gz` ha una griglia diversa e non è sovrapponibile.

**Spazio standard**: i cervelli hanno forme diverse. Per confrontarli si può deformare ciascuno su un
template comune, il più noto dei quali è lo spazio **MNI**. ATLAS è distribuito già in spazio MNI: è per
questo che tutti i 955 volumi hanno esattamente la stessa shape `197 x 233 x 189` e lo stesso spacing 1
mm. Sono stati tutti "raddrizzati" sullo stesso stampo.

Questo rende ATLAS molto più semplice da trattare di WMH, che è in **spazio nativo** — cioè ogni volume è
come è uscito dallo scanner, con la sua geometria.

---

## 6. Che cosa significa "segmentare una lesione"

**Segmentazione** = classificare **ogni singolo voxel**. Non "c'è una lesione in questa immagine?"
(classificazione) e nemmeno "dov'è, più o meno?" (detection), ma "questo preciso voxel è lesione, sì o
no?", ripetuto per tutti i milioni di voxel del volume.

L'output è quindi un volume della stessa forma dell'input, contenente 0 (background) e 1 (lesione). Lo si
chiama **maschera** o **label**.

La verità di riferimento (*ground truth*) è disegnata a mano da esperti, fetta per fetta. Due
conseguenze importanti:

- è un lavoro lentissimo, e questo spiega perché i dataset annotati siano piccoli (99 casi per WMH);
- **esperti diversi disegnano confini diversi**. Questa *variabilità inter-rater* è un limite superiore
  di fatto alle prestazioni: un modello che concorda con l'annotatore quanto due annotatori concordano
  fra loro è, in pratica, al massimo di ciò che si può misurare.

---

## 7. Perché è un problema fortemente sbilanciato

Un cervello adulto è circa 1200-1500 ml. Una lesione mediana in questi dataset è circa 5-11 ml. Vuol
dire che **meno dell'1% dei voxel** è lesione.

Un modello che predicesse "background" ovunque, sempre, avrebbe oltre il 99% di accuratezza per voxel.
L'accuratezza è quindi una metrica inutile qui, e serve qualcosa che guardi solo alla classe rara.

### Il coefficiente di Dice

La metrica standard in segmentazione medica è il **Dice**:

```
Dice = 2 * (voxel in comune fra predetto e vero) / (voxel predetti + voxel veri)
```

Vale 1 con sovrapposizione perfetta e 0 se non c'è alcuna sovrapposizione. Essendo definito solo sui
voxel di lesione (predetti e veri), ignora del tutto l'oceano di background e non è ingannato dallo
sbilanciamento.

### Perché il Dice da solo non basta

Il Dice ha due difetti seri proprio su questi dataset, ed è la ragione per cui
[training.md](training.md#metriche) chiede metriche aggiuntive.

**È instabile sulle lesioni piccole.** Su una lesione da 20 voxel, sbagliarne 10 fa crollare il Dice;
sulla stessa lesione da 20.000 voxel, sbagliarne 10 non si vede. Il Dice di un caso con lesione minuscola
è quasi rumore. Nel progetto, 19 casi ATLAS su 655 hanno meno di 100 voxel di lesione alla risoluzione
target: su quelli il valore è instabile per costruzione, ed è per questo che il Dice medio va sempre
letto insieme alla sua distribuzione.

**Pesa i voxel, non le lesioni.** In WMH un paziente ha tipicamente molte piccole lesioni sparse più
qualche placca grande. Il Dice è dominato dalle placche grandi: un modello che trovasse *solo* quelle e
mancasse tutte le piccole avrebbe comunque un Dice alto. Ma clinicamente contare le lesioni conta. Per
questo servono **metriche per lesione** (quante lesioni distinte sono state trovate, quante inventate),
che il Dice per voxel non cattura.

---

## 8. Il preprocessing, passo per passo

Ora ogni operazione della pipeline dovrebbe avere una motivazione riconoscibile.

| Operazione | Cosa fa | Perché serve |
| --- | --- | --- |
| **Correzione del bias field** | Rimuove una lenta variazione di luminosità sull'immagine | Le bobine dello scanner non ricevono in modo uniforme: lo stesso tessuto appare più chiaro al centro e più scuro ai bordi. Confonde qualunque metodo basato sull'intensità. **In WMH è già stata fatta**: è il `_BiasCorr` nei nomi dei file. |
| **Registrazione** | Allinea volumi diversi sulla stessa griglia | Serve per usare più canali. **In WMH è già fatta** (`_2FLAIR_Aligned`). |
| **Cropping** | Elimina il bordo di sfondo vuoto | Nel volume c'è molta aria attorno alla testa: sprecare voxel lì significa sprecare memoria GPU. Lo fa nnU-Net da solo. |
| **Resampling** | Porta tutti i volumi allo stesso spacing | La rete impara su una griglia fissa; se i casi hanno spacing diversi, la stessa lesione ha dimensioni diverse in voxel da caso a caso. Lo fa nnU-Net. |
| **Normalizzazione** | Riporta le intensità su una scala comune (z-score) | Il punto 1: i valori MRI non hanno unità assolute. Lo fa nnU-Net. |

### Resampling: due interpolazioni diverse

Nel resampling bisogna calcolare valori in posizioni intermedie fra i voxel originali. Il metodo va
scelto in base a **cosa** si sta interpolando:

- **Immagini**: interpolazione continua (lineare, cubica). Se fra un voxel da 100 e uno da 200 il valore
  intermedio è 150, va benissimo: le intensità sono grandezze continue.
- **Label**: interpolazione **nearest neighbour** (al vicino più prossimo). Fra un voxel "background"
  (0) e uno "lesione" (1) il valore intermedio 0.5 **non significa niente**: non esiste il mezzo
  tessuto. Le etichette sono categorie, e interpolarle in modo continuo produce valori privi di senso.

### Due regole del progetto che ora si spiegano da sole

**Non resamplare a mano prima di nnU-Net.** Ogni interpolazione sfoca leggermente l'immagine. Farlo due
volte (una a mano, una da nnU-Net) sfoca due volte, e a rimetterci sono esattamente le lesioni piccole,
già penalizzate dal volume parziale. Il notebook quindi **misura** la risoluzione e **prevede** cosa farà
il planner, ma non tocca i dati.

**Le label vanno convertite in `uint8`.** Le etichette sono categorie, non misure: servono numeri interi
piccoli. Salvare una maschera in virgola mobile occupa 4 volte lo spazio e apre la porta a confronti di
uguaglianza che falliscono per errori di arrotondamento — un problema che in questo progetto si è
presentato davvero (vedere [dataset-e-decisioni.md](dataset-e-decisioni.md)).

---

## 9. Come si approccia il problema: nnU-Net

### Che cos'è

nnU-Net **non è un'architettura di rete**. È una *ricetta auto-configurante*: dato un dataset, ne analizza
le proprietà e decide da solo tutti gli iperparametri, poi addestra una U-Net standard. La U-Net in sé è
un'architettura del 2015, non particolarmente esotica.

L'intuizione dietro nnU-Net è che, in segmentazione medica, quasi tutto il guadagno viene dal **fare bene
le cose noiose** — preprocessing, spacing, patch size, augmentation, post-processing — e non da
architetture nuove. È stato validato vincendo decine di challenge pubbliche contro modelli progettati a
mano, e per questo è la baseline giusta con cui iniziare: non perché sia insuperabile, ma perché batterlo
è la prova che una tua idea funziona davvero.

### Le fasi

1. **Fingerprint** — nnU-Net legge tutti i casi di training e ne estrae le proprietà: spacing, shape,
   distribuzione delle intensità, dimensione della regione non vuota.
2. **Planning** — dal fingerprint decide la target spacing, la patch size, la batch size, quanti livelli
   di downsampling, e quali configurazioni proporre. Il risultato è `nnUNetPlans.json`.
3. **Preprocessing** — applica il piano ai dati: crop, resampling, normalizzazione, e salva su disco.
4. **Training** — 5 modelli, uno per fold (vedi sotto).
5. **Inferenza** — applica i modelli a dati nuovi.

Il punto 1 è anche il motivo per cui il **test set non deve stare in `imagesTr`**: se ci stesse,
entrerebbe nel fingerprint, e le statistiche di normalizzazione e le scelte di pianificazione sarebbero
in parte derivate da dati che dovrebbero essere invisibili. È una forma sottile di *leakage*.

### La regola della target spacing

Vale la pena capirla perché è la decisione che determina la risoluzione a cui il modello lavora.

1. Prende la **mediana** degli spacing per ogni asse.
2. Guarda l'asse con lo spacing peggiore (il più grande).
3. Applica una correzione **solo se valgono entrambe** queste condizioni: quell'asse è più di 3 volte
   peggiore degli altri, **e** ha anche molti meno voxel degli altri.
4. In quel caso usa il 10o percentile invece della mediana, recuperando risoluzione dove i dati ce
   l'hanno.

Su WMH la prima condizione **non** scatta: `3.0 > 3 * 1.2 = 3.6` è falso. La target spacing resta quindi
la mediana, `1.2 x 0.9766 x 3.0` mm. Ma **separatamente** nnU-Net decide *come* resamplare, e lì usa il
semplice rapporto `max/min` degli spacing: `3.0 / 0.9766 = 3.07`, che supera 3. Quindi WMH riceve il
resampling con asse z trattato a parte, che evita di spalmare le lesioni fra fette adiacenti.

Sono **due decisioni indipendenti** con soglie simili, ed è facile confonderle: WMH non riceve
l'override della spacing ma riceve comunque il resampling separato.

### Training a patch

Un volume da 8,7 milioni di voxel non entra nella memoria di una GPU insieme alla rete e ai gradienti.
Perciò l'addestramento non usa volumi interi ma **patch**: ritagli 3D di dimensione fissa, estratti a caso
dai volumi, con un bilanciamento che garantisce che una parte contenga lesione (altrimenti, con meno
dell'1% di voxel positivi, quasi tutte le patch sarebbero solo background). In inferenza il volume viene
ricoperto con una finestra scorrevole e le predizioni ricucite.

Questo spiega perché la **patch size** nel piano è un numero da guardare: su WMH, con circa 83 fette,
indica quanto contesto attraverso il piano il modello riesce effettivamente a vedere.

### Cross-validation a 5 fold

Invece di un solo split train/validation, nnU-Net divide i dati in 5 parti e addestra 5 modelli: ognuno
usa 4 parti per addestrare e 1 per validare. Ogni caso finisce nella validation esattamente una volta.

I vantaggi: si ottiene una stima di prestazione su **tutti** i casi invece che sul solo 20%, e si vede la
**variabilità** fra fold, che su dataset piccoli è grande e informativa. Con 99 casi come in WMH, un
singolo split da 20 casi darebbe un numero troppo rumoroso per decidere qualunque cosa.

I 5 modelli possono poi essere usati insieme come **ensemble** in inferenza, mediando le predizioni.

---

## 10. Glossario

| Termine | Significato |
| --- | --- |
| **Affine** | Matrice 4x4 che converte indici di voxel in coordinate fisiche in mm |
| **Anisotropo** | Voxel non cubico, con spacing diverso fra gli assi |
| **Bias field** | Variazione lenta e spuria di luminosità dovuta alle bobine dello scanner |
| **BIDS** | Convenzione standard per organizzare cartelle e nomi in neuroimaging (usata da ATLAS) |
| **Dice** | Metrica di sovrapposizione fra maschera predetta e vera, da 0 a 1 |
| **FLAIR** | Sequenza MRI che sopprime il liquido cerebrospinale, ideale per lesioni della sostanza bianca |
| **FOV** | *Field of view*: spazio fisico coperto dal volume (shape per spacing) |
| **Fold** | Una delle 5 partizioni della cross-validation |
| **Ground truth** | Annotazione di riferimento disegnata da esperti |
| **Isotropo** | Voxel cubico, spacing uguale sui tre assi |
| **Label / maschera** | Volume di interi che marca ogni voxel come background o lesione |
| **Leakage** | Informazione dal test che filtra nel training, gonfiando i risultati |
| **MNI** | Spazio anatomico standard su cui i cervelli vengono deformati per essere confrontabili |
| **Modalità** | Tipo di immagine acquisita (T1w, T2w, FLAIR); in nnU-Net corrisponde a un canale |
| **NIfTI** | Formato di file (`.nii.gz`) per immagini neurologiche: array + header con l'affine |
| **Patch** | Ritaglio 3D di dimensione fissa usato per addestrare |
| **Registrazione** | Allineamento di due volumi sulla stessa griglia |
| **Resampling** | Conversione di un volume a uno spacing diverso |
| **Segmentazione** | Classificazione di ogni singolo voxel |
| **Shape** | Numero di voxel per lato (adimensionale) |
| **Spacing** | Dimensione fisica di un voxel, in mm |
| **Spazio nativo** | Geometria originale di acquisizione, non normalizzata su un template |
| **Volume parziale** | Perdita di contrasto perché un voxel media tessuti diversi al suo interno |
| **Voxel** | Elemento di volume: il pixel in 3D |
| **WMH** | *White Matter Hyperintensities*, iperintensità della sostanza bianca |

---

## Dove andare adesso

- [dataset-e-decisioni.md](dataset-e-decisioni.md) — i problemi concreti di WMH e ATLAS e cosa stiamo correggendo.
- [dataset-e-decisioni.md](dataset-e-decisioni.md) — la specifica operativa del preprocessing.
- [training.md](training.md) — dai dataset preparati al modello valutato.
