#!/usr/bin/env python3
"""Costruisce notebooks/aincp_primo_lotto.ipynb.

Il notebook viene scritto da uno script invece che a mano perche' cosi' e'
rigenerabile e sta in git in forma leggibile.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
co = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Primo lotto AInCP — 10 pazienti FSM

Analisi dell'archivio `dataset_FSM_v1.7z` consegnato l'11 settembre 2026. È il **bersaglio vero**
del progetto: tutto il lavoro sui dataset pubblici serviva a costruire i pesi di partenza per
questi dati.

## Che cosa contiene, e il nome che inganna

L'archivio è la versione ripulita del lotto: sono state tolte le immagini già preprocessate e
registrate, che erano state prodotte per un altro scopo. **Restano solo T1 e FLAIR originali** —
defacciate per anonimizzazione, nient'altro — più le ROI.

> **`COREGISTERED` è un nome fuorviante e va ignorato.** I volumi che contiene **non** sono
> co-registrati fra loro: sono gli originali. La sezione 2 lo verifica sui file invece di fidarsi,
> e il risultato decide il primo passo del preprocessing.

| usato | ignorato |
| --- | --- |
| `FSM_XXX_T1_original.nii.gz` | le cartelle `back_up/` |
| `FSM_XXX_FLAIR_original.nii.gz` | i file `.voi` |
| `ROI_*.nii` al primo livello di `COREGISTERED/` | |

## Le ROI

Da `ROI_type_definition.txt`, versione finale del 19 maggio 2026:

| ROI | significato | fenotipo secondo le nostre misure |
| --- | --- | --- |
| `ROI_1` | iperintensità FLAIR, tipo gliosi | **non cavitato** — rapporto di intensità 1.07, invisibile al T1 |
| `ROI_2` | lesione cistica | cavitato — rapporto 0.62 |
| `ROI_3` | ventricolo dell'emisfero leso | **escluso**: è una conseguenza, non una lesione |
| `ROI_4` | encefalomalacia, perdita di volume non cistica | cavitato |

La maschera di addestramento è quindi **`ROI_1 ∪ ROI_2 ∪ ROI_4`**, binaria.

Il documento contiene però **tre versioni successive** della legenda, e nella prima `ROI_1`
significava «perdita di tessuto *e/o* gliosi», cioè comprendeva anche l'attuale `ROI_4`. Quale
schema sia stato usato per ciascun paziente non è scritto da nessuna parte: la sezione 5 lo
**ricava dalle intensità** invece di assumerlo.
""")

co(r"""
from pathlib import Path
import re, json, warnings
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage

warnings.filterwarnings('ignore', category=UserWarning)

PROGETTO = Path('/home/ccantini/lesion_detection_project')
LOTTO    = PROGETTO / 'data/raw/AINCP/dataset_FSM_v1'
FIGURE   = PROGETTO / 'reports/figures/aincp'
FIGURE.mkdir(parents=True, exist_ok=True)

# I tre fenotipi lesionali prendono i primi tre slot della palette di riferimento, che sono
# validati su tutte le coppie; il ventricolo e' grigio perche' non e' una serie, e' escluso.
COL = {'ROI_1': '#2a78d6', 'ROI_2': '#eb6834', 'ROI_4': '#1baf7a', 'ROI_3': '#c3c2b7'}
NOME = {'ROI_1': 'Iperintensità (gliosi)', 'ROI_2': 'Cistica',
        'ROI_4': 'Encefalomalacia', 'ROI_3': 'Ventricolo (escluso)'}
LESIONE = ['ROI_1', 'ROI_2', 'ROI_4']          # ROI_3 non entra mai nella maschera

INK, INK2, GRIGLIA = '#0b0b0b', '#52514e', '#e2e2df'
plt.rcParams.update({
    'figure.dpi': 120, 'savefig.dpi': 150, 'font.size': 9,
    'axes.edgecolor': GRIGLIA, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})
print('lotto:', LOTTO, '| esiste:', LOTTO.exists())
""")

md(r"""
## 1. Quali file ci sono, soggetto per soggetto

Due dettagli che un `glob` ingenuo sbaglierebbe in silenzio, e che qui sono gestiti apposta:

- **`FSM_052` non ha `ROI_1.nii`**, ha `ROI_1 - Copia.nii`. Cercare il nome esatto lo salterebbe
  senza dire niente, e perderemmo un paziente su dieci.
- accanto a ogni `.nii` c'è un `.voi` della stessa dimensione. I `.voi` sono ignorati.
""")

co(r"""
def trova_roi(cartella):
    '''ROI al primo livello di COREGISTERED, indicizzate per numero.

    Il numero si estrae con una regex tollerante ai suffissi di lavorazione (` - Copia`),
    perche' la nomenclatura non e' uniforme fra i soggetti.
    '''
    out = {}
    for f in sorted(cartella.glob('ROI_*.nii')):
        m = re.match(r'ROI_(\d+)', f.name, re.I)
        if not m:
            continue
        chiave = f'ROI_{m.group(1)}'
        # a parita' di numero preferiamo il nome pulito
        if chiave not in out or len(f.name) < len(out[chiave].name):
            out[chiave] = f
    return out

soggetti = {}
for d in sorted(LOTTO.glob('FSM_*_anon')):
    co_dir = d / 'COREGISTERED'
    if not co_dir.is_dir():
        continue
    sig = d.name.split('_MR_')[0]                 # FSM_032_MR_1_anon -> FSM_032
    t1  = next(iter(co_dir.glob('*_T1_original.nii.gz')), None)
    fl  = next(iter(co_dir.glob('*_FLAIR_original.nii.gz')), None)
    soggetti[d.name] = {'paziente': sig, 'dir': co_dir, 'T1': t1, 'FLAIR': fl,
                        'roi': trova_roi(co_dir)}

print(f'{len(soggetti)} soggetti, {len({s["paziente"] for s in soggetti.values()})} pazienti distinti\n')
intest = f"{'soggetto':22s} {'T1':>3s} {'FLAIR':>6s}  " + '  '.join(f'{r:>6s}' for r in ['ROI_1','ROI_2','ROI_3','ROI_4'])
print(intest); print('-' * len(intest))
for k, s in soggetti.items():
    riga = f"{k:22s} {'ok' if s['T1'] else '--':>3s} {'ok' if s['FLAIR'] else '--':>6s}  "
    riga += '  '.join(f"{('ok' if r in s['roi'] else '--'):>6s}" for r in ['ROI_1','ROI_2','ROI_3','ROI_4'])
    note = ' <- ROI_1 con nome non standard' if any('Copia' in f.name for f in s['roi'].values()) else ''
    print(riga + note)
""")

md(r"""
## 2. Geometria

Due domande, e la risposta alla seconda decide il preprocessing.

1. **Le maschere sono nello spazio di quale immagine?** La segmentazione è stata fatta sulla
   **FLAIR**, quindi ci si aspetta che le ROI condividano la griglia della FLAIR.
2. **T1 e FLAIR sono già sulla stessa griglia?** Il nome della cartella dice `COREGISTERED`, ma
   secondo chi ha preparato i dati non lo sono. Qui si controlla.
""")

co(r"""
def geo(p):
    im = nib.load(str(p))
    return im.shape, tuple(round(float(z), 3) for z in im.header.get_zooms()[:3]), im.affine

righe = []
for k, s in soggetti.items():
    sh_t1, vx_t1, af_t1 = geo(s['T1'])
    sh_fl, vx_fl, af_fl = geo(s['FLAIR'])
    r1 = s['roi'].get('ROI_1') or next(iter(s['roi'].values()))
    sh_r,  vx_r,  af_r  = geo(r1)
    righe.append({
        'soggetto': k, 'T1_shape': sh_t1, 'T1_vox': vx_t1,
        'FLAIR_shape': sh_fl, 'FLAIR_vox': vx_fl,
        'ROI_su_FLAIR': sh_r == sh_fl and np.allclose(af_r, af_fl, atol=1e-3),
        'ROI_su_T1':    sh_r == sh_t1 and np.allclose(af_r, af_t1, atol=1e-3),
        'T1_uguale_FLAIR': sh_t1 == sh_fl and np.allclose(af_t1, af_fl, atol=1e-3),
    })

print(f"{'soggetto':22s} {'T1 (shape / vox)':34s} {'FLAIR (shape / vox)':34s} ROI su")
for r in righe:
    dove = 'FLAIR' if r['ROI_su_FLAIR'] else ('T1' if r['ROI_su_T1'] else '???')
    print(f"{r['soggetto']:22s} {str(r['T1_shape']):17s}{str(r['T1_vox']):17s} "
          f"{str(r['FLAIR_shape']):17s}{str(r['FLAIR_vox']):17s} {dove}")

print()
print('ROI sulla griglia della FLAIR in tutti i soggetti:', all(r['ROI_su_FLAIR'] for r in righe))
print('T1 e FLAIR gia\' sulla stessa griglia:            ', all(r['T1_uguale_FLAIR'] for r in righe))
""")

md(r"""
### Conseguenza

Se le due sequenze stanno su griglie diverse, **la co-registrazione è a carico nostro** ed è il
primo passo del preprocessing, esattamente come descritto in `docs/pipeline_preprocessing.docx`.

Resta però una scelta di direzione, e le due opzioni **non sono equivalenti**, perché le maschere
vivono nello spazio della FLAIR:

- **FLAIR → T1**, la direzione del documento: le maschere vanno trasformate insieme alla FLAIR, con
  interpolazione *nearest neighbour*. Una maschera binaria di lesioni piccole perde qualcosa a ogni
  resampling, e le lesioni piccole sono proprio la parte di bersaglio che ci costa di più.
- **T1 → FLAIR**: le maschere **non si toccano affatto**, ed è lo spazio in cui l'annotatore ha
  lavorato.

La sezione 7 misura quanto costa davvero la prima opzione, così la scelta si fa su un numero invece
che su un'intuizione. Il costo si paga una volta sola e in un verso solo: qualunque direzione si
scelga, le matrici vanno conservate per poter tornare agli spazi nativi.
""")

md(r"""
## 3. Volumi per fenotipo

Il volume conta più di quanto sembri: su questo compito il Dice dipende quasi solo dal volume del
bersaglio, quindi questi numeri dicono in anticipo che cosa aspettarsi dal Dice su AInCP e come va
letto rispetto ai dataset pubblici.
""")

co(r"""
def carica_mask(p):
    d = np.asanyarray(nib.load(str(p)).dataobj)
    return d > 0

dati = {}
for k, s in soggetti.items():
    im = nib.load(str(s['FLAIR']))
    ml_per_voxel = float(np.prod(im.header.get_zooms()[:3])) / 1000.0
    rec = {'ml_voxel': ml_per_voxel, 'vol': {}, 'mask': {}}
    for r, f in s['roi'].items():
        m = carica_mask(f)
        rec['mask'][r] = m
        rec['vol'][r] = float(m.sum()) * ml_per_voxel
    lez = np.zeros_like(next(iter(rec['mask'].values())), dtype=bool)
    for r in LESIONE:
        if r in rec['mask']:
            lez |= rec['mask'][r]
    rec['lesione'] = lez
    rec['vol_lesione'] = float(lez.sum()) * ml_per_voxel
    dati[k] = rec

print(f"{'soggetto':22s} {'ROI_1':>9s} {'ROI_2':>9s} {'ROI_4':>9s} {'LESIONE':>10s} {'ROI_3':>9s}")
print('-' * 72)
for k in soggetti:
    v = dati[k]['vol']
    f = lambda r: f"{v[r]:9.2f}" if r in v else f"{'--':>9s}"
    print(f"{k:22s} {f('ROI_1')} {f('ROI_2')} {f('ROI_4')} {dati[k]['vol_lesione']:10.2f} {f('ROI_3')}")

vols = np.array([dati[k]['vol_lesione'] for k in soggetti])
print('\nvolume di lesione (ml):  mediana %.1f   min %.1f   max %.1f' % (np.median(vols), vols.min(), vols.max()))
print('per confronto, mediana del foreground:  WMH 10.9 ml   ATLAS 5.3 ml')
""")

co(r"""
# Barre orizzontali impilate: composizione del volume per fenotipo, un soggetto per riga.
ordine = sorted(soggetti, key=lambda k: -dati[k]['vol_lesione'])
fig, ax = plt.subplots(figsize=(7.4, 4.2))
y = np.arange(len(ordine))
sinistra = np.zeros(len(ordine))
for r in LESIONE:
    w = np.array([dati[k]['vol'].get(r, 0.0) for k in ordine])
    ax.barh(y, w, left=sinistra, height=.62, color=COL[r], label=NOME[r],
            edgecolor='white', linewidth=1.4)          # 2px di distacco fra i segmenti
    sinistra += w
for i, k in enumerate(ordine):
    ax.text(sinistra[i] + max(sinistra) * .012, i, f'{sinistra[i]:.0f} ml',
            va='center', ha='left', fontsize=8, color=INK2)
ax.axvline(10.9, color=INK2, lw=1, ls=(0, (4, 3)), zorder=0)
ax.text(10.9, len(ordine) - .35, ' mediana WMH', fontsize=7.5, color=INK2, va='top')
ax.set_yticks(y, [k.replace('_MR_1_anon', '').replace('_MR_2_anon', ' (MR_2)') for k in ordine])
ax.set_xlabel('volume (ml)')
ax.set_xlim(0, max(sinistra) * 1.16)
ax.invert_yaxis()
ax.grid(axis='x', color=GRIGLIA, lw=.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='lower right', fontsize=8.5)
ax.set_title('Volume di lesione per fenotipo', loc='left', fontweight='bold', pad=10)
fig.tight_layout()
fig.savefig(FIGURE / 'volumi_per_fenotipo.png', bbox_inches='tight')
plt.show()
""")

md(r"""
## 4. Sovrapposizioni fra le ROI

Due cose da verificare prima di unire le maschere:

1. **le ROI lesionali si sovrappongono fra loro?** Se sì, l'unione è comunque corretta, ma vuol
   dire che i fenotipi non sono esclusivi e non si possono contare separatamente;
2. **la lesione tocca il ventricolo?** Una cavità può comunicare con il ventricolo dilatato. Se la
   sovrapposizione fosse grossa, «unire tutto tranne `ROI_3`» sarebbe ambiguo e bisognerebbe
   decidere se sottrarre `ROI_3` dall'unione.
""")

co(r"""
print(f"{'soggetto':22s} {'coppie lesionali sovrapposte':32s} {'lesione ∩ ventricolo':>22s}")
print('-' * 78)
for k in soggetti:
    m = dati[k]['mask']
    sovr = []
    for i, a in enumerate(LESIONE):
        for b in LESIONE[i+1:]:
            if a in m and b in m:
                n = int((m[a] & m[b]).sum())
                if n: sovr.append(f'{a}∩{b}={n}')
    if 'ROI_3' in m:
        nv = int((dati[k]['lesione'] & m['ROI_3']).sum())
        pv = 100.0 * nv / max(1, int(dati[k]['lesione'].sum()))
        vent = f'{nv} vox ({pv:.2f}%)'
    else:
        vent = 'ROI_3 assente'
    print(f"{k:22s} {(', '.join(sovr) if sovr else 'nessuna'):32s} {vent:>22s}")
""")

md(r"""
## 5. Quale schema di annotazione ha ciascun paziente — ricavato dalle intensità

È la domanda aperta lasciata dal file della legenda, e si può chiudere con una misura invece che
con una domanda al clinico.

Il ragionamento: sulla FLAIR i tre fenotipi hanno segni **opposti** rispetto al tessuto sano.

| ROI | che cosa deve fare sulla FLAIR | rapporto atteso |
| --- | --- | --- |
| `ROI_1` iperintensità | brilla | **> 1** |
| `ROI_2` cistica | segnale simile al liquido, quindi scura | **< 1** |
| `ROI_4` encefalomalacia | perdita di tessuto, scura | **< 1** |

Si calcola per ogni ROI il rapporto fra l'intensità mediana dentro la ROI e quella del tessuto
cerebrale sano dello stesso soggetto. Se `ROI_1` di un paziente risultasse **< 1**, quel paziente
seguirebbe la legenda vecchia, in cui `ROI_1` comprendeva anche la perdita di tessuto — e la sua
maschera non sarebbe confrontabile con quella degli altri.

È la stessa grandezza con cui avevamo separato i casi risolti da quelli falliti su ATLAS
(0.62 contro 1.07), quindi i numeri sono direttamente confrontabili con quelli.
""")

co(r"""
def riferimento_tessuto(vol, tutte_le_roi):
    '''Intensita' mediana del tessuto cerebrale sano.

    Il cervello si isola con una soglia di Otsu approssimata dal 60esimo percentile dei voxel
    non nulli: non serve precisione, serve un riferimento stabile che escluda l'aria e il fondo.
    Da li' si tolgono tutte le ROI annotate, ventricolo compreso.
    '''
    v = vol.astype(np.float32)
    non_nulli = v[v > 0]
    soglia = np.percentile(non_nulli, 60)
    sano = (v > soglia)
    for m in tutte_le_roi:
        sano &= ~m
    return float(np.median(v[sano])) if sano.any() else float('nan')

rapporti = {}
for k, s in soggetti.items():
    v = np.asanyarray(nib.load(str(s['FLAIR'])).dataobj).astype(np.float32)
    rif = riferimento_tessuto(v, list(dati[k]['mask'].values()))
    rapporti[k] = {r: float(np.median(v[m])) / rif for r, m in dati[k]['mask'].items() if m.any()}
    del v

print(f"{'soggetto':22s} " + ''.join(f'{r:>10s}' for r in ['ROI_1','ROI_2','ROI_4','ROI_3']) + '   coerente?')
print('-' * 80)
for k in soggetti:
    rp = rapporti[k]
    riga = f"{k:22s} " + ''.join((f"{rp[r]:10.2f}" if r in rp else f"{'--':>10s}") for r in ['ROI_1','ROI_2','ROI_4','ROI_3'])
    atteso = []
    if 'ROI_1' in rp: atteso.append(rp['ROI_1'] > 1.0)
    if 'ROI_2' in rp: atteso.append(rp['ROI_2'] < 1.0)
    if 'ROI_4' in rp: atteso.append(rp['ROI_4'] < 1.0)
    print(riga + ('   sì' if all(atteso) else '   NO  <-- da verificare'))
""")

co(r"""
# Un punto per ogni ROI di ogni soggetto, contro la linea del tessuto sano.
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ordine5 = sorted(soggetti, key=lambda k: -rapporti[k].get('ROI_1', 0))
y = np.arange(len(ordine5))
for r in ['ROI_1', 'ROI_2', 'ROI_4']:
    xs = [rapporti[k].get(r, np.nan) for k in ordine5]
    ax.scatter(xs, y, s=64, color=COL[r], label=NOME[r],
               edgecolor='white', linewidth=1.4, zorder=3)   # anello di 2px sulle sovrapposizioni
ax.axvline(1.0, color=INK2, lw=1.2, zorder=1)
ax.text(1.0, -0.85, ' tessuto sano', fontsize=7.5, color=INK2, va='bottom')
ax.set_yticks(y, [k.replace('_MR_1_anon', '').replace('_MR_2_anon', ' (MR_2)') for k in ordine5])
ax.set_xlabel('intensità mediana sulla FLAIR, rapporto sul tessuto sano')
ax.invert_yaxis()
ax.grid(axis='x', color=GRIGLIA, lw=.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='upper right', fontsize=8.5)
ax.set_title('I fenotipi si separano sulla FLAIR', loc='left', fontweight='bold', pad=10)
ax.set_ylim(len(ordine5) - .3, -1.3)
fig.tight_layout()
fig.savefig(FIGURE / 'rapporti_intensita.png', bbox_inches='tight')
plt.show()
""")

md(r"""
## 6. Quante lesioni, e di che dimensione

Il Dice medio non dice quante lesioni si perdono: è il risultato più importante che abbiamo
ottenuto sui dataset pubblici. Qui si conta in quante componenti connesse si divide la maschera di
ciascun soggetto, e come si distribuiscono per dimensione.

> **Le fasce sono in millimetri cubi, non in voxel, e non è un dettaglio.** Sulle FLAIR di AInCP un
> voxel vale circa **0.2 mm³**; su ATLAS ne vale **1.0**. Contare in voxel farebbe sembrare
> confrontabili fasce che differiscono di cinque volte, e su un compito in cui tutto dipende dalla
> dimensione del bersaglio sarebbe un errore grosso.

Accanto al conteggio c'è la **quota di volume** che ogni fascia rappresenta, ed è quella colonna a
dire se le componenti piccole sono lesioni o rumore di tracciamento.
""")

co(r"""
FASCE = [('< 1 mm³', 0, 1), ('1 - 10 mm³', 1, 10),
         ('10 - 100 mm³', 10, 100), ('> 100 mm³', 100, np.inf)]

# Connettivita' a 26 vicini: due voxel che si toccano anche solo per uno spigolo sono la stessa
# lesione. E' la stessa scelta di scripts/metriche_per_lesione.py, e i due conteggi devono
# coincidere o non sarebbero confrontabili. Con la connettivita' a 6 (il default di
# ndimage.label) le stesse dieci maschere danno 158 componenti invece di 65: una lesione
# sottile e obliqua si spezza in tanti frammenti che sembrano lesioni minuscole distinte.
STRUTTURA = np.ones((3, 3, 3), dtype=bool)

conteggi = {}
for k in soggetti:
    et, n = ndimage.label(dati[k]['lesione'], structure=STRUTTURA)
    # dimensione di ogni componente in mm cubi, non in voxel
    dim = (np.bincount(et.ravel())[1:] * dati[k]['ml_voxel'] * 1000.0) if n else np.array([])
    conteggi[k] = {'n': int(n), 'dim': dim,
                   'fasce': [int(((dim >= lo) & (dim < hi)).sum()) for _, lo, hi in FASCE]}

print(f"{'soggetto':22s} {'lesioni':>8s} {'<1mm³':>7s}  {'volume':>9s} {'quota in <1mm³':>16s}")
print('-' * 68)
for k in soggetti:
    d = conteggi[k]['dim']; pic = d[d < 1.0]
    print(f"{k:22s} {conteggi[k]['n']:8d} {len(pic):7d}  {d.sum()/1000:8.2f}ml "
          f"{100 * pic.sum() / max(1e-9, d.sum()):15.3f}%")

tutte = np.concatenate([conteggi[k]['dim'] for k in soggetti if len(conteggi[k]['dim'])])
print('-' * 68)
print(f"{'TOTALE':22s} {len(tutte):8d} {int((tutte < 1).sum()):7d}  {tutte.sum()/1000:8.2f}ml "
      f"{100 * tutte[tutte < 1].sum() / tutte.sum():15.3f}%")

print(f"\n{'fascia':14s} {'lesioni':>9s} {'quota':>8s} {'volume':>11s} {'quota volume':>14s}")
for nome, lo, hi in FASCE:
    sel = (tutte >= lo) & (tutte < hi)
    print(f'{nome:14s} {int(sel.sum()):9d} {100*sel.mean():7.1f}% {tutte[sel].sum()/1000:10.2f}ml '
          f'{100*tutte[sel].sum()/tutte.sum():13.2f}%')
print(f'\nmediana della singola componente: {np.median(tutte):.2f} mm³  '
      f'(la piu\' piccola {tutte.min():.2f}, la piu\' grande {tutte.max()/1000:.0f} ml)')
""")

co(r"""
# Distribuzione cumulativa delle dimensioni in mm cubi: una linea per soggetto, scala log.
fig, ax = plt.subplots(figsize=(7.4, 4.0))
for k in soggetti:
    d = np.sort(conteggi[k]['dim'])
    if not len(d): continue
    ax.plot(d, 100.0 * np.arange(1, len(d) + 1) / len(d), lw=2,
            color=COL['ROI_1'], alpha=.5, solid_capstyle='round')
tutte = np.sort(np.concatenate([conteggi[k]['dim'] for k in soggetti if len(conteggi[k]['dim'])]))
ax.plot(tutte, 100.0 * np.arange(1, len(tutte) + 1) / len(tutte), lw=2.6,
        color=COL['ROI_2'], solid_capstyle='round', zorder=4, label='tutti i soggetti insieme')
for x in (1, 10, 100):
    ax.axvline(x, color=GRIGLIA, lw=1, zorder=0)
    ax.text(x, 102, f'{x} mm³', fontsize=7.5, color=INK2, ha='center')
ax.set_xscale('log')
ax.set_xlabel('dimensione della lesione (mm³, scala logaritmica)')
ax.set_ylabel('quota cumulativa delle lesioni (%)')
ax.set_ylim(0, 108)
ax.grid(axis='y', color=GRIGLIA, lw=.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc='lower right', fontsize=8.5)
ax.set_title('Due terzi delle componenti stanno sotto i dieci millimetri cubi',
             loc='left', fontweight='bold', pad=14)
fig.tight_layout()
fig.savefig(FIGURE / 'dimensioni_lesioni.png', bbox_inches='tight')
plt.show()
""")

md(r"""
### Che cosa dice quella tabella, e che cosa non dice

Le due colonne vanno lette insieme, perché dicono cose opposte. **Per numero** due terzi delle
componenti stanno sotto i dieci millimetri cubi; **per volume** quelle stesse componenti non
valgono quasi nulla (0.08 ml su 244), mentre quindici lesioni grandi contengono tutto il resto.

Verrebbe da concludere che le componenti minuscole siano imprecisioni del disegno a mano, e che
convenga scartarle. **La misura dice il contrario**, ed è il motivo per cui questa conclusione è
stata verificata invece che assunta: `scripts/figura_lesioni_piccole.py` guarda una per una le
componenti sotto i 10 mm³, e per ciascuna misura l'intensità sulla FLAIR rispetto al tessuto sano
dello stesso paziente — lo stesso rapporto della sezione 5. Le componenti di gliosi, **anche
quelle da un solo voxel**, stanno quasi tutte fra 2.3 e 4.4: hanno il segnale di una lesione vera,
non del rumore. Solo tre componenti su 65 in tutto il lotto cadono dove l'immagine ha il livello
dell'aria, cioè fuori dal cervello, e quelle sì sono errori di tracciato.

Quindi, per ora, **nessuna soglia: tutto ciò che è annotato conta come lesione.** La scelta è
conservativa e va dichiarata prima di misurare, non aggiustata dopo. Resta una domanda per il
clinico, alla quale la figura degli esempi serve a rispondere guardando: un singolo voxel
iperintenso in mezzo a sostanza bianca normale è una lesione da trovare, o la punta di qualcosa
che sulla sua risoluzione non si vede? Se la risposta fosse «ignorabile», la soglia si introduce
allora, una volta, e si rimisura tutto.
""")

md(r"""
### Una scelta che cambia i numeri: come si contano le componenti

Due voxel che si toccano solo per uno spigolo sono la stessa lesione o due lesioni? Con la
connettività a **6 vicini** (solo le facce, il default di `scipy`) le dieci maschere danno **158**
componenti; con quella a **26 vicini** (facce, spigoli e angoli) ne danno **65**. Il volume totale
è identico: cambia solo quanti pezzi si conta.

Qui si usa **26**, per due ragioni. Una lesione sottile e obliqua, disegnata a mano su fette da
0.8 mm, con la connettività a 6 si spezza in tanti frammenti che sembrerebbero lesioni minuscole
distinte — e la sezione precedente le conterebbe come tali. E soprattutto è la stessa scelta di
`scripts/metriche_per_lesione.py`, che è lo strumento con cui misureremo quante lesioni il modello
trova: se i due conteggi usassero convenzioni diverse, il numeratore e il denominatore del tasso
di rilevamento verrebbero da due definizioni di «lesione».
""")

md(r"""
## 7. Quanto costerebbe portare le maschere nello spazio della T1

La domanda della sezione 2, con un numero. Si simula il percorso `FLAIR → T1` applicando alla
maschera solo il **cambio di griglia** (dalla griglia FLAIR a quella T1, nearest neighbour), senza
alcuna rotazione: è il caso *migliore*, perché la registrazione vera aggiunge un'interpolazione in
più. Quello che si misura è quindi un **limite inferiore** della perdita.
""")

co(r"""
from scipy.ndimage import affine_transform

print(f"{'soggetto':22s} {'componenti':>12s} {'dopo':>8s} {'Δ comp.':>9s} {'Δ volume':>10s}")
print('-' * 66)
for k, s in soggetti.items():
    fl, t1 = nib.load(str(s['FLAIR'])), nib.load(str(s['T1']))
    # matrice che porta gli indici della T1 negli indici della FLAIR
    M = np.linalg.inv(fl.affine) @ t1.affine
    m = dati[k]['lesione'].astype(np.uint8)
    rip = affine_transform(m, M[:3, :3], offset=M[:3, 3], output_shape=t1.shape,
                           order=0, mode='constant', cval=0) > 0
    n_pre = conteggi[k]['n']
    n_post = int(ndimage.label(rip, structure=STRUTTURA)[1])
    v_pre = dati[k]['vol_lesione']
    v_post = float(rip.sum()) * float(np.prod(t1.header.get_zooms()[:3])) / 1000.0
    dv = 100.0 * (v_post - v_pre) / v_pre if v_pre else 0.0
    # spessore di fetta delle due sequenze: e' li' che si rompono le lesioni sottili
    dz_f, dz_t = fl.header.get_zooms()[0], t1.header.get_zooms()[0]
    print(f'{k:22s} {n_pre:12d} {n_post:8d} {n_post - n_pre:+9d} {dv:+9.1f}%'
          f'   ({dz_f:.1f} -> {dz_t:.1f} mm)')
""")

md(r"""
### Che cosa dice quella tabella

**Il volume si conserva, il conteggio no.** La maschera non perde quasi niente in millilitri —
al massimo il 6% su FSM_046, che è anche il soggetto con la lesione più piccola in assoluto — ma
in sette casi su dieci si **spezza in più pezzi** di quanti ne avesse prima: FSM_040 passa da 30
componenti a 48, FSM_041 da 10 a 22.

La causa è nell'ultima colonna, ed è una sola. Nel piano la T1 è **più fine** della FLAIR
(0.43 contro 0.5 mm), quindi lì non si perde nulla; lungo l'asse delle fette invece la FLAIR ha
fette da **0.8 mm** e la T1 da **1.0 mm**. Ricampionare a nearest neighbour significa, lungo
quell'asse, **scartare una fetta ogni cinque**: una lesione sottile che sulla FLAIR occupa poche
fette contigue si ritrova con un buco in mezzo, e quello che era un oggetto solo diventa due.

La verifica è nei tre soggetti la cui FLAIR ha già fette da 1.0 mm — FSM_039, FSM_054, FSM_056:
**nessuno dei tre cambia numero di componenti**, nemmeno FSM_054 che ne ha 43. Non è un caso, è
esattamente la previsione. Lo zero di FSM_045 è invece di altra natura: quel soggetto ha due sole
lesioni, entrambe grandi, e non c'è niente di sottile da spezzare.

Due conseguenze pratiche:

- **la registrazione vera peggiora questo numero, non lo migliora**: qui c'è solo il cambio di
  griglia, mentre FLIRT aggiunge una rotazione e quindi un'interpolazione in più;
- **il problema riguarda le componenti piccole**, cioè in gran parte le stesse scorie di
  tracciamento della sezione 6. Se si applica la soglia minima discussa lì, la frammentazione che
  resta è molto meno preoccupante di quanto il conteggio grezzo faccia sembrare.

Resta però il motivo per cui questa misura serviva: **le annotazioni sono nate sulla FLAIR**, e
ogni spazio diverso dalla FLAIR le fa passare per un'interpolazione. È il primo argomento concreto
a favore di addestrare nello spazio nativo, e va pesato contro il fatto che i pesi di partenza di
ATLAS vivono in MNI.
""")

md(r"""
## 8. Controllo visivo, soggetto per soggetto

I relatori hanno chiesto esplicitamente di poter vedere le immagini. Per ogni soggetto si mostra
la fetta con più lesione, con le ROI sovrapposte nei colori dei fenotipi. Il ventricolo è in grigio
per ricordare che è annotato ma **non** entra nella maschera.
""")

co(r"""
def fetta_migliore(mask):
    conteggio = mask.reshape(mask.shape[0], -1).sum(axis=1)
    return int(np.argmax(conteggio))

def ritaglio(sl):
    '''La testa occupa il centro di un campo di vista 512x512 per meta' vuoto: senza ritaglio
    le lesioni piccole sarebbero di pochi pixel sullo schermo.'''
    pieno = sl > np.percentile(sl[sl > 0], 5) if (sl > 0).any() else sl > 0
    r = np.where(pieno.any(axis=1))[0]; c = np.where(pieno.any(axis=0))[0]
    if len(r) == 0: return slice(None), slice(None)
    m = 12
    return (slice(max(r[0] - m, 0), r[-1] + m), slice(max(c[0] - m, 0), c[-1] + m))

n = len(soggetti)
righe_g = (n + 4) // 5
fig, axes = plt.subplots(righe_g, 5, figsize=(15, 3.55 * righe_g))
fig.subplots_adjust(left=.005, right=.995, top=.90, bottom=.07, hspace=.20, wspace=.03)
for ax, (k, s) in zip(np.ravel(axes), soggetti.items()):
    v = np.asanyarray(nib.load(str(s['FLAIR'])).dataobj).astype(np.float32)
    z = fetta_migliore(dati[k]['lesione'])
    sl = np.rot90(v[z])
    box = ritaglio(sl)
    sl = sl[box]
    lo, hi = np.percentile(sl[sl > 0], [1, 99.5]) if (sl > 0).any() else (0, 1)
    ax.imshow(sl, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest', aspect='equal')
    for r in ['ROI_3'] + LESIONE:                    # il ventricolo sotto, i fenotipi sopra
        if r not in dati[k]['mask']: continue
        m = np.rot90(dati[k]['mask'][r][z])[box]
        if not m.any(): continue
        # Il ventricolo e' scuro sulla FLAIR quanto il fondo: riempirlo di grigio lo rende
        # invisibile, e una cavita' annotata sembrerebbe non annotata. Va quindi disegnato
        # come contorno, che si distingue senza nascondere l'immagine sotto.
        if r == 'ROI_3':
            ax.contour(m.astype(float), levels=[.5], colors=[COL[r]], linewidths=.9)
            continue
        rgba = np.zeros(m.shape + (4,))
        rgba[..., :3] = np.array([int(COL[r][i:i+2], 16) / 255 for i in (1, 3, 5)])
        rgba[..., 3] = m * 0.62
        ax.imshow(rgba, interpolation='nearest')
    ax.set_title(f"{k.replace('_anon','')}\nfetta {z}  ·  {dati[k]['vol_lesione']:.1f} ml",
                 fontsize=9, loc='left', pad=5, linespacing=1.4)
    ax.axis('off')
for ax in np.ravel(axes)[n:]:
    ax.axis('off')
manici = [plt.Line2D([], [], marker='s', ls='', ms=9, color=COL[r], label=NOME[r])
          for r in LESIONE]
manici.append(plt.Line2D([], [], ls='-', lw=1.4, color=COL['ROI_3'], label=NOME['ROI_3']))
fig.legend(handles=manici, frameon=False, ncol=4, loc='lower center', fontsize=9.5,
           bbox_to_anchor=(.5, .0))
fig.suptitle('Primo lotto AInCP — maschere sulla FLAIR', x=.008, y=.985,
             ha='left', fontweight='bold', fontsize=12)
fig.savefig(FIGURE / 'panoramica_maschere.png', bbox_inches='tight')
plt.show()
""")

md(r"""
## 9. Che cosa se ne ricava

Le conclusioni vanno lette dalle celle qui sopra, ma i punti che decidono i prossimi passi sono:

- **quanti soggetti sono davvero utilizzabili**, cioè hanno T1, FLAIR e almeno una ROI lesionale;
- **se lo schema di annotazione è coerente** fra i pazienti (sezione 5). Se non lo è, i pazienti con
  la legenda vecchia vanno o riletti o tenuti da parte, perché la loro maschera vuol dire un'altra
  cosa;
- **quanto sono grandi le lesioni** rispetto ai dataset pubblici (sezione 3), perché il Dice che
  riporteremo va letto con quella distribuzione accanto e non confrontato a occhio con lo 0.610 di
  ATLAS;
- **quante lesioni piccole ci sono** (sezione 6), che è la parte del bersaglio che il T1 da solo non
  vede e la ragione per cui il modello finale è a due canali;
- **in quale spazio lavorare** (sezioni 2 e 7).

Lo split va fissato **prima** di addestrare qualsiasi cosa e per **paziente**, non per sessione.
""")

nb.cells = C
nb.metadata.update({
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10'},
})
out = Path(__file__).resolve().parent.parent / 'notebooks' / 'aincp_primo_lotto.ipynb'
nbf.write(nb, str(out))
print('scritto', out, '-', len(C), 'celle')
