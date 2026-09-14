#!/usr/bin/env python
"""Le componenti piu' piccole delle annotazioni AInCP, viste una per una.

    python scripts/figura_lesioni_piccole.py

Produce:
    reports/figures/aincp/lesioni_piccole_esempi.png   ritagli della FLAIR attorno a venti
                                                       componenti minuscole, nella fetta in cui
                                                       sono piu' estese, per giudicarle a occhio
    reports/figures/aincp/lesioni_piccole_intensita.png  intensita' di ogni componente rispetto
                                                       al tessuto sano, in funzione del volume
    reports/lesioni_piccole.json                       i numeri di entrambe

LA DOMANDA A CUI SERVONO. Delle 158 componenti connesse delle dieci maschere, 93 stanno sotto il
millimetro cubo -- cioe' sono da uno a cinque voxel -- e insieme valgono lo 0.013% del volume
totale. Se sono lesioni vere, un modello che non le trova sta fallendo; se sono imprecisioni del
tracciato a mano, contarle come lesioni mancate rende il tasso di rilevamento una misura del
rumore di annotazione invece che della qualita' del modello. La differenza non si decide a
tavolino: va **guardata**, ed e' una domanda per il clinico che le ha disegnate.

Finche' non c'e' una risposta si tiene tutto: nessuna soglia, ogni voxel annotato e' lesione.

I DUE MODI DI GUARDARLE. La figura degli esempi mostra il contesto: una componente isolata in
mezzo a sostanza bianca normale ha un aspetto diverso da una che sta sul bordo di una lesione
grande (dove e' probabilmente un pezzo staccato dal ricampionamento o dalla mano) o dentro un
solco. La figura delle intensita' da' il numero: sulla FLAIR la gliosi **brilla**, quindi una
componente con intensita' pari al tessuto sano difficilmente e' gliosi. E' lo stesso rapporto con
cui e' stata verificata la legenda delle annotazioni (sezione 5 del notebook), applicato qui alla
singola componente invece che all'intera ROI.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

PROGETTO = Path(__file__).resolve().parent.parent
LOTTO = PROGETTO / 'data/raw/AINCP/dataset_FSM_v1'
FIGURE = PROGETTO / 'reports/figures/aincp'

# Palette del progetto (references/palette.md della skill dataviz). I tre fenotipi tengono i
# colori che hanno in tutte le altre figure, cosi' un lettore che ha visto la panoramica li
# riconosce; il grigio chiaro e' il ventricolo, che non entra nella maschera.
COL = {'ROI_1': '#2a78d6', 'ROI_2': '#eb6834', 'ROI_4': '#1baf7a', 'ROI_3': '#c3c2b7'}
NOME = {'ROI_1': 'gliosi', 'ROI_2': 'cistica', 'ROI_4': 'encefalomalacia'}
LESIONE = ['ROI_1', 'ROI_2', 'ROI_4']
INK, INK2, GRIGLIA = '#0b0b0b', '#52514e', '#e6e5e1'
STRUTTURA = np.ones((3, 3, 3), dtype=bool)

# Le fasce della sezione 6 del notebook, in mm3.
FASCE = [('< 1 mm³', 0, 1), ('1 – 10 mm³', 1, 10), ('10 – 100 mm³', 10, 100), ('> 100 mm³', 100, np.inf)]


def trova_roi(cartella):
    """ROI indicizzate per numero; la regex tollera i suffissi (` - Copia` di FSM_052)."""
    out = {}
    for f in sorted(cartella.glob('ROI_*.nii')):
        m = re.match(r'ROI_(\d+)', f.name, re.I)
        if not m:
            continue
        chiave = f'ROI_{m.group(1)}'
        if chiave not in out or len(f.name) < len(out[chiave].name):
            out[chiave] = f
    return out


def riferimento_tessuto(vol, maschere):
    """Intensita' mediana del tessuto cerebrale sano, come nella sezione 5 del notebook."""
    v = vol.astype(np.float32)
    sano = v > np.percentile(v[v > 0], 60)
    for m in maschere:
        sano &= ~m
    return float(np.median(v[sano])) if sano.any() else float('nan')


def raccogli():
    """Una riga per componente connessa, su tutti i soggetti."""
    righe, volumi = [], {}
    for d in sorted(LOTTO.glob('FSM_*_anon')):
        co = d / 'COREGISTERED'
        fl_file = next(iter(co.glob('*_FLAIR_original.nii.gz')), None)
        roi = trova_roi(co)
        if fl_file is None or not roi:
            continue
        img = nib.load(str(fl_file))
        flair = np.asanyarray(img.dataobj).astype(np.float32)
        mm3 = float(np.prod(img.header.get_zooms()[:3]))
        masc = {r: np.asanyarray(nib.load(str(f)).dataobj) > 0 for r, f in roi.items()}
        rif = riferimento_tessuto(flair, list(masc.values()))
        lesione = np.zeros(flair.shape, bool)
        for r in LESIONE:
            if r in masc:
                lesione |= masc[r]
        lab, n = ndimage.label(lesione, structure=STRUTTURA)
        conteggi = np.bincount(lab.ravel())[1:]
        centri = ndimage.center_of_mass(lesione, lab, range(1, n + 1))
        for i in range(1, n + 1):
            m = lab == i
            # il fenotipo della componente e' quello che ne copre piu' voxel
            quote = {r: int((m & masc[r]).sum()) for r in LESIONE if r in masc}
            fenotipo = max(quote, key=quote.get) if quote else 'ROI_1'
            righe.append({
                'soggetto': d.name, 'paziente': d.name.split('_MR_')[0], 'id': i,
                'voxel': int(conteggi[i - 1]), 'mm3': float(conteggi[i - 1] * mm3),
                'fenotipo': fenotipo,
                'intensita_rapporto': float(np.median(flair[m]) / rif) if rif else float('nan'),
                'centro': [float(c) for c in centri[i - 1]],
                'file_flair': str(fl_file),
            })
        volumi[d.name] = mm3
    righe.sort(key=lambda r: r['mm3'])
    return righe, volumi


def figura_esempi(righe, n_col=5, n_riga=4, meta_lato=18):
    """Ritagli della FLAIR attorno alle componenti piu' piccole.

    Si mostrano le piu' piccole in assoluto e un campione regolare fra quelle sotto i 10 mm3:
    scegliere solo le minime darebbe venti volte lo stesso caso, scegliere a caso nasconderebbe
    proprio quelle su cui si decide.
    """
    piccole = [r for r in righe if r['mm3'] < 10]
    n = n_col * n_riga
    if len(piccole) > n:
        # meta' dal fondo della distribuzione, meta' distribuite uniformemente fino a 10 mm3
        fisse = piccole[:n // 2]
        resto = piccole[n // 2:]
        passo = max(len(resto) // (n - len(fisse)), 1)
        scelte = fisse + resto[::passo][:n - len(fisse)]
    else:
        scelte = piccole
    scelte = sorted(scelte, key=lambda r: r['mm3'])

    fig, axes = plt.subplots(n_riga, n_col, figsize=(2.9 * n_col, 3.25 * n_riga))
    fig.subplots_adjust(left=.005, right=.995, top=.905, bottom=.055, hspace=.30, wspace=.03)
    cache = {}
    for ax, r in zip(np.ravel(axes), scelte):
        f = r['file_flair']
        if f not in cache:
            cache.clear()                      # un volume alla volta: sono 200-400 MB l'uno
            cache[f] = np.asanyarray(nib.load(f).dataobj).astype(np.float32)
        vol = cache[f]
        z, y, x = [int(round(c)) for c in r['centro']]
        fetta = np.rot90(vol[z])
        # rot90 manda (y, x) in (X - 1 - x, y): il centro va convertito nello stesso sistema
        ry, rx = vol.shape[2] - 1 - x, y
        y0, y1 = max(ry - meta_lato, 0), min(ry + meta_lato, fetta.shape[0])
        x0, x1 = max(rx - meta_lato, 0), min(rx + meta_lato, fetta.shape[1])
        crop = fetta[y0:y1, x0:x1]
        lo, hi = np.percentile(crop[crop > 0], [2, 99]) if (crop > 0).any() else (0, 1)
        ax.imshow(crop, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
        # la componente e' di pochi voxel: un contorno sarebbe invisibile, un cerchio la indica
        # senza coprirla, che e' il punto -- il clinico deve vedere il tessuto sotto
        ax.add_patch(plt.Circle((rx - x0, ry - y0), 5.5, fill=False, lw=1.6,
                                color=COL[r['fenotipo']]))
        ax.set_title(f"{r['paziente']}  ·  {r['mm3']:.2f} mm³  ·  {r['voxel']} vox\n"
                     f"{NOME[r['fenotipo']]}  ·  intensità {r['intensita_rapporto']:.2f}×",
                     fontsize=8.5, color=INK, linespacing=1.5, pad=5)
        ax.set_axis_off()
    for ax in np.ravel(axes)[len(scelte):]:
        ax.set_axis_off()
    fig.suptitle('Le componenti annotate più piccole, ingrandite  ·  ogni riquadro è circa '
                 '18 × 18 mm di FLAIR', x=.008, y=.985, ha='left', fontsize=12.5,
                 fontweight='bold', color=INK)
    fig.text(.008, .012, 'Il cerchio indica la componente senza coprirla. «intensità» è il '
                         'rapporto fra il segnale della componente e quello del tessuto sano '
                         'dello stesso paziente: la gliosi brilla, quindi dovrebbe stare sopra 1.',
             fontsize=9, color=INK2)
    fig.savefig(FIGURE / 'lesioni_piccole_esempi.png', dpi=140, facecolor='white')
    plt.close(fig)
    return scelte


def figura_intensita(righe):
    """Rapporto d'intensita' contro volume, una per componente."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for r in LESIONE:
        sel = [x for x in righe if x['fenotipo'] == r and np.isfinite(x['intensita_rapporto'])]
        if not sel:
            continue
        ax.scatter([x['mm3'] for x in sel], [x['intensita_rapporto'] for x in sel],
                   s=46, color=COL[r], label=NOME[r], alpha=.85,
                   edgecolor='white', linewidth=1.2, zorder=3)   # anello di 2px sulle sovrapposizioni
    ax.axhline(1.0, color=INK2, lw=1.2, zorder=2)
    ax.text(ax.get_xlim()[1], 1.02, 'come il tessuto sano', ha='right', va='bottom',
            fontsize=9, color=INK2)
    ax.axvline(10, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(10, ax.get_ylim()[1], ' 10 mm³ ', ha='left', va='top', fontsize=9, color=INK2)
    ax.set_xscale('log')
    ax.set_xlabel('volume della componente (mm³, scala logaritmica)', color=INK2)
    ax.set_ylabel('intensità FLAIR rispetto al tessuto sano', color=INK2)
    ax.grid(color=GRIGLIA, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRIGLIA)
    ax.tick_params(colors=INK2, length=0)
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    ax.set_title('Le componenti piccole hanno il segnale di una lesione?',
                 loc='left', fontsize=12, fontweight='bold', color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(FIGURE / 'lesioni_piccole_intensita.png', dpi=140, facecolor='white')
    plt.close(fig)


def riepilogo(righe):
    out = {'totale_componenti': len(righe), 'fasce': [], 'per_paziente': {}}
    for etichetta, lo, hi in FASCE:
        sel = [r for r in righe if lo <= r['mm3'] < hi]
        if not sel:
            continue
        rap = [r['intensita_rapporto'] for r in sel if np.isfinite(r['intensita_rapporto'])]
        out['fasce'].append({
            'fascia': etichetta, 'componenti': len(sel),
            'volume_ml': round(sum(r['mm3'] for r in sel) / 1000, 4),
            'intensita_mediana': round(float(np.median(rap)), 3) if rap else None,
            'quota_sopra_1': round(float(np.mean([x > 1 for x in rap])), 3) if rap else None,
        })
    for r in righe:
        p = out['per_paziente'].setdefault(r['paziente'], {'componenti': 0, 'sotto_1mm3': 0,
                                                           'sotto_10mm3': 0})
        p['componenti'] += 1
        p['sotto_1mm3'] += r['mm3'] < 1
        p['sotto_10mm3'] += r['mm3'] < 10
    return out


if __name__ == '__main__':
    FIGURE.mkdir(parents=True, exist_ok=True)
    righe, _ = raccogli()
    scelte = figura_esempi(righe)
    figura_intensita(righe)
    ris = riepilogo(righe)
    ris['esempi_mostrati'] = [{k: r[k] for k in ('paziente', 'mm3', 'voxel', 'fenotipo',
                                                 'intensita_rapporto')} for r in scelte]
    (PROGETTO / 'reports' / 'lesioni_piccole.json').write_text(
        json.dumps(ris, indent=1, ensure_ascii=False))
    print(f"{ris['totale_componenti']} componenti")
    print(f"{'fascia':14s} {'n':>4s} {'volume':>9s} {'intensità mediana':>18s} {'quota > 1':>10s}")
    for f in ris['fasce']:
        print(f"{f['fascia']:14s} {f['componenti']:4d} {f['volume_ml']:8.3f} ml "
              f"{f['intensita_mediana']:18.2f} {f['quota_sopra_1']:10.0%}")
