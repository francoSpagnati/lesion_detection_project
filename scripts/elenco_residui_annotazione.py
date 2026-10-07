#!/usr/bin/env python
"""Dove sono, nelle annotazioni originali, le componenti da togliere: l'elenco per il clinico.

    python scripts/elenco_residui_annotazione.py
    # -> reports/residui_annotazione.md, reports/residui_annotazione.csv,
    #    reports/figures/aincp/residui_annotazione.png

LA DECISIONE (incontro del 23 settembre). Le componenti connesse delle annotazioni fino a 1 mm³
sono residui del tracciato, non lesioni: si tolgono. Il clinico vuole poterle correggere nei file
originali, quindi serve sapere per ciascuna **quale paziente, quale file ROI, quale fetta e dove**.

TUTTO NELLO SPAZIO IN CUI SONO STATE DISEGNATE: la FLAIR originale di ogni paziente
(`COREGISTERED/*_FLAIR_original.nii.gz`) e i file `ROI_*.nii` accanto, non lo spazio T1 o MNI in
cui lavora la rete. Le coordinate sono date in tre forme, perche' i visualizzatori le mostrano in
modo diverso:
  - indici di voxel (i, j, k) nell'ordine del file, contati da 1 come li mostra MRIcron/MRIcroGL
    (i .voi accanto alle ROI sono il suo formato), e da 0 per ITK-SNAP e per il codice;
  - coordinate in millimetri (x, y, z) dello scanner, dall'header: valgono in qualunque
    visualizzatore, anche se riorienta l'immagine per mostrarla.
Le FLAIR sono acquisite in fette sagittali: l'indice i e' il numero della fetta sagittale.

«Fuori dal cervello» e' deciso con SynthStrip sulla FLAIR originale, non dall'intensita': una
componente e' fuori se nessuno dei suoi voxel cade nella maschera del cervello. Si elencano anche
le componenti fuori dal cervello piu' grandi di 1 mm³, se ce ne sono, perche' sono comunque errori.

La connettivita' e' a 26 vicini, lo standard del progetto.

Copre tutti i lotti AInCP in data/raw/AINCP/ (dal 1/10 anche il secondo). La maschera del cervello
sulla FLAIR originale, se manca (pazienti nuovi), si calcola qui con SynthStrip su CPU.
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Circle
from scipy import ndimage

PROGETTO = Path(__file__).resolve().parent.parent
LOTTI = PROGETTO / 'data/raw/AINCP'
PROC = PROGETTO / 'data/processed/AINCP'
FIG = PROGETTO / 'reports/figures/aincp'
SOGLIA_MM3 = 1.0
STRUTTURA = np.ones((3, 3, 3), dtype=bool)
LESIONE = ('ROI_1', 'ROI_2', 'ROI_4')
NOME = {'ROI_1': 'gliosis', 'ROI_2': 'cystic lesion', 'ROI_4': 'encephalomalacia'}
INK, INK2 = '#0b0b0b', '#52514e'


def maschera_flair(d: Path, fl: Path) -> Path:
    m = PROC / d.name / 'cervello_FLAIR_originale.nii.gz'
    if not m.exists():
        subprocess.run([sys.executable, str(PROGETTO / 'scripts/synthstrip/synthstrip.py'), '-i', str(fl),
                        '-m', str(m), '--model', str(PROGETTO / 'synthstrip.1.pt')], check=True,
                       capture_output=True)
    return m


def raccogli():
    righe = []
    for d in sorted(LOTTI.glob('*/FSM_*_anon')):
        co = d / 'COREGISTERED'
        fl = next(co.glob('*_FLAIR_original.nii.gz'))
        maschera_flair(d, fl)
        img = nib.load(str(fl)); vol = np.asarray(img.dataobj, dtype=np.float32)
        mm3 = float(np.prod(img.header.get_zooms()[:3]))
        cervello = np.asarray(nib.load(str(PROC / d.name / 'cervello_FLAIR_originale.nii.gz')).dataobj) > 0.5
        roi = {}
        for f in sorted(co.glob('ROI_*.nii')):
            m = re.match(r'ROI_(\d+)', f.name, re.I)             # FSM_052: 'ROI_1 - Copia.nii'
            if m and f'ROI_{m.group(1)}' in LESIONE:
                roi[f'ROI_{m.group(1)}'] = (f.name, np.squeeze(np.asarray(nib.load(str(f)).dataobj)) > 0)
        lesione = np.zeros(vol.shape, bool)
        for _, m in roi.values():
            lesione |= m
        lab, n = ndimage.label(lesione, structure=STRUTTURA)
        for i in range(1, n + 1):
            c = lab == i
            nv = int(c.sum()); v = nv * mm3
            fuori = not (c & cervello).any()
            if v > SOGLIA_MM3 + 1e-6 and not fuori:
                continue
            idx = np.argwhere(c)
            centro = idx.mean(axis=0)
            vox = tuple(int(round(x)) for x in centro)
            mm = nib.affines.apply_affine(img.affine, centro)
            file_roi = [nome for r, (nome, m) in roi.items() if (m & c).any()]
            fen = max(((r, int((m & c).sum())) for r, (_, m) in roi.items()), key=lambda t: t[1])[0]
            righe.append({'paziente': d.name.split('_MR_')[0], 'soggetto': d.name, 'fenotipo': NOME[fen],
                          'file_roi': ' + '.join(file_roi), 'voxel': nv, 'mm3': round(v, 2),
                          'fuori_dal_cervello': fuori,
                          'i1': vox[0] + 1, 'j1': vox[1] + 1, 'k1': vox[2] + 1,
                          'i0': vox[0], 'j0': vox[1], 'k0': vox[2],
                          'x_mm': round(float(mm[0]), 1), 'y_mm': round(float(mm[1]), 1), 'z_mm': round(float(mm[2]), 1),
                          'orientamento': ''.join(nib.aff2axcodes(img.affine)),
                          '_flair': str(fl), '_zoom': img.header.get_zooms()[:3]})
    return righe


def figura(righe, n_col=6, lato_mm=24):
    n = len(righe); n_riga = int(np.ceil(n / n_col))
    fig, axes = plt.subplots(n_riga, n_col, figsize=(2.55 * n_col, 2.95 * n_riga), squeeze=False)
    fig.subplots_adjust(left=.01, right=.99, top=.965, bottom=.01, hspace=.36, wspace=.06)
    cache = {}
    for ax, r in zip(axes.ravel(), righe):
        if r['_flair'] not in cache:
            cache[r['_flair']] = np.asarray(nib.load(r['_flair']).dataobj, dtype=np.float32)
        v = cache[r['_flair']]
        zi, zj, zk = r['_zoom']
        hj, hk = int(lato_mm / 2 / zj), int(lato_mm / 2 / zk)
        j0, k0 = r['j0'], r['k0']
        sl = v[r['i0'], max(j0 - hj, 0):j0 + hj, max(k0 - hk, 0):k0 + hk]
        vis = np.rot90(sl)                                   # superiore in alto, come nelle altre figure
        lo, hi = np.percentile(v[v > 0], [1, 99.5]) if (v > 0).any() else (0, 1)
        ax.imshow(vis, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
        # centro della componente nel ritaglio ruotato: rot90 manda (a, b) in (W-1-b, a)
        a, b = j0 - max(j0 - hj, 0), k0 - max(k0 - hk, 0)
        col = '#d6452b' if r['fuori_dal_cervello'] else '#eda100'
        ax.add_patch(Circle((a, sl.shape[1] - 1 - b), radius=4 / zj, fill=False, lw=1.6, ec=col))
        ax.set_title(f"{r['paziente']} · {r['fenotipo']}\nsagittal slice {r['i1']} · ({r['j1']}, {r['k1']})"
                     + ('\nOUTSIDE THE BRAIN' if r['fuori_dal_cervello'] else f"\n{r['voxel']} voxel{'s' if r['voxel'] > 1 else ''} · {r['mm3']} mm³"),
                     fontsize=7.8, color=INK, loc='left', linespacing=1.35, pad=3)
        ax.set_axis_off()
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / 'residui_annotazione.png', dpi=115, facecolor='white')


def main():
    righe = sorted(raccogli(), key=lambda r: (r['paziente'], r['i1'], r['j1']))
    campi = [k for k in righe[0] if not k.startswith('_')]
    with open(PROGETTO / 'reports/residui_annotazione.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=campi); w.writeheader()
        for r in righe:
            w.writerow({k: r[k] for k in campi})
    md = ['# Componenti da togliere dalle annotazioni AInCP', '',
          'Prodotto da `scripts/elenco_residui_annotazione.py`, **non modificare a mano**.', '',
          f'Componenti fino a {SOGLIA_MM3:g} mm³ (decisione del 23 settembre: residui del tracciato), più quelle fuori '
          'dal cervello di qualunque dimensione. Spazio: la FLAIR originale di ciascun paziente. Indici di voxel '
          'contati da 1 (MRIcron); coordinate in mm dello scanner dall\'header. L\'indice *i* è la fetta sagittale.', '',
          '| paziente | file ROI | fenotipo | voxel | mm³ | fetta sagittale *i* | *j* | *k* | x mm | y mm | z mm | nota |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in righe:
        md.append(f"| {r['paziente']} | `{r['file_roi']}` | {r['fenotipo']} | {r['voxel']} | {r['mm3']} | {r['i1']} | {r['j1']} | "
                  f"{r['k1']} | {r['x_mm']} | {r['y_mm']} | {r['z_mm']} | {'**fuori dal cervello**' if r['fuori_dal_cervello'] else ''} |")
    per = {}
    for r in righe:
        per[r['paziente']] = per.get(r['paziente'], 0) + 1
    md += ['', f"Totale: **{len(righe)}** componenti, di cui {sum(r['fuori_dal_cervello'] for r in righe)} fuori dal cervello. "
           'Per paziente: ' + ', '.join(f'{p} {n}' for p, n in per.items()) + '.']
    (PROGETTO / 'reports/residui_annotazione.md').write_text('\n'.join(md) + '\n')
    figura(righe)
    print(f'{len(righe)} componenti; fuori dal cervello {sum(r["fuori_dal_cervello"] for r in righe)}')


if __name__ == '__main__':
    main()
