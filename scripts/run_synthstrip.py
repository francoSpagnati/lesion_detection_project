#!/usr/bin/env python
"""Applica SynthStrip a tutti i casi di DBB e ne misura la qualita' contro le
maschere cerebrali vere.

**Perche' DBB e' il banco di prova giusto.** Fornisce 244 maschere di riferimento,
di cui 37 su cervelli malformati (ACC, HD, MCD, PFM). E' l'unico posto dove si puo'
rispondere alla domanda che conta prima di applicare lo strumento a WMH e ATLAS,
dove nessun riferimento esiste: **lo skull stripping regge sulle anatomie alterate?**
E' il caso in cui BET fallisce, ed e' il motivo per cui BET e' stato escluso.

Il riferimento e' la parcellazione binarizzata, che coincide con la maschera
fornita (verificato: Dice 1.0000). Include il CSF, quindi va confrontata con il
modello **principale** di SynthStrip e non con le varianti "without CSF".

Produce:
  - le maschere in `data/processed/synthstrip/DBB/`
  - un'immagine di controllo per caso in `reports/figures/synthstrip/`, **con il
    Dice come prefisso del nome**: ordinando la cartella per nome si vedono per
    primi i casi peggiori, che sono quelli da guardare davvero
  - `reports/synthstrip_dbb.md` con il Dice per patologia

Uso:
    python scripts/run_synthstrip.py                # tutti i 244 casi
    python scripts/run_synthstrip.py --n 20         # prova rapida
"""
import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

PROGETTO = Path(__file__).resolve().parent.parent
DBB = PROGETTO / 'data' / 'raw' / 'DBB'
SCRIPT = PROGETTO / 'scripts' / 'synthstrip' / 'synthstrip.py'
PESI = PROGETTO / 'synthstrip.1.pt'
USCITA = PROGETTO / 'data' / 'processed' / 'synthstrip' / 'DBB'
FIGURE = PROGETTO / 'reports' / 'figures' / 'synthstrip'


def casi_dbb():
    """Un record per soggetto con immagine: percorso T1, maschera vera, patologia."""
    m = pd.read_csv(DBB / 'manifest.csv')
    m['gruppo'] = m.tags.str.split('|').str[1]
    con_img = set(m[m.datatype_slug == 'anat-t1w'].subject)
    m = m[m.subject.isin(con_img)]
    out = []
    for s in sorted(con_img):
        r = m[m.subject == s]
        def trova(slug):
            rr = r[r.datatype_slug == slug]
            if not len(rr):
                return None
            f = glob.glob(str(DBB / rr.iloc[0].path / '*.nii*'))
            return f[0] if f else None
        img, mask = trova('anat-t1w'), trova('mask')
        if img and mask:
            out.append(dict(soggetto=f'sub-{s:04d}', gruppo=r.iloc[0].gruppo, img=img, mask=mask))
    return out


def figura(caso, img_p, pred, vero, dice, dest):
    """Tre fette assiali con i due contorni sovrapposti: quello che si guarda a occhio."""
    img = np.asarray(nib.load(img_p).dataobj, dtype=np.float32)
    carico = vero.sum(axis=(0, 1))
    ks = np.linspace(max(carico.argmax() - 30, 5), min(carico.argmax() + 30, img.shape[2] - 5), 3).astype(int)
    lo, hi = np.percentile(img[img > 0], (1, 99)) if (img > 0).any() else (0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
    for ax, k in zip(axes, ks):
        ax.imshow(np.rot90(img[:, :, k]), cmap='gray', vmin=lo, vmax=max(hi, lo + 1e-6))
        p, v = np.rot90(pred[:, :, k]), np.rot90(vero[:, :, k])
        if v.any():
            ax.contour(v, [.5], colors='#00d67d', linewidths=1.0, linestyles='--')
        if p.any():
            ax.contour(p, [.5], colors='#ff4d4d', linewidths=1.3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'fetta {k}', fontsize=8, color='#666')
    fig.suptitle(f'{caso["soggetto"]} — {caso["gruppo"]} — Dice {dice:.4f}   '
                 f'(verde tratteggiato = riferimento, rosso = SynthStrip)', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(dest, dpi=95, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=0, help='limita il numero di casi (0 = tutti)')
    ap.add_argument('--gpu', action='store_true', help='usa la GPU (default CPU, ~6 s per caso)')
    args = ap.parse_args()

    if not PESI.exists():
        sys.exit(f'pesi non trovati: {PESI}')
    USCITA.mkdir(parents=True, exist_ok=True)
    FIGURE.mkdir(parents=True, exist_ok=True)

    casi = casi_dbb()
    if args.n:
        # campiona fra tutti i gruppi, non i primi n che sarebbero tutti sani
        df = pd.DataFrame(casi)
        casi = pd.concat([g.head(max(1, args.n // df.gruppo.nunique()))
                          for _, g in df.groupby('gruppo')]).to_dict('records')
    print(f'{len(casi)} casi, modello {PESI.name}, {"GPU" if args.gpu else "CPU"}')

    righe, t0 = [], time.time()
    for i, c in enumerate(casi, 1):
        out_mask = USCITA / f'{c["soggetto"]}_mask.nii.gz'
        if not out_mask.exists():
            cmd = [sys.executable, str(SCRIPT), '-i', c['img'], '-m', str(out_mask), '--model', str(PESI)]
            if args.gpu:
                cmd.append('-g')
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f'  {c["soggetto"]}: FALLITO\n{r.stderr[-300:]}')
                continue
        pred = np.asarray(nib.load(out_mask).dataobj) > 0
        rif = nib.load(c['mask'])
        vero = np.asarray(rif.dataobj) > 0
        vox = abs(np.linalg.det(rif.affine[:3, :3]))
        dice = 2 * (pred & vero).sum() / (pred.sum() + vero.sum())
        righe.append(dict(soggetto=c['soggetto'], gruppo=c['gruppo'], dice=dice,
                          mm3_synthstrip=pred.sum() * vox, mm3_riferimento=vero.sum() * vox,
                          mm3_solo_synthstrip=(pred & ~vero).sum() * vox,
                          mm3_solo_riferimento=(~pred & vero).sum() * vox))
        # il Dice come prefisso: ordinando per nome, i peggiori vengono per primi
        figura(c, c['img'], pred, vero, dice, FIGURE / f'{dice:.3f}_{c["soggetto"]}_{c["gruppo"]}.png')
        if i % 20 == 0 or i == len(casi):
            print(f'  {i}/{len(casi)}  ({time.time()-t0:.0f}s)')

    d = pd.DataFrame(righe)
    d.to_csv(PROGETTO / 'reports' / 'synthstrip_dbb.csv', index=False)

    patologici = d[d.gruppo != 'Healthy']
    md = ['# SynthStrip su DBB: qualita\' contro le maschere di riferimento', '',
          'Generato da `scripts/run_synthstrip.py`, **non modificare a mano**.', '',
          f'Modello: `synthstrip.1.pt` (principale, con CSF). Casi: **{len(d)}**.', '',
          'Il riferimento e\' la maschera cerebrale fornita da DBB. La domanda a cui risponde questa',
          'misura: **lo skull stripping regge sulle anatomie alterate?** — il caso in cui BET fallisce.', '',
          '## Dice per gruppo', '',
          '| gruppo | n | Dice medio | mediana | minimo |', '| --- | --- | --- | --- | --- |']
    for g, gg in d.groupby('gruppo'):
        md.append(f'| {g} | {len(gg)} | {gg.dice.mean():.4f} | {gg.dice.median():.4f} | {gg.dice.min():.4f} |')
    if len(patologici):
        md += ['', f'**Sani: {d[d.gruppo=="Healthy"].dice.mean():.4f}   |   '
                   f'patologici: {patologici.dice.mean():.4f}** '
                   f'(differenza {patologici.dice.mean()-d[d.gruppo=="Healthy"].dice.mean():+.4f})', '']
    peggiori = d.nsmallest(10, 'dice')
    md += ['## I dieci casi peggiori', '',
           '| soggetto | gruppo | Dice | mm³ SynthStrip | mm³ riferimento | mm³ solo SynthStrip | mm³ solo riferimento |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for _, r in peggiori.iterrows():
        md.append((f'| {r.soggetto} | {r.gruppo} | {r.dice:.4f} | {r.mm3_synthstrip:,.0f} | '
                   f'{r.mm3_riferimento:,.0f} | {r.mm3_solo_synthstrip:,.0f} | {r.mm3_solo_riferimento:,.0f} |').replace(',', '\u202f'))
    md += ['', f'Immagini di controllo in `reports/figures/synthstrip/`, **una per caso, con il Dice come',
           'prefisso del nome**: ordinando la cartella per nome si guardano prima i casi peggiori.',
           f'Maschere in `data/processed/synthstrip/DBB/`.', '']
    (PROGETTO / 'reports' / 'synthstrip_dbb.md').write_text('\n'.join(md) + '\n')
    print(f'\nDice medio {d.dice.mean():.4f} (min {d.dice.min():.4f}) -> reports/synthstrip_dbb.md')


if __name__ == '__main__':
    main()
