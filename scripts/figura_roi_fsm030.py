#!/usr/bin/env python
"""Le due ROI_1 di FSM_030 a confronto, per decidere quale usare.

    python scripts/figura_roi_fsm030.py    # -> reports/figures/aincp/fsm030_due_roi1.png

PERCHE'. Nel secondo lotto FSM_030 ha due file per la gliosi: `ROI_1.nii` (31/3, 16:05, 5151 voxel)
e `FSM_030_FLAIR_original_ROI_1 - Copia.nii` (31/3, 12:25, 6460 voxel), con Dice 0.72 fra loro. Non
sono una copia dell'altra: chi deve scegliere ha bisogno di vederle sulla FLAIR, fetta per fetta.

COME. Tutto nella FLAIR originale, dove sono state disegnate. Si mostrano le fette in cui le due ROI
differiscono di piu' (piu' voxel in una sola delle due), in tre orientamenti, ritagliate attorno alla
lesione perche' e' piccola (~5 000 mm³). Blu continuo = ROI_1.nii, arancio tratteggiato = la copia: lo
stile della linea distingue le due anche senza colore. Gli indici delle fette sono contati da 1, come
li mostra MRIcron, cosi' si ritrovano nel visualizzatore.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D

PROGETTO = Path(__file__).resolve().parent.parent
CO = PROGETTO / 'data/raw/AINCP/dataset_FSM_v2/FSM_030_MR_2_anon/COREGISTERED'
OUT = PROGETTO / 'reports/figures/aincp/fsm030_due_roi1.png'
BLU, ARANCIO = '#2a78d6', '#eb6834'          # slot 1 e 2 della palette, validati (CVD dE 24.7)
INK, INK2 = '#0b0b0b', '#52514e'


def carica():
    fl = nib.load(str(CO / 'FSM_030_FLAIR_original.nii.gz'))
    a = np.squeeze(np.asanyarray(nib.load(str(CO / 'ROI_1.nii')).dataobj)) > 0
    b = np.squeeze(np.asanyarray(nib.load(str(CO / 'FSM_030_FLAIR_original_ROI_1 - Copia.nii')).dataobj)) > 0
    return np.asanyarray(fl.dataobj).astype(np.float32), a, b, fl


def fette_migliori(a, b, asse, n):
    """Le n fette con piu' voxel in disaccordo, distanziate di almeno 3 per non ripetersi."""
    somma = tuple(x for x in range(3) if x != asse)
    dis = (a ^ b).sum(axis=somma)
    scelte = []
    for i in np.argsort(-dis):
        if dis[i] == 0 or len(scelte) == n:
            break
        if all(abs(i - j) >= 3 for j in scelte):
            scelte.append(int(i))
    return sorted(scelte)


def ritaglio(a, b, asse, margine=18):
    """Riquadro attorno all'unione delle due ROI, nel piano della fetta."""
    m = a | b
    idx = np.argwhere(m)
    lo, hi = idx.min(0) - margine, idx.max(0) + margine
    assi = [x for x in range(3) if x != asse]
    return [(max(0, lo[x]), min(m.shape[x], hi[x] + 1)) for x in assi]


def fetta(v, asse, i):
    return np.take(v, i, axis=asse)


def main():
    fl, a, b, img = carica()
    vox = float(np.prod(img.header.get_zooms()[:3]))
    nome_asse = {0: 'sagittale', 1: 'coronale', 2: 'assiale'}
    # orientamento del file: LAS. Asse 0 cresce verso sinistra (a sinistra dell'immagine c'e' la
    # destra del paziente), asse 1 verso anteriore, asse 2 verso superiore
    etichette = {0: ('P', 'A', 'I', 'S'), 1: ('R', 'L', 'I', 'S'), 2: ('R', 'L', 'P', 'A')}
    righe = [(0, 4), (2, 4), (1, 4)]
    vmax = np.percentile(fl[fl > 0], 99.5)
    fig, ax = plt.subplots(len(righe), 4, figsize=(16, 15), facecolor='#fcfcfb')
    for r, (asse, n) in enumerate(righe):
        fette = fette_migliori(a, b, asse, n)
        (u0, u1), (w0, w1) = ritaglio(a, b, asse)
        for c in range(4):
            x = ax[r, c]
            x.set_facecolor('#fcfcfb')
            x.set_xticks([]); x.set_yticks([])
            for s in x.spines.values():
                s.set_visible(False)
            if c >= len(fette):
                continue
            i = fette[c]
            sl = fetta(fl, asse, i)[u0:u1, w0:w1]
            sa, sb = fetta(a, asse, i)[u0:u1, w0:w1], fetta(b, asse, i)[u0:u1, w0:w1]
            # .T con origin='lower': orizzontale il primo asse rimasto, verticale il secondo
            x.imshow(sl.T, cmap='gray', origin='lower', vmin=0, vmax=vmax, interpolation='nearest')
            if sa.any():
                x.contour(sa.T.astype(float), levels=[0.5], colors=BLU, linewidths=2.0, origin='lower')
            if sb.any():
                x.contour(sb.T.astype(float), levels=[0.5], colors=ARANCIO, linewidths=2.0,
                          linestyles='--', origin='lower')
            x.set_title(f'{nome_asse[asse]}, fetta {i + 1}\nvoxel: solo blu {int((sa & ~sb).sum())} · '
                        f'solo arancio {int((sb & ~sa).sum())} · comuni {int((sa & sb).sum())}',
                        fontsize=10, color=INK2, loc='left')
            e = etichette[asse]
            for testo, pos, ha, va in ((e[0], (0.01, 0.5), 'left', 'center'), (e[1], (0.99, 0.5), 'right', 'center'),
                                       (e[2], (0.5, 0.01), 'center', 'bottom'), (e[3], (0.5, 0.99), 'center', 'top')):
                x.text(*pos, testo, transform=x.transAxes, ha=ha, va=va, color='white', fontsize=9, fontweight='bold')
    entrambe = int((a & b).sum())
    fig.suptitle('FSM_030 — le due ROI_1 (gliosi) sulla FLAIR originale: quale è quella giusta?',
                 x=0.012, ha='left', fontsize=15, color=INK, fontweight='bold', y=0.995)
    va, vb = (f'{x.sum() * vox:,.0f}'.replace(',', '\u202f') for x in (a, b))
    fig.text(0.012, 0.962,
             f'Blu continuo: ROI_1.nii (31/3 ore 16:05) — {int(a.sum())} voxel, {va} mm³.   '
             f'Arancio tratteggiato: FSM_030_FLAIR_original_ROI_1 - Copia.nii (31/3 ore 12:25) — {int(b.sum())} voxel, '
             f'{vb} mm³.\nIn comune {entrambe} voxel (Dice {2 * entrambe / (a.sum() + b.sum()):.2f}). '
             'Fette scelte dove le due differiscono di più, ritagliate attorno alla lesione; numero di fetta contato da 1 '
             '(come in MRIcron). Righe: sagittale, assiale, coronale.',
             ha='left', va='top', fontsize=10.5, color=INK2)
    fig.legend(handles=[Line2D([], [], color=BLU, lw=2.4, label='ROI_1.nii (16:05)'),
                        Line2D([], [], color=ARANCIO, lw=2.4, ls='--', label='… ROI_1 - Copia.nii (12:25)')],
               loc='upper right', bbox_to_anchor=(0.995, 0.995), frameon=False, fontsize=11, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2, w_pad=1.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, facecolor=fig.get_facecolor())
    print('scritto', OUT)


if __name__ == '__main__':
    main()
