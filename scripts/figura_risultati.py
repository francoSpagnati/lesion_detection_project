#!/usr/bin/env python
"""Figura riassuntiva dei risultati, per l'incontro con i relatori.

    python scripts/figura_risultati.py     # -> reports/figures/risultati.png

Due pannelli, perche' rispondono a due domande diverse e mescolarli su un asse solo
sarebbe scorretto:

  A. **Quanto vale il canale, su WMH.** Tre configurazioni sullo stesso dataset, stessi
     casi, stessa architettura: e' un confronto di grandezze, quindi barre.
  B. **Le quattro modifiche di pipeline provate dopo le baseline.** Qui il dato e' una
     differenza appaiata rispetto alla propria baseline, cioe' ha un segno: quindi punti
     attorno allo zero, con il valore di p accanto a ciascuno.

Tutti i numeri sono letti dai report di `scripts/confronto_appaiato.py`, non riscritti
a mano: se un confronto viene rifatto, questa figura cambia con lui.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROGETTO = Path(__file__).resolve().parent.parent
REPORT = PROGETTO / 'reports'

# Token del sistema di colori: pareggio caldo/freddo per il segno, inchiostri neutri
# per il testo. Il colore non e' l'unico canale — la posizione rispetto allo zero
# porta gia' il segno — quindi non serve legenda.
SUPERFICIE = '#fcfcfb'
INK, INK2, MUTO = '#0b0b0b', '#52514e', '#898781'
GRIGLIA, ASSE = '#e1e0d9', '#c3c2b7'
BLU, ROSSO = '#2a78d6', '#d03b3b'

CONFRONTI = [
    ('confronto_512_due_canali.md',    'Aggiungere il T1 alla FLAIR\nWMH: 512 contro 501'),
    ('test_esclusione_1_5T.md',        'Togliere i casi a 1.5 T\nATLAS: 521 contro 502'),
    ('confronto_522_skullstrip.md',    'Skull stripping\nWMH: 522 contro 501'),
    ('confronto_523_skullstrip.md',    'Skull stripping\nATLAS: 523 contro 502'),
]

CANALI = [('solo T1', 0.6257), ('FLAIR + T1', 0.7535), ('solo FLAIR', 0.7739)]


def leggi(nome: str):
    """Estrae differenza appaiata e p dal report, invece di ricopiarli."""
    t = (REPORT / nome).read_text()
    d = float(re.search(r'\*\*differenza\*\* \| \*\*([+-][\d.]+)\*\*', t).group(1))
    p = float(re.search(r'Wilcoxon appaiato: \*\*p = ([\d.e-]+)\*\*', t).group(1))
    return d, p


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 4.4), facecolor=SUPERFICIE,
                                   gridspec_kw={'width_ratios': [1, 1.5]})

    # --- A: il valore del canale ------------------------------------------
    nomi = [n for n, _ in CANALI]
    val = [v for _, v in CANALI]
    ax1.barh(nomi, val, height=0.5, color=BLU, zorder=3)
    for y, v in enumerate(val):
        ax1.text(v + 0.012, y, f'{v:.3f}', va='center', fontsize=11, color=INK,
                 fontweight='bold')
    ax1.set_xlim(0, 0.92)
    ax1.set_title('Quanto vale il canale, su WMH\nstessi 16 casi, stessa rete',
                  fontsize=12, color=INK, loc='left', pad=12)
    ax1.set_xlabel('Dice di validation', fontsize=9.5, color=INK2)
    ax1.xaxis.grid(True, color=GRIGLIA, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)

    # --- B: le quattro modifiche ------------------------------------------
    dati = [(et, *leggi(f)) for f, et in CONFRONTI][::-1]
    y = range(len(dati))
    for i, (et, d, p) in zip(y, dati):
        colore = BLU if d > 0 else ROSSO
        ax2.plot([0, d], [i, i], color=colore, linewidth=2, zorder=3,
                 solid_capstyle='round')
        ax2.plot([d], [i], 'o', color=colore, markersize=9, zorder=4,
                 markeredgecolor=SUPERFICIE, markeredgewidth=2)
        # I valori vanno in una colonna fissa fuori dagli assi: appesi al punto si
        # sovrapporrebbero alle etichette di riga quando la differenza e' negativa.
        significativo = p < 0.05
        ax2.text(1.05, i, f'{d:+.4f}', transform=ax2.get_yaxis_transform(),
                 va='center', ha='left', fontsize=10.5,
                 color=INK if significativo else INK2,
                 fontweight='bold' if significativo else 'normal')
        ax2.text(1.30, i, f'p = {p:.2g}' + ('  significativo' if significativo else ''),
                 transform=ax2.get_yaxis_transform(), va='center', ha='left', fontsize=9.5,
                 color=INK if significativo else MUTO,
                 fontweight='bold' if significativo else 'normal')
    ax2.axvline(0, color=ASSE, linewidth=1.4, zorder=2)
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([et for et, _, _ in dati], fontsize=9.5, color=INK2)
    ax2.set_ylim(-0.6, len(dati) - 0.25)
    ax2.set_xlim(-0.026, 0.014)
    ax2.set_title('Le quattro modifiche provate dopo le baseline\n'
                  'differenza di Dice sugli stessi casi, rispetto alla propria baseline',
                  fontsize=12, color=INK, loc='left', pad=12)
    ax2.set_xlabel('← peggiora            invariato            migliora →',
                   fontsize=9.5, color=INK2)
    ax2.set_xticks([-0.02, -0.01, 0.0, 0.01])
    ax2.xaxis.grid(True, color=GRIGLIA, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        ax.set_facecolor(SUPERFICIE)
        for lato in ('top', 'right', 'left'):
            ax.spines[lato].set_visible(False)
        ax.spines['bottom'].set_color(ASSE)
        ax.tick_params(colors=MUTO, labelsize=9.5, length=0)
        for e in ax.get_yticklabels():
            e.set_color(INK2)

    # Il margine destro deve ospitare la colonna dei valori, che sta fuori dagli assi.
    fig.tight_layout(pad=1.8, w_pad=3.5, rect=(0, 0, 0.83, 1))
    out = REPORT / 'figures' / 'risultati.png'
    fig.savefig(out, dpi=150, facecolor=SUPERFICIE)
    print(f'scritta {out}')


if __name__ == '__main__':
    main()
