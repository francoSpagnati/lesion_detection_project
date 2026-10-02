#!/usr/bin/env python
"""Controllo della registrazione AInCP verso MNI, e misura dell'ingrandimento che comporta.

    python scripts/figura_registrazione_aincp.py

Produce due figure e un file di numeri:

    reports/figures/aincp/registrazione_mni.png   T1 di ogni paziente portata in MNI, con sopra
                                                  il contorno del template: se la registrazione
                                                  e' buona l'anatomia segue le linee
    reports/figures/aincp/ingrandimento_mni.png   quanto cresce il volume della lesione passando
                                                  in MNI, per paziente, confrontato con quanto
                                                  prevede la matrice affine
    reports/aincp_scala_mni.json                  gli stessi numeri, per il testo

Il punto che le due figure devono rendere guardabile: MNI152 e' un template ADULTO. Per far
combaciare il cervello di un bambino la registrazione lo ingrandisce, e con lui ingrandisce ogni
volume, lesione compresa. Il determinante della parte lineare della trasformazione e' esattamente
il fattore di cui cambiano i volumi, e va confrontato con l'aumento misurato sulle maschere: se
coincidono, l'aumento e' un ingrandimento e non un errore di registrazione o della lesione.

La parte lineare (la matrice .mat) e' gia' su disco, quindi il confronto non richiede una nuova
registrazione: basta applicare alla maschera solo quella.
"""
import json
from pathlib import Path

import ants
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
PROC = PROGETTO / 'data/processed/AINCP'
FIG = PROGETTO / 'reports/figures/aincp'
TEMPLATE = Path('/data01/atlas_mri_data/ATLAS_2/MNI152NLin2009aSym.nii.gz')

# Palette del progetto (references/palette.md della skill dataviz): i primi tre slot categorici
# sono validati a tutte le coppie. Il giallo (slot 4) e' l'unica serie della figura di
# registrazione, quindi non ha coppie da cui distinguersi: serve solo che si veda sul grigio, e
# non e' nessuno dei tre colori dei fenotipi, cosi' non viene scambiato per una ROI.
BLU, ARANCIO, ACQUA, GIALLO = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INCHIOSTRO, INCHIOSTRO_2, GRIGLIA = '#0b0b0b', '#52514e', '#e6e5e1'


def mm3(img):
    return float(img.numpy().sum()) * float(np.prod(img.spacing))


def misura_ingrandimento():
    """Per ogni paziente: fattore previsto dal determinante, e aumento misurato in due modi."""
    template = ants.image_read(str(TEMPLATE))
    righe = []
    for m in sorted(PROC.glob('*/manifesto.json')):
        d = json.loads(m.read_text())
        c = m.parent
        mat = c / 'transforms/T1_to_MNI_fwd_1.mat'
        # ANTs salva la matrice nella direzione template -> paziente: il fattore
        # paziente -> template e' il reciproco del determinante.
        M = np.asarray(ants.read_transform(str(mat)).parameters[:9]).reshape(3, 3)
        fattore = 1.0 / float(np.linalg.det(M))
        lesione = ants.image_read(str(c / 'lesione_inT1.nii.gz'))
        solo_affine = ants.apply_transforms(template, lesione, [str(mat)],
                                            interpolator='nearestNeighbor')
        nativo = d['stadi']['1_spazio_T1']['mm3']
        righe.append({
            'soggetto': d['soggetto'],
            'fattore_volume_affine': round(fattore, 3),
            'previsto_%': round(100 * (fattore - 1), 1),
            'misurato_solo_affine_%': round(100 * (mm3(solo_affine) / nativo - 1), 1),
            'misurato_syn_%': round(100 * (d['stadi']['2_spazio_MNI']['mm3'] / nativo - 1), 1),
            'nativo_mm3': nativo,
        })
    return righe


def figura_ingrandimento(righe):
    nomi = [r['soggetto'].split('_')[1] for r in righe]
    serie = [('previsto dal determinante', 'previsto_%', BLU),
             ('misurato, solo affine', 'misurato_solo_affine_%', ARANCIO),
             ('misurato, affine + SyN (la pipeline)', 'misurato_syn_%', ACQUA)]
    y = np.arange(len(righe))
    h = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for i, (nome, chiave, col) in enumerate(serie):
        v = [r[chiave] for r in righe]
        ax.barh(y + (i - 1) * h, v, height=h - 0.03, color=col, label=nome, zorder=3)
    ax.set_yticks(y, [f'FSM_{n}' for n in nomi])
    ax.invert_yaxis()
    ax.set_xlabel('aumento del volume della lesione passando in MNI (%)', color=INCHIOSTRO_2)
    ax.xaxis.set_major_formatter(lambda v, _: f'+{v:.0f}%')
    ax.grid(axis='x', color=GRIGLIA, zorder=0)
    ax.axvline(0, color=INCHIOSTRO_2, lw=1)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRIGLIA)
    ax.tick_params(colors=INCHIOSTRO_2, length=0)
    ax.legend(frameon=False, loc='lower right', fontsize=9)
    # etichette dirette solo dove il messaggio e' piu' forte: la coppia che coincide
    for i, r in enumerate(righe):
        ax.text(r['misurato_solo_affine_%'] + 2, i, f"{r['misurato_solo_affine_%']:+.0f}%",
                va='center', fontsize=8, color=INCHIOSTRO_2)
    ax.set_title('Il passaggio a MNI ingrandisce le lesioni, e la matrice lo prevede',
                 loc='left', fontsize=11, color=INCHIOSTRO)
    fig.tight_layout()
    fig.savefig(FIG / 'ingrandimento_mni.png', dpi=150)
    plt.close(fig)


def figura_registrazione():
    """T1 in MNI di ogni paziente, tre viste, col contorno del template sopra."""
    from scipy import ndimage
    template = ants.image_read(str(TEMPLATE)).numpy()
    # due contorni: la testa (fondo vs tessuto) e la sostanza bianca, che e' il tessuto piu'
    # chiaro in T1. Il primo dice se la taglia combacia, il secondo se combacia l'interno.
    dentro = template[template > 1]
    soglie = (np.percentile(dentro, 40), np.percentile(dentro, 80))

    def contorno(fetta, soglia):
        # il template ha un po' di rumore fuori dalla testa: si tiene solo l'oggetto piu' grande
        m = ndimage.binary_opening(fetta > soglia, iterations=2)
        lab, n = ndimage.label(m)
        if n == 0:
            return m
        return lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    casi = sorted(PROC.glob('*/T1_MNI.nii.gz'))
    per_riga = 2
    righe = (len(casi) + per_riga - 1) // per_riga
    fig, axes = plt.subplots(righe, 3 * per_riga, figsize=(3.0 * 3 * per_riga, 3.4 * righe),
                             gridspec_kw={'hspace': .22, 'wspace': .02})
    axes = np.atleast_2d(axes)
    cx, cy, cz = [s // 2 for s in template.shape]

    def viste(v):
        return [np.rot90(v[cx, :, :]), np.rot90(v[:, cy, :]), np.rot90(v[:, :, cz])]

    tpl = viste(template)
    for k, f in enumerate(casi):
        v = ants.image_read(str(f)).numpy()
        lo, hi = np.percentile(v[v > 0], [1, 99.5])
        r, c0 = divmod(k, per_riga)
        for j, (img, tp) in enumerate(zip(viste(v), tpl)):
            ax = axes[r, 3 * c0 + j]
            ax.imshow(img, cmap='gray', vmin=lo, vmax=hi, interpolation='nearest')
            for sg, lw in zip(soglie, (1.2, 0.7)):
                ax.contour(contorno(tp, sg), levels=[.5], colors=[GIALLO], linewidths=lw)
            ax.set_axis_off()
            if j == 1:
                ax.set_title(f.parent.name.replace('_anon', ''), fontsize=10, color=INCHIOSTRO)
    for ax in np.ravel(axes)[3 * len(casi):]:
        ax.set_axis_off()
    fig.suptitle('T1 di ogni paziente portata in MNI; in giallo i contorni del template adulto '
                 '(testa e sostanza bianca)', x=0.01, y=0.995, ha='left', fontsize=12,
                 color=INCHIOSTRO)
    fig.subplots_adjust(top=0.965, bottom=0.005, left=0.005, right=0.995)
    fig.savefig(FIG / 'registrazione_mni.png', dpi=110, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    FIG.mkdir(parents=True, exist_ok=True)
    righe = misura_ingrandimento()
    (PROGETTO / 'reports/aincp_scala_mni.json').write_text(
        json.dumps(righe, indent=1, ensure_ascii=False))
    figura_ingrandimento(righe)
    figura_registrazione()
    for r in righe:
        print(f"{r['soggetto']:20s} fattore {r['fattore_volume_affine']:.2f}  previsto "
              f"{r['previsto_%']:+6.1f}%  affine {r['misurato_solo_affine_%']:+6.1f}%  "
              f"SyN {r['misurato_syn_%']:+6.1f}%")
