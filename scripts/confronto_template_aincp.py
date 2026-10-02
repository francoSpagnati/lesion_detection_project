#!/usr/bin/env python
"""Template adulto o pediatrico per standardizzare i pazienti AInCP: quale deforma meno le lesioni.

    python scripts/confronto_template_aincp.py   # -> reports/confronto_template_aincp.{md,json}

LA DECISIONE. All'incontro del 23 settembre il responsabile ha indicato di lavorare in uno spazio
standard (MNI) e di riportare le predizioni nello spazio del paziente solo alla fine: la
pipeline per un paziente nuovo e' nativo -> template -> modello -> nativo. L'obiezione misurata
l'11 settembre era che il template MNI152NLin2009aSym e' adulto e ingrandisce i cervelli dei
bambini di 1.6-2.5 volte; la risposta del responsabile e' che conta la *proporzione* di cervello
lesionato, che dovrebbe restare la stessa. Qui si misurano entrambe le cose su due template dello
stesso spazio MNI:

    adulto     MNI152NLin2009aSym, quello di ATLAS e del preprocessing gia' fatto
    pediatrico MNIPediatricAsym cohort-1 (4.5-18.5 anni; Fonov et al. 2011, NeuroImage), da
               TemplateFlow. Si usa la coorte intera perche' le eta' dei pazienti non le abbiamo.

COSA SI MISURA, per paziente e per template, con la stessa registrazione (ANTs SyN: rigida,
affine, non lineare) e la stessa maschera del cervello (SynthStrip sulla T1 nativa, portata nel
template con la trasformazione del paziente):
  - fattore di scala dell'affine (determinante): quanto viene ingrandito il cervello;
  - volume della lesione nel template rispetto al nativo;
  - percentuale di cervello lesionato nel template rispetto al nativo: e' la grandezza che il
    responsabile dice conservata. Se lesione e cervello vengono stirati allo stesso modo resta
    uguale; se la parte non lineare stira la lesione piu' del resto, cambia;
  - sovrapposizione (Dice) fra il cervello del paziente portato nel template e la maschera del
    template: quanto bene la registrazione allinea.

La registrazione adulta e' quella gia' calcolata da preprocess_aincp.py (trasformazioni salvate);
quella pediatrica si calcola qui e si salva in data/processed/AINCP/<soggetto>/transforms_ped/.
"""
import json
import shutil
import urllib.request
from pathlib import Path

import ants
import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
PROC = PROGETTO / 'data/processed/AINCP'
TPL_DIR = PROGETTO / 'data/templates'
S3 = 'https://templateflow.s3.amazonaws.com/tpl-MNIPediatricAsym/cohort-1/'
PED = {'T1w': 'tpl-MNIPediatricAsym_cohort-1_res-1_T1w.nii.gz',
       'mask': 'tpl-MNIPediatricAsym_cohort-1_res-1_desc-brain_mask.nii.gz'}
ADULTO = Path('/data01/atlas_mri_data/ATLAS_2/MNI152NLin2009aSym.nii.gz')


def scarica():
    TPL_DIR.mkdir(parents=True, exist_ok=True)
    for f in PED.values():
        if not (TPL_DIR / f).exists():
            urllib.request.urlretrieve(S3 + f, TPL_DIR / f)
    return ants.image_read(str(TPL_DIR / PED['T1w'])), ants.image_read(str(TPL_DIR / PED['mask']))


def maschera_adulto():
    """MNI152NLin2009aSym di ATLAS contiene tutta la testa: la maschera del cervello e' quella
    ufficiale di TemplateFlow, sulla stessa griglia (verificato: stessa forma e stesso affine)."""
    f = TPL_DIR / 'tpl-MNI152NLin2009aSym_res-1_desc-brain_mask.nii.gz'
    if not f.exists():
        urllib.request.urlretrieve('https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009aSym/' + f.name, f)
    return ants.image_read(str(f))


def det_affine(mat):
    # la trasformazione salvata da ANTs porta i punti del template in quelli del paziente (e' quella
    # che serve per ricampionare il paziente sul template): il suo determinante e' quindi
    # paziente/template, e il fattore di ingrandimento del paziente e' l'inverso
    t = ants.read_transform(str(mat))
    return 1.0 / float(abs(np.linalg.det(np.array(t.parameters[:9]).reshape(3, 3))))


def misura(sogg, fisso, maschera_fissa, trasf, det):
    t1 = ants.image_read(str(sogg / 'T1_N4.nii.gz'))
    lez = ants.image_read(str(sogg / 'lesione_inT1.nii.gz'))
    cer = ants.image_read(str(sogg / 'cervello_inT1.nii.gz'))
    vn = np.prod(t1.spacing)
    Ln, Bn = lez.numpy().astype(bool).sum() * vn, (cer.numpy() > .5).sum() * vn
    lt = ants.apply_transforms(fixed=fisso, moving=lez, transformlist=trasf, interpolator='nearestNeighbor').numpy() > .5
    bt = ants.apply_transforms(fixed=fisso, moving=cer, transformlist=trasf, interpolator='linear').numpy() > .5
    vt = np.prod(fisso.spacing)
    mf = maschera_fissa.numpy() > .5
    return {'scala_affine': det, 'lesione_mm3_nativo': Ln, 'lesione_mm3_template': lt.sum() * vt,
            'perc_nativo': 100 * Ln / Bn, 'perc_template': 100 * lt.sum() / bt.sum(),
            'lesione_fuori_cervello_%': 100 * (lt & ~bt).sum() / max(lt.sum(), 1),
            'dice_cervello_template': 2 * (bt & mf).sum() / (bt.sum() + mf.sum())}


def main():
    ped, ped_mask = scarica()
    adulto = ants.image_read(str(ADULTO)); adulto_mask = maschera_adulto()
    ris = {}
    for s in sorted(PROC.glob('FSM_*')):
        tr = s / 'transforms'
        adu = misura(s, adulto, adulto_mask, [str(tr / 'T1_to_MNI_fwd_0.nii.gz'), str(tr / 'T1_to_MNI_fwd_1.mat')],
                     det_affine(tr / 'T1_to_MNI_fwd_1.mat'))
        tp = s / 'transforms_ped'
        if not (tp / 'T1_to_PED_fwd_1.mat').exists():
            tp.mkdir(exist_ok=True)
            reg = ants.registration(fixed=ped, moving=ants.image_read(str(s / 'T1_N4.nii.gz')), type_of_transform='SyN')
            for i, t in enumerate(reg['fwdtransforms']):
                shutil.copy(t, tp / f'T1_to_PED_fwd_{i}{".mat" if t.endswith(".mat") else ".nii.gz"}')
            for i, t in enumerate(reg['invtransforms']):
                shutil.copy(t, tp / f'T1_to_PED_inv_{i}{".mat" if t.endswith(".mat") else ".nii.gz"}')
        pe = misura(s, ped, ped_mask, [str(tp / 'T1_to_PED_fwd_0.nii.gz'), str(tp / 'T1_to_PED_fwd_1.mat')],
                    det_affine(tp / 'T1_to_PED_fwd_1.mat'))
        ris[s.name[:7]] = {'adulto': adu, 'pediatrico': pe}
        print(s.name[:7], {k: round(v['scala_affine'], 2) for k, v in ris[s.name[:7]].items()}, flush=True)
    (PROGETTO / 'reports/confronto_template_aincp.json').write_text(json.dumps(ris, indent=1))

    r = ['# Template adulto o pediatrico per i pazienti AInCP', '',
         'Prodotto da `scripts/confronto_template_aincp.py`, **non modificare a mano**.', '',
         '| paziente | scala adulto | scala pediatrico | lesione adulto / nativo | lesione pediatrico / nativo | % lesionata nativo | % adulto | % pediatrico | Dice cervello adulto | Dice cervello pediatrico |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for p, v in ris.items():
        a, q = v['adulto'], v['pediatrico']
        r.append(f"| {p} | {a['scala_affine']:.2f} | {q['scala_affine']:.2f} | {a['lesione_mm3_template']/a['lesione_mm3_nativo']:.2f} | "
                 f"{q['lesione_mm3_template']/q['lesione_mm3_nativo']:.2f} | {a['perc_nativo']:.2f} | {a['perc_template']:.2f} | "
                 f"{q['perc_template']:.2f} | {a['dice_cervello_template']:.3f} | {q['dice_cervello_template']:.3f} |")
    for nome in ('adulto', 'pediatrico'):
        rel = [100 * (v[nome]['perc_template'] - v[nome]['perc_nativo']) / v[nome]['perc_nativo'] for v in ris.values()]
        sc = [v[nome]['scala_affine'] for v in ris.values()]
        r.append(f"\n**{nome}**: scala mediana {np.median(sc):.2f} ({min(sc):.2f}–{max(sc):.2f}); variazione relativa della "
                 f"percentuale lesionata mediana {np.median(rel):+.0f}%, da {min(rel):+.0f}% a {max(rel):+.0f}%, "
                 f"scarto assoluto mediano {np.median(np.abs(rel)):.0f}%.")
    (PROGETTO / 'reports/confronto_template_aincp.md').write_text('\n'.join(r) + '\n')
    print('scritto reports/confronto_template_aincp.md')


if __name__ == '__main__':
    main()
