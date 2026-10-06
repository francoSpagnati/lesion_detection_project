#!/usr/bin/env python
"""Ventricoli con SynthSeg sui 17 pazienti AInCP, e la loro prova come postprocessing di Z.

    python scripts/synthseg_aincp.py segmenta     # SynthSeg sulla T1 di ogni paziente (ambiente TensorFlow a parte)
    python scripts/synthseg_aincp.py ventricoli   # maschera dei ventricoli nella griglia T1 nativa
    python scripts/synthseg_aincp.py valuta       # -> reports/ventricoli_synthseg.{md,json}

PERCHE'. Sul test Z ha segmentato il ventricolo laterale dilatato di FSM_046 (9 458 mm³ predetti
contro 454 veri): ha scambiato il liquor per una cavita'. Il filtro per volume non toglie una
componente cosi' grande. Una maschera dei ventricoli puo' toglierla, ma deve essere **automatica**:
la ROI_3 del clinico (il ventricolo dell'emisfero colpito) su un paziente nuovo non c'e', e usarla
darebbe al modello un pezzo della risposta. Qui la ROI_3 serve solo a controllare SynthSeg.

SYNTHSEG. Billot et al. 2023 (Medical Image Analysis): segmenta il cervello in strutture su
qualunque contrasto e risoluzione, perche' e' addestrato su immagini sintetiche. Versione 1.0
(`--v1`): i pesi della 2.0 stanno su un archivio che richiede il browser, quelli della 1.0 nel
repository. Gira in un ambiente a parte (`~/.venv_synthseg`, TensorFlow 2.10 CPU) per non toccare
quello di nnU-Net; sorgente in `~/strumenti/SynthSeg`. Circa 30 s per paziente su 16 thread.
Ventricoli = etichette 4, 43 (laterali), 5, 44 (corni inferiori), 14 (terzo), 15 (quarto).
L'uscita di SynthSeg e' a 1 mm, ritagliata; si riporta sulla griglia T1 nativa con il vicino piu'
prossimo, usando le affini (stesso spazio, nessuna registrazione).

LA REGOLA, decisa prima di guardare i risultati e senza griglia di parametri: dopo il postprocessing
congelato di Z si toglie una componente predetta se **almeno meta' dei suoi voxel** cade nei
ventricoli. E' la stessa forma della regola gia' prevista per la maschera del cervello.

LA VALUTAZIONE (`valuta`), sui 12 pazienti fuori-fold, mai sul test:
1. **SynthSeg contro la ROI_3**: quanta parte del ventricolo annotato dal clinico cade nella maschera;
2. **rischio**: quanta lesione annotata cade nei ventricoli, e quante lesioni vere la regola toglie;
3. **effetto** su Z: falsi positivi, lesioni trovate, Dice, F1, con e senza la regola, anche sulle
   predizioni senza postprocessing (dove i falsi positivi sono 20 invece di 1).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocessing_aincp as pp  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
PROC = PROGETTO / 'data/processed/AINCP'
SYNTHSEG = Path.home() / 'strumenti/SynthSeg'
PYTHON_TF = Path.home() / '.venv_synthseg/bin/python'
VENTRICOLI = (4, 43, 5, 44, 14, 15)
QUOTA = 0.5
RUN_Z = (Path(os.environ['nnUNet_results']) / 'Dataset506_AINCP_FINALE'
         / 'nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres')
OUT = PROGETTO / 'reports/ventricoli_synthseg'


def sessioni():
    return sorted(d for d in PROC.iterdir() if d.is_dir() and (d / 'T1_N4.nii.gz').exists())


def segmenta():
    for s in sessioni():
        if (s / 'synthseg.nii.gz').exists():
            continue
        subprocess.run([str(PYTHON_TF), str(SYNTHSEG / 'scripts/commands/SynthSeg_predict.py'), '--v1',
                        '--i', str(s / 'T1_N4.nii.gz'), '--o', str(s / 'synthseg.nii.gz'),
                        '--vol', str(s / 'synthseg_volumi.csv'), '--threads', '16'],
                       cwd=SYNTHSEG, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(s.name, 'fatto', flush=True)


def ventricoli():
    import nibabel as nib
    from nibabel.processing import resample_from_to
    for s in sessioni():
        t1 = nib.load(str(s / 'T1_N4.nii.gz'))
        seg = nib.load(str(s / 'synthseg.nii.gz'))
        assert nib.aff2axcodes(seg.affine) == nib.aff2axcodes(t1.affine)
        r = np.asanyarray(resample_from_to(seg, (t1.shape[:3], t1.affine), order=0).dataobj)
        m = np.isin(np.rint(r).astype(int), VENTRICOLI)
        nib.save(nib.Nifti1Image(m.astype(np.uint8), t1.affine), str(s / 'ventricoli_synthseg_inT1.nii.gz'))
        print(s.name, f'ventricoli {pp.mm3(m.sum() * float(np.prod(t1.header.get_zooms()[:3])))}', flush=True)


def roi3_in_t1(caso: str, s: Path):
    """La ROI_3 nella griglia T1, con la stessa strada delle altre ROI (pipeline_aincp.py)."""
    import ants
    import pipeline_aincp as pa
    c = pa.trova_casi()[caso]
    if 'ROI_3' not in c['roi']:
        return None
    t1 = ants.image_read(str(s / 'T1_N4.nii.gz'))
    rif = ants.image_read(str(c['FLAIR']))
    m = np.squeeze(ants.image_read(str(c['roi']['ROI_3'])).numpy()) > 0
    a = ants.apply_transforms(fixed=t1, moving=rif.new_image_like(m.astype('float32')),
                              transformlist=[str(s / 'transforms' / 'FLAIR_to_T1_0.mat')], interpolator='linear')
    return a.numpy() >= 0.5


def togli_ventricoli(pred, ven):
    from scipy import ndimage
    lab, n = ndimage.label(pred, structure=pp.STRUTTURA)
    if n == 0:
        return pred, 0
    idx = np.arange(1, n + 1)
    dentro = ndimage.mean(ven, lab, idx) >= QUOTA
    return np.isin(lab, idx[~dentro]), int(dentro.sum())


def valuta_caso(c: str):
    import nibabel as nib
    s = pp.sessione(c)
    ref, cer, vox = pp.carica_caso(c)
    ven = np.asanyarray(nib.load(str(s / 'ventricoli_synthseg_inT1.nii.gz')).dataobj) > 0
    r3 = roi3_in_t1(c, s)
    par = json.loads((PROGETTO / 'reports' / f'postprocessing_{RUN_Z.parent.name}__{RUN_Z.name}.json')
                     .read_text())['scelto']
    prob = pp.prob_nativa(pp.predizioni_fuori_fold(RUN_Z)[c], RUN_Z / 'prob_nativa')
    out = {'ventricoli_mm3': float(ven.sum() * vox), 'lesione_mm3': float(ref.sum() * vox),
           'lesione_nei_ventricoli_mm3': float((ref & ven).sum() * vox),
           'roi3_mm3': None if r3 is None else float(r3.sum() * vox),
           'roi3_coperta': None if r3 is None or not r3.any() else float((r3 & ven).sum() / r3.sum())}
    for nome, (t, v) in {'grezza': (0.5, 0), 'postprocessing': (par['t'], par['vmin'])}.items():
        pred = pp.postprocessa(prob, t, v, par['cervello'], cer, vox)
        senza, tolte = togli_ventricoli(pred, ven)
        out[nome] = {'prima': pp.misure(pred, ref, cer, vox), 'dopo': pp.misure(senza, ref, cer, vox),
                     'componenti_tolte': tolte}
    return c, out


def valuta():
    from multiprocessing import Pool
    casi = sorted(pp.predizioni_fuori_fold(RUN_Z))
    with Pool(6) as p:
        R = dict(p.map(valuta_caso, casi))
    r = ['# Ventricoli di SynthSeg come postprocessing di Z (fuori-fold, 12 pazienti)', '',
         'Prodotto da `scripts/synthseg_aincp.py valuta`, **non modificare a mano**. SynthSeg 1.0 sulla T1; '
         f'regola fissata prima: si toglie una componente predetta con almeno il {QUOTA:.0%} dei voxel nei ventricoli. '
         'Il test non entra.', '', '## SynthSeg e la ROI_3 del clinico, e quanta lesione cade nei ventricoli', '',
         '| paziente | ventricoli SynthSeg | ROI_3 (ventricolo annotato) | ROI_3 dentro SynthSeg | lesione | lesione nei ventricoli |',
         '| --- | --- | --- | --- | --- | --- |']
    for c in casi:
        x = R[c]
        r.append(f"| {c} | {pp.mm3(x['ventricoli_mm3'])} | {pp.mm3(x['roi3_mm3']) if x['roi3_mm3'] else '—'} | "
                 f"{'—' if x['roi3_coperta'] is None else format(x['roi3_coperta'], '.0%')} | {pp.mm3(x['lesione_mm3'])} | "
                 f"{pp.mm3(x['lesione_nei_ventricoli_mm3'])} ({x['lesione_nei_ventricoli_mm3'] / x['lesione_mm3']:.1%}) |")
    for nome, et in (('postprocessing', 'con il postprocessing congelato di Z (soglia 0.3, volume minimo 50 mm³)'),
                     ('grezza', 'sulle predizioni senza postprocessing (soglia 0.5, nessun filtro)')):
        r += ['', f'## Effetto {et}', '',
              '| | lesioni trovate | falsi positivi | F1 lesioni | Dice medio | componenti tolte |', '| --- | --- | --- | --- | --- | --- |']
        for fase in ('prima', 'dopo'):
            ris = [R[c][nome][fase] for c in casi]
            F, _, _ = pp.f1(ris)
            les = [l for x in ris for l in x['lesioni']]
            tolte = sum(R[c][nome]['componenti_tolte'] for c in casi) if fase == 'dopo' else '—'
            r.append(f"| {'senza' if fase == 'prima' else 'con'} la regola dei ventricoli | {sum(l['rilevata'] for l in les)}/{len(les)} | "
                     f"{sum(x['fp'] for x in ris)} | {F:.3f} | {np.mean([x['dice'] for x in ris]):.3f} | {tolte} |")
        persi = [(c, round(a['mm3'], 1)) for c in casi for a, b in zip(R[c][nome]['prima']['lesioni'], R[c][nome]['dopo']['lesioni'])
                 if a['rilevata'] and not b['rilevata']]
        r.append('')
        r.append(f"Lesioni vere perse per la regola: {len(persi)}" + (f" ({', '.join(f'{c} {pp.mm3(v)}' for c, v in persi)})" if persi else '') + '.')
    OUT.with_suffix('.md').write_text('\n'.join(r) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps(R, indent=1, default=float))
    print('\n'.join(r))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('azione', choices=['segmenta', 'ventricoli', 'valuta'])
    {'segmenta': segmenta, 'ventricoli': ventricoli, 'valuta': valuta}[ap.parse_args().azione]()
