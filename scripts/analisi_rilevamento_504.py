#!/usr/bin/env python
"""Rilevamento su Dataset504 per fenotipo di lesione e per dimensione rispetto alla griglia della rete.

    python scripts/analisi_rilevamento_504.py      # -> reports/analisi_rilevamento_504.{md,json}

Risponde a tre domande che il Dice complessivo e il tasso di rilevamento globale non separano.

1. **Che cosa vede ciascun canale.** Il progetto usa T1 e FLAIR insieme perche' vedono lesioni
   diverse: la gliosi (ROI_1) e' iperintensa in FLAIR, le cavita' -- cistica (ROI_2) ed
   encefalomalacia (ROI_4) -- sono ipointense in T1. Qui la copertura e il rilevamento si
   misurano fenotipo per fenotipo. C'e' un esperimento naturale gia' pronto: nella variante
   `zero` a passo 1e-4 il filtro FLAIR e' rimasto a 0.05 della norma del T1, cioe' quel modello
   e' di fatto un modello a sola T1; nella variante a passo differenziato la FLAIR e' attiva.
   Il confronto fra le due, sulla gliosi, dice che cosa aggiunge la FLAIR senza addestrare un
   modello a un canale.

2. **Quali lesioni la rete puo' vedere.** Il piano nnUNetPlansDa502 lavora a 1 mm isotropo
   (quello di 502, necessario per trasferire i pesi) mentre le immagini originali hanno voxel da
   0.2 mm^3. nnU-Net ricampiona l'annotazione con interpolazione lineare sulla codifica one-hot
   e soglia 0.5: una componente piu' piccola di un voxel della griglia puo' sparire dal
   bersaglio di addestramento. Qui ogni componente viene ricampionata da sola con la funzione
   di nnU-Net, sulla stessa griglia del preprocessing, e si conta chi sopravvive. La procedura
   si verifica da sola: ricampionando l'annotazione intera deve riprodurre voxel per voxel il
   bersaglio che nnU-Net ha salvato in nnUNet_preprocessed.

3. **Che cosa distingue davvero le varianti.** Il Dice e il numero di lesioni rilevate non
   distinguono le varianti su dieci pazienti; i falsi positivi forse si'. Qui si misurano
   appaiati, con la loro dimensione.

Le lesioni sono componenti connesse a 26 vicini (standard del progetto), nella griglia T1 in
cui il dataset e' stato costruito. Una lesione e' rilevata se almeno un voxel predetto la tocca.
Le predizioni sono quelle fuori-fold: ogni paziente dal fold in cui era in validazione.
"""
import json
import os
import pickle
import re
from pathlib import Path

import ants
import blosc2
import nibabel as nib
import numpy as np
from scipy import ndimage, stats

from nnunetv2.preprocessing.resampling.default_resampling import (compute_new_shape,
                                                                   resample_data_or_seg_to_shape)

PROGETTO = Path(__file__).resolve().parent.parent
RAW = Path(os.environ['nnUNet_raw']) / 'Dataset504_AINCP'
PREP = Path(os.environ['nnUNet_preprocessed']) / 'Dataset504_AINCP'
RES = Path(os.environ['nnUNet_results']) / 'Dataset504_AINCP'
PROC = PROGETTO / 'data/processed/AINCP'
LOTTO = PROGETTO / 'data/raw/AINCP/dataset_FSM_v1'
OUT = PROGETTO / 'reports'
PIANO = 'nnUNetPlansDa502'
STRUTTURA = np.ones((3, 3, 3), dtype=bool)
FENOTIPI = {'ROI_1': 'gliosi', 'ROI_2': 'cistica', 'ROI_4': 'encefalomalacia'}

VARIANTI = [
    ('da zero', 'nnUNetTrainerWandb_250ep_arresto50'),
    ('B zero, passo 1e-4 (FLAIR spenta)', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50'),
    ('B duplica, passo 1e-4', 'nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50'),
    ('B passo differenziato (FLAIR attiva)', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_arresto50'),
    ('B passo differenziato, freno corretto', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_arresto50'),
    ('B passo 1e-3 uniforme', 'nnUNetTrainerEspansioneCanali_zero_250ep_lr1e3_arresto50'),
]


def fenotipi_in_t1(soggetto: str, riferimento_t1: ants.ANTsImage) -> dict:
    """Le singole ROI lesionali portate nello spazio T1 con la trasformazione gia' calcolata.

    Stessa strada di preprocess_aincp.py: le ROI stanno sulla griglia della FLAIR (se ne
    prende la geometria), la trasformazione e' la rigida FLAIR->T1 salvata, l'interpolazione e'
    nearest neighbour. L'unione di queste maschere deve coincidere con labelsTr: e' verificato
    in main()."""
    co = LOTTO / soggetto / 'COREGISTERED'
    flair = ants.image_read(str(next(co.glob('*_FLAIR_original.nii.gz'))))
    trasf = [str(PROC / soggetto / 'transforms' / 'FLAIR_to_T1_0.mat')]
    out = {}
    for f in sorted(co.glob('ROI_*.nii')):
        m = re.match(r'ROI_(\d+)', f.name, re.I)            # FSM_052 ha 'ROI_1 - Copia.nii'
        chiave = f'ROI_{m.group(1)}' if m else None
        if chiave not in FENOTIPI:
            continue
        a = np.squeeze(ants.image_read(str(f)).numpy()) > 0
        img = ants.from_numpy(a.astype('float32'), origin=flair.origin, spacing=flair.spacing,
                              direction=flair.direction)
        t1 = ants.apply_transforms(fixed=riferimento_t1, moving=img, transformlist=trasf,
                                   interpolator='nearestNeighbor')
        out[chiave] = t1.numpy() > 0.5
    return out


def griglia_nnunet(caso: str):
    """Geometria del preprocessing di nnU-Net per il caso: crop, forma e spaziatura."""
    with open(PREP / f'{PIANO}_3d_fullres' / f'{caso}.pkl', 'rb') as f:
        prop = pickle.load(f)
    piano = json.load(open(PREP / f'{PIANO}.json'))
    nuova = piano['configurations']['3d_fullres']['spacing']
    return prop, nuova


def ricampiona_come_nnunet(maschera_nib: np.ndarray, prop: dict, nuova_spaz) -> np.ndarray:
    """La maschera (ordine nibabel) portata sulla griglia di addestramento, come fa nnU-Net.

    nnU-Net legge con SimpleITK, quindi il suo array e' quello di nibabel con gli assi
    invertiti; poi ritaglia sul riquadro salvato e ricampiona con resample_data_or_seg_to_shape
    in modalita' segmentazione (one-hot, lineare, soglia 0.5)."""
    a = maschera_nib.transpose(2, 1, 0)
    (x0, x1), (y0, y1), (z0, z1) = prop['bbox_used_for_cropping']
    a = a[x0:x1, y0:y1, z0:z1]
    vecchia = prop['spacing']
    forma = compute_new_shape(a.shape, vecchia, nuova_spaz)
    r = resample_data_or_seg_to_shape(a[None].astype(np.int16), forma, vecchia, nuova_spaz,
                                      is_seg=True, order=1, order_z=0, force_separate_z=None)
    return r[0] > 0


def predizioni_fuori_fold(trainer: str) -> dict:
    cartella = RES / f'{trainer}__{PIANO}__3d_fullres'
    out = {}
    for s in sorted(cartella.glob('fold_*/validation/summary.json')):
        for m in json.load(open(s))['metric_per_case']:
            nome = Path(m['reference_file']).name
            out[nome[:-7]] = s.parent / nome
    return out


def main():
    casi = sorted(p.name[:-7] for p in (RAW / 'labelsTr').glob('*.nii.gz'))
    soggetto = {c: next(d.name for d in PROC.iterdir() if d.name.startswith(c + '_')) for c in casi}
    varianti = [(e, t) for e, t in VARIANTI if len(predizioni_fuori_fold(t)) == len(casi)]
    pred_path = {t: predizioni_fuori_fold(t) for _, t in varianti}

    lesioni, fp, controlli = [], [], []
    copertura = {t: {f: [] for f in FENOTIPI} for _, t in varianti}
    for c in casi:
        gt_img = nib.load(str(RAW / 'labelsTr' / f'{c}.nii.gz'))
        gt = np.asarray(gt_img.dataobj) > 0
        vox = float(np.prod(gt_img.header.get_zooms()[:3]))
        rif = ants.image_read(str(PROC / soggetto[c] / 'T1_N4.nii.gz'))
        fen = fenotipi_in_t1(soggetto[c], rif)
        unione = np.zeros_like(gt)
        for m in fen.values():
            unione |= m

        # controllo 1: le ROI trasportate ricostruiscono esattamente il bersaglio
        # controllo 2: il ricampionamento riproduce il bersaglio salvato da nnU-Net
        prop, nuova = griglia_nnunet(c)
        salvato = blosc2.open(urlpath=str(PREP / f'{PIANO}_3d_fullres' / f'{c}_seg.b2nd'), mode='r')[:][0] > 0
        mio = ricampiona_come_nnunet(gt, prop, nuova)
        controlli.append({'caso': c, 'roi_uguali_a_labelsTr': bool((unione == gt).all()),
                          'voxel_diversi_da_labelsTr': int((unione != gt).sum()),
                          'ricampionamento_uguale_a_nnunet': bool(mio.shape == salvato.shape and (mio == salvato).all()),
                          'voxel_diversi_da_nnunet': int((mio != salvato).sum()) if mio.shape == salvato.shape else -1,
                          'componenti_bersaglio_1mm': int(ndimage.label(salvato, structure=STRUTTURA)[1])})

        pred = {t: np.asarray(nib.load(str(pred_path[t][c])).dataobj) > 0 for _, t in varianti}
        etich, n = ndimage.label(gt, structure=STRUTTURA)
        for i in range(1, n + 1):
            comp = etich == i
            quote = {f: int((comp & m).sum()) for f, m in fen.items()}
            fenotipo = max(quote, key=quote.get)
            sopravvive = int(ricampiona_come_nnunet(comp, prop, nuova).sum())
            lesioni.append({'caso': c, 'mm3': float(comp.sum() * vox), 'voxel_originali': int(comp.sum()),
                            'fenotipo': fenotipo, 'voxel_a_1mm': sopravvive,
                            'rilevata': {t: bool((pred[t] & comp).any()) for _, t in varianti}})
        for _, t in varianti:
            for f, m in fen.items():
                if m.any():
                    copertura[t][f].append({'caso': c, 'voxel': int(m.sum()),
                                            'coperti': int((m & pred[t]).sum())})
            pe, pn = ndimage.label(pred[t], structure=STRUTTURA)
            for j in range(1, pn + 1):
                pc = pe == j
                if not (pc & gt).any():
                    fp.append({'caso': c, 'trainer': t, 'mm3': float(pc.sum() * vox)})
        print(f'{c}: {n} lesioni, controlli {controlli[-1]}', flush=True)

    ris = {'varianti': varianti, 'controlli': controlli, 'lesioni': lesioni, 'falsi_positivi': fp,
           'copertura': copertura}
    (OUT / 'analisi_rilevamento_504.json').write_text(json.dumps(ris, indent=1))
    scrivi_md(ris, casi)


def scrivi_md(ris, casi):
    V, L, FP, COP = ris['varianti'], ris['lesioni'], ris['falsi_positivi'], ris['copertura']
    r = ['# Rilevamento su `Dataset504_AINCP` per fenotipo e per dimensione',
         '', 'Prodotto da `scripts/analisi_rilevamento_504.py`, **non modificare a mano**.', '',
         '## Controlli della procedura', '']
    c = ris['controlli']
    r.append(f"- ROI trasportate in T1 identiche a `labelsTr`: **{sum(x['roi_uguali_a_labelsTr'] for x in c)} casi su {len(c)}**"
             f" (voxel diversi in totale: {sum(x['voxel_diversi_da_labelsTr'] for x in c)})")
    r.append(f"- ricampionamento identico al bersaglio salvato da nnU-Net: **{sum(x['ricampionamento_uguale_a_nnunet'] for x in c)} casi su {len(c)}**"
             f" (voxel diversi in totale: {sum(max(0, x['voxel_diversi_da_nnunet']) for x in c)})")
    r += ['', '## 1. Quali lesioni arrivano alla rete', '',
          'Il piano lavora a 1 mm isotropo. Una componente «scompare» se, ricampionata da sola con la '
          'funzione di nnU-Net sulla griglia del preprocessing, non lascia nemmeno un voxel.', '',
          '| volume originale | lesioni | scompaiono a 1 mm | di cui gliosi / cistica / encefalomalacia |',
          '| --- | --- | --- | --- |']
    for lo, hi, et in [(0, 1, '< 1 mm³'), (1, 10, '1 – 10 mm³'), (10, 50, '10 – 50 mm³'), (50, np.inf, '> 50 mm³')]:
        sel = [x for x in L if lo <= x['mm3'] < hi]
        via = [x for x in sel if x['voxel_a_1mm'] == 0]
        per = ' / '.join(str(sum(x['fenotipo'] == f for x in via)) for f in FENOTIPI)
        r.append(f'| {et} | {len(sel)} | {len(via)} | {per} |')
    via = [x for x in L if x['voxel_a_1mm'] == 0]
    r += ['', f"Totale: **{len(via)} lesioni su {len(L)}** non esistono nel bersaglio di addestramento. "
          f"Il bersaglio a 1 mm salvato da nnU-Net ha {sum(x['componenti_bersaglio_1mm'] for x in c)} componenti "
          f"contro le {len(L)} della griglia originale.", '']

    r += ['## 2. Rilevamento per fenotipo', '',
          '| variante | ' + ' | '.join(f'{n} (lesioni)' for n in FENOTIPI.values()) + ' | tutte | solo quelle che esistono a 1 mm |',
          '| --- |' + ' --- |' * (len(FENOTIPI) + 2)]
    for e, t in V:
        celle = []
        for f in FENOTIPI:
            sel = [x for x in L if x['fenotipo'] == f]
            celle.append(f"{sum(x['rilevata'][t] for x in sel)}/{len(sel)}")
        vis = [x for x in L if x['voxel_a_1mm'] > 0]
        celle.append(f"{sum(x['rilevata'][t] for x in L)}/{len(L)}")
        celle.append(f"**{sum(x['rilevata'][t] for x in vis)}/{len(vis)}**")
        r.append(f'| {e} | ' + ' | '.join(celle) + ' |')

    r += ['', '## 3. Copertura dei voxel per fenotipo', '',
          'Frazione dei voxel di ciascun fenotipo coperta dalla predizione (sensibilita\' a livello di voxel), '
          'sommata sui pazienti che hanno quel fenotipo. Per le lesioni grandi il rilevamento e\' sempre 100% e non '
          'distingue niente; la copertura si\'.', '',
          '| variante | ' + ' | '.join(FENOTIPI.values()) + ' |', '| --- |' + ' --- |' * len(FENOTIPI)]
    for e, t in V:
        celle = []
        for f in FENOTIPI:
            v = COP[t][f]
            tot, cop = sum(x['voxel'] for x in v), sum(x['coperti'] for x in v)
            celle.append(f"{cop / tot:.3f} ({len(v)} pazienti)" if tot else '—')
        r.append(f'| {e} | ' + ' | '.join(celle) + ' |')

    r += ['', 'Paziente per paziente, gliosi (ROI_1): FLAIR spenta contro FLAIR attiva.', '']
    spenta, attiva = V[1][1], V[3][1] if len(V) > 3 else None
    if attiva:
        r += ['| paziente | voxel di gliosi | coperti, FLAIR spenta | coperti, FLAIR attiva |', '| --- | --- | --- | --- |']
        a = {x['caso']: x for x in COP[spenta]['ROI_1']}
        b = {x['caso']: x for x in COP[attiva]['ROI_1']}
        d = []
        for caso in sorted(a):
            fa, fb = a[caso]['coperti'] / a[caso]['voxel'], b[caso]['coperti'] / b[caso]['voxel']
            d.append(fb - fa)
            r.append(f"| {caso} | {a[caso]['voxel']} | {fa:.3f} | {fb:.3f} |")
        if len(d) >= 5:
            w = stats.wilcoxon(d) if any(x != 0 for x in d) else None
            r += ['', f"Migliora in {sum(x > 0 for x in d)} pazienti su {len(d)}"
                  + (f", Wilcoxon p = {w.pvalue:.3f}." if w else '.')]

    r += ['', '## 4. Falsi positivi', '',
          '| variante | falsi positivi | per paziente | volume mediano (mm³) | sotto 10 mm³ | sopra 200 mm³ |',
          '| --- | --- | --- | --- | --- | --- |']
    for e, t in V:
        v = [x['mm3'] for x in FP if x['trainer'] == t]
        r.append(f"| {e} | {len(v)} | {len(v) / len(casi):.1f} | {np.median(v) if v else 0:.1f} | "
                 f"{sum(x < 10 for x in v)} | {sum(x > 200 for x in v)} |")
    rif = V[0][1]
    r += ['', f'Appaiati contro «{V[0][0]}», per paziente:', '']
    for e, t in V[1:]:
        a = np.array([sum(1 for x in FP if x['trainer'] == rif and x['caso'] == c) for c in casi])
        b = np.array([sum(1 for x in FP if x['trainer'] == t and x['caso'] == c) for c in casi])
        d = b - a
        w = stats.wilcoxon(b, a) if (d != 0).any() else None
        r.append(f"- {e}: meno falsi positivi in **{int((d < 0).sum())}** pazienti su {len(casi)}, "
                 f"di piu' in {int((d > 0).sum())}" + (f", Wilcoxon p = {w.pvalue:.3f}" if w else ''))
    (OUT / 'analisi_rilevamento_504.md').write_text('\n'.join(r) + '\n')
    print('scritto', OUT / 'analisi_rilevamento_504.md')


if __name__ == '__main__':
    main()
