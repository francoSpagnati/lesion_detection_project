#!/usr/bin/env python
"""Ablazione dei canali in inferenza sul modello della tesi (ricetta Z di Dataset506), fuori-fold.

    python scripts/ablazione_canali_506.py predici [--processi 4] [--thread 16]   # a CPU
    python scripts/ablazione_canali_506.py valuta     # -> reports/ablazione_canali_506.{md,json}

LA DOMANDA. Il modello riceve due sequenze, FLAIR e T1, ed e' stato addestrato con il modality
dropout perche' funzioni anche con una sola. Due cose da sapere: quanto perde quando una manca
(si puo' usare su un paziente che ha solo la T1?) e se usa davvero entrambe, cioe' se ciascuna
porta qualcosa che l'altra non ha.

COME. Gli stessi 12 pazienti di addestramento, ognuno predetto dal modello del fold che non l'ha
visto, in tre condizioni: entrambe le sequenze, FLAIR azzerata (solo T1), T1 azzerata (solo
FLAIR). Un canale azzerato e' esattamente cio' che la rete vede in inferenza quando la sequenza
manca, e cio' che ha visto in addestramento col dropout (un'immagine nulla resta nulla dopo la
normalizzazione z-score). La condizione "entrambe" sono le predizioni fuori-fold gia' esistenti:
ripredetto a CPU con questa stessa strada, FSM_001 differisce di al massimo 0.003 di probabilita'
e di nessun voxel alla soglia. Poi tutto come per il modello: ritorno al nativo delle probabilita',
postprocessing **congelato** di Z (reports/postprocessing_<...>.json), metriche contro
l'etichetta pulita. Niente si sceglie qui, e il test non si tocca.

LE MISURE. Dice per paziente (con confronto appaiato di Wilcoxon contro "entrambe"), lesioni
trovate per fascia di volume (confronto appaiato lesione per lesione: test esatto di McNemar sulle
lesioni trovate in una condizione e non nell'altra), falsi positivi; e la **copertura per
fenotipo**: quanta parte dei voxel di gliosi (ROI_1), lesione cistica (ROI_2) ed encefalomalacia
(ROI_4) la predizione copre. La gliosi e' iperintensa in FLAIR e quasi invisibile in T1, le cavita'
sono nette in T1: se il modello usa le due sequenze per quello che vedono, togliere la FLAIR deve
far perdere soprattutto la gliosi.
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
RAW = Path(os.environ['nnUNet_raw']) / 'Dataset506_AINCP_FINALE'
PREP = Path(os.environ['nnUNet_preprocessed']) / 'Dataset506_AINCP_FINALE'
RUN = (Path(os.environ['nnUNet_results']) / 'Dataset506_AINCP_FINALE'
       / 'nnUNetTrainerWandb_250ep_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres')
ABL = RUN / 'ablazione_canali'
OUT = PROGETTO / 'reports/ablazione_canali_506'
# canale da azzerare: 0 = FLAIR, 1 = T1 (dataset.json)
CONDIZIONI = {'entrambe': None, 'solo_T1': 0, 'solo_FLAIR': 1}
NOME = {'entrambe': 'FLAIR + T1', 'solo_T1': 'solo T1 (FLAIR azzerata)', 'solo_FLAIR': 'solo FLAIR (T1 azzerata)'}
FASCE = [(1, 10, '1–10 mm³'), (10, 50, '10–50 mm³'), (50, 200, '50–200 mm³'), (200, 1000, '200–1 000 mm³'),
         (1000, 10000, '1 000–10 000 mm³'), (10000, np.inf, '> 10 000 mm³')]
FENOTIPI = {'ROI_1': 'gliosi', 'ROI_2': 'cistica', 'ROI_4': 'encefalomalacia'}


def lavori():
    """(condizione, fold, caso) per tutte le predizioni da fare: ogni caso col fold che non l'ha visto."""
    out = []
    for k, f in enumerate(json.loads((PREP / 'splits_final.json').read_text())):
        for c in f['val']:
            for cond, canale in CONDIZIONI.items():
                if canale is not None:
                    out.append((cond, k, c))
    return out


def predici_parte(parte: int, di: int, thread: int):
    import torch
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(thread)
    io = SimpleITKIO()
    predittori = {}
    for cond, k, c in lavori()[parte::di]:
        dest = ABL / cond
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / f'{c}.npz').exists():
            continue
        if k not in predittori:
            # le impostazioni di nnUNetv2_predict: finestra scorrevole al 50%, pesi gaussiani,
            # specchiamento; checkpoint_final, che con l'arresto contiene i pesi dell'epoca migliore
            p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=False, device=torch.device('cpu'),
                                verbose=False, allow_tqdm=False)
            p.initialize_from_trained_model_folder(str(RUN), use_folds=(k,), checkpoint_name='checkpoint_final.pth')
            predittori[k] = p
        img, prop = io.read_images([str(RAW / 'imagesTr' / f'{c}_000{i}.nii.gz') for i in (0, 1)])
        img[CONDIZIONI[cond]] = 0
        predittori[k].predict_single_npy_array(img, prop, None, str(dest / c), True)
        print(f'{cond} {c} (fold {k}) fatto', flush=True)


def predici(processi: int, thread: int):
    # processi separati e non un Pool: ognuno ha il suo numero di thread di torch, e
    # nnUNetv2_predict a CPU ne prenderebbe 96 per processo
    figli = [subprocess.Popen([sys.executable, __file__, '_parte', str(i), str(processi), str(thread)])
             for i in range(processi)]
    codici = [f.wait() for f in figli]
    assert not any(codici), codici


def fenotipi_in_t1(caso: str, sessione: Path) -> dict:
    """Le ROI di ciascun fenotipo nella griglia T1 nativa, con la stessa strada dell'etichetta pulita
    (pipeline_aincp.py): geometria della FLAIR originale, rigida FLAIR->T1 salvata, lineare e
    soglia 0.5. Poi intersecate con l'etichetta pulita, cosi' i residui tolti restano fuori."""
    import ants
    import pipeline_aincp as pa
    c = pa.trova_casi()[caso]
    t1 = ants.image_read(str(sessione / 'T1_N4.nii.gz'))
    rif = ants.image_read(str(c['FLAIR']))
    out = {}
    for r in FENOTIPI:
        if r not in c['roi']:
            continue
        m = np.squeeze(ants.image_read(str(c['roi'][r])).numpy()) > 0
        a = ants.apply_transforms(fixed=t1, moving=rif.new_image_like(m.astype('float32')),
                                  transformlist=[str(sessione / 'transforms' / 'FLAIR_to_T1_0.mat')],
                                  interpolator='linear')
        out[r] = a.numpy() >= 0.5
    return out


def valuta_caso(c: str):
    par = json.loads((PROGETTO / 'reports' / f'postprocessing_{RUN.parent.name}__{RUN.name}.json').read_text())['scelto']
    ref, cer, vox = pp.carica_caso(c)
    fen = {r: m & ref for r, m in fenotipi_in_t1(c, pp.sessione(c)).items()}
    unione = np.zeros_like(ref)
    for m in fen.values():
        unione |= m
    # controllo: l'unione delle ROI ricostruite deve essere l'etichetta pulita
    accordo = 2 * float((unione & ref).sum()) / float(unione.sum() + ref.sum())
    # fenotipo di ogni lesione annotata: quello con piu' voxel nella componente
    from scipy import ndimage
    lab, n = ndimage.label(ref, structure=pp.STRUTTURA)
    fen_lesione = []
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() * vox < pp.SOGLIA_LESIONE_MM3:
            continue
        q = {r: int((m & comp).sum()) for r, m in fen.items()}
        fen_lesione.append(max(q, key=q.get) if q and max(q.values()) else None)
    ris = {'accordo_roi_etichetta': accordo}
    for cond in CONDIZIONI:
        if cond == 'entrambe':
            npz = pp.predizioni_fuori_fold(RUN)[c]
            cache = RUN / 'prob_nativa'
        else:
            npz = ABL / cond / f'{c}.npz'
            cache = ABL / cond / 'prob_nativa'
            cache.mkdir(exist_ok=True)
        pred = pp.postprocessa(pp.prob_nativa(npz, cache), par['t'], par['vmin'], par['cervello'], cer, vox)
        m = pp.misure(pred, ref, cer, vox)
        assert len(m['lesioni']) == len(fen_lesione)
        for l, f in zip(m['lesioni'], fen_lesione):
            l['fenotipo'] = f
        m['copertura'] = {r: [int((mk & pred).sum()), int(mk.sum())] for r, mk in fen.items()}
        ris[cond] = m
    return c, ris


def fp_(p: float) -> str:
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def mcnemar(a, b):
    """Test esatto di McNemar: le lesioni trovate da una sola delle due condizioni."""
    from scipy.stats import binomtest
    solo_a = sum(x and not y for x, y in zip(a, b))
    solo_b = sum(y and not x for x, y in zip(a, b))
    p = binomtest(solo_a, solo_a + solo_b, 0.5).pvalue if solo_a + solo_b else 1.0
    return solo_a, solo_b, p


def valuta():
    from multiprocessing import Pool
    from scipy.stats import wilcoxon
    casi = sorted(pp.predizioni_fuori_fold(RUN))
    with Pool(6) as p:
        R = dict(p.map(valuta_caso, casi))
    tot = {}
    for cond in CONDIZIONI:
        ris = [R[c][cond] for c in casi]
        F, se, pc = pp.f1(ris)
        les = [l for r in ris for l in r['lesioni']]
        inter = sum(r['dice'] * (r['mm3_vero'] + r['mm3_pred']) for r in ris)
        tot[cond] = {'f1': F, 'sens': se, 'prec': pc, 'dice': float(np.mean([r['dice'] for r in ris])),
                     'mediana': float(np.median([r['dice'] for r in ris])),
                     'globale': inter / sum(r['mm3_vero'] + r['mm3_pred'] for r in ris),
                     'rilevate': sum(l['rilevata'] for l in les), 'lesioni': len(les), 'fp': sum(r['fp'] for r in ris),
                     'trovate': [l['rilevata'] for l in les], 'volumi': [l['mm3'] for l in les],
                     'fenotipi': [l['fenotipo'] for l in les],
                     'per_paziente': {c: R[c][cond]['dice'] for c in casi},
                     'copertura': {f: [sum(R[c][cond]['copertura'].get(f, [0, 0])[i] for c in casi) for i in (0, 1)]
                                   for f in FENOTIPI}}
    base = tot['entrambe']
    r = ['# Ablazione dei canali in inferenza: ricetta Z di Dataset506 (fuori-fold)', '',
         'Prodotto da `scripts/ablazione_canali_506.py`, **non modificare a mano**. 12 pazienti di addestramento, '
         'ognuno predetto dal modello del fold che non l\'ha visto; la sequenza tolta e\' azzerata in ingresso. '
         'Postprocessing congelato di Z (soglia 0.3, volume minimo 50 mm³), metriche nel nativo. '
         f'Controllo delle ROI: l\'unione dei fenotipi ricostruiti coincide con l\'etichetta pulita con Dice minimo '
         f'{min(R[c]["accordo_roi_etichetta"] for c in casi):.3f}.', '',
         '## Complessivo', '',
         '| ingresso | Dice medio | mediana | globale | Δ Dice vs entrambe | meglio/peggio | p (Wilcoxon) | lesioni trovate | '
         'solo qui / solo con entrambe | p (McNemar) | falsi positivi | F1 lesioni |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    zb = np.array(list(base['per_paziente'].values()))
    for cond, t in tot.items():
        if cond == 'entrambe':
            conf = ('—',) * 5
        else:
            e = np.array(list(t['per_paziente'].values()))
            sa, sb, pm = mcnemar(t['trovate'], base['trovate'])
            conf = (f'{(e - zb).mean():+.3f}', f'{(e > zb).sum()}/{(e < zb).sum()}', fp_(wilcoxon(e, zb).pvalue),
                    f'{sa} / {sb}', fp_(pm))
        r.append(f"| {NOME[cond]} | {t['dice']:.3f} | {t['mediana']:.3f} | {t['globale']:.3f} | {conf[0]} | {conf[1]} | "
                 f"{conf[2]} | {t['rilevate']}/{t['lesioni']} | {conf[3]} | {conf[4]} | {t['fp']} | {t['f1']:.3f} |")
    r += ['', '## Lesioni trovate per fascia di volume', '', '| fascia | lesioni | ' +
          ' | '.join(NOME[c] for c in CONDIZIONI) + ' |', '| --- | --- |' + ' --- |' * len(CONDIZIONI)]
    v = np.array(base['volumi'])
    for lo, hi, et in FASCE:
        sel = (v >= lo) & (v < hi)
        r.append(f'| {et} | {sel.sum()} | ' + ' | '.join(
            f"{np.array(tot[c]['trovate'])[sel].sum()}/{sel.sum()}" for c in CONDIZIONI) + ' |')
    r += ['', '## Per fenotipo', '',
          'Copertura: voxel del fenotipo coperti dalla predizione su voxel totali, sommati sui pazienti (pesa chi ha '
          'piu\' lesione, come il Dice). Lesioni trovate: ogni lesione annotata ha il fenotipo con piu\' voxel.', '',
          '| fenotipo | voxel | ' + ' | '.join(f'copertura {NOME[c]}' for c in CONDIZIONI) + ' | lesioni | ' +
          ' | '.join(f'trovate {NOME[c]}' for c in CONDIZIONI) + ' |',
          '| --- | --- |' + ' --- |' * (2 * len(CONDIZIONI) + 1)]
    fl = np.array(base['fenotipi'], dtype=object)
    for f, nome in FENOTIPI.items():
        n_vox = base['copertura'][f][1]
        sel = fl == f
        r.append(f'| {nome} | {n_vox} | ' + ' | '.join(
            f"{tot[c]['copertura'][f][0] / n_vox:.0%}" if n_vox else '—' for c in CONDIZIONI) +
            f' | {sel.sum()} | ' + ' | '.join(f"{np.array(tot[c]['trovate'])[sel].sum()}/{sel.sum()}" for c in CONDIZIONI) + ' |')
    r += ['', '## Dice per paziente', '', '| paziente | ' + ' | '.join(NOME[c] for c in CONDIZIONI) + ' |',
          '| --- |' + ' --- |' * len(CONDIZIONI)]
    for c in casi:
        r.append(f'| {c} | ' + ' | '.join(f"{tot[k]['per_paziente'][c]:.3f}" for k in CONDIZIONI) + ' |')
    OUT.with_suffix('.md').write_text('\n'.join(r) + '\n')
    OUT.with_suffix('.json').write_text(json.dumps({'totali': tot, 'per_caso': R}, indent=1, default=float))
    print('\n'.join(r))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '_parte':
        predici_parte(*map(int, sys.argv[2:5]))
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument('azione', choices=['predici', 'valuta'])
    ap.add_argument('--processi', type=int, default=4)
    ap.add_argument('--thread', type=int, default=16)
    a = ap.parse_args()
    predici(a.processi, a.thread) if a.azione == 'predici' else valuta()
