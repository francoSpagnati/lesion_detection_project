#!/usr/bin/env python
"""Pipeline dei pazienti AInCP per il modello finale: dai file ricevuti al dataset nnU-Net.

    python scripts/pipeline_aincp.py elenco                  # che cosa trova, in che formato
    python scripts/pipeline_aincp.py prepara [--casi ...]    # N4, registrazioni, cervello, template
    python scripts/pipeline_aincp.py dataset --id 506        # split 70/30 per centro + nnUNet_raw
    # poi: nnUNetv2_extract_fingerprint -d 506; pipeline_aincp.py piano --id 506; nnUNetv2_preprocess

PERCHE' UNO SCRIPT NUOVO. Il primo lotto e' stato preparato in piu' passaggi (preprocess_aincp.py,
confronto_template_aincp.py, make_aincp_mni_dataset.py, SynthStrip a mano) pensati per dieci
pazienti in un formato solo. Qui la stessa strada, decisa il 23/9, diventa un comando per
paziente che funziona anche sui casi nuovi:

    T1 --N4--> T1_N4 --SyN--> template pediatrico (trasformazioni e inverse salvate)
    FLAIR --N4--> --rigida--> T1 --> template            (una sola interpolazione)
    etichetta: ROI_1|ROI_2|ROI_4, meno le componenti <= 1 mm3 --> template (lineare, soglia 0.5)
    SynthStrip su T1_N4 --> cervello_inT1 (postprocessing e % di cervello lesionato)

DUE FORMATI DI INGRESSO, perche' sono gia' due:
  - primo lotto (data/raw/AINCP/dataset_FSM_v1/<caso>/COREGISTERED): T1 e FLAIR originali, una ROI
    per fenotipo disegnata sulla FLAIR (ROI_1 gliosi, ROI_2 cistica, ROI_3 ventricolo, escluso,
    ROI_4 encefalomalacia);
  - /data01/fsm_mri_data/<caso>/ (FSM_028): T1 originale, FLAIR gia' allineata alla T1, e
    un'etichetta **unica e binaria** gia' nello spazio T1. Senza fenotipi: su questi casi la
    copertura per fenotipo non si puo' misurare.
Un formato nuovo va aggiunto a `trova_casi`, non adattato a mano.

FLAIR MANCANTE. Fra i 1.5 T ci sono esami senza FLAIR 3D (foglio Excel, colonne "Sub_optimal").
Il caso si tiene: il canale FLAIR e' tutto zero, che e' cio' che il modality dropout ha insegnato
alla rete a gestire (ATLAS in 543 e' interamente cosi').

Le uscite stanno in data/processed/AINCP/<caso>/, accanto a quelle gia' esistenti del primo lotto:
per quei dieci casi i file gia' calcolati si riusano (stesse trasformazioni del 23/9).
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
LOTTO1 = PROGETTO / 'data/raw/AINCP/dataset_FSM_v1'
NUOVI = Path('/data01/fsm_mri_data')
PROC = PROGETTO / 'data/processed/AINCP'
TEMPLATE = PROGETTO / 'data/templates/tpl-MNIPediatricAsym_cohort-1_res-1_T1w.nii.gz'
EXCEL = PROGETTO / 'docs/AINCP_SERIES_TYPE_dataset_FSM_v1.xlsx'
SYNTHSTRIP = PROGETTO / 'scripts/synthstrip/synthstrip.py'
PESI_SYNTHSTRIP = PROGETTO / 'synthstrip.1.pt'
ROI_LESIONE = ('ROI_1', 'ROI_2', 'ROI_4')      # ROI_3, il ventricolo, resta fuori (relatore)
SOGLIA_MM3 = 1.0                               # residui del tracciato (incontro del 23/9)
STRUTTURA = np.ones((3, 3, 3), dtype=bool)
QUOTA_TEST = 0.30                              # 70/30 (incontro del 23/9)
SEME = 20260923


def paziente(nome: str) -> str:
    """FSM_032_MR_1_anon -> FSM_032."""
    return '_'.join(nome.split('_')[:2])


# --- 1. che cosa c'e' ---------------------------------------------------------------------------

def trova_casi() -> dict:
    casi = {}
    for d in sorted(LOTTO1.glob('FSM_*_anon')):
        co = d / 'COREGISTERED'
        t1 = next(iter(co.glob('*_T1_original.nii.gz')), None)
        fl = next(iter(co.glob('*_FLAIR_original.nii.gz')), None)
        roi = {}
        for f in sorted(co.glob('ROI_*.nii')):
            m = re.match(r'ROI_(\d+)', f.name, re.I)            # FSM_052 ha 'ROI_1 - Copia.nii'
            if m:
                roi.setdefault(f'ROI_{m.group(1)}', f)
        if t1:
            casi[paziente(d.name)] = {'formato': 'lotto1', 'sessione': d.name, 'T1': t1, 'FLAIR': fl,
                                      'roi': roi, 'etichetta_T1': None}
    if NUOVI.exists():
        for d in sorted(NUOVI.glob('FSM_*')):
            p = paziente(d.name)
            if p in casi:
                continue
            t1 = next(iter(d.glob('*_T1_original.nii.gz')), None)
            fl = next(iter(d.glob('*flair_in_T1.nii.gz')), None) or next(iter(d.glob('*FLAIR*.nii.gz')), None)
            seg = next(iter(d.glob('*seg_in_T1.nii.gz')), None)
            if t1:
                casi[p] = {'formato': 'allineato_T1', 'sessione': d.name, 'T1': t1, 'FLAIR': fl,
                           'roi': {}, 'etichetta_T1': seg}
    return casi


def metadati() -> dict:
    """Centro, campo ed eta' per paziente, dal foglio `SITE & MRFIELD`.

    Il foglio ha piu' sessioni per paziente; si prende quella il cui nome coincide con la
    sessione del caso, altrimenti la prima. Le grafie del centro non sono uniformi (Gaslini ne ha
    due, Meyer due, Stella Maris due che pero' corrispondono a 1.5 e 3 T): si normalizzano."""
    import openpyxl
    ws = openpyxl.load_workbook(EXCEL, read_only=True, data_only=True)['SITE & MRFIELD']
    righe = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] and r[1]]
    out = {}
    for sogg, sess, _, campo, eta, sesso, sito, *_ in righe:
        s = str(sito).upper()
        centro = ('Stella Maris' if 'STELLA' in s else 'Gaslini' if 'GASLINI' in s
                  else 'Meyer' if 'MEYER' in s else str(sito).strip())
        out.setdefault(sogg, []).append({'sessione': sess, 'campo_T': float(campo), 'centro': centro,
                                         'eta': None if eta in (None, '#N/A') else str(eta)})
    return out


def elenco():
    casi, meta = trova_casi(), metadati()
    for p, c in casi.items():
        m = next((x for x in meta.get(p, []) if x['sessione'].startswith(c['sessione'][:12])),
                 (meta.get(p) or [{}])[0])
        print(f"{p}: {c['formato']:13s} FLAIR {'si' if c['FLAIR'] else 'NO':2s}  "
              f"etichetta {'ROI ' + ','.join(sorted(c['roi'])) if c['roi'] else ('unica in T1' if c['etichetta_T1'] else 'NESSUNA')}  "
              f"| {m.get('centro', '?')} {m.get('campo_T', '?')} T, eta' {m.get('eta', '?')}")
    print(f'{len(casi)} casi')


# --- 2. preparazione di un caso -------------------------------------------------------------------

def togli_residui(m: np.ndarray, mm3_voxel: float) -> tuple[np.ndarray, int]:
    from scipy import ndimage
    lab, n = ndimage.label(m, structure=STRUTTURA)
    vol = np.bincount(lab.ravel())[1:] * mm3_voxel
    piccole = np.flatnonzero(vol <= SOGLIA_MM3) + 1
    m = m.copy()
    m[np.isin(lab, piccole)] = False
    return m, int(len(piccole))


def salva_trasf(lista, dest: Path, prefisso: str):
    for i, t in enumerate(lista):
        shutil.copy(t, dest / f'{prefisso}_{i}{".mat" if t.endswith(".mat") else ".nii.gz"}')


def prepara(p: str, c: dict, tpl, gpu: bool):
    import ants
    from scipy import ndimage
    dest = PROC / c['sessione']
    (dest / 'transforms').mkdir(parents=True, exist_ok=True)
    (dest / 'transforms_ped').mkdir(exist_ok=True)
    man_f = dest / 'pipeline.json'
    if man_f.exists():
        print(f'{p}: gia\' preparato')
        return json.loads(man_f.read_text())
    man = {'paziente': p, 'formato': c['formato'], 'sessione': c['sessione'],
           'FLAIR': c['FLAIR'] is not None, 'fenotipi': bool(c['roi'])}

    # N4 e registrazioni: si riusano se ci sono gia' (primo lotto, 22-23/9)
    if not (dest / 'T1_N4.nii.gz').exists():
        ants.image_write(ants.n4_bias_field_correction(ants.image_read(str(c['T1']))), str(dest / 'T1_N4.nii.gz'))
    t1n = ants.image_read(str(dest / 'T1_N4.nii.gz'))
    tp = dest / 'transforms_ped'
    if not (tp / 'T1_to_PED_fwd_1.mat').exists():
        reg = ants.registration(fixed=tpl, moving=t1n, type_of_transform='SyN')
        salva_trasf(reg['fwdtransforms'], tp, 'T1_to_PED_fwd')
        salva_trasf(reg['invtransforms'], tp, 'T1_to_PED_inv')
    avanti = [str(tp / 'T1_to_PED_fwd_0.nii.gz'), str(tp / 'T1_to_PED_fwd_1.mat')]

    # FLAIR: nello spazio originale (lotto 1) serve anche la rigida verso la T1
    fl_t = tpl.new_image_like(np.zeros(tpl.shape, 'float32'))
    fl_trasf = None
    if c['FLAIR'] is not None:
        if not (dest / 'FLAIR_N4.nii.gz').exists():
            ants.image_write(ants.n4_bias_field_correction(ants.image_read(str(c['FLAIR']))), str(dest / 'FLAIR_N4.nii.gz'))
        fln = ants.image_read(str(dest / 'FLAIR_N4.nii.gz'))
        if c['formato'] == 'lotto1':
            if not (dest / 'transforms' / 'FLAIR_to_T1_0.mat').exists():
                reg = ants.registration(fixed=t1n, moving=fln, type_of_transform='Rigid')
                salva_trasf(reg['fwdtransforms'], dest / 'transforms', 'FLAIR_to_T1')
            fl_trasf = avanti + [str(dest / 'transforms' / 'FLAIR_to_T1_0.mat')]
        else:
            fl_trasf = avanti
        fl_t = ants.apply_transforms(fixed=tpl, moving=fln, transformlist=fl_trasf, interpolator='linear')

    # etichetta, nello spazio in cui e' stata disegnata, senza i residui
    if c['roi']:
        rif = ants.image_read(str(c['FLAIR']))
        m = np.zeros(rif.shape, bool)
        for r in ROI_LESIONE:
            if r in c['roi']:
                m |= np.squeeze(ants.image_read(str(c['roi'][r])).numpy()) > 0
        lez_trasf = fl_trasf
    else:
        rif = t1n
        m = ants.image_read(str(c['etichetta_T1'])).numpy() > 0
        lez_trasf = avanti
    m, tolte = togli_residui(m, float(np.prod(rif.spacing)))
    lez = rif.new_image_like(m.astype('float32'))
    lez_t = ants.apply_transforms(fixed=tpl, moving=lez, transformlist=lez_trasf, interpolator='linear')
    lez_t = lez_t.new_image_like((lez_t.numpy() >= 0.5).astype('float32'))
    # riferimento per le metriche: la stessa etichetta pulita nella griglia della T1 nativa
    if c['roi']:
        lez_nat = ants.apply_transforms(fixed=t1n, moving=lez, transformlist=[fl_trasf[-1]], interpolator='linear')
        lez_nat = lez_nat.new_image_like((lez_nat.numpy() >= 0.5).astype('float32'))
    else:
        lez_nat = lez
    ants.image_write(lez_nat, str(dest / 'lesione_pulita_inT1.nii.gz'))

    t1_t = ants.apply_transforms(fixed=tpl, moving=t1n, transformlist=avanti, interpolator='linear')
    for nome, img in (('T1_PED', t1_t), ('FLAIR_PED', fl_t), ('lesione_PED', lez_t)):
        ants.image_write(img, str(dest / f'{nome}.nii.gz'))

    # maschera del cervello: postprocessing e percentuale di cervello lesionato
    cer = dest / 'cervello_inT1.nii.gz'
    if not cer.exists():
        r = subprocess.run([sys.executable, str(SYNTHSTRIP), '-i', str(dest / 'T1_N4.nii.gz'), '-m', str(cer),
                            '--model', str(PESI_SYNTHSTRIP)] + (['-g'] if gpu else []), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'SynthStrip fallito su {p}: {r.stderr[-300:]}')
    cer_a = ants.image_read(str(cer)).numpy() > 0.5
    vox = float(np.prod(t1n.spacing))
    man.update({'residui_tolti': tolte, 'lesione_ml_nativo': round(float(lez_nat.numpy().sum()) * vox / 1000, 3),
                'cervello_ml': round(float(cer_a.sum()) * vox / 1000, 1),
                'lesione_ml_template': round(float(lez_t.numpy().sum()) / 1000, 3),
                'componenti_template': int(ndimage.label(lez_t.numpy() > 0, structure=STRUTTURA)[1])})
    man['percentuale_cervello_lesionato'] = round(100 * man['lesione_ml_nativo'] / man['cervello_ml'], 3)
    man_f.write_text(json.dumps(man, indent=1))
    print(p, man, flush=True)
    return man


# --- 3. split e dataset -------------------------------------------------------------------------

def strato(m: dict) -> str:
    """Lo strato dello split. Centro e campo sono parametri di acquisizione diversi (incontro del
    23/9); con la composizione attesa (un solo centro a 3 T, Stella Maris anche a 1.5 T, gli altri
    centri con uno o due pazienti ciascuno) tre strati sono il massimo che regge 27 pazienti."""
    if m.get('centro') == 'Stella Maris':
        return f"Stella Maris {m['campo_T']:g} T"
    return 'altri centri'


def dividi(pazienti: dict) -> tuple[list, list]:
    """70/30 per strato, e dentro lo strato alternando per volume di lesione: il test riceve
    lesioni grandi e piccole in proporzione, non per caso. Deterministico (seme fisso)."""
    rng = random.Random(SEME)
    test, train = [], []
    per_strato = {}
    for p, m in pazienti.items():
        per_strato.setdefault(strato(m), []).append(p)
    for s, ps in sorted(per_strato.items()):
        ps = sorted(ps, key=lambda x: pazienti[x]['lesione_ml_nativo'])
        n_test = round(len(ps) * QUOTA_TEST)
        # un paziente di test ogni ~1/QUOTA, con partenza casuale: copre tutta la scala dei volumi
        passo = len(ps) / n_test if n_test else 0
        off = rng.random() * passo if n_test else 0
        scelti = {ps[min(len(ps) - 1, int(off + i * passo))] for i in range(n_test)}
        test += sorted(scelti)
        train += [p for p in ps if p not in scelti]
    return sorted(train), sorted(test)


def fold(train: list, pazienti: dict, k: int = 5) -> list:
    """Cinque fold sul 70%: ordinati per strato e volume e distribuiti a serpentina, cosi' ogni
    fold di validazione ha strati e volumi misti."""
    ordine = sorted(train, key=lambda p: (strato(pazienti[p]), pazienti[p]['lesione_ml_nativo']))
    gruppi = [[] for _ in range(k)]
    for i, p in enumerate(ordine):
        giro, pos = divmod(i, k)
        gruppi[pos if giro % 2 == 0 else k - 1 - pos].append(p)
    return [{'train': sorted(p for p in train if p not in g), 'val': sorted(g)} for g in gruppi]


def dataset(id_ds: int, prova: bool = False):
    import nibabel as nib
    casi, meta = trova_casi(), metadati()
    pazienti = {}
    for p, c in casi.items():
        f = PROC / c['sessione'] / 'pipeline.json'
        if not f.exists():
            sys.exit(f'{p} non e\' preparato: pipeline_aincp.py prepara')
        m = json.loads(f.read_text())
        mm = next((x for x in meta.get(p, []) if x['sessione'].startswith(c['sessione'][:12])), None)
        if mm is None:
            sys.exit(f'{p}: centro e campo non sono nel foglio Excel, lo split non si puo\' fare')
        pazienti[p] = {**m, **mm}
    if prova:
        # solo lo split, senza scrivere niente: per controllarlo prima di fissarlo
        train, test = dividi(pazienti)
        for p in sorted(pazienti):
            print(f"{p} {'TEST' if p in test else 'train'} {strato(pazienti[p])} {pazienti[p]['lesione_ml_nativo']:.2f} ml")
        for i, f in enumerate(fold(train, pazienti)):
            print(f'fold {i}: val {f["val"]}')
        return
    nome = f'Dataset{id_ds}_AINCP_FINALE'
    raw = Path(os.environ['nnUNet_raw']) / nome
    if raw.exists():
        sys.exit(f'{raw} esiste gia\': non si sovrascrive (e lo split non si rigenera)')
    train, test = dividi(pazienti)
    for sub in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        (raw / sub).mkdir(parents=True)
    for p in train + test:
        d = PROC / pazienti[p]['sessione']
        tr = 'Tr' if p in train else 'Ts'
        shutil.copy(d / 'FLAIR_PED.nii.gz', raw / f'images{tr}' / f'{p}_0000.nii.gz')
        shutil.copy(d / 'T1_PED.nii.gz', raw / f'images{tr}' / f'{p}_0001.nii.gz')
        lab = nib.load(str(d / 'lesione_PED.nii.gz'))
        nib.save(nib.Nifti1Image((np.asanyarray(lab.dataobj) > 0).astype(np.uint8), lab.affine),
                 str(raw / f'labels{tr}' / f'{p}.nii.gz'))
    (raw / 'dataset.json').write_text(json.dumps({
        'name': nome, 'channel_names': {'0': 'FLAIR', '1': 'T1'}, 'labels': {'background': 0, 'lesione': 1},
        'numTraining': len(train), 'file_ending': '.nii.gz',
        'description': 'AInCP, template MNIPediatricAsym cohort-1, preparato da scripts/pipeline_aincp.py'}, indent=2))
    split = {'seme': SEME, 'quota_test': QUOTA_TEST, 'train': train, 'test': test,
             'fold': fold(train, pazienti), 'pazienti': pazienti}
    (raw / 'split_aincp.json').write_text(json.dumps(split, indent=1, default=str))
    righe = [f'# Split di `{nome}`', '', 'Prodotto da `scripts/pipeline_aincp.py dataset`, **non modificare a mano**. '
             'Fissato prima di qualunque training; il test si usa una volta sola.', '',
             '| paziente | insieme | strato | centro | campo | FLAIR | fenotipi | lesione (ml) | % cervello |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for p in sorted(pazienti):
        m = pazienti[p]
        righe.append(f"| {p} | {'**test**' if p in test else 'addestr.'} | {strato(m)} | {m['centro']} | {m['campo_T']:g} T | "
                     f"{'si' if m['FLAIR'] else 'no'} | {'si' if m['fenotipi'] else 'no'} | {m['lesione_ml_nativo']:.2f} | "
                     f"{m['percentuale_cervello_lesionato']:.2f} |")
    for i, f in enumerate(split['fold']):
        righe.append(f'\nFold {i}, validazione: {", ".join(f["val"])}')
    (PROGETTO / 'reports' / f'split_{nome}.md').write_text('\n'.join(righe) + '\n')
    print('\n'.join(righe))


def piano(id_ds: int):
    """Piano di 502 con le tre correzioni per canale, e i fold scritti da `dataset`."""
    prep = next(Path(os.environ['nnUNet_preprocessed']).glob(f'Dataset{id_ds}_*'))
    raw = Path(os.environ['nnUNet_raw']) / prep.name
    if not (prep / 'nnUNetPlansDa502.json').exists():
        subprocess.run(['nnUNetv2_move_plans_between_datasets', '-s', '502', '-t', str(id_ds),
                        '-sp', 'nnUNetPlans', '-tp', 'nnUNetPlansDa502'], check=True)
    p = json.load(open(prep / 'nnUNetPlansDa502.json'))
    fp = json.load(open(prep / 'dataset_fingerprint.json'))
    for c in p['configurations'].values():
        if 'normalization_schemes' in c:
            c['normalization_schemes'] = ['ZScoreNormalization'] * 2
            c['use_mask_for_norm'] = [False] * 2
    p['foreground_intensity_properties_per_channel'] = fp['foreground_intensity_properties_per_channel']
    json.dump(p, open(prep / 'nnUNetPlansDa502.json', 'w'), indent=2)
    shutil.copy(raw / 'dataset.json', prep / 'dataset.json')
    dest = prep / 'splits_final.json'
    if dest.exists():
        sys.exit(f'{dest} esiste gia\' e non si rigenera')
    dest.write_text(json.dumps(json.loads((raw / 'split_aincp.json').read_text())['fold'], indent=2))
    print('piano e fold pronti in', prep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('azione', choices=['elenco', 'prepara', 'dataset', 'piano'])
    ap.add_argument('--casi', nargs='*')
    ap.add_argument('--id', type=int)
    ap.add_argument('--gpu', action='store_true', help='SynthStrip su GPU')
    ap.add_argument('--prova', action='store_true', help='dataset: mostra lo split senza scrivere')
    a = ap.parse_args()
    if a.azione == 'elenco':
        elenco()
    elif a.azione == 'prepara':
        import ants
        tpl = ants.image_read(str(TEMPLATE))
        casi = trova_casi()
        for p in (a.casi or casi):
            try:
                prepara(p, casi[p], tpl, a.gpu)
            except Exception as e:                                   # noqa: BLE001
                print(f'{p}: FALLITO -> {e!r}', flush=True)
    elif a.azione == 'dataset':
        dataset(a.id, a.prova)
    else:
        piano(a.id)


if __name__ == '__main__':
    main()
