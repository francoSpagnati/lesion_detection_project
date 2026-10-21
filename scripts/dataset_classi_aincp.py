#!/usr/bin/env python
"""Due dataset a piu' classi dai pazienti AInCP, identici a Dataset506 in tutto tranne l'etichetta.

    python scripts/dataset_classi_aincp.py crea  --tipo C     # -> nnUNet_raw/Dataset507_AINCP_VENTRICOLO
    python scripts/dataset_classi_aincp.py crea  --tipo F     # -> nnUNet_raw/Dataset508_AINCP_FENOTIPI
    python scripts/dataset_classi_aincp.py piano --tipo C|F   # fingerprint, piano Da502, preprocessing, split di 506

PERCHE' (21/10/2026, dopo la ricetta B). Due domande dell'utente e del responsabile, una per dataset,
cosi' che ognuno cambi una cosa sola rispetto a Z:
  C  lesione + **ventricolo dilatato** (ROI_3) come seconda classe. La rete impara esplicitamente che
     la cavita' del ventricolo non e' lesione: e' l'errore di Z su FSM_046 nel test. La lesione resta
     quella di 506 (il relatore ha voluto il ventricolo fuori dal bersaglio, e fuori resta).
  F  **gliosi, cistica, encefalomalacia** come tre classi (ROI_1, ROI_2, ROI_4). La rete dice anche il
     tipo di lesione. L'unione delle tre classi e' esattamente l'etichetta di 506.

COME. Immagini: collegamenti fisici a quelle di Dataset506 (stessi file). Etichette: ogni ROI, nello
spazio della FLAIR in cui e' stata disegnata, ristretta all'etichetta pulita dai residui, portata nel
template con le trasformazioni della pipeline (lineare); poi
  C  1 = lesione di 506; 2 = ROI_3 >= 0.5 fuori dalla lesione. Nei 4 pazienti di training senza ROI_3
     (FSM_033, 039, 045, 056) il ventricolo resta sfondo: la ROI_3 e' il ventricolo *dilatato*, e
     l'assenza si legge come ventricolo normale (decisione dell'utente, 21/10, da confermare col clinico);
  F  dentro la lesione di 506, la classe con il valore interpolato piu' alto; a parita' (le ROI si
     sovrappongono: gliosi ed encefalomalacia per circa 5 000 mm³, cistica ed encefalomalacia per
     circa 6 400 mm³ sui 17) vince cistica, poi encefalomalacia, poi gliosi: la cavita' e' il reperto
     piu' netto, la gliosi spesso ne e' il bordo.
Split: copiati da 506 (split_aincp.json, splits_final.json), mai rigenerati.

IL CAMPIONAMENTO DELLE PATCH. Con piu' classi nnU-Net centra la patch forzata su una classe scelta a
caso fra quelle presenti, poi su un suo voxel: in C meta' delle patch forzate andrebbe sul ventricolo,
in F le cisti verrebbero scelte quanto la gliosi. Sarebbe una seconda variabile. Per questo `piano`
sostituisce, nei .pkl del preprocessing, le posizioni di campionamento con quelle di 506 (stesse
immagini, stessa lesione): le patch forzate sono campionate come in Z, voxel per voxel.
"""
import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_aincp as pa  # noqa: E402

RAW = Path(os.environ['nnUNet_raw'])
PREP = Path(os.environ['nnUNet_preprocessed'])
BASE = 'Dataset506_AINCP_FINALE'
TIPI = {
    'C': {'id': 507, 'nome': 'Dataset507_AINCP_VENTRICOLO',
          'labels': {'background': 0, 'lesione': 1, 'ventricolo_dilatato': 2}},
    'F': {'id': 508, 'nome': 'Dataset508_AINCP_FENOTIPI',
          'labels': {'background': 0, 'gliosi': 1, 'cistica': 2, 'encefalomalacia': 3}},
}
PRECEDENZA_F = [('ROI_2', 2), ('ROI_4', 3), ('ROI_1', 1)]      # a parita' vince la prima


def roi_nel_template(c: dict, roi: str, pulita: np.ndarray | None, tpl):
    """Valore interpolato (0-1) della ROI nel template, dopo averla ristretta a `pulita` (se data)."""
    import ants
    dest = pa.PROC / c['sessione']
    tp = dest / 'transforms_ped'
    trasf = [str(tp / 'T1_to_PED_fwd_0.nii.gz'), str(tp / 'T1_to_PED_fwd_1.mat')]
    if c['formato'] == 'coregistered':
        trasf.append(str(dest / 'transforms' / 'FLAIR_to_T1_0.mat'))
    rif = ants.image_read(str(c['FLAIR']))
    m = np.squeeze(ants.image_read(str(c['roi'][roi])).numpy()) > 0
    if pulita is not None:
        m &= pulita
    return ants.apply_transforms(fixed=tpl, moving=rif.new_image_like(m.astype('float32')),
                                 transformlist=trasf, interpolator='linear').numpy()


def etichetta(p: str, c: dict, tipo: str, tpl, lesione: np.ndarray) -> tuple[np.ndarray, dict]:
    import ants
    rif = ants.image_read(str(c['FLAIR']))
    unione = np.zeros(rif.shape, bool)
    for r in pa.ROI_LESIONE:
        if r in c['roi']:
            unione |= np.squeeze(ants.image_read(str(c['roi'][r])).numpy()) > 0
    pulita, _ = pa.togli_residui(unione, float(np.prod(rif.spacing)))
    out = lesione.astype(np.uint8)                            # C: 1 = lesione di 506
    info = {}
    if tipo == 'C':
        if 'ROI_3' in c['roi']:
            v = (roi_nel_template(c, 'ROI_3', None, tpl) >= 0.5) & ~lesione
            out[v] = 2
            info['ventricolo_voxel'] = int(v.sum())
        else:
            info['ventricolo_voxel'] = 0
    else:
        valori = np.stack([roi_nel_template(c, r, pulita, tpl) if r in c['roi'] else np.zeros(lesione.shape, np.float32)
                           for r, _ in PRECEDENZA_F])
        classe = np.array([k for _, k in PRECEDENZA_F], np.uint8)[np.argmax(valori, 0)]
        out = np.where(lesione, classe, 0).astype(np.uint8)
        info.update({n: int((out == k).sum()) for n, k in TIPI['F']['labels'].items() if k})
        info['lesione_senza_roi'] = int((lesione & (valori.max(0) == 0)).sum())   # atteso 0
    return out, info


def crea(tipo: str):
    import ants
    import nibabel as nib
    t = TIPI[tipo]
    dest, base = RAW / t['nome'], RAW / BASE
    if dest.exists():
        sys.exit(f'{dest} esiste gia\': non si sovrascrive')
    split = json.loads((base / 'split_aincp.json').read_text())
    tpl = ants.image_read(str(pa.TEMPLATE))
    casi = pa.trova_casi()
    for sub in ('imagesTr', 'labelsTr', 'imagesTs', 'labelsTs'):
        (dest / sub).mkdir(parents=True)
    righe = []
    for p in split['train'] + split['test']:
        tr = 'Tr' if p in split['train'] else 'Ts'
        for k in ('0000', '0001'):
            os.link(base / f'images{tr}' / f'{p}_{k}.nii.gz', dest / f'images{tr}' / f'{p}_{k}.nii.gz')
        lab = nib.load(str(base / f'labels{tr}' / f'{p}.nii.gz'))
        lesione = np.asanyarray(lab.dataobj) > 0
        out, info = etichetta(p, casi[p], tipo, tpl, lesione)
        # la lesione (C: classe 1; F: unione delle tre classi) deve essere quella di 506, voxel per voxel
        assert (((out == 1) if tipo == 'C' else (out > 0)) == lesione).all(), f'{p}: la lesione non coincide con 506'
        nib.save(nib.Nifti1Image(out, lab.affine, lab.header), str(dest / f'labels{tr}' / f'{p}.nii.gz'))
        righe.append(f'{p} ({tr}): {info}')
        print(righe[-1], flush=True)
    (dest / 'dataset.json').write_text(json.dumps({
        'name': t['nome'], 'channel_names': {'0': 'FLAIR', '1': 'T1'}, 'labels': t['labels'],
        'numTraining': len(split['train']), 'file_ending': '.nii.gz',
        'description': f'Come {BASE} con piu\' classi, preparato da scripts/dataset_classi_aincp.py'}, indent=2))
    shutil.copy(base / 'split_aincp.json', dest / 'split_aincp.json')
    (dest / 'etichette.txt').write_text('\n'.join(righe) + '\n')


def piano(tipo: str):
    t = TIPI[tipo]
    env = {**os.environ}
    subprocess.run(['nnUNetv2_extract_fingerprint', '-d', str(t['id']), '--verify_dataset_integrity'], check=True, env=env)
    prep = PREP / t['nome']
    # stesso piano di 506 (Da502 con le correzioni per canale di pipeline_aincp.piano): copiato, non ricalcolato
    p = json.loads((PREP / BASE / 'nnUNetPlansDa502.json').read_text())
    p['dataset_name'] = t['nome']
    (prep / 'nnUNetPlansDa502.json').write_text(json.dumps(p, indent=2))
    shutil.copy(RAW / t['nome'] / 'dataset.json', prep / 'dataset.json')
    if (prep / 'splits_final.json').exists():
        sys.exit('splits_final.json esiste gia\' e non si rigenera')
    shutil.copy(PREP / BASE / 'splits_final.json', prep / 'splits_final.json')
    subprocess.run(['nnUNetv2_preprocess', '-d', str(t['id']), '-plans_name', 'nnUNetPlansDa502', '-c', '3d_fullres',
                    '-np', '8'], check=True, env=env)
    # campionamento delle patch come in Z: le posizioni della lesione di 506 (vedi il docstring)
    cartella = prep / 'nnUNetPlansDa502_3d_fullres'
    for f in sorted(cartella.glob('*.pkl')):
        b = pickle.load(open(PREP / BASE / 'nnUNetPlansDa502_3d_fullres' / f.name, 'rb'))
        q = pickle.load(open(f, 'rb'))
        for k in ('shape_before_cropping', 'bbox_used_for_cropping', 'shape_after_cropping_and_before_resampling'):
            assert np.array_equal(np.asarray(q[k]), np.asarray(b[k])), f'{f.name}: {k} diverso da 506'
        q['class_locations'] = b['class_locations']
        pickle.dump(q, open(f, 'wb'))
    print('pronto:', prep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('azione', choices=['crea', 'piano'])
    ap.add_argument('--tipo', choices=list(TIPI), required=True)
    a = ap.parse_args()
    crea(a.tipo) if a.azione == 'crea' else piano(a.tipo)


if __name__ == '__main__':
    main()
