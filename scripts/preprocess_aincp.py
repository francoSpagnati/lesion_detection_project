#!/usr/bin/env python3
"""Preprocessing del lotto AInCP, secondo `docs/pipeline_preprocessing.docx`.

Da T1 e FLAIR originali allo spazio MNI, con le maschere che seguono le stesse
trasformazioni. La registrazione non lineare al template si calcola **una sola volta**,
sulla T1; la FLAIR viene prima portata nello spazio T1 e poi eredita le trasformazioni
T1 -> MNI, senza una seconda registrazione non lineare.

    T1 originale ---[N4]---> T1_N4 ---[Rigid+Affine+SyN]---> T1_MNI
                                |                    |
    FLAIR originale -[N4]-> FLAIR_N4 -[Rigid, MI]-> FLAIR_inT1 -[eredita]-> FLAIR_MNI
                                                        |
    ROI_1/2/4 (spazio FLAIR) ---------[stessa trasformazione, NN]---------> ROI_MNI

Perche' ANTsPy e non FSL/ANTs da riga di comando
------------------------------------------------
Su questa macchina non sono installati ne' `flirt` ne' `antsRegistration`, mentre
`antspyx` c'e' gia' nel venv. E' lo **stesso motore di registrazione** di ANTs, quindi
la pipeline e' quella del documento; cambia solo l'interfaccia. Il passo 2 del documento
usa FLIRT con `-dof 6 -cost mutualinfo`: qui e' `type_of_transform='Rigid'`, che in ANTs
usa mutual information ed e' anch'esso a 6 gradi di liberta'.

Quale template
--------------
`MNI152NLin2009aSym`, che **non e' una scelta libera**: e' esattamente il template in cui
si trova gia' `Dataset502_ATLAS` (verificato: stessa griglia 197x233x189 a 1 mm, stesso
affine). Usare lo stesso mette AInCP nello spazio del dataset di pretraining piu' grande
che abbiamo, invece che in uno spazio simile ma non identico.

Le maschere
-----------
Sono disegnate sulla FLAIR e vengono trasportate con interpolazione **nearest neighbour**,
perche' sono binarie. Ogni resampling costa qualcosa su lesioni di pochi voxel: per questo
lo script **misura** il volume e il numero di componenti connesse a ogni stadio e lo scrive
nel manifesto, invece di dare per scontato che non si perda nulla.

Che cosa viene scritto, per ogni soggetto, sotto `data/processed/AINCP/<soggetto>/`:

    T1_N4.nii.gz  FLAIR_N4.nii.gz          dopo la correzione di campo
    FLAIR_inT1.nii.gz  lesione_inT1.nii.gz spazio T1 nativo
    T1_MNI.nii.gz  FLAIR_MNI.nii.gz  lesione_MNI.nii.gz
    transforms/                            tutte le matrici e i warp, per tornare indietro
    manifesto.json                         volumi e conteggi a ogni stadio

Lo spazio T1 nativo viene salvato **oltre** a quello MNI perche' costa quasi nulla una
volta che le trasformazioni esistono, e permette di decidere in quale spazio addestrare
confrontando due numeri invece che discutendone.

    source scripts/env.sh
    python scripts/preprocess_aincp.py                 # tutti i soggetti
    python scripts/preprocess_aincp.py --soggetti FSM_032_MR_1_anon --forza
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from scipy import ndimage

PROGETTO = Path('/home/ccantini/lesion_detection_project')
LOTTO = PROGETTO / 'data/raw/AINCP/dataset_FSM_v1'
USCITA = PROGETTO / 'data/processed/AINCP'
TEMPLATE = Path('/data01/atlas_mri_data/ATLAS_2/MNI152NLin2009aSym.nii.gz')

# ROI_3 e' il ventricolo dell'emisfero leso: e' una conseguenza della lesione, non la
# lesione, e per decisione del relatore resta fuori dalla maschera.
ROI_LESIONE = ('ROI_1', 'ROI_2', 'ROI_4')


def estensione(percorso) -> str:
    """Estensione completa, non l'ultima.

    `Path('x1Warp.nii.gz').suffix` restituisce `.gz`, e un campo di deformazione salvato
    come `.gz` non e' piu' rileggibile da ANTs: le trasformazioni servono per riportare le
    predizioni nello spazio nativo del paziente, quindi il nome deve restare valido.
    """
    nome = Path(percorso).name
    return '.nii.gz' if nome.endswith('.nii.gz') else Path(nome).suffix


def trova_roi(cartella: Path) -> dict:
    """ROI al primo livello di COREGISTERED, indicizzate per numero.

    Il numero si estrae con una regex tollerante ai suffissi di lavorazione, perche' la
    nomenclatura non e' uniforme: `FSM_052` ha `ROI_1 - Copia.nii` e non `ROI_1.nii`, e
    cercare il nome esatto lo salterebbe in silenzio perdendo un paziente su dieci.
    Le cartelle `back_up/` non vengono guardate, e i `.voi` sono ignorati.
    """
    out = {}
    for f in sorted(cartella.glob('ROI_*.nii')):
        m = re.match(r'ROI_(\d+)', f.name, re.I)
        if not m:
            continue
        chiave = f'ROI_{m.group(1)}'
        if chiave not in out or len(f.name) < len(out[chiave].name):
            out[chiave] = f
    return out


def elenca_soggetti() -> dict:
    soggetti = {}
    for d in sorted(LOTTO.glob('FSM_*_anon')):
        co = d / 'COREGISTERED'
        if not co.is_dir():
            continue
        t1 = next(iter(co.glob('*_T1_original.nii.gz')), None)
        fl = next(iter(co.glob('*_FLAIR_original.nii.gz')), None)
        if t1 is None or fl is None:
            print(f'  {d.name}: manca T1 o FLAIR, saltato')
            continue
        soggetti[d.name] = {'T1': t1, 'FLAIR': fl, 'roi': trova_roi(co)}
    return soggetti


def statistiche_maschera(img: ants.ANTsImage) -> dict:
    """Volume, voxel e componenti connesse. E' la misura con cui si verifica quanto
    costa ogni resampling, invece di assumere che sia gratis."""
    a = img.numpy() > 0.5
    _, n = ndimage.label(a)
    mm3 = float(np.prod(img.spacing))
    return {'voxel': int(a.sum()), 'mm3': round(float(a.sum()) * mm3, 1), 'componenti': int(n)}


def unione_lesione(roi: dict, riferimento: ants.ANTsImage) -> ants.ANTsImage:
    """ROI_1 | ROI_2 | ROI_4, binaria, sulla griglia della FLAIR."""
    acc = None
    for nome in ROI_LESIONE:
        if nome not in roi:
            continue
        a = ants.image_read(str(roi[nome])).numpy() > 0
        acc = a if acc is None else (acc | a)
    if acc is None:
        raise RuntimeError('nessuna ROI lesionale')
    return ants.from_numpy(acc.astype('float32'), origin=riferimento.origin,
                           spacing=riferimento.spacing, direction=riferimento.direction)


def elabora(nome: str, info: dict, template: ants.ANTsImage, forza: bool) -> dict:
    dest = USCITA / nome
    if dest.exists() and not forza:
        print(f'{nome}: gia\' fatto, salto (--forza per rifarlo)')
        return json.loads((dest / 'manifesto.json').read_text())
    if dest.exists():
        shutil.rmtree(dest)
    (dest / 'transforms').mkdir(parents=True)

    avvio = time.time()
    man = {'soggetto': nome, 'roi_presenti': sorted(info['roi']),
           'roi_nella_lesione': [r for r in ROI_LESIONE if r in info['roi']], 'stadi': {}}

    t1 = ants.image_read(str(info['T1']))
    fl = ants.image_read(str(info['FLAIR']))
    man['geometria'] = {
        'T1': {'shape': list(t1.shape), 'spacing': [round(s, 4) for s in t1.spacing]},
        'FLAIR': {'shape': list(fl.shape), 'spacing': [round(s, 4) for s in fl.spacing]},
        # se fosse gia' vero, la cartella COREGISTERED direbbe la verita'
        'T1_e_FLAIR_stessa_griglia': bool(t1.shape == fl.shape and
                                          np.allclose(t1.direction, fl.direction, atol=1e-3) and
                                          np.allclose(t1.origin, fl.origin, atol=1e-3)),
    }

    lez = unione_lesione(info['roi'], fl)
    man['stadi']['0_originale_spazio_FLAIR'] = statistiche_maschera(lez)

    # --- 1. correzione del campo di bias -------------------------------------
    # Il documento la dice "se necessario". Sui dataset pubblici avevamo misurato che non
    # era giustificata, ma AInCP viene da 18 siti di acquisizione diversi, quindi qui il
    # campo puo' pesare davvero. Si applica e si registra di quanto ha cambiato le
    # intensita', cosi' l'effetto resta misurato e non assunto.
    t1n = ants.n4_bias_field_correction(t1)
    fln = ants.n4_bias_field_correction(fl)
    man['n4'] = {
        'T1_variazione_mediana_%': round(float(
            100 * np.median(np.abs(t1n.numpy() - t1.numpy())) / max(1e-6, np.median(np.abs(t1.numpy())))), 2),
        'FLAIR_variazione_mediana_%': round(float(
            100 * np.median(np.abs(fln.numpy() - fl.numpy())) / max(1e-6, np.median(np.abs(fl.numpy())))), 2),
    }
    ants.image_write(t1n, str(dest / 'T1_N4.nii.gz'))
    ants.image_write(fln, str(dest / 'FLAIR_N4.nii.gz'))

    # --- 2. FLAIR -> T1, rigida --------------------------------------------
    # Multimodale, quindi la metrica e' mutual information: e' quello che fa ANTs con
    # 'Rigid', ed e' la stessa scelta di `flirt -dof 6 -cost mutualinfo` del documento.
    reg_ft = ants.registration(fixed=t1n, moving=fln, type_of_transform='Rigid')
    for i, t in enumerate(reg_ft['fwdtransforms']):
        shutil.copy(t, dest / 'transforms' / f'FLAIR_to_T1_{i}{estensione(t)}')
    ants.image_write(reg_ft['warpedmovout'], str(dest / 'FLAIR_inT1.nii.gz'))

    lez_t1 = ants.apply_transforms(fixed=t1n, moving=lez,
                                   transformlist=reg_ft['fwdtransforms'],
                                   interpolator='nearestNeighbor')
    ants.image_write(lez_t1, str(dest / 'lesione_inT1.nii.gz'))
    man['stadi']['1_spazio_T1'] = statistiche_maschera(lez_t1)

    # --- 3. T1 -> MNI, Rigid -> Affine -> SyN -------------------------------
    reg_mni = ants.registration(fixed=template, moving=t1n, type_of_transform='SyN')
    for i, t in enumerate(reg_mni['fwdtransforms']):
        shutil.copy(t, dest / 'transforms' / f'T1_to_MNI_fwd_{i}{estensione(t)}')
    for i, t in enumerate(reg_mni['invtransforms']):
        # servono per riportare le predizioni nello spazio nativo del paziente
        shutil.copy(t, dest / 'transforms' / f'T1_to_MNI_inv_{i}{estensione(t)}')
    ants.image_write(reg_mni['warpedmovout'], str(dest / 'T1_MNI.nii.gz'))

    # --- 4. la FLAIR e le maschere ereditano le trasformazioni --------------
    # Nessuna seconda registrazione non lineare: si concatenano quelle gia' stimate.
    fl_mni = ants.apply_transforms(fixed=template, moving=fln,
                                   transformlist=reg_mni['fwdtransforms'] + reg_ft['fwdtransforms'],
                                   interpolator='linear')
    ants.image_write(fl_mni, str(dest / 'FLAIR_MNI.nii.gz'))

    lez_mni = ants.apply_transforms(fixed=template, moving=lez,
                                    transformlist=reg_mni['fwdtransforms'] + reg_ft['fwdtransforms'],
                                    interpolator='nearestNeighbor')
    ants.image_write(lez_mni, str(dest / 'lesione_MNI.nii.gz'))
    man['stadi']['2_spazio_MNI'] = statistiche_maschera(lez_mni)

    # Le singole ROI seguono la stessa strada: servono per misurare il rilevamento
    # separatamente sui due fenotipi, che e' il confronto che ci interessa davvero.
    for r in ROI_LESIONE:
        if r not in info['roi']:
            continue
        m = ants.image_read(str(info['roi'][r]))
        m = ants.from_numpy((m.numpy() > 0).astype('float32'), origin=fl.origin,
                            spacing=fl.spacing, direction=fl.direction)
        ants.image_write(
            ants.apply_transforms(fixed=template, moving=m,
                                  transformlist=reg_mni['fwdtransforms'] + reg_ft['fwdtransforms'],
                                  interpolator='nearestNeighbor'),
            str(dest / f'{r}_MNI.nii.gz'))

    a, b = man['stadi']['0_originale_spazio_FLAIR'], man['stadi']['2_spazio_MNI']
    man['perdita'] = {
        'variazione_volume_%': round(100 * (b['mm3'] - a['mm3']) / a['mm3'], 2) if a['mm3'] else None,
        'componenti_perse': a['componenti'] - b['componenti'],
    }
    man['secondi'] = round(time.time() - avvio, 1)
    (dest / 'manifesto.json').write_text(json.dumps(man, indent=2, ensure_ascii=False))

    p = man['perdita']
    print(f"{nome}: {a['mm3']:.0f} mm³ / {a['componenti']} comp. -> MNI {b['mm3']:.0f} mm³ / "
          f"{b['componenti']} comp.  (volume {p['variazione_volume_%']:+.1f}%, "
          f"componenti {-p['componenti_perse']:+d})  [{man['secondi']:.0f}s]")
    return man


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--soggetti', nargs='*', help='sottoinsieme da elaborare')
    ap.add_argument('--forza', action='store_true', help='rifai anche i soggetti gia\' fatti')
    args = ap.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f'Template assente: {TEMPLATE}')
    template = ants.image_read(str(TEMPLATE))
    print(f'template: {TEMPLATE.name} {template.shape} @ '
          f'{tuple(round(s, 2) for s in template.spacing)} mm  '
          '(lo stesso in cui si trova gia\' Dataset502_ATLAS)\n')

    soggetti = elenca_soggetti()
    if args.soggetti:
        soggetti = {k: v for k, v in soggetti.items() if k in set(args.soggetti)}
    print(f'{len(soggetti)} soggetti da elaborare\n')

    USCITA.mkdir(parents=True, exist_ok=True)
    manifesti, falliti = [], []
    for nome, info in soggetti.items():
        try:
            manifesti.append(elabora(nome, info, template, args.forza))
        except Exception as e:                                   # noqa: BLE001
            falliti.append((nome, repr(e)))
            print(f'{nome}: FALLITO -> {e!r}')

    (USCITA / 'manifesti.json').write_text(json.dumps(manifesti, indent=2, ensure_ascii=False))
    print(f'\n{len(manifesti)} completati, {len(falliti)} falliti')
    if manifesti:
        dv = [m['perdita']['variazione_volume_%'] for m in manifesti
              if m['perdita']['variazione_volume_%'] is not None]
        cp = [m['perdita']['componenti_perse'] for m in manifesti]
        print(f'variazione di volume verso MNI: mediana {np.median(dv):+.1f}%, '
              f'estremi {min(dv):+.1f}% / {max(dv):+.1f}%')
        print(f'componenti connesse perse: totale {sum(cp)}, massimo su un soggetto {max(cp)}')
    for nome, err in falliti:
        print('  ', nome, err)


if __name__ == '__main__':
    main()
