#!/usr/bin/env python
"""Riassume la validation di cross-validation prodotta da nnU-Net.

Legge i `summary.json` scritti in `fold_X/validation/` e produce una tabella
leggibile in `reports/baseline_validation.md`.

Perche' non basta il Dice medio che nnU-Net stampa a fine training: su questi
compiti la media nasconde la distribuzione. Con lesioni molto piccole il Dice e'
quasi rumore (sbagliare 10 voxel su 20 fa crollare il punteggio, sulla stessa
lesione da 20.000 voxel non si nota), quindi vanno guardate mediana, quartili e
i casi peggiori. Su WMH serve anche il Dice **per sito**, perche' la
generalizzazione fra scanner e' la domanda interessante di quel dataset e una
media unica la nasconde.

Uso:
    python scripts/summarize_validation.py            # tutti i dataset trovati
    python scripts/summarize_validation.py 501 502    # solo alcuni
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

PROGETTO = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get('nnUNet_results', PROGETTO / 'nnUNet' / 'nnUNet_results'))
OUT = PROGETTO / 'reports' / 'baseline_validation.md'

# Nomi leggibili delle classi, per dataset. DBB ha 6 tessuti, gli altri una lesione.
CLASSI = {
    '503': {'1': 'CSF', '2': 'sostanza grigia', '3': 'sostanza bianca',
            '4': 'grigia profonda', '5': 'tronco encefalico', '6': 'cervelletto'},
}


def gruppo_di(caso: str) -> str:
    """Sito per WMH (`Amsterdam_GE3T_44` -> `Amsterdam_GE3T`), coorte per ATLAS
    (`sub-r001s002` -> `r001`). Per gli altri, tutto in un gruppo unico."""
    m = re.match(r'^sub-(r\d+)s\d+$', caso)
    if m:
        return m.group(1)
    m = re.match(r'^(.+)_\d+$', caso)
    if m:
        return m.group(1)
    return 'tutti'


def quartili(v):
    v = np.asarray(v, dtype=float)
    return dict(n=len(v), media=v.mean(), mediana=np.median(v),
                q1=np.percentile(v, 25), q3=np.percentile(v, 75),
                minimo=v.min(), massimo=v.max())


def riga(nome, s):
    return (f"| {nome} | {s['n']} | {s['media']:.4f} | {s['mediana']:.4f} | "
            f"{s['q1']:.4f} | {s['q3']:.4f} | {s['minimo']:.4f} | {s['massimo']:.4f} |")


INTESTAZIONE = ('| | n | media | mediana | Q1 | Q3 | min | max |\n'
                '| --- | --- | --- | --- | --- | --- | --- | --- |')


def analizza(cartella: Path, did: str, righe: list):
    summary = cartella / 'validation' / 'summary.json'
    if not summary.exists():
        return False
    s = json.loads(summary.read_text())
    per_caso = s['metric_per_case']
    casi = [Path(c['reference_file']).name.replace('.nii.gz', '') for c in per_caso]
    classi = sorted(s['mean'].keys(), key=int)
    etichette = CLASSI.get(did, {})

    righe.append(f"\n## {cartella.parents[1].name} — {cartella.parent.name}\n")
    righe.append(f"Fold: `{cartella.name}`, {len(per_caso)} casi di validation.\n")

    # --- distribuzione del Dice, per classe
    righe.append('### Dice per classe, sui casi interi\n')
    righe.append(INTESTAZIONE)
    for cl in classi:
        dice = [c['metrics'][cl]['Dice'] for c in per_caso]
        nome = f"classe {cl}" + (f" ({etichette[cl]})" if cl in etichette else '')
        righe.append(riga(nome, quartili(dice)))
    righe.append('')

    # --- per gruppo (sito/coorte), solo sulla prima classe e solo se i gruppi sono piu' d'uno
    gruppi = sorted({gruppo_di(c) for c in casi})
    if len(gruppi) > 1 and len(classi) == 1:
        cl = classi[0]
        righe.append('### Dice per gruppo (sito per WMH, coorte per ATLAS)\n')
        righe.append(INTESTAZIONE)
        for g in gruppi:
            dice = [c['metrics'][cl]['Dice'] for c, n in zip(per_caso, casi) if gruppo_di(n) == g]
            righe.append(riga(g, quartili(dice)))
        righe.append('')

    # --- relazione fra dimensione del bersaglio e Dice: e' il punto che la media nasconde
    if len(classi) == 1:
        cl = classi[0]
        dati = sorted(((c['metrics'][cl]['n_ref'], c['metrics'][cl]['Dice'], n)
                       for c, n in zip(per_caso, casi)), key=lambda t: t[0])
        righe.append('### I cinque casi con il bersaglio piu' + "'" + ' piccolo\n')
        righe.append('| caso | voxel di riferimento | Dice |')
        righe.append('| --- | --- | --- |')
        for n_ref, dice, nome in dati[:5]:
            righe.append(f'| {nome} | {n_ref:.0f} | {dice:.4f} |')
        meta = len(dati) // 2
        piccoli = np.mean([d for _, d, _ in dati[:meta]])
        grandi = np.mean([d for _, d, _ in dati[meta:]])
        righe.append(f'\nDice medio sulla meta\' di casi con bersaglio piu\' piccolo: **{piccoli:.4f}**; '
                     f'sulla meta\' con bersaglio piu\' grande: **{grandi:.4f}**.\n')
        zero = [(n, c['metrics'][cl]['n_ref'], c['metrics'][cl]['n_pred'])
                for c, n in zip(per_caso, casi) if c['metrics'][cl]['Dice'] == 0]
        if zero:
            righe.append('### Casi con Dice esattamente 0\n')
            righe.append('| caso | voxel di riferimento | voxel predetti |')
            righe.append('| --- | --- | --- |')
            for nome_caso, nref, npred in sorted(zero, key=lambda t: -t[1]):
                righe.append(f'| {nome_caso} | {nref:.0f} | {npred:.0f} |')
            righe.append('\nUn Dice nullo su un bersaglio grande, con voxel predetti in quantita\', non e\'')
            righe.append('una rifinitura imprecisa dei bordi: e\' un errore di **localizzazione**, il modello ha')
            righe.append('segmentato altrove. Vanno guardati come categoria a se\'.\n')
    return True


def main():
    voluti = sys.argv[1:]
    righe = ['# Validation delle baseline (cross-validation)',
             '',
             'Generato da `scripts/summarize_validation.py`, **non modificare a mano**.',
             '',
             'Sono i risultati della validation di nnU-Net sui **casi interi** ricostruiti con la',
             'finestra scorrevole, non lo "pseudo Dice" stampato durante il training (che e\' calcolato',
             'sulle patch di addestramento e serve solo a seguire la convergenza).',
             '',
             'Questi numeri vengono dalla **cross-validation**, non dal test holdout, che resta',
             'intoccato fino alla scelta definitiva della configurazione.']
    trovati = 0
    for ds in sorted(RESULTS.iterdir()):
        did = ds.name[7:10] if ds.name.startswith('Dataset') else ''
        if not did.isdigit() or (voluti and did not in voluti) or did == '599':
            continue
        for trainer in sorted(ds.iterdir()):
            for fold in sorted(trainer.glob('fold_*')):
                trovati += analizza(fold, did, righe)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text('\n'.join(righe) + '\n')
    print(f'{trovati} validation riassunte -> {OUT.relative_to(PROGETTO)}')


if __name__ == '__main__':
    main()
