#!/usr/bin/env python
"""Confronta due trainer sullo stesso dataset unendo i fold: ogni paziente conta una volta.

    python scripts/confronto_fold.py 504 \\
        nnUNetTrainerWandb_250ep_arresto50 \\
        nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50 \\
        --soglia-mm3 10 --out reports/confronto_504_T0_T1.md

PERCHE'. Sui dieci pazienti AInCP la validazione e' a cinque fold, due pazienti per fold. Il Dice
medio di un fold e' la media di due numeri: confrontare due trainer fold per fold vorrebbe dire
cinque confronti da due casi l'uno, e nessuno direbbe niente. Ogni paziente pero' e' fuori dal
training esattamente in un fold, quindi mettendo insieme le cinque validazioni si ottengono dieci
predizioni **fuori-fold**, una per paziente, per ciascun trainer: e' su quelle che si fa il
confronto appaiato, come in scripts/confronto_appaiato.py ma con un campione di dieci invece
che di due.

Con dieci coppie il test di Wilcoxon ha poca potenza (il p minimo raggiungibile e' 0.002 se
tutti e dieci vanno nello stesso verso): il numero che conta e' quante coppie migliorano e di
quanto, e il p va letto come cautela, non come verdetto.

COSA CONFRONTA. Due grandezze per paziente, lette da due fonti:
  - il Dice, da validation/summary.json di ogni fold (lo scrive nnU-Net);
  - il tasso di rilevamento per lesione e i falsi positivi, dal JSON per caso scritto da
    scripts/metriche_per_lesione.py, che va lanciato prima con `--fold all` e la stessa soglia.
Se il JSON manca lo script lo genera chiamando metriche_per_lesione.py.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROGETTO = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get('nnUNet_results', PROGETTO / 'nnUNet' / 'nnUNet_results'))
PER_CASO = PROGETTO / 'reports' / 'lesioni_per_caso'


def cartella(dataset: str, trainer: str) -> tuple[str, Path]:
    nome = next(d.name for d in sorted(RES.iterdir()) if d.name.startswith(f'Dataset{dataset}'))
    c = [d for d in (RES / nome).iterdir() if d.is_dir() and d.name.startswith(trainer + '__')]
    if len(c) != 1:
        raise SystemExit(f'{nome}: {len(c)} cartelle per il trainer {trainer}: {[x.name for x in c]}')
    return nome, c[0]


def dice_fuori_fold(run: Path, classe: str) -> dict:
    """caso -> (Dice, fold). Ogni caso deve comparire in un solo fold."""
    out = {}
    for s in sorted(run.glob('fold_*/validation/summary.json')):
        fold = int(s.parent.parent.name.split('_')[1])
        for m in json.load(open(s))['metric_per_case']:
            caso = Path(m['reference_file']).name.replace('.nii.gz', '')
            if caso in out:
                raise SystemExit(f'{caso} compare in due fold ({out[caso][1]} e {fold}): '
                                 f'lo split non e\' per paziente')
            out[caso] = (float(m['metrics'][classe]['Dice']), fold)
    if not out:
        raise SystemExit(f'nessun summary.json in {run}')
    return out


def lesioni_fuori_fold(dataset: str, nome: str, run: Path, soglia: float, classe: str) -> dict:
    """caso -> (lesioni rilevate, lesioni totali, falsi positivi)."""
    f = PER_CASO / f'{nome}__{run.name}.json'
    j = json.load(open(f)) if f.exists() else None
    if j is None or abs(j.get('soglia_mm3', -1) - soglia) > 1e-9 or str(j.get('classe')) != classe:
        subprocess.run([sys.executable, str(PROGETTO / 'scripts' / 'metriche_per_lesione.py'),
                        dataset, '--trainer', run.name, '--fold', 'all',
                        '--soglia-mm3', str(soglia), '--classe', classe,
                        '--out', str(PROGETTO / 'reports' / f'metriche_per_lesione_{nome}.md')],
                       check=True, stdout=subprocess.DEVNULL)
        j = json.load(open(f))
    return {r['caso']: (sum(l['rilevata'] for l in r['lesioni']), len(r['lesioni']),
                        r['falsi_positivi']) for r in j['casi']}


def wilcoxon(x, y):
    d = np.asarray(y) - np.asarray(x)
    if len(d) < 2 or np.all(d == 0):
        return None
    return stats.wilcoxon(y, x).pvalue


def confronta(dataset: str, base: str, variante: str, soglia: float, classe: str) -> str:
    nome, rb = cartella(dataset, base)
    _, rv = cartella(dataset, variante)
    db, dv = dice_fuori_fold(rb, classe), dice_fuori_fold(rv, classe)
    lb = lesioni_fuori_fold(dataset, nome, rb, soglia, classe)
    lv = lesioni_fuori_fold(dataset, nome, rv, soglia, classe)
    casi = sorted(set(db) & set(dv))
    if set(db) != set(dv):
        print(f'attenzione: i due trainer non coprono gli stessi casi; in comune {len(casi)}')

    x = np.array([db[c][0] for c in casi]); y = np.array([dv[c][0] for c in casi])
    p_dice = wilcoxon(x, y)
    ril_b = sum(lb[c][0] for c in casi); ril_v = sum(lv[c][0] for c in casi)
    tot = sum(lb[c][1] for c in casi)
    fp_b = sum(lb[c][2] for c in casi); fp_v = sum(lv[c][2] for c in casi)
    # rilevamento per paziente, per il test appaiato: quota di lesioni trovate in quel paziente
    qb = np.array([lb[c][0] / lb[c][1] if lb[c][1] else np.nan for c in casi])
    qv = np.array([lv[c][0] / lv[c][1] if lv[c][1] else np.nan for c in casi])
    ok = ~np.isnan(qb)
    p_ril = wilcoxon(qb[ok], qv[ok])

    r = [f'# `{nome}`: {variante} contro {base}', '',
         f'Dieci pazienti in cinque fold: per ciascuno si usa la predizione del fold in cui era '
         f'fuori dal training. Soglia minima {soglia:g} mm3 sulle lesioni.', '',
         f'| | {base} | {variante} | differenza |', '| --- | --- | --- | --- |',
         f'| Dice medio sui {len(casi)} pazienti | {x.mean():.4f} | {y.mean():.4f} | '
         f'**{y.mean() - x.mean():+.4f}** |',
         f'| Dice mediano | {np.median(x):.4f} | {np.median(y):.4f} | '
         f'{np.median(y) - np.median(x):+.4f} |',
         f'| lesioni rilevate su {tot} | {ril_b} ({ril_b / tot:.1%}) | {ril_v} ({ril_v / tot:.1%}) | '
         f'**{ril_v - ril_b:+d}** |',
         f'| falsi positivi (per paziente) | {fp_b} ({fp_b / len(casi):.1f}) | '
         f'{fp_v} ({fp_v / len(casi):.1f}) | {fp_v - fp_b:+d} |', '',
         f'Dice: migliora in **{(y > x).sum()}** pazienti su {len(casi)}, peggiora in '
         f'**{(y < x).sum()}**' + (f', Wilcoxon p = {p_dice:.3g}.' if p_dice is not None else '.'),
         f'Rilevamento: migliora in **{(qv[ok] > qb[ok]).sum()}**, peggiora in '
         f'**{(qv[ok] < qb[ok]).sum()}**' + (f', Wilcoxon p = {p_ril:.3g}.' if p_ril is not None else '.'),
         '', '## Paziente per paziente', '',
         f'| paziente | fold | Dice {base} | Dice {variante} | Δ | lesioni | rilevate prima | '
         f'rilevate dopo | FP prima | FP dopo |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for c in sorted(casi, key=lambda c: dv[c][0] - db[c][0]):
        r.append(f'| {c} | {db[c][1]} | {db[c][0]:.4f} | {dv[c][0]:.4f} | '
                 f'{dv[c][0] - db[c][0]:+.4f} | {lb[c][1]} | {lb[c][0]} | {lv[c][0]} | '
                 f'{lb[c][2]} | {lv[c][2]} |')
    r += ['', 'Ordinati dal paziente che peggiora di piu\' a quello che migliora di piu\'. '
              'Con dieci coppie il p di Wilcoxon non scende sotto 0.002 nemmeno se tutte vanno '
              'nello stesso verso: contano le coppie e le grandezze, il p e\' una cautela.']
    return '\n'.join(r) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('base', help='nome del trainer di riferimento (es. nnUNetTrainerWandb_250ep_arresto50)')
    ap.add_argument('variante', help='nome del trainer da confrontare')
    ap.add_argument('--soglia-mm3', type=float, default=10.0)
    ap.add_argument('--classe', default='1')
    ap.add_argument('--out')
    a = ap.parse_args()
    testo = confronta(a.dataset, a.base, a.variante, a.soglia_mm3, a.classe)
    print(testo)
    if a.out:
        Path(a.out).write_text(testo)


if __name__ == '__main__':
    main()
