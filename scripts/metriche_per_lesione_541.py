#!/usr/bin/env python
"""Rilevamento per lesione dei bracci del test di pretraining (541), sugli stessi 35 casi.

    python scripts/metriche_per_lesione_541.py   # -> reports/metriche_per_lesione_541.md

PERCHE'. Il registro riporta per 541 solo il Dice. Qui, con lo stesso criterio di
metriche_per_lesione.py (rilevata = almeno un voxel predetto dentro; connettivita' 26; nessuna
soglia minima), lesioni trovate e falsi positivi di ogni braccio, sia all'ultima epoca
(`validation_final/`, i Dice del registro) sia con `checkpoint_best` (`validation/`, dopo
run_val_best.sh), piu' `521` applicato senza adattamento (data/processed/inferenza_15T/pred_521).
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metriche_per_lesione import analizza  # noqa: E402

PROGETTO = Path(__file__).resolve().parent.parent
RES = PROGETTO / 'nnUNet/nnUNet_results/Dataset541_SIM_AINCP'
GT = Path(os.environ.get('nnUNet_raw', PROGETTO / 'nnUNet/nnUNet_raw')) / 'Dataset541_SIM_AINCP/labelsTr'
BRACCI = [('da zero', 'nnUNetTrainerWandb_250epochs__nnUNetPlans__3d_fullres'),
          ('fine-tuning, passo 1e-2', 'nnUNetTrainerWandb_250epochs__nnUNetPlansFT__3d_fullres'),
          ('fine-tuning, passo 1e-3', 'nnUNetTrainerWandb_250epochs_lr1e3__nnUNetPlans__3d_fullres'),
          ('fine-tuning, passo 1e-4', 'nnUNetTrainerWandb_250epochs_lr1e4__nnUNetPlans__3d_fullres')]


def misura(cartella: Path, casi):
    with ProcessPoolExecutor(max_workers=12) as pool:
        ris = list(pool.map(analizza, [(c, cartella / f'{c}.nii.gz', GT / f'{c}.nii.gz', 1, 0.0) for c in casi]))
    les = [l for r in ris for l in r['lesioni']]
    return sum(l['rilevata'] for l in les), len(les), sum(r['falsi_positivi'] for r in ris)


def main():
    casi = sorted(p.name[:-7] for p in (RES / BRACCI[0][1] / 'fold_0/validation').glob('*.nii.gz'))
    r = ['# Rilevamento per lesione nel test di pretraining (541), 35 casi ATLAS a 1.5 T', '',
         'Prodotto da `scripts/metriche_per_lesione_541.py`, **non modificare a mano**. Rilevata = almeno un '
         'voxel predetto dentro la lesione; falso positivo = componente predetta che non tocca lesioni vere.', '',
         '| braccio | checkpoint | lesioni trovate | falsi positivi | per caso |', '| --- | --- | --- | --- | --- |']
    zs = PROGETTO / 'data/processed/inferenza_15T/pred_521'
    t, n, fp = misura(zs, casi)
    r.append(f'| `521` senza adattamento | — | {t}/{n} | {fp} | {fp / len(casi):.1f} |')
    for nome, run in BRACCI:
        for et, sub in (('ultima epoca', 'validation_final'), ('checkpoint_best', 'validation')):
            t, n, fp = misura(RES / run / 'fold_0' / sub, casi)
            r.append(f'| {nome} | {et} | {t}/{n} | {fp} | {fp / len(casi):.1f} |')
    (PROGETTO / 'reports/metriche_per_lesione_541.md').write_text('\n'.join(r) + '\n')
    print('\n'.join(r))


if __name__ == '__main__':
    main()
