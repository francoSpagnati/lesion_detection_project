#!/usr/bin/env python3
"""Costruisce un dataset nnU-Net minuscolo per lo smoke test della pipeline.

Serve a verificare che la catena giri (planning -> preprocessing -> training ->
logging su W&B) senza pagare il costo di un dataset intero. **Non produce
metriche interpretabili**: pochi casi, poche epoche, split arbitrario.

Il dataset di partenza deve essere gia' convertito in formato nnU-Net. Immagini e
label vengono esposte come link simbolici: il dataset di smoke occupa qualche KB.

Esempio:

    python scripts/make_smoke_dataset.py --sorgente Dataset502_ATLAS --n 16
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'processed' / 'nnUNet_raw'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sorgente', default='Dataset502_ATLAS')
    ap.add_argument('--id', type=int, default=599)
    ap.add_argument('--n', type=int, default=16, help='numero di casi da copiare')
    ap.add_argument('--nome', default=None, help='nome del dataset di smoke')
    args = ap.parse_args()

    src = RAW / args.sorgente
    if not src.exists():
        raise SystemExit(f'Dataset sorgente non trovato: {src}')

    src_json = json.loads((src / 'dataset.json').read_text())
    canali = src_json['channel_names']
    if not all(v.upper() in ('T1', 'T1W') for v in canali.values()):
        raise SystemExit(
            f'Lo smoke test e\' previsto solo su T1. Canali della sorgente: {canali}')

    nome = args.nome or f"Dataset{args.id}_SMOKE_{args.sorgente.split('_', 1)[1]}"
    dst = RAW / nome
    if dst.exists():
        shutil.rmtree(dst)
    (dst / 'imagesTr').mkdir(parents=True)
    (dst / 'labelsTr').mkdir(parents=True)

    ending = src_json['file_ending']
    casi = sorted(p.name[: -len(ending)] for p in (src / 'labelsTr').glob(f'*{ending}'))
    # passo costante invece dei primi N: evita di prendere un solo sito/coorte
    passo = max(1, len(casi) // args.n)
    scelti = casi[::passo][: args.n]

    for caso in scelti:
        for ch in canali:
            f = f'{caso}_{int(ch):04d}{ending}'
            (dst / 'imagesTr' / f).symlink_to((src / 'imagesTr' / f).resolve())
        (dst / 'labelsTr' / f'{caso}{ending}').symlink_to((src / 'labelsTr' / f'{caso}{ending}').resolve())

    (dst / 'dataset.json').write_text(json.dumps({
        'name': nome,
        'description': (f'Smoke test della pipeline. {len(scelti)} casi presi da {args.sorgente} '
                        f'con passo {passo}. Non usare per produrre metriche.'),
        'channel_names': canali,
        'labels': src_json['labels'],
        'numTraining': len(scelti),
        'file_ending': ending,
        'smoke_test': {'sorgente': args.sorgente, 'passo': passo, 'casi': scelti},
    }, indent=2) + '\n')

    link = ROOT / 'nnUNet' / 'nnUNet_raw' / nome
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(dst)

    print(f'{nome}: {len(scelti)} casi, canali {canali}, label {src_json["labels"]}')
    print(f'  {dst}')
    print(f'  esposto come {link}')


if __name__ == '__main__':
    main()
