#!/usr/bin/env python3
"""Loads the local secret, authenticates W&B and creates a small validation run."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_env(path: Path):
    for line in path.read_text().splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key, value)

load_env(ROOT / '.env')
if not os.environ.get('WANDB_API_KEY'):
    raise RuntimeError('WANDB_API_KEY non trovata in .env')

import wandb
cfg = json.loads((ROOT / 'configs' / 'wandb.json').read_text())
wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
run = wandb.init(project=cfg['project'], entity=cfg['entity'], mode=cfg['mode'], tags=cfg['tags'], config={'project_root': str(ROOT), 'nnunet_experiments': str(ROOT / 'configs' / 'experiments.json')})
wandb.log({'setup/complete': 1})
run.finish()
print(f"W&B pronto: project={cfg['project']}")
