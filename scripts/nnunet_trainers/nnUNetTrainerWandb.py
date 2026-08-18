"""Trainer nnU-Net con tracking su Weights & Biases.

nnU-Net 2.8 **ha gia'** un logger W&B integrato (`MetaLogger` -> `WandbLogger`),
attivato dalla variabile d'ambiente `nnUNet_wandb_enabled=1`. Questo file non lo
reimplementa: lo accende e ne copre i tre buchi che rendono un run difficile da
rileggere a distanza di settimane.

1. Il run W&B nasce senza nome: in progetto compare come `stellar-sunset-42` e non
   si capisce a quale dataset, configurazione e fold appartenga.
2. Il config del logger integrato non contiene la **target spacing effettiva del
   piano**. Senza quella due run non sono confrontabili
   (vedi `docs/experiments-and-tracking.md`), perche' la stessa rete su spacing
   diverse e' un altro esperimento.
3. Non registra canali di input ne' mappa delle label, cioe' esattamente le
   variabili che qui cambiano fra un esperimento e l'altro (T1 contro FLAIR,
   lesione contro 6 tessuti).

La logica di training e' quella del `nnUNetTrainer` di default: stessa rete,
stessa loss, stesso scheduler. Un run con questo trainer e uno con il default
restano confrontabili.

Il file vive nel repository, non in `site-packages`. nnU-Net lo trova tramite
`nnUNet_extTrainer` (vedi `scripts/env.sh`).

Se W&B non e' configurato il training **prosegue senza tracking**: un problema di
logging non deve far perdere ore di GPU.
"""
import json
import os
from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

PROGETTO = Path('/home/ccantini/lesion_detection_project')


def _carica_env():
    """Legge .env senza sovrascrivere quanto e' gia' nell'ambiente."""
    env_file = PROGETTO / '.env'
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            chiave, valore = line.split('=', 1)
            os.environ.setdefault(chiave.strip(), valore.strip())


def _tags_configurate():
    cfg_file = PROGETTO / 'configs' / 'wandb.json'
    if not cfg_file.exists():
        return []
    return list(json.loads(cfg_file.read_text()).get('tags', []))


def _consenti_riscrittura_config():
    """Rende ripartibile un training tracciato (`nnUNetv2_train ... --c`).

    Bug di nnU-Net 2.8: `WandbLogger.update_config` chiama `run.config.update(config)`
    senza `allow_val_change=True`. Alla ripresa il logger riaggancia il run W&B
    esistente (stesso id, preso da `wandb/latest-run`) e `nnUNetTrainer.__init__`
    riscrive la config con il piano; wandb rifiuta di cambiare una chiave gia'
    presente e solleva `ConfigError` **prima** che il training inizi.

    I due valori sono lo stesso piano: differiscono all'ultima cifra della
    serializzazione dei float (`0.9765999913215636` contro `...37`). Qui la
    riscrittura viene autorizzata.

    Costato una volta: 189 epoche su 501 non ripartivano dal checkpoint.
    """
    from nnunetv2.training.logging import nnunet_logger

    if getattr(nnunet_logger.WandbLogger.update_config, '_patch_ripresa', False):
        return

    def update_config(self, config: dict):
        self.run.config.update(config, allow_val_change=True)

    update_config._patch_ripresa = True
    nnunet_logger.WandbLogger.update_config = update_config


def _abilita_wandb():
    """Accende il logger integrato di nnU-Net. Va fatto prima di `nnUNetTrainer.__init__`,
    che e' dove il `MetaLogger` viene costruito.

    Ritorna False se W&B non e' utilizzabile: in quel caso il training gira comunque,
    senza tracking.
    """
    _carica_env()
    try:
        import wandb  # noqa: F401
    except ImportError:
        print('W&B non installato: training senza tracking.')
        os.environ['nnUNet_wandb_enabled'] = '0'
        return False
    if not os.environ.get('WANDB_API_KEY') and os.environ.get('nnUNet_wandb_mode') != 'offline':
        print('WANDB_API_KEY assente (attesa in .env): training senza tracking.')
        os.environ['nnUNet_wandb_enabled'] = '0'
        return False

    cfg_file = PROGETTO / 'configs' / 'wandb.json'
    cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
    os.environ['nnUNet_wandb_enabled'] = '1'
    os.environ.setdefault('nnUNet_wandb_project', cfg.get('project', 'lesion-detection-nnunet'))
    os.environ.setdefault('nnUNet_wandb_mode', cfg.get('mode', 'online'))
    if cfg.get('entity'):
        os.environ.setdefault('WANDB_ENTITY', cfg['entity'])
    _consenti_riscrittura_config()
    return True


class nnUNetTrainerWandb(nnUNetTrainer):
    """`nnUNetTrainer` con il logger W&B integrato acceso e il run reso identificabile."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        self._wandb_attivo = _abilita_wandb()
        if self._wandb_attivo:
            # Nome e tag vanno passati per ambiente: il `WandbLogger` di nnU-Net chiama
            # `wandb.init()` senza `name=` ne' `tags=`, e assegnarli dopo l'init non ha
            # effetto sul run gia' creato. Qui siamo prima di `super().__init__`, cioe'
            # prima che il MetaLogger esista, e i dati servono sono gia' negli argomenti.
            dataset = plans.get('dataset_name', 'dataset')
            os.environ['WANDB_NAME'] = f'{dataset}-{configuration}-fold{fold}-{type(self).__name__}'
            os.environ['WANDB_RUN_GROUP'] = dataset
            os.environ['WANDB_JOB_TYPE'] = configuration
            tag_cfg = ','.join(_tags_configurate())
            os.environ['WANDB_TAGS'] = ','.join(
                t for t in (tag_cfg, dataset, configuration, f'fold{fold}') if t)
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self._wandb_attivo:
            self._arricchisci_config()

    def _run_wandb(self):
        for logger in getattr(self.logger, 'loggers', []):
            run = getattr(logger, 'run', None)
            if run is not None:
                return run
        return None

    def _arricchisci_config(self):
        run = self._run_wandb()
        if run is None:
            self._wandb_attivo = False
            return
        cfg_plan = self.configuration_manager.configuration
        try:
            self.logger.update_config({
                'dataset_name': self.plans_manager.dataset_name,
                'plans_name': self.plans_manager.plans_name,
                # senza questa due run non sono confrontabili
                'target_spacing': list(cfg_plan['spacing']),
                'patch_size': list(cfg_plan['patch_size']),
                'normalization': cfg_plan.get('normalization_schemes'),
                'channel_names': self.dataset_json.get('channel_names'),
                'labels': self.dataset_json.get('labels'),
                'num_training_cases': self.dataset_json.get('numTraining'),
                'torch_version': torch.__version__,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            })
            self.print_to_log_file(f'W&B attivo: {run.url}')
        except Exception as e:                                   # noqa: BLE001
            self.print_to_log_file(f'W&B: run non arricchito ({e}), il training prosegue.')

    def on_train_end(self):
        super().on_train_end()
        if not self._wandb_attivo:
            return
        # Il run NON va chiuso qui: nnU-Net esegue `perform_actual_validation` dopo
        # `run_training()` e vi scrive le metriche finali (`final_val/*`). Chiuderlo
        # ora le farebbe fallire. Ci pensa wandb alla fine del processo.
        try:
            if self._best_ema is not None:
                self.logger.log_summary('best_ema_pseudo_dice', float(self._best_ema))
        except Exception as e:                                   # noqa: BLE001
            self.print_to_log_file(f'W&B: summary non scritto ({e}).')


class nnUNetTrainerWandb_5epochs(nnUNetTrainerWandb):
    """Smoke test: verifica che la catena giri. Non produce metriche interpretabili."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5


class nnUNetTrainerWandb_250epochs(nnUNetTrainerWandb):
    """Primo giro rapido, per scegliere fra `2d` e `3d_fullres` senza pagare il prezzo pieno."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
