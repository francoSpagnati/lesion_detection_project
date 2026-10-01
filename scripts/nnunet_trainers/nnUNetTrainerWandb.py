"""Trainer nnU-Net con tracking su Weights & Biases.

nnU-Net 2.8 **ha gia'** un logger W&B integrato (`MetaLogger` -> `WandbLogger`),
attivato dalla variabile d'ambiente `nnUNet_wandb_enabled=1`. Questo file non lo
reimplementa: lo accende e ne copre i tre buchi che rendono un run difficile da
rileggere a distanza di settimane.

1. Il run W&B nasce senza nome: in progetto compare come `stellar-sunset-42` e non
   si capisce a quale dataset, configurazione e fold appartenga.
2. Il config del logger integrato non contiene la **target spacing effettiva del
   piano**. Senza quella due run non sono confrontabili
   (vedi `docs/modelli-ed-esperimenti.md`), perche' la stessa rete su spacing
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
from batchgenerators.utilities.file_and_folder_operations import isfile, join

import numpy as np
from batchgeneratorsv2.transforms.base.basic_transform import ImageOnlyTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class DropoutModalita(ImageOnlyTransform):
    """Azzera a caso UNA delle sequenze di un esempio di addestramento (modality dropout).

    Serve al requisito del responsabile: un modello a due canali che si possa usare anche con una
    sola sequenza. Con probabilita' `p` un esempio perde un canale scelto a caso, mai tutti; il
    canale vale 0, che e' esattamente cio' che la rete riceve in inferenza quando la sequenza manca
    (un'immagine nulla resta nulla dopo la normalizzazione z-score di nnU-Net).

    Eredita da ImageOnlyTransform e non da BasicTransform: la seconda, se la trasformazione non
    gestisce la segmentazione, la sostituisce con None."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def get_parameters(self, **data_dict) -> dict:
        c = data_dict['image'].shape[0]
        if c < 2 or np.random.rand() >= self.p:
            return {'canale': None}
        return {'canale': int(np.random.randint(c))}

    def _apply_to_image(self, img: torch.Tensor, canale=None, **params) -> torch.Tensor:
        if canale is None:
            return img
        img = img.clone()
        img[canale] = 0
        return img

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

    # --- arresto sulla perdita di validazione ---------------------------------
    #
    # nnU-Net non ha alcun arresto anticipato, e il suo `checkpoint_best` e' scelto sulla
    # pseudo-Dice mediata esponenzialmente, che sui pochi casi continua a salire anche quando
    # il modello sta peggiorando sui dati non visti (misurato sul test 541: rivalutare con quel
    # checkpoint da' 0.4491 contro 0.4507 dell'ultima epoca, cioe' nulla). La grandezza che
    # traccia l'overfitting e' la perdita di validazione, e su 541 il suo minimo cadeva fra
    # l'epoca 69 e la 128 su 250.
    #
    # Qui, se `pazienza_val` e' impostata:
    #   - a ogni miglioramento della perdita di validazione si salva `checkpoint_val_best.pth`;
    #   - dopo `pazienza_val` epoche senza miglioramento il training si ferma;
    #   - alla fine i pesi dell'ultima epoca restano in `checkpoint_ultima_epoca.pth`, e
    #     `checkpoint_final.pth` -- quello su cui nnU-Net esegue la validazione vera -- riceve
    #     i pesi dell'epoca migliore. Cosi' le due versioni restano confrontabili.
    # Con `pazienza_val = None` (default) il comportamento e' identico a nnU-Net, e i training
    # gia' fatti restano riproducibili con la stessa classe.
    #
    # Il contatore riparte da zero se il training viene ripreso con `--c`: non vale la pena
    # di serializzarlo per run da 250 epoche.

    # probabilita' di modality dropout per esempio (None = spento); vedi DropoutModalita
    p_dropout_modalita = None

    def get_training_transforms(self, *args, **kwargs):
        # nnU-Net la dichiara statica ma la chiama come self.get_training_transforms(...): un metodo
        # d'istanza qui la sostituisce e puo' leggere l'attributo della classe. Il dropout va in
        # coda, dopo le trasformazioni d'intensita', cosi' il canale azzerato resta esattamente 0.
        t = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        if self.p_dropout_modalita:
            t.transforms.append(DropoutModalita(self.p_dropout_modalita))
        return t

    pazienza_val = None
    _best_val_loss = None
    _epoca_best_val = None
    _fermato_in_anticipo = False

    def on_epoch_end(self):
        super().on_epoch_end()            # incrementa current_epoch: l'epoca appena chiusa e' -1
        if self.pazienza_val is None:
            return
        epoca = self.current_epoch - 1
        val = float(self.logger.get_value('val_losses', step=-1))
        if self._best_val_loss is None or val < self._best_val_loss:
            self._best_val_loss, self._epoca_best_val = val, epoca
            self._salva_con_epoca_giusta('checkpoint_val_best.pth')
            self.print_to_log_file(f'perdita di validazione migliore: {val:.4f} (epoca {epoca})')
        elif epoca - self._epoca_best_val >= self.pazienza_val:
            self.print_to_log_file(
                f'ARRESTO: nessun miglioramento della perdita di validazione da '
                f'{self.pazienza_val} epoche (migliore {self._best_val_loss:.4f} '
                f'all\'epoca {self._epoca_best_val}).')
            self._fermato_in_anticipo = True

    def run_training(self):
        # Copia del ciclo di nnUNetTrainer.run_training con la sola aggiunta dell'uscita
        # anticipata: `range(current_epoch, num_epochs)` e' fissato all'ingresso e non si puo'
        # interrompere dall'esterno.
        self.on_train_start()
        for _ in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = [self.train_step(next(self.dataloader_train))
                             for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [self.validation_step(next(self.dataloader_val))
                               for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
            if self._fermato_in_anticipo:
                break
        self.on_train_end()

    def _salva_con_epoca_giusta(self, nome: str):
        # save_checkpoint scrive current_epoch + 1, ed e' pensato per essere chiamato prima
        # dell'incremento di on_epoch_end. Qui siamo dopo: senza questo passo il checkpoint
        # dichiarerebbe un'epoca in piu' (lo stesso "dirty hack" di nnUNetTrainer.on_train_end).
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, nome))
        self.current_epoch += 1

    def _ripristina_pesi_migliori(self):
        f = join(self.output_folder, 'checkpoint_val_best.pth')
        if self.pazienza_val is None or not isfile(f):
            return
        self._salva_con_epoca_giusta('checkpoint_ultima_epoca.pth')
        ck = torch.load(f, map_location=self.device, weights_only=False)
        pesi = ck['network_weights']
        rete = self.network.module if self.is_ddp else self.network
        if hasattr(rete, '_orig_mod'):                     # rete compilata con torch.compile
            rete = rete._orig_mod
        rete.load_state_dict(pesi)
        self.print_to_log_file(
            f'checkpoint_final riceve i pesi dell\'epoca {self._epoca_best_val} '
            f'(perdita di validazione {self._best_val_loss:.4f}); l\'ultima epoca '
            f'({self.current_epoch - 1}) resta in checkpoint_ultima_epoca.pth')
        if self._wandb_attivo:
            try:
                self.logger.log_summary('epoca_best_val_loss', int(self._epoca_best_val))
                self.logger.log_summary('best_val_loss', float(self._best_val_loss))
                self.logger.log_summary('fermato_in_anticipo', bool(self._fermato_in_anticipo))
            except Exception as e:                           # noqa: BLE001
                self.print_to_log_file(f'W&B: summary non scritto ({e}).')

    def on_train_end(self):
        self._ripristina_pesi_migliori()
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


class nnUNetTrainerWandb_250ep_arresto50(nnUNetTrainerWandb_250epochs):
    """Da zero, 250 epoche, arresto dopo 50 epoche senza miglioramento della perdita di
    validazione. E' il riferimento T0 sui pazienti AInCP, con lo stesso criterio di arresto
    dei trainer di espansione cosi' che l'unica differenza sia il pretraining."""

    pazienza_val = 50


class nnUNetTrainerWandb_250ep_dropmod30_arresto50(nnUNetTrainerWandb_250ep_arresto50):
    """Il riferimento da zero per il dataset finale: come T0, con il modality dropout acceso come
    nelle due ricette pre-addestrate, cosi' che l'unica differenza resti il pretraining."""

    p_dropout_modalita = 0.3


# --- varianti per il fine-tuning su pochi casi -------------------------------
#
# Il test del pretraining (reports/test_pretraining.md) ha mostrato che partire da pesi
# pre-addestrati e poi rimettere il passo di apprendimento a 0.01 -- il valore di un
# addestramento completo -- su dieci soli casi **cancella** le feature pre-addestrate:
# il modello messo a punto rende meno del modello di partenza lasciato intatto.
#
# Queste varianti abbassano il passo iniziale. Lo scheduler resta quello polinomiale di
# nnU-Net, quindi il decadimento e' lo stesso: cambia solo il punto di partenza.

class nnUNetTrainerWandb_250epochs_lr1e3(nnUNetTrainerWandb_250epochs):
    """Fine-tuning prudente: passo iniziale dieci volte piu' basso del default."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3


class nnUNetTrainerWandb_250epochs_lr1e4(nnUNetTrainerWandb_250epochs):
    """Fine-tuning molto prudente: cento volte piu' basso del default."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4
