"""Trainer che espande il primo strato convoluzionale da N a M canali di ingresso.

E' la **proposta B**: pre-addestrare a un solo canale su tutti i casi solo-T1 che
abbiamo, poi aggiungere la FLAIR e continuare il fine-tuning sui dati che ce l'hanno.
Serve perche' nnU-Net pretende tutti i canali per tutti i casi, quindi un modello a due
canali non puo' vedere ATLAS e DBB, che la FLAIR non ce l'hanno.

Perche' nnU-Net non lo fa da solo
---------------------------------
`nnunetv2.run.load_pretrained_weights.load_pretrained_weights` copia tutti i parametri
tranne i `seg_layers` e verifica con un `assert` che ogni altro parametro abbia forma
identica. Il primo strato passa da `[32, 1, 3, 3, 3]` a `[32, 2, 3, 3, 3]`: l'assert
fallisce e il caricamento si interrompe.

Qui viene sostituita **quella sola funzione**. Tutto il resto -- pianificazione,
preprocessing, augmentation, ciclo di training, validazione -- resta di nnU-Net, e un
run con questo trainer resta confrontabile con gli altri del registro.

I canali si appaiano per NOME, non per indice
---------------------------------------------
E' il punto in cui questo codice puo' produrre in silenzio un esperimento sbagliato, e
il motivo per cui l'appaiamento non e' posizionale. In `Dataset512_WMH_T1_FLAIR` i
canali sono `{'0': 'FLAIR', '1': 'T1'}`: il canale 0 e' la FLAIR. I modelli di
pre-addestramento (502, 511) sono a un canale, e quel canale e' il T1. Copiare il
filtro pre-addestrato nella posizione 0 lo metterebbe sulla FLAIR -- un filtro
addestrato a leggere il T1 applicato a un'immagine con un altro contrasto. Il modello
non fallirebbe: partirebbe semplicemente da pesi senza senso, e il risultato sarebbe
illeggibile.

Il nome dei canali del modello di partenza sta nel checkpoint, sotto
`init_args['dataset_json']['channel_names']` (verificato su `Dataset502_ATLAS`).

Le due strategie di inizializzazione
------------------------------------
`zero`      i canali gia' presenti ricevono il loro filtro pre-addestrato, i canali
            nuovi partono da zero. Il modello all'inizio si comporta **esattamente**
            come quello a un canale -- proprieta' verificata da
            `scripts/verifica_espansione_canali.py` -- e impara a usare la FLAIR strada
            facendo. E' l'opzione conservativa: non puo' peggiorare il punto di partenza.
`duplica`   lo stesso filtro va su tutti i canali, diviso per il loro numero, cosi' la
            scala delle attivazioni resta invariata. Il modello parte trattando le due
            sequenze come intercambiabili e poi le differenzia: sfrutta subito la FLAIR
            ma da un punto meno affidabile.

Sono due esperimenti, non una scelta da fare a priori, e hanno due classi distinte
apposta: nnU-Net ricava dal nome del trainer la cartella dei risultati, quindi una
strategia scelta per variabile d'ambiente sovrascriverebbe l'altra.

Prerequisito da non dimenticare
-------------------------------
Il trasferimento dei pesi funziona solo fra architetture identiche a meno dei canali di
ingresso, e nnU-Net ricava l'architettura dal piano. Il dataset di partenza e quello di
arrivo devono quindi usare lo **stesso piano**, imposto con
`nnUNetv2_move_plans_between_datasets`. Con i piani di default non funziona: 502 lavora
a 1 mm isotropo con patch 128^3, mentre 512 lavora a 3.0 x 0.98 x 1.2 con patch
64 x 224 x 128, e le forme non combaciano gia' al secondo stadio.

Uso
---
    source scripts/env.sh
    nnUNetv2_train 512 3d_fullres 0 \\
        -tr nnUNetTrainerEspansioneCanali_zero \\
        -pretrained_weights .../Dataset5XX_.../fold_0/checkpoint_final.pth
"""
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from nnUNetTrainerWandb import nnUNetTrainerWandb


# Le convoluzioni finali che producono la mappa di segmentazione. nnU-Net non le
# trasferisce mai, perche' il numero di classi puo' cambiare fra i due compiti.
STRATI_SEGMENTAZIONE = ('.seg_layers.',)


def nomi_canali_ordinati(dataset_json: dict) -> list:
    """`{'0': 'FLAIR', '1': 'T1'}` -> `['FLAIR', 'T1']`.

    Le chiavi di `channel_names` sono stringhe di indici e un dizionario JSON non
    garantisce l'ordine: vanno ordinate numericamente, non alfabeticamente, altrimenti
    con dieci o piu' canali `'10'` finirebbe prima di `'2'`.
    """
    canali = dataset_json.get('channel_names') or dataset_json.get('modality') or {}
    return [canali[k] for k in sorted(canali, key=int)]


def espandi_primo_strato(peso_pre: torch.Tensor, canali_pre: list, canali_nuovi: list,
                         strategia: str = 'zero') -> torch.Tensor:
    """Ricostruisce il peso del primo strato convoluzionale per un nuovo insieme di canali.

    `peso_pre` ha forma `[uscite, len(canali_pre), k, k, k]`; il risultato ha forma
    `[uscite, len(canali_nuovi), k, k, k]`. L'appaiamento e' per nome (vedi il docstring
    del modulo).
    """
    if peso_pre.shape[1] != len(canali_pre):
        raise ValueError(
            f'Il peso pre-addestrato ha {peso_pre.shape[1]} canali di ingresso ma il suo '
            f'dataset ne dichiara {len(canali_pre)} ({canali_pre}): checkpoint e dataset.json '
            'non corrispondono.')

    nuovo = torch.zeros(
        (peso_pre.shape[0], len(canali_nuovi)) + tuple(peso_pre.shape[2:]),
        dtype=peso_pre.dtype, device=peso_pre.device)

    if strategia == 'duplica':
        # Il filtro medio del modello di partenza su ogni canale, diviso per il loro
        # numero: la somma sui canali resta quella di prima, quindi le attivazioni
        # partono con la stessa scala invece di raddoppiare.
        medio = peso_pre.mean(dim=1)
        for j in range(len(canali_nuovi)):
            nuovo[:, j] = medio / len(canali_nuovi)
        return nuovo

    if strategia != 'zero':
        raise ValueError(f"Strategia sconosciuta: {strategia!r}. Attese 'zero' o 'duplica'.")

    appaiati = 0
    for j, nome in enumerate(canali_nuovi):
        if nome in canali_pre:
            nuovo[:, j] = peso_pre[:, canali_pre.index(nome)]
            appaiati += 1
        # altrimenti resta a zero: e' un canale che il modello di partenza non aveva

    if appaiati == 0:
        # Nessun canale in comune significa che si sta mettendo un filtro addestrato su
        # una sequenza sopra una sequenza diversa. Meglio fermarsi qui che scoprirlo
        # dopo sette ore di GPU leggendo un Dice senza senso.
        raise ValueError(
            f'Nessun canale in comune fra il modello di partenza ({canali_pre}) e questo '
            f'dataset ({canali_nuovi}). Con la strategia "zero" i pesi pre-addestrati '
            'andrebbero tutti persi.')
    return nuovo


def carica_pesi_espandendo(network, fname: str, canali_nuovi: list, strategia: str,
                           log=print) -> dict:
    """Sostituto di `load_pretrained_weights` che tollera un numero di canali diverso.

    Fa esattamente quello che fa la versione di nnU-Net -- copia ogni parametro
    omonimo e di forma uguale, salta i `seg_layers` -- tranne che sui parametri la cui
    unica differenza e' il numero di canali di ingresso, che vengono espansi.

    Ritorna un riepilogo di cosa e' stato fatto, cosi' il chiamante puo' scriverlo nel
    log del training invece di doverlo dedurre.

    Nota su cui si inciampa una volta sola: **il primo strato compare sotto quattro
    chiavi diverse**, non una. `ConvDropoutNormReLU` tiene sia `...convs.0.conv.weight`
    sia `...convs.0.all_modules.0.weight`, che sono lo stesso tensore, e il decoder
    mantiene un riferimento all'encoder, quindi entrambe ricompaiono con il prefisso
    `decoder.`. Il ciclo qui sotto le tratta tutte allo stesso modo perche' scorre le
    chiavi del modello: espanderne solo una lascerebbe `load_state_dict` con due valori
    in conflitto per lo stesso parametro.
    """
    checkpoint = torch.load(fname, map_location='cpu', weights_only=False)
    pesi_pre = checkpoint['network_weights']
    canali_pre = nomi_canali_ordinati(checkpoint.get('init_args', {}).get('dataset_json', {}))
    if not canali_pre:
        raise ValueError(
            f'{fname} non dichiara i nomi dei canali. Senza quelli l\'appaiamento sarebbe '
            'posizionale, che e\' esattamente l\'errore che questo trainer evita.')

    modulo = network
    for attributo in ('module', '_orig_mod'):        # DDP e torch.compile avvolgono la rete
        modulo = getattr(modulo, attributo, modulo)
    pesi_modello = modulo.state_dict()

    espansi, copiati, saltati = [], 0, []
    for chiave, atteso in pesi_modello.items():
        if any(s in chiave for s in STRATI_SEGMENTAZIONE):
            continue
        if chiave not in pesi_pre:
            raise ValueError(
                f'Il parametro {chiave} manca nel modello di partenza: le due architetture '
                'non sono la stessa. Serve lo stesso piano su entrambi i dataset '
                '(nnUNetv2_move_plans_between_datasets).')
        pre = pesi_pre[chiave]
        if pre.shape == atteso.shape:
            pesi_modello[chiave] = pre
            copiati += 1
            continue
        # Unica differenza ammessa: il numero di canali di ingresso del primo strato.
        if pre.shape[0] == atteso.shape[0] and pre.shape[2:] == atteso.shape[2:] \
                and atteso.shape[1] == len(canali_nuovi):
            pesi_modello[chiave] = espandi_primo_strato(pre, canali_pre, canali_nuovi, strategia)
            espansi.append((chiave, tuple(pre.shape), tuple(atteso.shape)))
            continue
        raise ValueError(
            f'Il parametro {chiave} ha forma {tuple(pre.shape)} nel modello di partenza e '
            f'{tuple(atteso.shape)} qui, e la differenza non e\' solo nei canali di ingresso. '
            'Le due architetture non sono compatibili.')

    for chiave in pesi_modello:
        if any(s in chiave for s in STRATI_SEGMENTAZIONE):
            saltati.append(chiave)

    modulo.load_state_dict(pesi_modello)

    log(f'Pesi caricati da {fname}')
    log(f'  canali del modello di partenza: {canali_pre}')
    log(f'  canali di questo dataset:       {canali_nuovi}')
    log(f'  strategia di espansione:        {strategia}')
    for chiave, forma_pre, forma_post in espansi:
        log(f'  espanso {chiave}: {forma_pre} -> {forma_post}')
    log(f'  parametri copiati identici: {copiati}')
    log(f'  seg_layers non trasferiti:  {len(saltati)}')
    if not espansi:
        log('  ATTENZIONE: nessuno strato espanso, i canali erano gia\' compatibili. '
            'Questo trainer non serviva.')

    return {'canali_pre': canali_pre, 'canali_nuovi': canali_nuovi, 'strategia': strategia,
            'espansi': espansi, 'copiati': copiati, 'saltati': len(saltati)}


class nnUNetTrainerEspansioneCanali(nnUNetTrainerWandb):
    """Base: non usarla direttamente, usa una delle due sottoclassi con la strategia nel nome.

    L'aggancio e' in `__init__` e non a livello di modulo perche' cosi' la sostituzione
    vale **solo** quando si usa davvero questo trainer. `maybe_load_checkpoint` risolve
    `load_pretrained_weights` come nome globale del modulo `run_training` e viene chiamata
    dopo che il trainer e' stato costruito, quindi sostituirlo qui arriva in tempo.
    """

    strategia = 'zero'

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._installa_caricamento_espandibile()

    def _installa_caricamento_espandibile(self):
        from nnunetv2.run import run_training

        canali_nuovi = nomi_canali_ordinati(self.dataset_json)
        strategia = self.strategia

        def load_pretrained_weights(network, fname, verbose=False):
            riepilogo = carica_pesi_espandendo(
                network, fname, canali_nuovi, strategia, log=self.print_to_log_file)
            # Finisce nel config del run W&B: senza, a distanza di settimane non si
            # distingue quale dei due esperimenti di inizializzazione fosse.
            if getattr(self, '_wandb_attivo', False):
                try:
                    self.logger.update_config({
                        'pretrained_from': fname,
                        'espansione_strategia': strategia,
                        'espansione_canali_pre': riepilogo['canali_pre'],
                        'espansione_canali_post': riepilogo['canali_nuovi'],
                    })
                except Exception as e:                           # noqa: BLE001
                    self.print_to_log_file(f'W&B: espansione non registrata ({e}).')

        run_training.load_pretrained_weights = load_pretrained_weights


class nnUNetTrainerEspansioneCanali_zero(nnUNetTrainerEspansioneCanali):
    """Opzione conservativa: il canale nuovo parte da zero, il modello parte identico."""

    strategia = 'zero'


class nnUNetTrainerEspansioneCanali_duplica(nnUNetTrainerEspansioneCanali):
    """Il filtro va su entrambi i canali, dimezzato: usa subito la FLAIR."""

    strategia = 'duplica'


# --- varianti con passo di apprendimento ridotto -----------------------------
#
# Il test 541 ha misurato che ripartire dal passo di default (0.01) dopo un
# pre-addestramento cancella le feature invece di raffinarle. Li' i casi erano dieci;
# allo stadio 2 della proposta B sono 79, quindi il rischio e' minore ma non nullo, e
# il confronto va fatto invece che assunto.

class nnUNetTrainerEspansioneCanali_zero_lr1e3(nnUNetTrainerEspansioneCanali_zero):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3


class nnUNetTrainerEspansioneCanali_duplica_lr1e3(nnUNetTrainerEspansioneCanali_duplica):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3



# --- la ricetta misurata per i pochi casi (stadio 3, i pazienti AInCP) --------
#
# Il test 541 (docs/modelli-ed-esperimenti.md) ha misurato sui dieci casi simulati che il
# fine-tuning rende solo con passo 1e-4 -- cento volte sotto il default -- e che il minimo
# della perdita di validazione cade fra l'epoca 69 e la 128 su 250. Queste due classi sono
# quella ricetta applicata al trainer di espansione: 250 epoche, passo 1e-4. Il riferimento
# da zero con lo stesso budget e' nnUNetTrainerWandb_250epochs, sullo stesso piano.

class nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4(nnUNetTrainerEspansioneCanali_zero):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.initial_lr = 1e-4


class nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4(nnUNetTrainerEspansioneCanali_duplica):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.initial_lr = 1e-4


assert issubclass(nnUNetTrainerEspansioneCanali, nnUNetTrainer)
