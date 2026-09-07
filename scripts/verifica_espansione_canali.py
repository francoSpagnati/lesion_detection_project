#!/usr/bin/env python3
"""Verifica su CPU che l'espansione dei canali della proposta B faccia quello che dice.

Costruisce le due reti dal piano di 502 -- una a un canale, una a due -- carica nella
prima i pesi veri di `Dataset502_ATLAS` e usa
`nnUNetTrainerEspansioneCanali.carica_pesi_espandendo` per portarli sulla seconda.

Le quattro verifiche, in ordine di forza:

1. **Forme.** Il primo strato passa da `[32, 1, 3, 3, 3]` a `[32, 2, 3, 3, 3]` e ogni
   altro parametro non-`seg_layers` viene copiato identico.
2. **Posizione dei canali.** Con `Dataset512_WMH_T1_FLAIR` il canale 0 e' la FLAIR e il
   canale 1 e' il T1: il filtro pre-addestrato, che e' un filtro T1, deve finire nella
   posizione 1. E' l'errore che questo trainer esiste per evitare, quindi va controllato
   e non dedotto.
3. **Equivalenza funzionale della strategia `zero`.** La stessa immagine T1 data alla
   rete a un canale e alla rete a due canali -- con qualunque cosa nel canale FLAIR --
   deve produrre **la stessa uscita**. E' la dimostrazione che l'opzione conservativa
   parte davvero identica al modello di partenza e non puo' peggiorarlo.
4. **Conservazione della scala della strategia `duplica`.** La somma dei filtri sui due
   canali deve essere uguale al filtro di partenza, cosi' le attivazioni non raddoppiano.

Nessuna GPU, nessun dato: si legge solo il checkpoint gia' addestrato.

    source scripts/env.sh && python scripts/verifica_espansione_canali.py
"""
import sys
from pathlib import Path

import torch

PROGETTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROGETTO / 'scripts' / 'nnunet_trainers'))

from nnUNetTrainerEspansioneCanali import (                     # noqa: E402
    carica_pesi_espandendo, espandi_primo_strato, nomi_canali_ordinati)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402
from batchgenerators.utilities.file_and_folder_operations import load_json  # noqa: E402

CHECKPOINT = PROGETTO / ('nnUNet/nnUNet_results/Dataset502_ATLAS/'
                         'nnUNetTrainerWandb__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth')
PIANO = PROGETTO / 'nnUNet/nnUNet_preprocessed/Dataset502_ATLAS/nnUNetPlans.json'

# I canali del dataset di arrivo, nell'ordine in cui li dichiara Dataset512_WMH_T1_FLAIR.
# La FLAIR e' il canale 0: e' il motivo per cui l'appaiamento non puo' essere posizionale.
CANALI_ARRIVO = ['FLAIR', 'T1']

esiti = []


def controlla(descrizione, condizione, dettaglio=''):
    esiti.append(bool(condizione))
    print(f"  [{'ok ' if condizione else 'NO '}] {descrizione}" + (f'  {dettaglio}' if dettaglio else ''))


def costruisci_rete(n_canali_ingresso):
    """Rete del piano di 502 con un numero di canali di ingresso scelto.

    La supervisione profonda e' spenta: qui interessano i pesi e una singola uscita, non
    il ciclo di training.
    """
    piano = load_json(str(PIANO))
    plans_manager = PlansManager(piano)
    configuration_manager = plans_manager.get_configuration('3d_fullres')
    return nnUNetTrainer.build_network_architecture(
        plans_manager, configuration_manager,
        num_input_channels=n_canali_ingresso,
        num_output_channels=2,                                  # sfondo + lesione
        enable_deep_supervision=False)


def chiave_primo_strato(rete):
    """Il primo parametro convoluzionale, cioe' l'unico che dipende dai canali d'ingresso."""
    for chiave, valore in rete.state_dict().items():
        if chiave.endswith('.conv.weight') and valore.dim() == 5:
            return chiave
    raise RuntimeError('Primo strato convoluzionale non trovato.')


def main():
    if not CHECKPOINT.exists():
        print(f'Checkpoint assente: {CHECKPOINT}')
        return 1

    checkpoint = torch.load(str(CHECKPOINT), map_location='cpu', weights_only=False)
    canali_partenza = nomi_canali_ordinati(checkpoint['init_args']['dataset_json'])
    print(f'Modello di partenza: {CHECKPOINT.name}, canali {canali_partenza}')
    print(f'Dataset di arrivo simulato: canali {CANALI_ARRIVO}\n')

    rete1 = costruisci_rete(1).eval()
    rete1.load_state_dict(checkpoint['network_weights'])
    chiave = chiave_primo_strato(rete1)
    peso_pre = rete1.state_dict()[chiave].clone()

    # --- 1 e 2: forme e posizione dei canali ---------------------------------
    print('Strategia "zero"')
    rete2 = costruisci_rete(2).eval()
    riepilogo = carica_pesi_espandendo(
        rete2, str(CHECKPOINT), CANALI_ARRIVO, 'zero', log=lambda *_: None)
    peso_post = rete2.state_dict()[chiave]

    controlla('il primo strato e\' stato espanso a 2 canali',
              tuple(peso_post.shape) == (peso_pre.shape[0], 2) + tuple(peso_pre.shape[2:]),
              f'{tuple(peso_pre.shape)} -> {tuple(peso_post.shape)}')
    # nnU-Net espone lo stesso parametro sotto piu' chiavi: `ConvDropoutNormReLU` tiene
    # sia `conv.weight` sia `all_modules.0.weight`, e il decoder mantiene un riferimento
    # all'encoder (`decoder.encoder....`). Sono quattro chiavi e un solo tensore, e vanno
    # espanse tutte allo stesso modo: se una restasse indietro, `load_state_dict`
    # riceverebbe due valori diversi per lo stesso parametro.
    espansi = [rete2.state_dict()[k] for k, _, _ in riepilogo['espansi']]
    controlla('sono stati espansi solo gli alias del primo strato',
              len(espansi) == 4, f"{len(espansi)} chiavi: {[k for k, _, _ in riepilogo['espansi']]}")
    controlla('tutti gli alias hanno ricevuto lo stesso valore',
              all(torch.equal(e, espansi[0]) for e in espansi))
    controlla('gli altri parametri sono stati copiati',
              riepilogo['copiati'] > 0, f"{riepilogo['copiati']} parametri")
    controlla('i seg_layers non sono stati trasferiti',
              riepilogo['saltati'] > 0, f"{riepilogo['saltati']} tensori")
    controlla('il filtro pre-addestrato e\' finito sul canale T1 (indice 1)',
              torch.equal(peso_post[:, 1], peso_pre[:, 0]))
    controlla('il canale FLAIR (indice 0) parte da zero',
              torch.count_nonzero(peso_post[:, 0]).item() == 0)
    controlla('il filtro T1 NON e\' finito sul canale FLAIR',
              not torch.equal(peso_post[:, 0], peso_pre[:, 0]))

    # --- 3: equivalenza funzionale ------------------------------------------
    #
    # I `seg_layers` non vengono trasferiti da nnU-Net (il numero di classi puo'
    # cambiare), quindi nelle due reti sono inizializzati a caso in modo diverso. Per
    # confrontare le uscite vanno resi uguali a mano: qui il numero di classi e' lo
    # stesso, e cio' che si vuole verificare e' il corpo della rete, non la testa.
    pesi2 = rete2.state_dict()
    for k, v in rete1.state_dict().items():
        if '.seg_layers.' in k:
            pesi2[k] = v
    rete2.load_state_dict(pesi2)

    torch.manual_seed(0)
    t1 = torch.randn(1, 1, 64, 64, 64)
    flair_finta = torch.randn(1, 1, 64, 64, 64) * 5             # deliberatamente fuori scala
    with torch.no_grad():
        uscita1 = rete1(t1)
        uscita2 = rete2(torch.cat([flair_finta, t1], dim=1))
    scarto = (uscita1 - uscita2).abs().max().item()
    controlla('a parita\' di T1 le due reti danno la stessa uscita, qualunque sia la FLAIR',
              scarto < 1e-5, f'scarto massimo {scarto:.2e}')

    # --- 4: conservazione della scala di "duplica" ---------------------------
    print('\nStrategia "duplica"')
    duplicato = espandi_primo_strato(peso_pre, canali_partenza, CANALI_ARRIVO, 'duplica')
    controlla('la somma sui due canali riproduce il filtro di partenza',
              torch.allclose(duplicato.sum(dim=1), peso_pre.sum(dim=1), atol=1e-6))
    controlla('i due canali sono uguali fra loro',
              torch.equal(duplicato[:, 0], duplicato[:, 1]))
    controlla('nessun canale e\' a zero',
              torch.count_nonzero(duplicato[:, 0]).item() > 0)

    # --- i due errori che il trainer deve rifiutare --------------------------
    print('\nControlli di sicurezza')
    try:
        espandi_primo_strato(peso_pre, ['T1'], ['FLAIR', 'T2'], 'zero')
        controlla('rifiuta un dataset senza canali in comune', False)
    except ValueError:
        controlla('rifiuta un dataset senza canali in comune', True)
    try:
        espandi_primo_strato(peso_pre, ['T1', 'FLAIR'], ['FLAIR', 'T1'], 'zero')
        controlla('rifiuta un checkpoint incoerente con i suoi canali dichiarati', False)
    except ValueError:
        controlla('rifiuta un checkpoint incoerente con i suoi canali dichiarati', True)

    passati = sum(esiti)
    print(f'\n{passati}/{len(esiti)} verifiche superate.')
    return 0 if passati == len(esiti) else 1


if __name__ == '__main__':
    raise SystemExit(main())
