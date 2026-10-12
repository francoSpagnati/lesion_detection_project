"""Ricetta B: Z con la blob loss, perche' ogni lesione pesi uguale nell'addestramento.

PERCHE'. La loss di nnU-Net (Dice + cross-entropy) conta voxel per voxel: una lesione pesa quanto il
suo volume. Nel bersaglio a 1 mm dei 12 pazienti di sviluppo le lesioni sotto 50 mm³ sono il 73%
delle lesioni (71 su 97) ma lo 0.07% dei voxel di lesione, quindi mancarle tutte quasi non costa.
Le patch le contengono gia' (un cubo di 128 mm copre quasi tutto il cervello di un bambino: una
lesione piccola sta dentro la patch forzata in mediana nel 100% dei casi), e' la loss che non le
premia. Richiesta del responsabile della tesi, 12/10/2026.

LA BLOB LOSS (Kofler et al., IPMI 2023). Alla loss normale si aggiunge un termine calcolato lesione
per lesione: per ogni componente connessa k dell'etichetta nella patch, un Dice morbido fra la
probabilita' di lesione e la sola componente k, con la predizione azzerata sulle ALTRE lesioni
(cosi' una lesione grande vicina non conta come falso positivo di quella piccola; i falsi positivi
nel tessuto sano invece contano per tutte). Si media sulle lesioni di ogni patch, poi sulle patch
che ne hanno. Una lesione da 3 mm³ pesa quanto una da 50 000 mm³.

    loss = loss di nnU-Net (invariata, con la deep supervision) + PESO_BLOB * blob loss (solo
           all'uscita a risoluzione piena)

PESO_BLOB = 0.5: il rapporto 2 : 1 fra loss globale e blob loss della configurazione indicata
nell'articolo (alpha = 2, beta = 1), ottenuto senza raddoppiare la loss di nnU-Net, che cambierebbe
la scala dei gradienti e quindi in pratica il passo di apprendimento (una seconda variabile).

UNA VARIABILE SOLA. Tutto il resto e' Z (nnUNetTrainerWandb_250ep_dropmod30_arresto50): da zero,
passo 0.01, 250 epoche, modality dropout 0.3, arresto dopo 50 epoche senza miglioramento della
perdita di validazione. La perdita di validazione e' la stessa loss totale: l'arresto guarda la
grandezza che il modello ottimizza.

COME ARRIVANO LE LESIONI. Le componenti si contano nei processi che preparano i batch, dopo le
trasformazioni spaziali (una rotazione puo' staccare o unire due macchie), a 26 vicini come nel
resto del progetto: IstanzeLesione aggiunge alla segmentazione un secondo canale con l'indice
della componente. train_step e validation_step lo staccano prima di passare il bersaglio a
nnU-Net, che vede il bersaglio di sempre.
"""
import cc3d
import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from torch import nn

from nnUNetTrainerWandb import nnUNetTrainerWandb_250ep_dropmod30_arresto50

PESO_BLOB = 0.5


class IstanzeLesione(BasicTransform):
    """Aggiunge alla segmentazione (1, x, y, z) un canale con l'indice della componente connessa."""

    def apply(self, data_dict, **params):
        # int16 per entrambi i canali: nnU-Net tiene la segmentazione in int8, e oltre 127 componenti
        # l'indice tornerebbe negativo
        seg = data_dict['segmentation'].to(torch.int16)
        ist = cc3d.connected_components(seg[0].numpy() > 0, connectivity=26)
        data_dict['segmentation'] = torch.cat([seg, torch.from_numpy(ist.astype(np.int16))[None]])
        return data_dict


def aggiungi_istanze(t):
    """Mette IstanzeLesione subito prima del ricampionamento per la deep supervision (o in coda)."""
    i = next((j for j, x in enumerate(t.transforms) if isinstance(x, DownsampleSegForDSTransform)), len(t.transforms))
    t.transforms.insert(i, IstanzeLesione())
    return t


def blob_dice(logit: torch.Tensor, ist: torch.Tensor) -> torch.Tensor:
    """1 - Dice morbido medio per lesione. logit (B, 2, ...) a risoluzione piena, ist (B, ...)."""
    p = torch.softmax(logit.float(), 1)[:, 1]
    per_patch = []
    for b in range(p.shape[0]):
        ids = torch.unique(ist[b])
        ids = ids[ids > 0]
        if len(ids) == 0:
            continue
        lesione = ist[b] > 0
        perdite = []
        for k in ids:
            g = (ist[b] == k).float()
            pm = p[b] * (~lesione | (ist[b] == k)).float()       # predizione azzerata sulle altre lesioni
            perdite.append(1 - (2 * (pm * g).sum() + 1e-5) / (pm.sum() + g.sum() + 1e-5))
        per_patch.append(torch.stack(perdite).mean())
    if not per_patch:
        return logit.sum() * 0
    return torch.stack(per_patch).mean()


class LossConBlob(nn.Module):
    """La loss di nnU-Net piu' la blob loss; le istanze del batch corrente le passa il trainer."""

    def __init__(self, base: nn.Module, peso: float):
        super().__init__()
        self.base, self.peso = base, peso
        self.istanze = None
        self.ultima_blob = None

    def forward(self, output, target):
        l = self.base(output, target)
        if self.istanze is None:
            return l
        out0 = output[0] if isinstance(output, (list, tuple)) else output
        b = blob_dice(out0, self.istanze.to(out0.device, non_blocking=True))
        self.ultima_blob = float(b.detach())
        return l + self.peso * b


class nnUNetTrainerBlob_250ep_dropmod30_arresto50(nnUNetTrainerWandb_250ep_dropmod30_arresto50):
    """Z + blob loss (ricetta B)."""

    peso_blob = PESO_BLOB

    def _build_loss(self):
        return LossConBlob(super()._build_loss(), self.peso_blob)

    def get_training_transforms(self, *args, **kwargs):
        return aggiungi_istanze(super().get_training_transforms(*args, **kwargs))

    def get_validation_transforms(self, *args, **kwargs):
        # nnU-Net la dichiara statica ma la chiama come self.get_validation_transforms(...)
        return aggiungi_istanze(nnUNetTrainerWandb_250ep_dropmod30_arresto50.get_validation_transforms(*args, **kwargs))

    def _separa(self, batch: dict) -> dict:
        t = batch['target']
        if isinstance(t, list):
            self.loss.istanze = t[0][:, 1].clone()
            batch['target'] = [x[:, :1] for x in t]
        else:
            self.loss.istanze = t[:, 1].clone()
            batch['target'] = t[:, :1]
        return batch

    def train_step(self, batch: dict) -> dict:
        out = super().train_step(self._separa(batch))
        out['blob'] = np.array(self.loss.ultima_blob)
        return out

    def validation_step(self, batch: dict) -> dict:
        out = super().validation_step(self._separa(batch))
        out['blob'] = np.array(self.loss.ultima_blob)
        return out

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        self.print_to_log_file('train_blob', np.round(np.mean([o['blob'] for o in train_outputs]), 4))

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        self.print_to_log_file('val_blob', np.round(np.mean([o['blob'] for o in val_outputs]), 4))
