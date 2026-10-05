"""Trainer per il training misto (Dataset532): pazienti AInCP e dati pubblici nello stesso addestramento.

E' la ricetta Z (nnUNetTrainerWandb_250ep_dropmod30_arresto50: da zero, passo 1e-2, 250 epoche al
massimo, arresto dopo 50 epoche senza miglioramento della perdita di validazione, modality dropout
0.3) con **una sola differenza**: da quale caso si estrae ogni patch.

PERCHE' SERVE UNA REGOLA. nnU-Net estrae i casi in modo uniforme. In 532 i pazienti sono 12 su
647: la rete vedrebbe un paziente in meno di una patch su cinquanta, e su 250 iterazioni per epoca
quasi solo ATLAS, che ha la FLAIR vuota. Imparerebbe soprattutto a lavorare con la sola T1, cioe'
il contrario di cio' che serve (la FLAIR e' la sequenza migliore: 0.774 contro 0.626 di Dice su
WMH, reports dell'ablazione 511). Qui ogni patch viene da una sorgente scelta con le quote sotto,
e dentro la sorgente da un caso uniforme. Con 1/2 ai pazienti ogni paziente esce in media ~10
volte per epoca invece di ~0.4 (in Z, dove sono tutto il dataset, ~42).

PERCHE' 250 EPOCHE E NON DI PIU'. Ogni epoca vede la meta' dei pazienti di Z, e verrebbe da
allungare. Ma il passo di apprendimento di nnU-Net scende con un polinomio legato al numero
massimo di epoche: con 500 la rete avrebbe un passo diverso da Z a ogni epoca, cioe' una seconda
variabile. Si tiene lo stesso programma di Z e si lascia decidere l'arresto, che in Z e' scattato
fra l'epoca 76 e la 198; se qui i fold arrivassero a 250 senza arresto, la durata andrebbe rivista.

LA VALIDAZIONE resta sui soli pazienti (splits di 532 = validazione di 506): la perdita su cui si
decide l'arresto e il checkpoint finale e' quella dei pazienti, non dei casi pubblici.

Il modality dropout e' quello corretto il 5/10/2026 (nnUNetTrainerWandb.DropoutModalita): sui casi
ATLAS, che hanno solo la T1, non azzera mai nulla.
"""
from collections import Counter

import numpy as np
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

from nnUNetTrainerWandb import nnUNetTrainerWandb_250ep_dropmod30_arresto50


class nnUNetTrainerMisto_250ep_dropmod30_arresto50(nnUNetTrainerWandb_250ep_dropmod30_arresto50):
    """Z sul dataset misto, con le patch estratte per sorgente: 1/2 pazienti, 1/4 WMH, 1/4 ATLAS.

    WMH e ATLAS a pari quota anche se ATLAS ha sette volte i casi: WMH e' l'unica sorgente
    pubblica con la FLAIR e con lesioni piccole della bianca, cioe' la fascia in cui Z sbaglia."""

    # prefisso del nome del caso -> frazione delle patch di addestramento
    quote_sorgenti = {'FSM_': 0.5, 'wmh_': 0.25, 'atlas_': 0.25}

    def probabilita_di_campionamento(self, identificativi) -> np.ndarray:
        sorgente = []
        for c in identificativi:
            s = [p for p in self.quote_sorgenti if c.startswith(p)]
            assert len(s) == 1, f'{c}: sorgente non riconosciuta'
            sorgente.append(s[0])
        n = Counter(sorgente)
        assert set(n) == set(self.quote_sorgenti), f'sorgenti mancanti in addestramento: {n}'
        p = np.array([self.quote_sorgenti[s] / n[s] for s in sorgente])
        self.print_to_log_file('campionamento per sorgente: ' + ', '.join(
            f'{s} {n[s]} casi, quota {q:.2f} ({q / n[s]:.4f} per caso)' for s, q in self.quote_sorgenti.items()))
        if self._wandb_attivo:
            try:
                self.logger.update_config({'quote_sorgenti': self.quote_sorgenti, 'casi_per_sorgente': dict(n)})
            except Exception as e:                           # noqa: BLE001
                self.print_to_log_file(f'W&B: config non aggiornato ({e}).')
        return p / p.sum()

    def get_dataloaders(self):
        # Copia di nnUNetTrainer.get_dataloaders (nnU-Net 2.8.1) con una sola modifica: il data
        # loader di addestramento riceve le probabilita' di campionamento, che nnU-Net supporta ma
        # passa sempre a None. Vanno impostate prima che partano i processi di augmentation, che
        # ricevono una copia del loader.
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size, initial_patch_size,
                                 self.configuration_manager.patch_size, self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_tr.sampling_probabilities = self.probabilita_di_campionamento(list(dl_tr.indices))
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size, self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val, transform=None,
                                                      num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda', wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
