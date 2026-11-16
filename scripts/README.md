# Indice degli script

Prima di ogni comando nnU-Net: `source scripts/env.sh`. L'intestazione di ogni script spiega perché esiste e come si lancia; i report in `reports/` sono prodotti da questi script. Le voci marcate *storico* servivano a esperimenti superati e restano per riproducibilità.

## Gli script principali

| Script | Cosa fa |
| --- | --- |
| [pipeline_aincp.py](pipeline_aincp.py) | Dai file ricevuti al dataset nnU-Net dei pazienti AInCP (Dataset506). |
| [postprocessing_aincp.py](postprocessing_aincp.py) | Postprocessing e valutazione del modello nello spazio del paziente. |
| [run_stadio3.sh](run_stadio3.sh) | Stadio 3: addestra sul dataset finale i fold scelti, con ricetta e trainer a scelta. |
| [nnunet_trainers/nnUNetTrainerWandb.py](nnunet_trainers/nnUNetTrainerWandb.py) | Trainer del modello della tesi (Z): classe `nnUNetTrainerWandb_250ep_dropmod30_arresto50`, 250 epoche, modality dropout 0.3, arresto anticipato, tracking W&B opzionale. |
| [confronto_appaiato.py](confronto_appaiato.py) | Confronta due training sugli stessi casi, non sulle medie. |
| [metriche_per_lesione.py](metriche_per_lesione.py) | Rilevamento per singola lesione, non per voxel. |
| [esporta_slicer_506.py](esporta_slicer_506.py) | Esporta le segmentazioni di Dataset506 per 3D Slicer. |

Gli altri trainer (blob loss, training misto, espansione dei canali) sono nella sezione sul training. Gli script principali compaiono anche nel loro gruppo.

## Dataset pubblici (WMH, ATLAS, DBB)

| Script | Descrizione |
| --- | --- |
| [download_dbb.py](download_dbb.py) | Scarica il progetto brainlife "DBB Distorted Brain Benchmark" in data/raw/DBB *(storico)* |
| [explore_datasets.py](explore_datasets.py) | Esplorazione read-only di struttura e header NIfTI *(storico)* |
| [analyze_dbb.py](analyze_dbb.py) | Analisi read-only del terzo dataset (Distorted Brain Benchmark) *(storico)* |
| [compare_datasets.py](compare_datasets.py) | Confronto a tre fra WMH, ATLAS e DBB a partire dai manifest per caso *(storico)* |
| [convert_dbb_to_nnunet.py](convert_dbb_to_nnunet.py) | Converte il Distorted Brain Benchmark in formato nnU-Net (`Dataset503_DBB`) *(storico)* |
| [measure_bias_field.py](measure_bias_field.py) | Fase 0: misura il bias di campo residuo su WMH, ATLAS e DBB *(storico)* |
| [run_synthstrip.py](run_synthstrip.py) | Applica SynthStrip a tutti i casi di DBB e ne misura la qualita' contro le *(storico)* |
| [make_wmh_t1_dataset.py](make_wmh_t1_dataset.py) | Costruisce `Dataset511_WMH_T1`: WMH con il solo canale T1 al posto della FLAIR *(storico)* |
| [make_wmh_two_channel_dataset.py](make_wmh_two_channel_dataset.py) | Costruisce `Dataset512_WMH_T1_FLAIR`: WMH con **due** canali di input *(storico)* |
| [make_atlas_3t_dataset.py](make_atlas_3t_dataset.py) | Costruisce `Dataset521_ATLAS_3T`: ATLAS senza le coorti a 1.5 Tesla *(storico)* |
| [make_skullstripped_datasets.py](make_skullstripped_datasets.py) | Costruisce le versioni skull-stripped di WMH e ATLAS come dataset nnU-Net a se' *(storico)* |
| [make_union_dataset.py](make_union_dataset.py) | Costruisce la vecchia variante D dei dati pubblici (Dataset531, non la ricetta D della tesi): un solo dataset con tutti e tre i sorgenti e 7 classi *(storico)* |
| [make_union_atlas_wmh_mni.py](make_union_atlas_wmh_mni.py) | Costruisce Dataset543_ATLAS_WMH_MNI: gli stadi 1 e 2 fusi in un solo addestramento *(storico)* |
| [make_wmh_mni_dataset.py](make_wmh_mni_dataset.py) | Costruisce `Dataset542_WMH_MNI`: WMH a due canali portato nello spazio MNI di ATLAS *(storico)* |
| [bilancio_classi_union.py](bilancio_classi_union.py) | Conta i voxel di ogni classe in Dataset531_UNION_D, separando i tre sorgenti *(storico)* |
| [analisi_union_d.py](analisi_union_d.py) | Legge il risultato della vecchia variante D (Dataset531) separandolo per sorgente e per classe *(storico)* |
| [oblio_atlas_stadio2.py](oblio_atlas_stadio2.py) | Lo stadio 2 ha fatto dimenticare le cavita' di ATLAS? Rete di 542 sulla validazione di 502 *(storico)* |
| [inferenza_incrociata_15T.py](inferenza_incrociata_15T.py) | Applica il modello addestrato **solo su 3 T** ai casi a 1.5 T che non ha mai visto *(storico)* |
| [run_baselines.sh](run_baselines.sh) | Esegue in sequenza le baseline fold 0 di 3d_fullres su 501, 502 e 503 *(storico)* |
| [run_stadio2_542.sh](run_stadio2_542.sh) | Stadio 2 della Proposta B: la rete espansa da ATLAS (502) impara la FLAIR su WMH in MNI (542) *(storico)* |

## Preprocessing e pipeline AInCP

| Script | Descrizione |
| --- | --- |
| [pipeline_aincp.py](pipeline_aincp.py) | Pipeline dei pazienti AInCP per il modello finale: dai file ricevuti al dataset nnU-Net |
| [preprocess_aincp.py](preprocess_aincp.py) | Preprocessing del lotto AInCP, secondo `docs/pipeline_preprocessing.docx` |
| [make_aincp_dataset.py](make_aincp_dataset.py) | Costruisce Dataset504_AINCP: i dieci pazienti nello spazio T1 nativo, due canali |
| [make_aincp_mni_dataset.py](make_aincp_mni_dataset.py) | Costruisce Dataset505_AINCP_MNI: i dieci pazienti nel template pediatrico, due canali *(storico)* |
| [make_misto_dataset.py](make_misto_dataset.py) | Costruisce Dataset532_MISTO_AINCP: i pazienti AInCP di addestramento insieme ad ATLAS e WMH |
| [dataset_classi_aincp.py](dataset_classi_aincp.py) | Due dataset a piu' classi dai pazienti AInCP, identici a Dataset506 in tutto tranne l'etichetta |
| [confronto_template_aincp.py](confronto_template_aincp.py) | Template adulto o pediatrico per standardizzare i pazienti AInCP: quale deforma meno le lesioni |
| [bias_campo_aincp.py](bias_campo_aincp.py) | Ampiezza del campo di bias che N4 ha tolto ai 17 pazienti AInCP, dentro il cervello |
| [eta_aincp.py](eta_aincp.py) | Eta' dei 17 pazienti AInCP, dai fogli di metadati DICOM dei due file di elenco |
| [synthseg_aincp.py](synthseg_aincp.py) | Ventricoli con SynthSeg sui 17 pazienti AInCP, e la loro prova come postprocessing di Z |
| [lesioni_fuori_maschera.py](lesioni_fuori_maschera.py) | Misura quanta parte delle lesioni annotate cadrebbe FUORI dalla maschera di |
| [conta_lesioni_griglie.py](conta_lesioni_griglie.py) | Perche' le lesioni annotate sono 90 nella griglia della FLAIR e 108 nella griglia T1 nativa |
| [elenco_residui_annotazione.py](elenco_residui_annotazione.py) | Dove sono, nelle annotazioni originali, le componenti da togliere: l'elenco per il clinico |
| [sopravvivenza_1mm_506.py](sopravvivenza_1mm_506.py) | Quali lesioni annotate esistono ancora nel bersaglio di addestramento a 1 mm (17 pazienti, Dataset506) |
| [make_finetuning_sim_dataset.py](make_finetuning_sim_dataset.py) | Costruisce il dataset che simula AInCP: pochi casi di target, mai visti da nessun pretraining *(storico)* |
| [make_smoke_dataset.py](make_smoke_dataset.py) | Costruisce un dataset nnU-Net minuscolo per lo smoke test della pipeline (Dataset599) *(storico)* |

## Training e code GPU

| Script | Descrizione |
| --- | --- |
| [run_stadio3.sh](run_stadio3.sh) | Stadio 3 sul dataset finale dei pazienti (scripts/pipeline_aincp.py dataset --id <ID>), fold scelti |
| [attendi_gpu_e_lancia.sh](attendi_gpu_e_lancia.sh) | Aspetta che una GPU si liberi e lancia la coda di un fold di Dataset504 su di essa |
| [attendi_memoria_e_lancia.sh](attendi_memoria_e_lancia.sh) | Coda per run_stadio3.sh quando le schede sono condivise: un fold alla volta, sulla scheda con piu' |
| [run_aincp_504.sh](run_aincp_504.sh) | Coda degli esperimenti sui dieci pazienti AInCP (Dataset504, spazio T1 nativo, 2 canali) |
| [run_aincp_505.sh](run_aincp_505.sh) | Stadio 3 sui dieci pazienti nel template pediatrico (Dataset505), i cinque fold in fila *(storico)* |
| [run_test_pretraining.sh](run_test_pretraining.sh) | Test del pretraining: gli stessi 10 casi di target, addestrati da zero e con pesi pre-addestrati *(storico)* |
| [run_test_finetuning_lr.sh](run_test_finetuning_lr.sh) | Seguito del test del pretraining: lo stesso fine-tuning con passo di apprendimento ridotto *(storico)* |
| [run_val_best.sh](run_val_best.sh) | Rivaluta i quattro bracci di 541 usando checkpoint_best invece di checkpoint_final *(storico)* |
| [wandb_setup.py](wandb_setup.py) | Autentica Weights & Biases con il segreto locale e crea una piccola run di prova |
| [nnunet_trainers/nnUNetTrainerBlob.py](nnunet_trainers/nnUNetTrainerBlob.py) | Ricetta B: Z con la blob loss, perche' ogni lesione pesi uguale nell'addestramento |
| [nnunet_trainers/nnUNetTrainerEspansioneCanali.py](nnunet_trainers/nnUNetTrainerEspansioneCanali.py) | Trainer che espande il primo strato convoluzionale da N a M canali di ingresso |
| [nnunet_trainers/nnUNetTrainerMisto.py](nnunet_trainers/nnUNetTrainerMisto.py) | Trainer per il training misto (Dataset532): pazienti AInCP e dati pubblici nello stesso addestramento |
| [nnunet_trainers/nnUNetTrainerWandb.py](nnunet_trainers/nnUNetTrainerWandb.py) | Trainer nnU-Net con tracking su Weights & Biases |
| [verifica_espansione_canali.py](verifica_espansione_canali.py) | Verifica su CPU che l'espansione dei canali della proposta B faccia quello che dice |

## Postprocessing e valutazione

| Script | Descrizione |
| --- | --- |
| [postprocessing_aincp.py](postprocessing_aincp.py) | Postprocessing e valutazione del modello AInCP, nello spazio del paziente |
| [predici_test.sh](predici_test.sh) | Il test, una volta sola, con la ricetta e il postprocessing gia' scelti in cross-validation |
| [analisi_ricette_BCF.sh](analisi_ricette_BCF.sh) | Postprocessing scelto sui fuori-fold e confronto con Z per le ricette B, C e F (21-22/10/2026) |
| [confronto_appaiato.py](confronto_appaiato.py) | Confronta due training sugli **stessi casi**, non sulle rispettive medie |
| [confronto_fold.py](confronto_fold.py) | Confronta due trainer sullo stesso dataset unendo i fold: ogni paziente conta una volta |
| [confronto_misto_532.py](confronto_misto_532.py) | Confronto fuori-fold fra il training misto (M, Dataset532) e il modello della tesi (Z, Dataset506) |
| [confronto_505_504.py](confronto_505_504.py) | Stadio 3 nel template (505) contro lo spazio nativo (504), misurati nello stesso posto *(storico)* |
| [porta_505_nativo.py](porta_505_nativo.py) | Riporta le predizioni fuori-fold di Dataset505 (template pediatrico) nello spazio T1 del paziente *(storico)* |
| [metriche_per_lesione.py](metriche_per_lesione.py) | Metriche di rilevamento per singola lesione, non per voxel |
| [metriche_per_lesione_541.py](metriche_per_lesione_541.py) | Rilevamento per lesione dei bracci del test di pretraining (541), sugli stessi 35 casi *(storico)* |
| [summarize_validation.py](summarize_validation.py) | Riassume la validation di cross-validation prodotta da nnU-Net |
| [inspect_failures.py](inspect_failures.py) | Esamina i casi in cui la segmentazione e' fallita, invece di mediarli con gli altri *(storico)* |
| [ensemble_506.py](ensemble_506.py) | Ensemble delle ricette A, D, Z di Dataset506, sulle predizioni fuori-fold (mai sul test) |
| [tabelle_postprocessing_506.py](tabelle_postprocessing_506.py) | Due tabelle per la tesi: un modello prima e dopo il postprocessing, e che cosa toglie il volume minimo |

## Analisi

| Script | Descrizione |
| --- | --- |
| [analisi_curve_training.py](analisi_curve_training.py) | Curve di training di tutti i modelli, e misura quantitativa dell'overfitting |
| [analisi_rilevamento_504.py](analisi_rilevamento_504.py) | Rilevamento su Dataset504 per fenotipo di lesione e per dimensione rispetto alla griglia della rete |
| [analisi_test_pretraining.py](analisi_test_pretraining.py) | Confronta i due bracci del test del pretraining. -> reports/test_pretraining.md *(storico)* |
| [analisi_volume_minimo_506.py](analisi_volume_minimo_506.py) | Che cosa toglie il volume minimo di 50 mm³ dal postprocessing di Z, e che cosa si puo' usare al suo posto |
| [ablazione_canali_506.py](ablazione_canali_506.py) | Ablazione dei canali in inferenza sul modello della tesi (ricetta Z di Dataset506), fuori-fold |
| [dice_per_dimensione_506.py](dice_per_dimensione_506.py) | Dice per singola lesione, per fascia di volume: come si comportano le tre ricette di 506 |
| [fenotipi_508.py](fenotipi_508.py) | Ricetta F (Dataset508): quanto la rete riconosce il tipo di lesione, sui fuori-fold dei 12 |
| [contrasto_fenotipi_aincp.py](contrasto_fenotipi_aincp.py) | Rapporto d'intensita' lesione / cervello per fenotipo, in T1 e in FLAIR, sui 17 pazienti AInCP |

## Figure

| Script | Descrizione |
| --- | --- |
| [figura_3d_506.py](figura_3d_506.py) | Viste 3D del modello della tesi (Z) su pazienti di test: lesioni trovate, mancate, falsi positivi |
| [figura_ablazione_canali_506.py](figura_ablazione_canali_506.py) | Figura dell'ablazione dei canali su Z: che cosa perde il modello senza FLAIR o senza T1 |
| [figura_annotazioni_aincp.py](figura_annotazioni_aincp.py) | Le annotazioni dei 17 pazienti AInCP, una fetta per paziente, e il contrasto per fenotipo |
| [figura_curve_506.py](figura_curve_506.py) | Curve di addestramento dei 15 run di Dataset506: c'e' overfitting, e l'arresto lo ferma? |
| [figura_dati_pubblici.py](figura_dati_pubblici.py) | Due figure per la tesi sui dati pubblici, in inglese |
| [figura_fenotipi_504.py](figura_fenotipi_504.py) | Le due figure dell'analisi per fenotipo su Dataset504 |
| [figura_flair_contro_t1.py](figura_flair_contro_t1.py) | Confronto visivo fra il modello sulla FLAIR e quello sul T1, sugli stessi casi WMH *(storico)* |
| [figura_lesioni_piccole.py](figura_lesioni_piccole.py) | Le componenti piu' piccole delle annotazioni AInCP, viste una per una |
| [figura_modello_unico.py](figura_modello_unico.py) | Che cosa produce il modello unico: lesione e sei tessuti nella stessa uscita |
| [figura_predizioni_504.py](figura_predizioni_504.py) | Le tre varianti di 504 a confronto, paziente per paziente, sulla predizione fuori-fold |
| [figura_registrazione_aincp.py](figura_registrazione_aincp.py) | Controllo della registrazione AInCP verso MNI, e misura dell'ingrandimento che comporta |
| [figura_risultati.py](figura_risultati.py) | Figura riassuntiva dei risultati, per l'incontro con i relatori |
| [figura_roi_fsm030.py](figura_roi_fsm030.py) | Le due ROI_1 di FSM_030 a confronto, per decidere quale usare |
| [figura_segmentazioni_506.py](figura_segmentazioni_506.py) | Come segmenta il modello della tesi (Z): la predizione accanto all'annotazione, sui 12 pazienti |
| [figura_skullstrip_prima_dopo.py](figura_skullstrip_prima_dopo.py) | Figura prima/dopo dello skull stripping su WMH e ATLAS, con la lesione sovrapposta *(storico)* |
| [figura_t1_flair_intro.py](figura_t1_flair_intro.py) | Figura introduttiva della tesi: la stessa fetta in T1 e in FLAIR, con i tre fenotipi di lesione |
| [plot_predictions.py](plot_predictions.py) | Figure di confronto fra predizione e ground truth, sui casi di validation |

## Utilità

| Script | Descrizione |
| --- | --- |
| [env.sh](env.sh) | Variabili d'ambiente di nnU-Net e del progetto (`source scripts/env.sh`) |
| [esporta_slicer_506.py](esporta_slicer_506.py) | Esporta le segmentazioni di Dataset506 in file pronti per 3D Slicer |
| [apri_in_slicer.py](apri_in_slicer.py) | Apre in 3D Slicer un paziente esportato da esporta_slicer_506.py, con le viste gia' impostate |
| [costruisci_notebook_aincp.py](costruisci_notebook_aincp.py) | Costruisce notebooks/aincp_primo_lotto.ipynb |
| [LEGGIMI_slicer.txt](LEGGIMI_slicer.txt) | Istruzioni per aprire in 3D Slicer i pazienti esportati |
| [synthstrip/synthstrip.py](synthstrip/synthstrip.py) | Script di SynthStrip (FreeSurfer; Hoopes et al., 2022), non del progetto: maschera del cervello, usato da run_synthstrip.py |
