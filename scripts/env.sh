#!/usr/bin/env bash
# Variabili d'ambiente del progetto. Uso:  source scripts/env.sh
#
# nnUNet_extTrainer e' la variabile (nnU-Net >= 2.8) che permette di tenere i
# trainer personalizzati nel repository invece che dentro site-packages.

export nnUNet_raw="/home/ccantini/lesion_detection_project/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/ccantini/lesion_detection_project/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/ccantini/lesion_detection_project/nnUNet/nnUNet_results"
export nnUNet_extTrainer="/home/ccantini/lesion_detection_project/scripts/nnunet_trainers"

# La GPU va scelta a mano: la macchina e' condivisa.
# Controllare prima quale e' libera:
#   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
