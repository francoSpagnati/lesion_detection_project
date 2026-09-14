#!/usr/bin/env bash
# Variabili d'ambiente del progetto. Uso:  source scripts/env.sh
#
# nnUNet_extTrainer e' la variabile (nnU-Net >= 2.8) che permette di tenere i
# trainer personalizzati nel repository invece che dentro site-packages.

export nnUNet_raw="/home/ccantini/lesion_detection_project/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/ccantini/lesion_detection_project/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/ccantini/lesion_detection_project/nnUNet/nnUNet_results"
export nnUNet_extTrainer="/home/ccantini/lesion_detection_project/scripts/nnunet_trainers"

# I comandi nnUNetv2_* stanno nel venv, non in /usr/bin. Da una shell interattiva con il venv
# gia' attivo funzionano lo stesso, e per questo l'assenza non si era mai vista; ma uno script
# lanciato con `setsid nohup` parte senza quell'ambiente e ogni comando fallisce con il codice
# 127 (command not found). Costato una notte di tentativi a vuoto il 13 settembre.
export PATH="/home/ccantini/.venv/bin:$PATH"

# La GPU va scelta a mano: la macchina e' condivisa.
# Controllare prima quale e' libera:
#   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
