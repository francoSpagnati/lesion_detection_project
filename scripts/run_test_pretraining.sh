#!/usr/bin/env bash
# Test del pretraining: gli stessi 10 casi di target, addestrati da zero e con pesi pre-addestrati.
#
#   CUDA_VISIBLE_DEVICES=0 setsid nohup bash scripts/run_test_pretraining.sh \
#       > reports/logs/coda_541.log 2>&1 < /dev/null &
#
# I due bracci differiscono **solo** per l'inizializzazione dei pesi: stesso dataset, stessa
# partizione, stessa architettura, stesso numero di epoche. Usano due identificativi di piano
# diversi (nnUNetPlans e nnUNetPlansFT) unicamente perche' nnU-Net scrive i risultati in una
# cartella che dipende da quel nome: i due file di piano sono identici e puntano agli stessi dati
# preprocessati, quindi non c'e' nessuna differenza sostanziale fra i bracci.
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRAINER=nnUNetTrainerWandb_250epochs
PESI="$nnUNet_results/Dataset521_ATLAS_3T/nnUNetTrainerWandb__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

if [ ! -f "$PESI" ]; then
    echo "STOP: pesi di partenza non trovati: $PESI"; exit 1
fi

gpu_usata=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
if [ "$gpu_usata" -gt 30000 ]; then
    echo "STOP: GPU $CUDA_VISIBLE_DEVICES occupata (${gpu_usata}MiB)"; exit 1
fi

echo "[$(date +%H:%M)] braccio 1/2: da zero"
nnUNetv2_train 541 3d_fullres 0 --npz -tr "$TRAINER" -p nnUNetPlans \
    >> reports/logs/train_541_zero.log 2>&1
rc=$?; echo "[$(date +%H:%M)] da zero: codice $rc"
[ "$rc" -ne 0 ] && { echo "STOP: braccio 1 fallito"; exit 1; }

echo "[$(date +%H:%M)] braccio 2/2: fine-tuning da Dataset521_ATLAS_3T"
nnUNetv2_train 541 3d_fullres 0 --npz -tr "$TRAINER" -p nnUNetPlansFT \
    -pretrained_weights "$PESI" \
    >> reports/logs/train_541_finetune.log 2>&1
rc=$?; echo "[$(date +%H:%M)] fine-tuning: codice $rc"

echo "[$(date +%H:%M)] coda finita"
