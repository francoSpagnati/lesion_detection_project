#!/usr/bin/env bash
# Seguito del test del pretraining: lo stesso fine-tuning con passo di apprendimento ridotto.
#
#   CUDA_VISIBLE_DEVICES=0 setsid nohup bash scripts/run_test_finetuning_lr.sh \
#       > reports/logs/coda_541_lr.log 2>&1 < /dev/null &
#
# Il primo giro ha mostrato che il fine-tuning con la ricetta predefinita (passo 0.01) rende meno
# del modello di partenza lasciato intatto: 0.5642 contro 0.6772. L'ipotesi e' che con dieci casi
# quel passo cancelli le feature pre-addestrate. Questi due bracci la mettono alla prova.
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PESI="$nnUNet_results/Dataset521_ATLAS_3T/nnUNetTrainerWandb__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"

for TR in nnUNetTrainerWandb_250epochs_lr1e3 nnUNetTrainerWandb_250epochs_lr1e4; do
    out="$nnUNet_results/Dataset541_SIM_AINCP/${TR}__nnUNetPlans__3d_fullres/fold_0"
    if [ -f "$out/checkpoint_final.pth" ]; then
        echo "[$(date +%H:%M)] $TR: gia' completato, salto."; continue
    fi
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    if [ "$gpu" -gt 30000 ]; then echo "[$(date +%H:%M)] STOP: GPU occupata (${gpu}MiB)"; exit 1; fi
    echo "[$(date +%H:%M)] avvio $TR"
    nnUNetv2_train 541 3d_fullres 0 --npz -tr "$TR" -p nnUNetPlans \
        -pretrained_weights "$PESI" >> "reports/logs/train_541_${TR}.log" 2>&1
    rc=$?; echo "[$(date +%H:%M)] $TR: codice $rc"
    [ "$rc" -ne 0 ] && { echo "STOP: $TR fallito"; exit 1; }
done
echo "[$(date +%H:%M)] coda finita"
