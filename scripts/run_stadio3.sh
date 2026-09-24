#!/usr/bin/env bash
# Stadio 3 sul dataset finale dei pazienti (scripts/pipeline_aincp.py dataset --id <ID>), fold scelti.
#
#   CUDA_VISIBLE_DEVICES=1 setsid nohup bash scripts/run_stadio3.sh 506 A "0 1" > reports/logs/s3_506_A_g1.log 2>&1 < /dev/null &
#
# Le due ricette candidate, entrambe con il modality dropout, si confrontano in cross-validation sui
# soli pazienti di addestramento (70%); il test non entra in nessuna scelta.
#   A  dai pesi di ATLAS (502), passo differenziato col freno corretto (la ricetta scelta su 504)
#   D  dai pesi dello stadio 1+2 congiunto, passo 1e-4 uniforme. PESI_2C sceglie quale variante di
#      543 (default: quella del 23/9); va fissata prima di lanciare, guardando solo ATLAS e WMH.
# Il secondo argomento facoltativo divide i fold fra piu' GPU.
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
ID="${1:?id del dataset}"; GRUPPO="${2:?gruppo (A o D)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DS=$(basename "$(ls -d "$nnUNet_preprocessed"/Dataset${ID}_*)")
case "$GRUPPO" in
    A) TR=nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_dropmod30_arresto50
       W="$nnUNet_results/Dataset502_ATLAS/nnUNetTrainerWandb__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth" ;;
    D) TR=nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_dropmod30_arresto50_dapre2c
       W="${PESI_2C:-$nnUNet_results/Dataset543_ATLAS_WMH_MNI/nnUNetTrainerEspansioneCanali_zero_500ep_lr1e3_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres/fold_0/checkpoint_final.pth}" ;;
    *) echo "gruppo sconosciuto: $GRUPPO"; exit 1 ;;
esac
[ -f "$W" ] || { echo "pesi non trovati: $W"; exit 1; }
for FOLD in ${3:-0 1 2 3 4}; do
    out="$nnUNet_results/$DS/${TR}__nnUNetPlansDa502__3d_fullres/fold_${FOLD}"
    if [ -f "$out/checkpoint_final.pth" ]; then echo "[$(date +%H:%M)] fold $FOLD gia' fatto"; continue; fi
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    if [ "$gpu" -gt "${SOGLIA_MEM:-30000}" ]; then echo "STOP: GPU $CUDA_VISIBLE_DEVICES occupata (${gpu}MiB)"; exit 1; fi
    echo "[$(date +%H:%M)] avvio $TR fold $FOLD su $DS (pesi: $W)"; df -h /home | tail -1
    nnUNetv2_train "$ID" 3d_fullres "$FOLD" --npz -tr "$TR" -p nnUNetPlansDa502 -pretrained_weights "$W" \
        >> "reports/logs/train_${ID}_${TR}_fold${FOLD}.log" 2>&1
    rc=$?; echo "[$(date +%H:%M)] fold $FOLD: codice $rc"; [ "$rc" -ne 0 ] && exit 1
done
