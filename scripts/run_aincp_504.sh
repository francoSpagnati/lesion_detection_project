#!/usr/bin/env bash
# Coda degli esperimenti sui dieci pazienti AInCP (Dataset504, spazio T1 nativo, 2 canali).
#
#   CUDA_VISIBLE_DEVICES=0 setsid nohup bash scripts/run_aincp_504.sh 0 \
#       > reports/logs/coda_504_fold0.log 2>&1 < /dev/null &
#
# Il primo argomento e' il fold (0-4). Per ogni fold la coda lancia, nell'ordine:
#   T1  espansione da ATLAS (502), canale FLAIR a zero, 250 epoche, passo 1e-4   <- Proposta B
#   T0  da zero, stesso piano e stesso budget di epoche                            <- riferimento
#   T1' come T1 ma con il filtro duplicato sui due canali
# Tutti sul piano nnUNetPlansDa502 (1 mm, patch 128^3), che e' quello di 502: il trasferimento
# dei pesi funziona solo fra architetture identiche a meno dei canali. Una sola variabile cambia
# fra T1 e T0 (il pretraining) e fra T1 e T1' (l'inizializzazione del canale nuovo).
#
# Lo split e' quello scritto da make_aincp_dataset.py e non si rigenera.
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
FOLD="${1:?fold mancante (0-4)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PESI="$nnUNet_results/Dataset502_ATLAS/nnUNetTrainerWandb__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
[ -f "$PESI" ] || { echo "pesi di 502 non trovati: $PESI"; exit 1; }

lancia() {   # trainer, [pesi]
    local TR="$1" W="${2:-}"
    local out="$nnUNet_results/Dataset504_AINCP/${TR}__nnUNetPlansDa502__3d_fullres/fold_${FOLD}"
    if [ -f "$out/checkpoint_final.pth" ]; then
        echo "[$(date +%H:%M)] $TR fold $FOLD: gia' completato, salto."; return 0
    fi
    local gpu; gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    if [ "$gpu" -gt 30000 ]; then echo "[$(date +%H:%M)] STOP: GPU $CUDA_VISIBLE_DEVICES occupata (${gpu}MiB)"; exit 1; fi
    echo "[$(date +%H:%M)] avvio $TR fold $FOLD ${W:+(pesi: 502)}"
    df -h /home | tail -1
    nnUNetv2_train 504 3d_fullres "$FOLD" --npz -tr "$TR" -p nnUNetPlansDa502 \
        ${W:+-pretrained_weights "$W"} >> "reports/logs/train_504_${TR}_fold${FOLD}.log" 2>&1
    local rc=$?; echo "[$(date +%H:%M)] $TR fold $FOLD: codice $rc"
    [ "$rc" -ne 0 ] && { echo "STOP: $TR fallito"; exit 1; }
    return 0
}

lancia nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4 "$PESI"
lancia nnUNetTrainerWandb_250epochs
lancia nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4 "$PESI"
echo "[$(date +%H:%M)] coda fold $FOLD finita"
