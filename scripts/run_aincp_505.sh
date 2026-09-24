#!/usr/bin/env bash
# Stadio 3 sui dieci pazienti nel template pediatrico (Dataset505), i cinque fold in fila.
#
#   CUDA_VISIBLE_DEVICES=2 setsid nohup bash scripts/run_aincp_505.sh A \
#       > reports/logs/coda_505_A.log 2>&1 < /dev/null &
#
# Tre esperimenti, ognuno a una variabile da un confronto gia' fatto:
#   A  la ricetta scelta su 504 (passo differenziato, freno corretto) dai pesi di ATLAS (502).
#      Contro la stessa ricetta su 504 misura l'effetto dello spazio: nativo o template.
#   B  dai pesi dello stadio 2 (542), passo 1e-4 uniforme. Contro A misura lo stadio 2.
#   C  come B con passo 1e-3.
#   D  come B dai pesi del congiunto ATLAS + WMH (543). Contro B misura sequenza contro congiunto.
# Lo split e' quello di 504, copiato da make_aincp_mni_dataset.py --piano: confronti appaiati.
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
GRUPPO="${1:?gruppo mancante (A, B, C o D)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
P502="$nnUNet_results/Dataset502_ATLAS/nnUNetTrainerWandb__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
# lo stadio 2 scelto: passo 1e-3, meglio dell'altra variante in 16 casi WMH su 16 e (vedi
# reports/oblio_atlas_stadio2.md) per l'oblio delle cavita'
P542="$nnUNet_results/Dataset542_WMH_MNI/${STADIO2:-nnUNetTrainerEspansioneCanali_zero_500ep_lr1e3_dropmod30_arresto50}__nnUNetPlansDa502__3d_fullres/fold_0/checkpoint_final.pth"
case "$GRUPPO" in
    A) TR=nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_arresto50; W="$P502" ;;
    B) TR=nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50_da542; W="$P542" ;;
    C) TR=nnUNetTrainerEspansioneCanali_zero_250ep_lr1e3_arresto50_da542; W="$P542" ;;
    D) TR=nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50_da543
       W="$nnUNet_results/Dataset543_ATLAS_WMH_MNI/nnUNetTrainerEspansioneCanali_zero_500ep_lr1e3_dropmod30_arresto50__nnUNetPlansDa502__3d_fullres/fold_0/checkpoint_final.pth" ;;
    *) echo "gruppo sconosciuto: $GRUPPO"; exit 1 ;;
esac
[ -f "$W" ] || { echo "pesi non trovati: $W"; exit 1; }

# secondo argomento facoltativo: i fold da fare, per dividerli fra piu' GPU (default tutti)
for FOLD in ${2:-0 1 2 3 4}; do
    out="$nnUNet_results/Dataset505_AINCP_MNI/${TR}__nnUNetPlansDa502__3d_fullres/fold_${FOLD}"
    if [ -f "$out/checkpoint_final.pth" ]; then echo "[$(date +%H:%M)] $TR fold $FOLD: gia' fatto"; continue; fi
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    if [ "$gpu" -gt "${SOGLIA_MEM:-30000}" ]; then echo "[$(date +%H:%M)] STOP: GPU $CUDA_VISIBLE_DEVICES occupata (${gpu}MiB)"; exit 1; fi
    echo "[$(date +%H:%M)] avvio $TR fold $FOLD (pesi: $W)"; df -h /home | tail -1
    nnUNetv2_train 505 3d_fullres "$FOLD" --npz -tr "$TR" -p nnUNetPlansDa502 -pretrained_weights "$W" \
        >> "reports/logs/train_505_${TR}_fold${FOLD}.log" 2>&1
    rc=$?; echo "[$(date +%H:%M)] $TR fold $FOLD: codice $rc"
    [ "$rc" -ne 0 ] && { echo "STOP: $TR fallito"; exit 1; }
done
