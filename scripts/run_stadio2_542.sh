#!/usr/bin/env bash
# Stadio 2 della Proposta B: la rete espansa da ATLAS (502) impara la FLAIR su WMH in MNI (542).
#
#   CUDA_VISIBLE_DEVICES=1 setsid nohup bash scripts/run_stadio2_542.sh <trainer> \
#       > reports/logs/stadio2_<breve>.log 2>&1 < /dev/null &
#
# Fold 0 di 542 (split copiato da 512): 63 casi di addestramento, 16 di validazione, lo stesso
# confronto del Dice 0.753 di 512. Non serve la cross-validation completa: di questo stadio si
# usano solo i pesi, come per 502.
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
TR="${1:?trainer mancante}"
PESI="$nnUNet_results/Dataset502_ATLAS/nnUNetTrainerWandb__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
[ -f "$PESI" ] || { echo "pesi di 502 non trovati"; exit 1; }
echo "[$(date '+%d/%m %H:%M')] avvio $TR su 542 fold 0, GPU $CUDA_VISIBLE_DEVICES"; df -h /home | tail -1
nnUNetv2_train 542 3d_fullres 0 -tr "$TR" -p nnUNetPlansDa502 -pretrained_weights "$PESI"
rc=$?; echo "[$(date '+%d/%m %H:%M')] $TR: codice $rc"
[ "$rc" -eq 127 ] && echo "AMBIENTE: nnUNetv2_train non e' nel PATH"
exit $rc
