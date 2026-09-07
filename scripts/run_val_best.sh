#!/usr/bin/env bash
# Rivaluta i quattro bracci di 541 usando checkpoint_best invece di checkpoint_final.
#
#   setsid nohup bash scripts/run_val_best.sh > reports/logs/coda_val_best.log 2>&1 < /dev/null &
#
# Perche' su CPU: le GPU sono tutte al 100% per un lavoro di altri utenti, e questa e' una
# validazione di 35 casi, non un training. Lenta ma a impatto zero su chi sta lavorando.
#
# `--val_best` riscrive la cartella `validation`: i risultati con checkpoint_final sono gia'
# stati copiati in `validation_final` prima di lanciare, altrimenti andrebbero persi.
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh

for spec in \
    "nnUNetTrainerWandb_250epochs nnUNetPlans" \
    "nnUNetTrainerWandb_250epochs nnUNetPlansFT" \
    "nnUNetTrainerWandb_250epochs_lr1e3 nnUNetPlans" \
    "nnUNetTrainerWandb_250epochs_lr1e4 nnUNetPlans" ; do
    set -- $spec
    TR=$1; PL=$2
    echo "[$(date +%H:%M)] $TR / $PL"
    nnUNetv2_train 541 3d_fullres 0 -tr "$TR" -p "$PL" --val --val_best -device cpu \
        >> "reports/logs/val_best_${TR}_${PL}.log" 2>&1
    echo "[$(date +%H:%M)] $TR / $PL: codice $?"
done
echo "[$(date +%H:%M)] finito"
