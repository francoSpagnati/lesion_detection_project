#!/usr/bin/env bash
# Il test, una volta sola, con la ricetta e il postprocessing gia' scelti in cross-validation.
#
#   CUDA_VISIBLE_DEVICES=1 bash scripts/predici_test.sh 506 <trainer>
#
# Insieme dei cinque modelli dei fold (media delle probabilita', con il ribaltamento a specchio di
# nnU-Net come augmentation al test), probabilita' salvate, poi postprocessing e metriche nel nativo
# con i parametri congelati in reports/postprocessing_<dataset>__<run>.json.
set -eu
cd /home/ccantini/lesion_detection_project
source scripts/env.sh
ID="${1:?id}"; TR="${2:?trainer}"
DS=$(basename "$(ls -d "$nnUNet_raw"/Dataset${ID}_*)")
RUN="$nnUNet_results/$DS/${TR}__nnUNetPlansDa502__3d_fullres"
[ -f "reports/postprocessing_${DS}__$(basename "$RUN").json" ] || { echo "manca la scelta del postprocessing: prima 'scegli'"; exit 1; }
OUT="$RUN/test_insieme"
[ -d "$OUT" ] && { echo "$OUT esiste: il test si fa una volta sola"; exit 1; }
nnUNetv2_predict -i "$nnUNet_raw/$DS/imagesTs" -o "$OUT" -d "$ID" -c 3d_fullres -tr "$TR" -p nnUNetPlansDa502 \
    -f 0 1 2 3 4 --save_probabilities
python scripts/postprocessing_aincp.py valuta --run "$RUN" --test "$OUT"
