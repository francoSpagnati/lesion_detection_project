#!/usr/bin/env bash
# Postprocessing scelto sui fuori-fold e confronto con Z per le ricette B, C e F (21-22/10/2026).
#   setsid nohup bash scripts/analisi_ricette_BCF.sh > reports/logs/analisi_BCF.log 2>&1 < /dev/null &
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
R=$nnUNet_results; T=nnUNetPlansDa502__3d_fullres
declare -A RUN=([B]=$R/Dataset506_AINCP_FINALE/nnUNetTrainerBlob_250ep_dropmod30_arresto50__$T
                [C]=$R/Dataset507_AINCP_VENTRICOLO/nnUNetTrainerWandb_250ep_dropmod30_arresto50__$T
                [F]=$R/Dataset508_AINCP_FENOTIPI/nnUNetTrainerWandb_250ep_dropmod30_arresto50__$T)
for k in B C F; do
    ( python scripts/postprocessing_aincp.py scegli --run "${RUN[$k]}" > reports/logs/postproc_$k.log 2>&1 \
      && python scripts/confronto_misto_532.py $k > reports/logs/confronto_$k.log 2>&1; echo "[$(date +%H:%M)] $k: codice $?" ) &
done
wait
echo "[$(date +%H:%M)] finito"
