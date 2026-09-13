#!/usr/bin/env bash
# Aspetta che una GPU si liberi e lancia la coda di un fold di Dataset504 su di essa.
#
#   setsid nohup bash scripts/attendi_gpu_e_lancia.sh > reports/logs/attesa_gpu.log 2>&1 < /dev/null &
#
# Sulla macchina non c'e' uno scheduler (ne' SLURM ne' PBS): le schede si prendono a chi arriva
# prima, e da giorni le occupa tutte un altro utente. Questo script fa da coda: ogni 5 minuti
# guarda le quattro schede e, se ne trova una libera (meno di 5 GB usati e utilizzo sotto il
# 20% -- i processi residui da 400 MB non contano), ci lancia il prossimo fold non ancora
# iniziato con scripts/run_aincp_504.sh. Per non occupare tutta la macchina appena si libera
# usa al massimo MAX_GPU schede insieme; i fold restanti aspettano il loro turno.
#
# Un fold conta come "iniziato" se esiste la cartella dei risultati del suo primo trainer, e
# come "finito" se run_aincp_504.sh ha scritto "coda fold N finita" nel suo log. Lo script
# termina quando tutti e cinque i fold sono finiti; se un fold fallisce lo dice e passa oltre.
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
MAX_GPU="${MAX_GPU:-2}"
INTERVALLO="${INTERVALLO:-300}"
PRIMO_TRAINER=nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50

log() { echo "[$(date '+%d/%m %H:%M')] $*"; }

fold_iniziato() { [ -d "$nnUNet_results/Dataset504_AINCP/${PRIMO_TRAINER}__nnUNetPlansDa502__3d_fullres/fold_$1" ]; }
fold_finito()   { grep -q "coda fold $1 finita" "reports/logs/coda_504_fold$1.log" 2>/dev/null; }
fold_in_corso() { pgrep -f "run_aincp_504.sh $1\$" >/dev/null; }

gpu_libere() {
    # indice delle schede con poca memoria usata e poco carico; le schede su cui ho appena
    # lanciato qualcosa vengono escluse dal chiamante
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -F', *' '$2 < 5000 && $3 < 20 {print $1}'
}

log "in attesa di una GPU libera (max $MAX_GPU in parallelo, controllo ogni ${INTERVALLO}s)"
while :; do
    da_fare=()
    for f in 0 1 2 3 4; do
        fold_finito "$f" && continue
        fold_in_corso "$f" && continue
        da_fare+=("$f")
    done
    if [ "${#da_fare[@]}" -eq 0 ]; then
        log "tutti i fold sono finiti o in corso: esco"; break
    fi
    attivi=$(pgrep -fc "run_aincp_504.sh")
    for gpu in $(gpu_libere); do
        [ "$attivi" -ge "$MAX_GPU" ] && break
        [ "${#da_fare[@]}" -eq 0 ] && break
        f="${da_fare[0]}"; da_fare=("${da_fare[@]:1}")
        log "GPU $gpu libera: lancio il fold $f"
        echo "fold $f -> GPU $gpu, $(date '+%d/%m %H:%M')" >> "reports/logs/coda_504_assegnazioni.log"
        CUDA_VISIBLE_DEVICES="$gpu" setsid nohup bash scripts/run_aincp_504.sh "$f" \
            >> "reports/logs/coda_504_fold$f.log" 2>&1 < /dev/null &
        attivi=$((attivi + 1))
        # nnU-Net impiega qualche minuto prima di allocare la memoria (caricamento dati,
        # compilazione della rete): senza questa pausa la stessa scheda sembrerebbe ancora
        # libera al giro successivo e ci finirebbero due fold.
        sleep 240
    done
    sleep "$INTERVALLO"
done
