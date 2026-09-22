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
# Soglie per considerare una scheda disponibile. Con i default (5 GB, 20%) si aspetta una
# scheda del tutto libera. Il 13 settembre l'utente ha deciso di condividere le schede con il
# job dell'altro utente (~50 GB su 80 per scheda): SOGLIA_MEM=60000 SOGLIA_UTIL=101.
SOGLIA_MEM="${SOGLIA_MEM:-5000}"
SOGLIA_UTIL="${SOGLIA_UTIL:-20}"
export SOGLIA_MEM
# L'ULTIMO trainer della coda di run_aincp_504.sh: un fold e' finito quando ha il suo
# checkpoint. Cosi' aggiungere una variante in fondo alla coda basta a far ripartire i fold,
# e le variante gia' fatte vengono saltate da run_aincp_504.sh stesso.
ULTIMO_TRAINER=nnUNetTrainerEspansioneCanali_zero_250ep_lr1e3_arresto50

log() { echo "[$(date '+%d/%m %H:%M')] $*"; }

fold_finito()   { [ -f "$nnUNet_results/Dataset504_AINCP/${ULTIMO_TRAINER}__nnUNetPlansDa502__3d_fullres/fold_$1/checkpoint_final.pth" ]; }
fold_in_corso() { pgrep -f "run_aincp_504.sh $1\$" >/dev/null; }
# Un fold che e' fallito non va ritentato: se fallisce, fallisce per un motivo che il tempo non
# cambia. La notte del 13 settembre la coda ha rilanciato lo stesso fold ogni tredici minuti per
# diciassette ore perche' `nnUNetv2_train` non era nel PATH, e nessuno se ne e' accorto fino al
# mattino. Per riprovare dopo aver corretto il problema si cancella il log del fold.
# Conta solo uno STOP successivo all'ultima riga "coda fold N finita": il log e' cumulativo
# fra un giro e l'altro, e un fallimento gia' superato non deve bloccare il giro nuovo.
fold_fallito()  { [ -f "reports/logs/coda_504_fold$1.log" ] && awk 'BEGIN{s=0;e=0} /coda fold .* finita/{s=NR} /^STOP: .* fallito/{if (NR>s) e=1} END{exit !e}' "reports/logs/coda_504_fold$1.log"; }

gpu_nostre() {
    # schede su cui gira gia' un nostro fold: l'ultima assegnazione di ogni fold in corso
    for f in 0 1 2 3 4; do
        fold_in_corso "$f" || continue
        grep "^fold $f ->" reports/logs/coda_504_assegnazioni.log 2>/dev/null | tail -1 \
          | sed 's/.*GPU \([0-9]*\).*/\1/'
    done
}

gpu_libere() {
    # indice delle schede sotto le soglie, dalla meno carica alla piu' carica, escluse quelle
    # su cui gira gia' un nostro fold (una scheda, un fold)
    local nostre; nostre=$(gpu_nostre | tr '\n' ' ')
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
      | awk -F', *' -v m="$SOGLIA_MEM" -v u="$SOGLIA_UTIL" '$2 < m && $3 < u {print $2, $1}' \
      | sort -n | awk -v n=" $nostre " '{ if (index(n, " " $2 " ") == 0) print $2 }'
}

log "in attesa di una GPU disponibile (max $MAX_GPU in parallelo, soglie ${SOGLIA_MEM} MiB / ${SOGLIA_UTIL}%, controllo ogni ${INTERVALLO}s)"
while :; do
    da_fare=(); falliti=0
    for f in 0 1 2 3 4; do
        fold_finito "$f" && continue
        fold_fallito "$f" && { falliti=$((falliti + 1)); continue; }
        fold_in_corso "$f" && continue
        da_fare+=("$f")
    done
    if [ "${#da_fare[@]}" -eq 0 ]; then
        if [ "$falliti" -gt 0 ] && ! pgrep -f "run_aincp_504.sh" >/dev/null; then
            log "ESCO: $falliti fold falliti e nessuno in corso. Leggere reports/logs/coda_504_fold*.log,"
            log "      correggere, cancellare il log del fold e rilanciare questo script."
        else
            log "tutti i fold sono finiti o in corso: esco"
        fi
        break
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
