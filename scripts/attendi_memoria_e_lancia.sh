#!/usr/bin/env bash
# Coda per run_stadio3.sh quando le schede sono condivise: un fold alla volta, sulla scheda con piu'
# memoria libera, appena ce n'e' abbastanza.
#
#   setsid nohup bash scripts/attendi_memoria_e_lancia.sh 506 B "0 2 4" > reports/logs/coda_506_B_a.log 2>&1 < /dev/null &
#   setsid nohup bash scripts/attendi_memoria_e_lancia.sh 506 B "1 3"   > reports/logs/coda_506_B_b.log 2>&1 < /dev/null &
#
# PERCHE'. Dal 12/10 le quattro A100 sono occupate da un altro utente al 100% con 76 GB su 80, e
# l'utente ha chiesto di lanciare lo stesso dividendo il lavoro, come per M (5/10, circa 7 GB a scheda).
# attendi_gpu_e_lancia.sh aspetta una scheda libera e serve solo a 504; questa aspetta la memoria.
#
# COME. Prima di ogni fold si aspetta una scheda con almeno MIN_LIBERA MiB liberi (default 10 000:
# il training ne usa circa 7 000, il resto e' margine per le oscillazioni dell'altro job). Due code
# in parallelo non prendono la stessa scheda: la scheda si prenota con una cartella in /tmp (mkdir
# e' atomico) e si libera alla fine del fold. Un fold che fallisce si riprova una volta sola
# (tipicamente memoria esaurita perche' l'altro job e' cresciuto); al secondo fallimento la coda si
# ferma: un errore vero non va ripetuto per ore (vedi attendi_gpu_e_lancia.sh, 13 settembre).
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
ID="${1:?id del dataset}"; GRUPPO="${2:?ricetta}"; FOLDS="${3:-0 1 2 3 4}"
MIN_LIBERA="${MIN_LIBERA:-10000}"; INTERVALLO="${INTERVALLO:-120}"
log() { echo "[$(date '+%d/%m %H:%M')] $*"; }

prenota() {   # stampa l'indice della scheda prenotata, o niente
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -F', *' -v m="$MIN_LIBERA" '$3 - $2 >= m {print $3 - $2, $1}' | sort -rn | awk '{print $2}' \
      | while read -r g; do mkdir "/tmp/ccantini_gpu_$g.lock" 2>/dev/null && { echo "$g"; break; }; done
}

for f in $FOLDS; do
    for tentativo in 1 2; do
        log "fold $f, tentativo $tentativo: aspetto una scheda con almeno $MIN_LIBERA MiB liberi"
        while :; do g=$(prenota); [ -n "$g" ] && break; sleep "$INTERVALLO"; done
        log "fold $f -> GPU $g"
        CUDA_VISIBLE_DEVICES="$g" SOGLIA_MEM=81920 bash scripts/run_stadio3.sh "$ID" "$GRUPPO" "$f"
        rc=$?; rmdir "/tmp/ccantini_gpu_$g.lock"
        [ "$rc" -eq 0 ] && break
        log "fold $f fallito (codice $rc)"
        [ "$tentativo" -eq 2 ] && { log "STOP: fold $f fallito due volte, leggere reports/logs/train_${ID}_*fold${f}.log"; exit 1; }
        sleep 600
    done
done
log "coda finita: fold $FOLDS"
