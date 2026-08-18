#!/usr/bin/env bash
# Esegue in sequenza le baseline fold 0 di 3d_fullres su 501, 502 e 503.
#
# Va lanciato in modo che sopravviva alla chiusura del terminale (o della sessione
# di un assistente): il training dura ore e non deve morire con chi lo ha avviato.
#
#   setsid nohup bash scripts/run_baselines.sh > reports/logs/baselines.log 2>&1 < /dev/null &
#
# `setsid` mette lo script in una sessione propria: `nohup` da solo protegge dal
# segnale di riaggancio del terminale, ma non dalla terminazione dell'intero gruppo
# di processi. Un training perso all'epoca 189 e' costato l'esperienza.
#
# Se un dataset ha gia' un checkpoint_latest.pth il training RIPRENDE da li' (--c),
# invece di ricominciare da zero.
set -u
cd /home/ccantini/lesion_detection_project || exit 1
source scripts/env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TRAINER=nnUNetTrainerWandb
CONFIG=3d_fullres
FOLD=0
MIN_GB=60        # spazio minimo su /home: il training scompatta i .npz in .npy
MAX_GPU_MIB=30000  # se la GPU scelta ha piu' di cosi' occupato, e' di qualcun altro

for d in "$@"; do
    nome=$(ls -d "$nnUNet_raw/Dataset${d}_"* 2>/dev/null | head -1 | xargs -r basename)
    out="$nnUNet_results/$nome/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}"

    if [ -f "$out/checkpoint_final.pth" ]; then
        echo "[$(date +%H:%M)] $d: gia' completato, salto."
        continue
    fi

    liberi=$(df -BG --output=avail /home | tail -1 | tr -d ' G')
    if [ "$liberi" -lt "$MIN_GB" ]; then
        echo "[$(date +%H:%M)] STOP: solo ${liberi}G liberi su /home, non lancio $d."
        exit 1
    fi
    gpu_usata=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    if [ "$gpu_usata" -gt "$MAX_GPU_MIB" ]; then
        echo "[$(date +%H:%M)] STOP: GPU $CUDA_VISIBLE_DEVICES occupata (${gpu_usata}MiB), non lancio $d."
        exit 1
    fi

    riprendi=''
    if [ -f "$out/checkpoint_latest.pth" ]; then
        riprendi='--c'
        echo "[$(date +%H:%M)] $d: trovato checkpoint_latest.pth, riprendo."
    fi

    echo "[$(date +%H:%M)] $d: avvio ($nome, ${liberi}G liberi, GPU $CUDA_VISIBLE_DEVICES)"
    nnUNetv2_train "$d" "$CONFIG" "$FOLD" --npz -tr "$TRAINER" $riprendi \
        >> "reports/logs/train_${d}_${CONFIG}_f${FOLD}.log" 2>&1
    # Il codice va catturato subito: una `$(...)` dentro l'echo lo azzererebbe,
    # e la coda proseguirebbe come se il training fosse andato a buon fine.
    rc=$?
    echo "[$(date +%H:%M)] $d: terminato con codice $rc"
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%H:%M)] STOP: $d fallito, non proseguo con i successivi."
        exit "$rc"
    fi
done
echo "[$(date +%H:%M)] coda finita"
