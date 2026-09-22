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
# Tutti e tre con l'arresto sulla perdita di validazione (pazienza 50 epoche): checkpoint_final
# riceve i pesi dell'epoca migliore, l'ultima epoca resta in checkpoint_ultima_epoca.pth.
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
    # SOGLIA_MEM: memoria oltre la quale la scheda si considera occupata. 30 GB per default;
    # 60 GB quando si e' deciso di condividere la scheda con il job di un altro utente.
    if [ "$gpu" -gt "${SOGLIA_MEM:-30000}" ]; then echo "[$(date +%H:%M)] STOP: GPU $CUDA_VISIBLE_DEVICES occupata (${gpu}MiB)"; exit 1; fi
    echo "[$(date +%H:%M)] avvio $TR fold $FOLD ${W:+(pesi: 502)}"
    df -h /home | tail -1
    nnUNetv2_train 504 3d_fullres "$FOLD" --npz -tr "$TR" -p nnUNetPlansDa502 \
        ${W:+-pretrained_weights "$W"} >> "reports/logs/train_504_${TR}_fold${FOLD}.log" 2>&1
    local rc=$?; echo "[$(date +%H:%M)] $TR fold $FOLD: codice $rc"
    # Codice 127 = comando non trovato: un problema di ambiente, non del training. Va detto
    # esplicitamente perche' il sintomo (un fold che "fallisce" subito) e' identico a quello di
    # un errore vero, e la coda esterna lo ritenterebbe all'infinito.
    [ "$rc" -eq 127 ] && echo "AMBIENTE: nnUNetv2_train non e' nel PATH (manca il venv?)"
    [ "$rc" -ne 0 ] && { echo "STOP: $TR fallito"; exit 1; }
    return 0
}

# Secondo giro (15 settembre): il passo differenziato sul canale FLAIR, una variabile rispetto
# alla variante `zero` gia' misurata. Le tre del primo giro restano nella coda e vengono saltate
# perche' hanno gia' il checkpoint_final.
lancia nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_arresto50 "$PESI"
lancia nnUNetTrainerWandb_250ep_arresto50
lancia nnUNetTrainerEspansioneCanali_duplica_250ep_lr1e4_arresto50 "$PESI"
lancia nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_arresto50 "$PESI"

# Terzo giro (22 settembre). Il freno del passo differenziato scalava il gradiente ma non il
# decadimento dei pesi, che SGD aggiunge dopo l'hook e moltiplica per il passo del gruppo: nei run
# del 16 il filtro T1 e' stato decaduto cento volte piu' del dovuto (norma da 8.36 a 3.50, coseno
# 0.996 con la partenza -- si era solo ristretto). Corretto il freno, la ricetta si ripete sotto un
# nome nuovo per non sovrascrivere quei risultati, e si aggiunge il controllo che le da' senso:
# passo 1e-3 uniforme, per sapere se serviva un passo *differenziato* o solo un passo *piu' grande*.
lancia nnUNetTrainerEspansioneCanali_zero_250ep_lr1e4_primo1e2_wdfix_arresto50 "$PESI"
lancia nnUNetTrainerEspansioneCanali_zero_250ep_lr1e3_arresto50 "$PESI"
echo "[$(date +%H:%M)] coda fold $FOLD finita"
