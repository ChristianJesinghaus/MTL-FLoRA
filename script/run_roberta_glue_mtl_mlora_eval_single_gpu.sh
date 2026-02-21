#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash script/run_roberta_glue_mtl_mlora_eval_single_gpu.sh <MODEL_DIR> [OUTPUT_DIR] [extra args...]
#
# <MODEL_DIR>   = Verzeichnis des Trainingslaufs (enthält global_model_final.pt, adapter_state_final.pt, heads_state_final.pt)
# [OUTPUT_DIR]  = optionales Ausgabeverzeichnis; Standard: <MODEL_DIR>/eval_only
# [extra args]  = werden direkt an das Python-Eval-Skript weitergereicht (z. B. --eval_details_max_examples -1)

# Erstes Argument: Verzeichnis mit dem trainierten Modell
MODEL_DIR="${1:?Usage: $0 <MODEL_DIR> [OUTPUT_DIR] [extra args...] }"
shift || true

# Wenn das nächste Argument kein Flag ist (--*), als OUTPUT_DIR interpretieren
if [[ $# -gt 0 && $1 != --* ]]; then
  OUTPUT_DIR="$1"
  shift || true
else
  OUTPUT_DIR="${MODEL_DIR}/eval_only"
fi

# Alle weiteren Argumente sind optionale Flags für das Python-Skript
EXTRA_ARGS=("$@")

# Lade Container-/Umgebungs-Helfer
source "$(dirname "${BASH_SOURCE[0]}")/common_env.sh"

# Ausgabeordner anlegen
mkdir -p "${OUTPUT_DIR}"

# Name des Python-Skripts
SCRIPT="run_glue_roberta_mtl_mlora_eval_single_gpu.py"

# Standard-Argumente (können via EXTRA_ARGS überschrieben werden)
# Passe lora_r und weitere mLoRA-Parameter an das Training an (zwei Clients → r=16)
ARGS=(
  --output_dir "${OUTPUT_DIR}"
  --model_name roberta-base
  --lora_r 16
  --lora_alpha 16
  --lora_dropout 0.05
  --num_B 3
  --temperature 0.1
  --eval_batch_size 32
  --max_length 256
  --load_global_model "${MODEL_DIR}/global_model_final.pt"
)

# Baue den Python-Befehl zusammen
CMD=(python3 -u "${SCRIPT}" "${ARGS[@]}" "${EXTRA_ARGS[@]}")
CMD_STR="$(printf '%q ' "${CMD[@]}")"

# Ausführliche Ausgaben
echo "[RUN] MODEL_DIR=${MODEL_DIR}"
echo "[RUN] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[RUN] REPO_DIR=${REPO_DIR}"
echo "[RUN] CONTAINER_IMAGE=${CONTAINER_IMAGE}"
echo "[RUN] OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
echo "[RUN] CMD=${CMD_STR}"

# Starte die Evaluation im Container
run_in_container "cd '${REPO_DIR}' && ${CMD_STR}"