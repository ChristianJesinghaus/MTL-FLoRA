#!/usr/bin/env bash
set -euo pipefail

# Minimal federated smoke test using a tiny HF model to verify that
# the Flower server/client wiring and tokenization pipelines run end-to-end.
# Requirements:
#   - network access to download sshleifer/tiny-gpt2 (or pre-download it)
#   - bash + python environment with dependencies from requirements.txt installed

SERVER_ADDR="127.0.0.1:8080"
BASE_MODEL="sshleifer/tiny-gpt2"
DATA_PATH="/tmp/federated_toy_data.json"

mkdir -p /tmp
cat >"${DATA_PATH}" <<'JSON'
[
  {
    "instruction": "Schreibe eine kurze Zusammenfassung von MTL-LoRA.",
    "input": "",
    "output": "MTL-LoRA nutzt Low-Rank Adapter für Multi-Task-Lernen.",
    "task_id": 0
  },
  {
    "instruction": "Zähle zwei Vorteile von Adapter-basiertem Training auf.",
    "input": "",
    "output": "Geringere Parameterzahl und leichteres Fine-Tuning.",
    "task_id": 1
  }
]
JSON

python federated_mlora.py server --server_address "${SERVER_ADDR}" --num_rounds 1 &
SERVER_PID=$!
echo "Started Flower server (pid=${SERVER_PID})"
sleep 3

python federated_mlora.py client \
  --server_address "${SERVER_ADDR}" \
  --base_model "${BASE_MODEL}" \
  --data_path "${DATA_PATH}" \
  --adapter_name mlora \
  --lora_target_modules "('c_attn','c_proj')" \
  --batch_size 1 \
  --num_epochs 1 \
  --learning_rate 1e-4 \
  --cutoff_len 128 \
  --val_split 0.5 \
  --train_on_inputs

kill "${SERVER_PID}" >/dev/null 2>&1 || true

echo "Federated smoke test completed"
