#!/bin/bash
##
## Automatic evaluation of TinyLlama mLoRA freezeR experiments.
## Hard-coded for linear_freeze: lora_r=8, num_B stays at training value.
##

#SBATCH --job-name=tinyllama_mtl_eval_auto
#SBATCH --output=logs/tinyllama_mtl_eval_auto_%j.out
#SBATCH --error=logs/tinyllama_mtl_eval_auto_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=volta,ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-04:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"

PARAM_FILE="${1:-param_combinations.txt}"
PARAM_FILE="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${PARAM_FILE}")"

if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Parameter file ${PARAM_FILE} not found." >&2
  exit 1
fi

## Config – these match train_tinyllama_mtl_mlora.sh defaults
LOCAL_R=8
LORA_ALPHA=16
NUM_CLIENTS=4

echo "Evaluating freezeR experiments from ${PARAM_FILE}..."

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/outputs"

pick_first_existing_file() {
  for f in "$@"; do
    if [[ -f "$f" ]]; then
      echo "$f"
      return 0
    fi
  done
  return 1
}

while read -r strat rounds num_B epochs client_p; do
  [[ -z "${strat:-}" || "${strat}" =~ ^# ]] && continue
  [[ "${strat}" != "federated" && "${strat}" != "fedit" ]] && continue

  client_p=${client_p:-1.0}
  p_tag=$(echo "${client_p}" | sed 's/\./_/g')

  # Try both naming conventions for train dir
  train_prefix_new="tinyllama_train_${strat}_epoch${epochs}_flround${rounds}_numB${num_B}_p${p_tag}_4Clients_freezeR"
  train_prefix_old="tinyllama_train_${strat}_epoch${epochs}_flround${rounds}_numB${num_B}_p${p_tag}_4Clients_freezeR"

  eval_prefix="tinyllama_eval_${strat}_epoch${epochs}_flround${rounds}_numB${num_B}_p${p_tag}_4Clients_freezeR"

  # Find the training directory (try new naming first, then old)
  train_dir=""
  for candidate in "${REPO_ROOT}/outputs/${train_prefix_new}" "${REPO_ROOT}/outputs/${train_prefix_old}"; do
    if [[ -d "${candidate}" ]]; then
      train_dir="${candidate}"
      break
    fi
  done

  if [[ -z "${train_dir}" ]]; then
    echo "[WARN] Train dir not found for: ${train_prefix_new} or ${train_prefix_old} (skipping)" >&2
    continue
  fi

  out_dir="${REPO_ROOT}/outputs/${eval_prefix}"
  logfile_out="${REPO_ROOT}/logs/${eval_prefix}.out"
  logfile_err="${REPO_ROOT}/logs/${eval_prefix}.err"

  load_path=""
  if load_path="$(pick_first_existing_file \
      "${train_dir}/checkpoints/ckpt_global_final.pt" \
      "${train_dir}/checkpoints/ckpt_global_best.pt" \
      "${train_dir}/checkpoints/ckpt_best.pt" \
      "${train_dir}/checkpoints/ckpt_last.pt" \
    )"; then
    :
  else
    if [[ -f "${train_dir}/adapter_state.pt" && -f "${train_dir}/heads_state.pt" ]]; then
      load_path="${train_dir}"
    else
      echo "[WARN] No usable checkpoint in: ${train_dir} (skipping)" >&2
      continue
    fi
  fi

  # ─── Compute eval dimensions matching what the training script saved ───
  #
  # freezeR / linear_freeze (federated, round > 0):
  #   Per round: each of K clients adds add_r=LOCAL_R rank and add_B=num_B B-matrices
  #   After R rounds (round 0 is exponential, rounds 1..R-1 are linear_freeze):
  #     Round 0 (exponential): r = LOCAL_R * K,  B = num_B * K
  #     Round 1 (linear):      r += K * LOCAL_R, B += K * num_B
  #     ...
  #     Round R-1 (linear):    r += K * LOCAL_R, B += K * num_B
  #   Total: r = LOCAL_R*K + (R-1)*K*LOCAL_R = K*LOCAL_R*R
  #          B = num_B*K   + (R-1)*K*num_B   = K*num_B*R
  #
  # fedit: simple FedAvg, no stacking at all
  #   r = LOCAL_R, B = num_B (always)
  #
  # For freezeR with only 1 round: round 0 is exponential
  #   r = LOCAL_R * K, B = num_B * K

  if [[ "${strat}" == "fedit" ]]; then
    # FedIT: no stacking, dimensions stay local
    eval_r="${LOCAL_R}"
    eval_num_B="${num_B}"
    eval_global_num_B="${num_B}"
    eval_block_size="${num_B}"
  elif [[ "${strat}" == "federated" ]]; then
    # FLoRA with linear_freeze
    # Round 0 always does exponential stacking: r*K, B*K
    # Rounds 1..R-1 do linear_freeze: each adds K*add_r, K*add_B
    #
    # But looking at the training code more carefully:
    #   - do_linear_freeze is only True when round_idx > 0
    #   - Round 0: exponential -> r = LOCAL_R * K, B = num_B * K
    #   - Round 1+: linear_freeze -> r += K * LOCAL_R, B += K * num_B per round
    #
    # Total after R rounds:
    #   r = LOCAL_R * K  +  (R-1) * K * LOCAL_R = K * LOCAL_R * R
    #   B = num_B  * K   +  (R-1) * K * num_B   = K * num_B  * R

    eval_r=$(( NUM_CLIENTS * LOCAL_R * rounds ))
    eval_global_num_B=$(( NUM_CLIENTS * num_B * rounds ))
    eval_num_B="${num_B}"
    eval_block_size="${num_B}"
  fi

  echo "------------------------------------------"
  echo "Evaluating: strat=${strat} epochs=${epochs} rounds=${rounds} numB=${num_B} p=${client_p}"
  echo "  Train dir:    ${train_dir}"
  echo "  Load:         ${load_path}"
  echo "  lora_r:       ${eval_r}"
  echo "  num_B:        ${eval_num_B}"
  echo "  global_num_B: ${eval_global_num_B}"
  echo "  block_size:   ${eval_block_size}"

  mkdir -p "${out_dir}"

  bash "${REPO_ROOT}/eval_tinyllama_mtl_mlora.sh" "${load_path}" "${out_dir}" \
    --lora_r "${eval_r}" \
    --lora_alpha "${LORA_ALPHA}" \
    --num_B "${eval_num_B}" \
    --global_num_B "${eval_global_num_B}" \
    --block_size "${eval_block_size}" \
    > "${logfile_out}" 2> "${logfile_err}"

  echo "Finished: ${eval_prefix}"
done < "${PARAM_FILE}"