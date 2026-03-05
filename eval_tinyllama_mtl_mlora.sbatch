#!/bin/bash
##
## Automatic evaluation of TinyLlama mLoRA federated experiments.
##
## This Slurm job reads the parameter combinations used for training and
## performs evaluation of the corresponding global models for the
## federated runs. For each entry in the parameter file where the
## strategy is "federated" or "fedit", it constructs the appropriate
## training prefix, derives the matching evaluation prefix (using "eval"
## in place of "train"), computes the global LoRA hyperparameters, and
## invokes the evaluation shell script (`eval_tinyllama_mtl_mlora.sh`).
##
## IMPORTANT:
## - We evaluate the last global aggregated checkpoint:
##   <train_dir>/checkpoints/ckpt_global_final.pt
## - We pass it via --load_ckpt.
##
## Parameter file format (non-comment lines):
##   strat num_fl_rounds num_B epochs [client_p] [dirichlet_alpha]
## - client_p defaults to 1.0 if omitted
## - dirichlet_alpha defaults to 10.0 if omitted
## - Naming must match training: ..._p{p_tag}_alpha{alpha}_newclienP

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

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

PARAM_FILE="${1:-param_combinations.txt}"
if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Parameter file ${PARAM_FILE} not found." >&2
  exit 1
fi

## Configuration for federated evaluations
NUM_CLIENTS=4
LOCAL_R=8
LORA_ALPHA=16

echo "Evaluating federated experiments from ${PARAM_FILE}..."
echo "Using local rank ${LOCAL_R}, clients ${NUM_CLIENTS}, alpha ${LORA_ALPHA}"

mkdir -p logs

# Now supports optional 5th field: client_p and optional 6th field: dirichlet_alpha
while read -r strat rounds num_B epochs client_p dirichlet_alpha; do
  [[ -z "${strat}" || "${strat}" =~ ^# ]] && continue

  if [[ "${strat}" != "federated" && "${strat}" != "fedit" ]]; then
    continue
  fi

  client_p=${client_p:-1.0}
  dirichlet_alpha=${dirichlet_alpha:-10.0}

  p_tag=$(echo "${client_p}" | sed 's/\./_/g')

  # Must match training naming (includes _alphaX.X)
  train_prefix="tinyllama_train_${strat}_epoch${epochs}_flround${rounds}_numB${num_B}_p${p_tag}_alpha${dirichlet_alpha}_4Clients_newclienP"
  eval_prefix="tinyllama_eval_${strat}_epoch${epochs}_flround${rounds}_numB${num_B}_p${p_tag}_alpha${dirichlet_alpha}_4Clients_newclienP"

  train_dir="${REPO_ROOT}/outputs/${train_prefix}"
  load_ckpt="${train_dir}/checkpoints/ckpt_global_final.pt"

  out_dir="${REPO_ROOT}/outputs/${eval_prefix}"
  logfile_out="${REPO_ROOT}/logs/${eval_prefix}.out"
  logfile_err="${REPO_ROOT}/logs/${eval_prefix}.err"

  if [[ ! -d "${train_dir}" ]]; then
    echo "[WARN] Training directory not found: ${train_dir} (skipping)" >&2
    continue
  fi
  if [[ ! -f "${load_ckpt}" ]]; then
    echo "[WARN] Global final checkpoint not found: ${load_ckpt} (skipping)" >&2
    continue
  fi

  echo "------------------------------------------"
  echo "Evaluating ${train_prefix} -> ${eval_prefix}"
  echo "Training dir: ${train_dir}"
  echo "Loading global checkpoint: ${load_ckpt}"
  echo "Writing evaluation results to ${out_dir}"
  echo "Capturing stdout to ${logfile_out} and stderr to ${logfile_err}"

  local_B="${num_B}"
  if [[ "${strat}" == "federated" ]]; then
    global_r=$((LOCAL_R * NUM_CLIENTS**rounds))
    global_B=$((num_B * NUM_CLIENTS**rounds))
  else
    # FEDIT case
    global_r=$((LOCAL_R ))
    global_B=$((num_B ))
  fi
  block_size="${num_B}"

  mkdir -p "${out_dir}"

  bash eval_tinyllama_mtl_mlora.sh "${train_dir}" "${out_dir}" \
    --load_ckpt "${load_ckpt}" \
    --lora_r "${global_r}" \
    --lora_alpha "${LORA_ALPHA}" \
    --num_B "${local_B}" \
    --global_num_B "${global_B}" \
    --block_size "${block_size}" \
    > "${logfile_out}" 2> "${logfile_err}"

  echo "Finished evaluation for ${train_prefix}."
done < "${PARAM_FILE}"