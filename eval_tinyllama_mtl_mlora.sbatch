#!/bin/bash
##
## Automatic evaluation of TinyLlama mLoRA experiments.
##
## This Slurm batch script mirrors the training automation:
## - reads parameter combinations from the same param file (default: param_combinations.txt)
## - derives the *exact* training output directory naming
## - evaluates the FINAL GLOBAL aggregated model that lives inside each training output dir:
##     * outputs/<train_prefix>/checkpoints/ckpt_global_final.pt   (preferred)
##       OR
##     * outputs/<train_prefix>/adapter_state_final.pt + heads_state_final.pt
##
## Evaluation results are written to:
##   outputs/<eval_prefix>
## where eval_prefix mirrors train_prefix but uses "tinyllama_eval_..." instead of "tinyllama_train_...".
##
## The actual model hyperparameters for evaluation are auto-resolved from
## outputs/<train_prefix>/run_config.json by the Python eval script, so you
## don't have to keep global-rank formulas in sync manually.
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
#SBATCH --time=0-06:00:00

set -euo pipefail

# Repository root (submission directory)
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

mkdir -p logs outputs

# Parameter file (same format as training auto script):
#   strat num_fl_rounds num_B epochs [client_p]
PARAM_FILE="${1:-param_combinations.txt}"
if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Parameter file '${PARAM_FILE}' not found." >&2
  exit 1
fi

echo "Reading parameter combinations from ${PARAM_FILE} ..."
echo "Evaluations will load FINAL GLOBAL models from outputs/<tinyllama_train_...>"

while read -r strat fl_rounds num_B epochs client_p; do
  # Skip empty lines and comments
  [[ -z "${strat}" || "${strat}" =~ ^# ]] && continue

  # Default client_p if omitted
  client_p="${client_p:-1.0}"
  p_tag=$(echo "${client_p}" | sed 's/\./_/g')

  # Must match training auto naming exactly
  train_prefix="tinyllama_train_${strat}_epoch${epochs}_flround${fl_rounds}_numB${num_B}_p${p_tag}"
  eval_prefix="tinyllama_eval_${strat}_epoch${epochs}_flround${fl_rounds}_numB${num_B}_p${p_tag}"

  train_dir="${REPO_ROOT}/outputs/${train_prefix}"
  out_dir="${REPO_ROOT}/outputs/${eval_prefix}"
  logfile_out="${REPO_ROOT}/logs/${eval_prefix}.out"
  logfile_err="${REPO_ROOT}/logs/${eval_prefix}.err"

  echo "------------------------------------------"
  echo "Evaluating: ${train_prefix} -> ${eval_prefix}"
  echo "Train dir:  ${train_dir}"
  echo "Out dir:    ${out_dir}"
  echo "Logs:       ${logfile_out} / ${logfile_err}"

  if [[ ! -d "${train_dir}" ]]; then
    echo "[SKIP] Training output directory not found: ${train_dir}" | tee -a "${logfile_err}"
    continue
  fi

  # Require a clearly global final artifact to avoid accidentally evaluating a client-local state.
  ckpt_global_final="${train_dir}/checkpoints/ckpt_global_final.pt"
  adapter_final="${train_dir}/adapter_state_final.pt"
  heads_final="${train_dir}/heads_state_final.pt"

  if [[ -f "${ckpt_global_final}" ]]; then
    :
  elif [[ -f "${adapter_final}" && -f "${heads_final}" ]]; then
    :
  else
    echo "[SKIP] No final GLOBAL model found in ${train_dir}." | tee -a "${logfile_err}"
    echo "       Expected either:" | tee -a "${logfile_err}"
    echo "         - ${ckpt_global_final}" | tee -a "${logfile_err}"
    echo "         - ${adapter_final} AND ${heads_final}" | tee -a "${logfile_err}"
    continue
  fi

  # We rely on run_config.json for auto-resolving the correct global LoRA hyperparameters.
  run_cfg="${train_dir}/run_config.json"
  if [[ ! -f "${run_cfg}" ]]; then
    echo "[SKIP] Missing run_config.json in ${train_dir} (needed to auto-resolve eval hyperparameters)." | tee -a "${logfile_err}"
    continue
  fi

  mkdir -p "${out_dir}"

  # Run eval (inside container) and capture output to per-run logs
  bash eval_tinyllama_mtl_mlora.sh "${train_dir}" "${out_dir}" \
    > "${logfile_out}" 2> "${logfile_err}"

  echo "Finished evaluation: ${eval_prefix}"
done < "${PARAM_FILE}"

echo "All evaluations completed."