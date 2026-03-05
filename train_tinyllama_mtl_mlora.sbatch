#!/bin/bash

#SBATCH --job-name=tinyllama_mtl_auto
#SBATCH --output=logs/tinyllama_mtl_auto_job.out
#SBATCH --error=logs/tinyllama_mtl_auto_job.err
#SBATCH --ntasks=1
#SBATCH --partition=volta,ampere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00

#
# SLURM batch script to run multiple TinyLlama mLoRA training experiments
# sequentially within a single job. Parameter combinations are read from
# a text file (param_combinations.txt). For each line in the file, the
# training launcher script (train_tinyllama_mtl_mlora.sh) is invoked with
# overridden hyperparameters. Output directories and log files are
# named dynamically based on the current combination.
#
# Format per non-comment line (optionally with client_p as 5th field):
#   strat num_fl_rounds num_B epochs [client_p]
#
# Example:
#   centralized 1 3 2
#   federated   3 2 2 1.0
#
# NEW BEHAVIOR:
# - For strat == "federated" ONLY, we automatically enable linear-freeze growth:
#     --flora_growth linear_freeze
#   (optionally you can also pass --linear_add_r / --linear_add_B here if desired)

set -euo pipefail

# Ensure we are in the directory where this script was submitted.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

# Create directories for logs and outputs if necessary.
mkdir -p logs
mkdir -p outputs

# Path to the parameter list.
PARAM_FILE=${1:-param_combinations.txt}
if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Parameter file '${PARAM_FILE}' not found. Please create it with lines of 'strat fl_rounds num_B epochs [client_p]'." >&2
  exit 1
fi

echo "Reading parameter combinations from ${PARAM_FILE} ..."

# Iterate over each line of the parameter file. Optional 5th field: client_p.
while read -r strat fl_rounds num_B epochs client_p; do
  # Skip empty lines and comments.
  [[ -z "${strat}" || "${strat}" =~ ^# ]] && continue

  # Determine number of clients.
  clients=1
  if [[ "${strat}" == "federated" || "${strat}" == "fedit" ]]; then
    clients=4
  fi

  # Default client_p if none provided (ignored for centralized where clients=1).
  if [[ -z "${client_p:-}" ]]; then
    client_p="1.0"
  fi

  # Sanitize client_p for use in directory names (replace '.' with '_').
  p_tag=$(echo "${client_p}" | sed 's/\./_/g')

  # Unique experiment prefix (include p_tag).
  exp_prefix="tinyllama_train_${strat}_epoch${epochs}_flround${fl_rounds}_numB${num_B}_p${p_tag}_alpha10"
  outdir="outputs/${exp_prefix}_freezeR"
  logfile_out="logs/${exp_prefix}_freezeR.out"
  logfile_err="logs/${exp_prefix}_freezeR.err"

  mkdir -p "${outdir}" logs

  echo "Starting experiment: strat=${strat}, epochs=${epochs}, fl_rounds=${fl_rounds}, num_B=${num_B}, client_p=${client_p}" | tee -a "${logfile_out}"

  # Enable linear_freeze ONLY for federated runs.
  EXTRA_FL_FLAGS=()
  if [[ "${strat}" == "federated" ]]; then
    EXTRA_FL_FLAGS+=(--flora_growth linear_freeze)
    # Optional overrides (uncomment if you want them fixed here):
    # EXTRA_FL_FLAGS+=(--linear_add_r 8)
    # EXTRA_FL_FLAGS+=(--linear_add_B "${num_B}")
  fi

  # Launch training. Extra flags are appended only when enabled above.
  bash train_tinyllama_mtl_mlora.sh "${outdir}" \
       --epochs "${epochs}" \
       --num_B "${num_B}" \
       --num_fl_rounds "${fl_rounds}" \
       --num_clients "${clients}" \
       --strat "${strat}" \
       --client_p "${client_p}" \
       "${EXTRA_FL_FLAGS[@]}" \
       > "${logfile_out}" 2> "${logfile_err}"

  echo "Finished experiment: ${exp_prefix}" | tee -a "${logfile_out}"
  echo "Output saved to ${outdir}, logs: ${logfile_out}, ${logfile_err}" | tee -a "${logfile_out}"
done < "${PARAM_FILE}"

echo "All experiments completed."