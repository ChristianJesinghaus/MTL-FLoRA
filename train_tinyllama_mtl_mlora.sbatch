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
#SBATCH --time=10:00:00

#
# SLURM batch script to run multiple TinyLlama mLoRA training experiments
# sequentially within a single job.
#
# Parameter combinations are read from a text file (param_combinations.txt).
# For each line in the file, the training launcher script
# (train_tinyllama_mtl_mlora.sh) is invoked with overridden hyperparameters.
#
# Expected format of each non-comment, non-empty line in param_combinations.txt:
#   strat num_fl_rounds num_B epochs [client_p] [dirichlet_alpha]
#
# Examples:
#   centralized 1 3 2
#   federated   3 3 2 0.05 10.0
#
# Notes:
# - client_p defaults to 1.0 if omitted
# - dirichlet_alpha defaults to 10.0 if omitted
# - Naming includes both p and alpha: ..._p{p_tag}_alpha{alpha}...

set -euo pipefail

# Ensure we are in the directory where this script was submitted.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

# Create directories for logs and outputs if necessary.
mkdir -p logs
mkdir -p outputs

# Path to the parameter list. You can override via SBATCH argument
# or environment variable PARAM_FILE.
PARAM_FILE=${1:-param_combinations.txt}
if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Parameter file '${PARAM_FILE}' not found. Please create it with lines of 'strat fl_rounds num_B epochs [client_p] [dirichlet_alpha]'." >&2
  exit 1
fi

echo "Reading parameter combinations from ${PARAM_FILE} ..."

# Now supports optional 5th field: client_p and optional 6th field: dirichlet_alpha
while read -r strat fl_rounds num_B epochs client_p dirichlet_alpha; do
  # Skip empty lines and comments (lines starting with '#').
  [[ -z "${strat}" || "${strat}" =~ ^# ]] && continue

  # Determine the number of clients.
  clients=1
  if [[ "${strat}" == "federated" || "${strat}" == "fedit" ]]; then
    clients=4
  fi

  # Defaults if not provided.
  if [[ -z "${client_p}" ]]; then
    client_p="1.0"
  fi
  if [[ -z "${dirichlet_alpha}" ]]; then
    dirichlet_alpha="10.0"
  fi

  # Sanitize client_p for use in directory names (replace '.' with '_').
  p_tag=$(echo "${client_p}" | sed 's/\./_/g')

  # Include dirichlet alpha in naming EXACTLY as requested: _alphaX.X
  exp_prefix="tinyllama_train_${strat}_epoch${epochs}_flround${fl_rounds}_numB${num_B}_p${p_tag}_alpha${dirichlet_alpha}_4Clients"
  outdir="outputs/${exp_prefix}_newclienP"
  logfile_out="logs/${exp_prefix}_newclienP.out"
  logfile_err="logs/${exp_prefix}_newclienP.err"

  mkdir -p "${outdir}" logs

  echo "Starting experiment: strat=${strat}, epochs=${epochs}, fl_rounds=${fl_rounds}, num_B=${num_B}, client_p=${client_p}, dirichlet_alpha=${dirichlet_alpha}" \
    | tee -a "${logfile_out}"

  # Invoke the top-level training launcher with overrides, including --dirichlet_alpha.
  bash train_tinyllama_mtl_mlora.sh "${outdir}" \
       --epochs "${epochs}" \
       --num_B "${num_B}" \
       --num_fl_rounds "${fl_rounds}" \
       --num_clients "${clients}" \
       --strat "${strat}" \
       --client_p "${client_p}" \
       --dirichlet_alpha "${dirichlet_alpha}" \
       > "${logfile_out}" 2> "${logfile_err}"

  echo "Finished experiment: ${exp_prefix}" | tee -a "${logfile_out}"
  echo "Output saved to ${outdir}, logs: ${logfile_out}, ${logfile_err}" | tee -a "${logfile_out}"
done < "${PARAM_FILE}"

echo "All experiments completed."