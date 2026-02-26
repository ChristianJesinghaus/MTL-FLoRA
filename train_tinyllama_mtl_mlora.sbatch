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
#SBATCH --time=14:00:00

#
# SLURM batch script to run multiple TinyLlama mLoRA training experiments
# sequentially within a single job.  Parameter combinations are read from
# a text file (param_combinations.txt).  For each line in the file, the
# training launcher script (train_tinyllama_mtl_mlora.sh) is invoked with
# overridden hyperparameters.  Output directories and log files are
# named dynamically based on the current combination.
#
# The expected format of each non‑comment, non‑empty line in
# param_combinations.txt is:
#   strat num_fl_rounds num_B epochs
# For example:
#   centralized 1 3 2
#   federated 2 2 3

set -euo pipefail

# Ensure we are in the directory where this script was submitted.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

# Create directories for logs and outputs if necessary.
mkdir -p logs
mkdir -p outputs

# Path to the parameter list.  You can override via SBATCH argument
# or environment variable PARAM_FILE.
PARAM_FILE=${1:-param_combinations.txt}
if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Parameter file '${PARAM_FILE}' not found. Please create it with lines of 'strat fl_rounds num_B epochs'." >&2
  exit 1
fi

echo "Reading parameter combinations from ${PARAM_FILE} ..."

# Iterate over each line of the parameter file.  The expected format now includes an optional
# fifth field specifying the client_p weight.  For example:
#   federated 3 2 2 1.0
# If the client_p field is missing or empty, it defaults to 1.0 (equal weighting).
while read -r strat fl_rounds num_B epochs client_p; do
  # Skip empty lines and comments (lines starting with '#').
  [[ -z "${strat}" || "${strat}" =~ ^# ]] && continue

  # Determine the number of clients.  Centralized experiments run with
  # a single client.  Both federated and FEDIT experiments use two
  # clients by default, since FEDIT is a variant of federated learning
  # where all LoRA adapters are jointly tuned.  If your training
  # launcher allows a different number of clients for FEDIT, adjust
  # this logic accordingly.
  clients=1
  if [[ "${strat}" == "federated" || "${strat}" == "fedit" ]]; then
    clients=2
  fi

  # Default client_p if none provided.  For centralized runs (clients=1) this is ignored.
  if [[ -z "${client_p}" ]]; then
    client_p="1.0"
  fi

  # Sanitize client_p for use in directory names (replace '.' with '_').
  p_tag=$(echo "${client_p}" | sed 's/\./_/g')

  # Derive a unique experiment prefix based on the current parameters.  Include
  # the p_tag to distinguish runs with different client_p values.
  exp_prefix="tinyllama_train_${strat}_epoch${epochs}_flround${fl_rounds}_numB${num_B}_p${p_tag}"
  outdir="outputs/${exp_prefix}"
  logfile_out="logs/${exp_prefix}.out"
  logfile_err="logs/${exp_prefix}.err"

  # Create the experiment output directory and log files directories.
  mkdir -p "${outdir}" logs

  echo "Starting experiment: strat=${strat}, epochs=${epochs}, fl_rounds=${fl_rounds}, num_B=${num_B}" | tee -a "${logfile_out}"

  # Invoke the top‑level training launcher.  We pass extra flags to override
  # the defaults defined in train_tinyllama_mtl_mlora.sh.  The training
  # output is redirected into per‑run log files.
  #
  # Note: --epochs, --num_B and --num_fl_rounds override the defaults.  For
  # federated runs we also set --strat to 'federated'; for centralized
  # runs we explicitly set --strat to 'centralized' to override any default.
  # The --client_p argument overrides the automatic client weighting in the
  # Python training script (if provided).
  bash train_tinyllama_mtl_mlora.sh "${outdir}" \
       --epochs "${epochs}" \
       --num_B "${num_B}" \
       --num_fl_rounds "${fl_rounds}" \
       --num_clients "${clients}" \
       --strat "${strat}" \
       --client_p "${client_p}" \
       > "${logfile_out}" 2> "${logfile_err}"

  echo "Finished experiment: ${exp_prefix}" | tee -a "${logfile_out}"
  echo "Output saved to ${outdir}, logs: ${logfile_out}, ${logfile_err}" | tee -a "${logfile_out}"
done < "${PARAM_FILE}"

echo "All experiments completed."