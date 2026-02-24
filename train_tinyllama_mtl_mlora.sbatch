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
#SBATCH --time=48:00:00

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

# Iterate over each line of the parameter file.
while read -r strat fl_rounds num_B epochs; do
  # Skip empty lines and comments (lines starting with '#').
  [[ -z "${strat}" || "${strat}" =~ ^# ]] && continue

  # Determine the number of clients:
  # federated runs use 2 clients, centralized runs remain at 1.  We
  # explicitly pass this to the training script to override its default.
  clients=1
  if [[ "${strat}" == "federated" ]]; then
    clients=2
  fi

  # Derive a unique experiment prefix based on the current parameters.
  exp_prefix="tinyllama_train_${strat}_epoch${epochs}_flround${fl_rounds}_numB${num_B}"
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
  bash train_tinyllama_mtl_mlora.sh "${outdir}" \
       --epochs "${epochs}" \
       --num_B "${num_B}" \
       --num_fl_rounds "${fl_rounds}" \
       --num_clients "${clients}" \
       --strat "${strat}" \
       > "${logfile_out}" 2> "${logfile_err}"

  echo "Finished experiment: ${exp_prefix}" | tee -a "${logfile_out}"
  echo "Output saved to ${outdir}, logs: ${logfile_out}, ${logfile_err}" | tee -a "${logfile_out}"
done < "${PARAM_FILE}"

echo "All experiments completed."