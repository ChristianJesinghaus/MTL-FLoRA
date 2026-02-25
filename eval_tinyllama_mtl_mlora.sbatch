#!/bin/bash
##
## Automatic evaluation of TinyLlama mLoRA federated experiments.
##
## This Slurm job reads the parameter combinations used for training and
## performs evaluation of the corresponding global models for the
## federated runs.  For each entry in the parameter file where the
## strategy is "federated", it constructs the appropriate training
## prefix, derives the matching evaluation prefix (using "eval" in
## place of "train"), computes the global LoRA hyperparameters based
## on the local settings and number of clients, and then invokes the
## existing evaluation shell script (`eval_tinyllama_mtl_mlora.sh`).  The
## evaluation outputs and logs are stored in directories and files
## whose names mirror those used during training, suffixed with
## "eval" for clarity.

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

# The directory from which the job was submitted.  All relative paths
# are resolved relative to this repository root.
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "${REPO_ROOT}"

# Parameter combinations file.  If a filename is provided as the first
# positional argument, use it; otherwise fall back to the default
# param_combinations.txt in the repository root.
PARAM_FILE="${1:-param_combinations.txt}"

# Verify that the parameter file exists.
if [[ ! -f "${PARAM_FILE}" ]]; then
  echo "Parameter file ${PARAM_FILE} not found." >&2
  exit 1
fi

## Configuration for federated evaluations
# Number of clients in federated experiments.  The global LoRA rank
# equals LOCAL_R * NUM_CLIENTS, and the global number of B matrices
# equals num_B * NUM_CLIENTS.
NUM_CLIENTS=2
# Local LoRA rank used during training.  If you change the rank in
# training, adjust this value accordingly.
LOCAL_R=8
# LoRA alpha hyperparameter (assumed constant across runs).  If you
# modify this in training, adjust here as well.
LORA_ALPHA=16

echo "Evaluating federated experiments from ${PARAM_FILE}..."
echo "Using local rank ${LOCAL_R}, clients ${NUM_CLIENTS}, alpha ${LORA_ALPHA}"

# Ensure the logs directory exists.  Individual evaluations will
# allocate their own output directories under outputs/ and log files
# under logs/.
mkdir -p logs

# Iterate over each line in the parameter file.  Lines beginning with
# '#' or blank lines are ignored.  Each non-comment line is expected
# to have five whitespace-separated fields: strat num_fl_rounds num_B
# epochs client_p.  If the last field is omitted, a default of 1.0 is assumed.
while read -r strat rounds num_B epochs client_p; do
  # Skip empty lines and comments.
  [[ -z "${strat}" || "${strat}" =~ ^# ]] && continue

  # Only handle federated and FEDIT runs; centralized runs have no
  # global model separate from the local one and are therefore not
  # evaluated here.  For FEDIT (where existing adapters are further
  # trained instead of freezing them), the global rank does not grow
  # with the number of rounds.
  if [[ "${strat}" != "federated" && "${strat}" != "fedit" ]]; then
    continue
  fi

  # Default client_p to 1.0 if not provided.  Sanitize for use in path names by
  # replacing decimal points with underscores.  This tag must match the
  # corresponding suffix used during training so that the evaluation picks up
  # the correct model directory.  When the parameter file includes a
  # fifth column, it specifies the client_p value used for training; otherwise
  # we assume p=1.0 (uniform weighting).
  client_p=${client_p:-1.0}
  p_tag=$(echo "${client_p}" | sed 's/\./_/g')

  # Build the training and evaluation prefixes.  The training prefix
  # matches the naming used in train_tinyllama_mtl_mlora_auto.sbatch,
  # including the p_tag suffix.  The evaluation prefix mirrors this but
  # uses "eval" instead of "train" to differentiate outputs.
  train_prefix="tinyllama_train_${strat}_epoch${epochs}_flround${rounds}_numB${num_B}_p${p_tag}"
  eval_prefix="tinyllama_eval_${strat}_epoch${epochs}_flround${rounds}_numB${num_B}_p${p_tag}"

  # Determine paths for the model checkpoint to evaluate, the output
  # directory for evaluation results, and the log files capturing
  # stdout and stderr.  The evaluation script expects the global
  # checkpoint to reside in a 'checkpoints' subdirectory inside the
  # training output directory.  If your training script stores the
  # best checkpoint elsewhere, adjust the load_dir accordingly.
  load_dir="${REPO_ROOT}/outputs/${train_prefix}/checkpoints/ckpt_best"
  out_dir="${REPO_ROOT}/outputs/${eval_prefix}"
  logfile_out="${REPO_ROOT}/logs/${eval_prefix}.out"
  logfile_err="${REPO_ROOT}/logs/${eval_prefix}.err"

  # Report progress on the job's stderr/stdout.  These messages
  # provide a high-level overview of what the script is doing but do
  # not include the full evaluation output, which goes to the
  # per-run log files.
  echo "------------------------------------------"
  echo "Evaluating ${train_prefix} -> ${eval_prefix}"
  echo "Loading model from ${load_dir}"
  echo "Writing evaluation results to ${out_dir}"
  echo "Capturing stdout to ${logfile_out} and stderr to ${logfile_err}"

  # Compute the global LoRA hyperparameters.  For standard federated
  # runs, each round introduces a new set of LoRA weights per client,
  # so the global rank and number of B matrices grow linearly with
  # rounds (LOCAL_R * NUM_CLIENTS * rounds).  In FEDIT, all LoRA
  # adapters are jointly trained across rounds, so the rank does not
  # accumulate across rounds – it stays LOCAL_R * NUM_CLIENTS.  In
  # both cases the block size equals the local B count.
  local_B="${num_B}"
  if [[ "${strat}" == "federated" ]]; then
    global_r=$((LOCAL_R * NUM_CLIENTS * rounds))
    global_B=$((num_B * NUM_CLIENTS * rounds))
  else
    # FEDIT case
    global_r=$((LOCAL_R * NUM_CLIENTS))
    global_B=$((num_B * NUM_CLIENTS))
  fi
  block_size="${num_B}"

  # Ensure the evaluation output directory exists.
  mkdir -p "${out_dir}"

  # Launch the evaluation script.  We explicitly pass the computed
  # hyperparameters so that the evaluation uses the correct global
  # settings.  The evaluation output and errors are redirected to the
  # per-run log files for later inspection.  Additional flags can
  # easily be appended here.
  bash eval_tinyllama_mtl_mlora.sh "${load_dir}" "${out_dir}" \
    --lora_r "${global_r}" \
    --lora_alpha "${LORA_ALPHA}" \
    --num_B "${local_B}" \
    --global_num_B "${global_B}" \
    --block_size "${block_size}" \
    > "${logfile_out}" 2> "${logfile_err}"

  echo "Finished evaluation for ${train_prefix}."
done < "${PARAM_FILE}"