#!/bin/bash
# Interactive version of train_4xA100.sh — use with salloc:
#   salloc --partition=boost_usr_prod --account=IscrC_YENDRI --qos=boost_qos_dbg \
#     --gres=gpu:4 --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 \
#     --mem=120G --time=00:30:00 bash scripts/train_4xA100_interactive.sh

set -euo pipefail

cd /leonardo_work/IscrC_YENDRI/paerle/parameter-golf

module load python/3.11.7
module load cuda/12.2

mkdir -p slurm results

# Ensure ~/.local/bin is in PATH (needed for uv, both here and in srun)
export PATH="$HOME/.local/bin:$PATH"

# Load .env if present (must come before proxy config)
set -a; [ -f .env ] && source .env; set +a

# Squid proxy for internet access on compute nodes
if [ -n "${PROXY_PORT:-}" ]; then
  export http_proxy="http://login01:${PROXY_PORT}"
  export https_proxy="http://login01:${PROXY_PORT}"
fi

# Install uv if not found (must be AFTER proxy is configured)
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing to $HOME/.local/bin..."
    curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 BINDIR=$HOME/.local/bin sh
fi

# ---- Configuration ----
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# ---- Script Configuration ----
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_gpt.py}"
SCRIPT_BASENAME=$(basename "${TRAIN_SCRIPT}" .py)
export RUN_ID="${RUN_ID:-${SCRIPT_BASENAME}_4x}"

# Override wallclock (slower than 8xH100, so give more time)
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

# Periodic validation logging
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"

# ---- Log file ----
LOGFILE="./slurm/${RUN_ID}-$(date +%Y%m%d-%H%M%S).log"
echo "Logging to: ${LOGFILE}"

# ---- Launch training ----
# Prepend the code to the log
echo "--- BEGIN SOURCE CODE: ${TRAIN_SCRIPT} ---"
cat "${TRAIN_SCRIPT}"
echo "--- END SOURCE CODE: ${TRAIN_SCRIPT} ---"

srun uv run torchrun --standalone --nproc_per_node=4 "${TRAIN_SCRIPT}" 2>&1 | tee -a "${LOGFILE}"
