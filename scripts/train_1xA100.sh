#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/paerle/parameter-golf
#SBATCH --job-name=pgolf-train
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_YENDRI

set -euo pipefail

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
export RUN_ID="${RUN_ID:-baseline_sp1024}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Override wallclock for single-GPU (slower than 8xH100, so give more time)
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

# Periodic validation logging
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"

# ---- Launch training ----
srun uv run torchrun --standalone --nproc_per_node=1 train_gpt.py
