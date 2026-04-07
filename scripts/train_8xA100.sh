#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/paerle/parameter-golf
#SBATCH --job-name=pgolf-train-8x
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=120G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
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
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# ---- Script Configuration ----
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_gpt.py}"
SCRIPT_BASENAME=$(basename "${TRAIN_SCRIPT}" .py)
export RUN_ID="${RUN_ID:-${SCRIPT_BASENAME}_8x}"

# 8 GPUs matches the canonical challenge setup (8xH100)
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"

# Periodic validation logging
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"

# ---- Multi-node rendezvous via SLURM ----
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# ---- Launch training ----
echo "--- BEGIN SOURCE CODE: ${TRAIN_SCRIPT} ---"
cat "${TRAIN_SCRIPT}"
echo "--- END SOURCE CODE: ${TRAIN_SCRIPT} ---"

# 2 nodes x 4 GPUs = 8 GPUs total
srun uv run torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id="${SLURM_JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    "${TRAIN_SCRIPT}"

#  TRAIN_SCRIPT=train_gpt_pe_r2.py sbatch scripts/train_8xA100.sh
