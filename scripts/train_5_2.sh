#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/paerle/parameter-golf
#SBATCH --job-name=heuron_5_2_xsi_lw
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --account=IscrC_YENDRI

set -euo pipefail

module load python/3.11.7
module load cuda/12.2

mkdir -p slurm results

export PATH="$HOME/.local/bin:$PATH"

set -a; [ -f .env ] && source .env; set +a

if [ -n "${PROXY_PORT:-}" ]; then
  export http_proxy="http://login01:${PROXY_PORT}"
  export https_proxy="http://login01:${PROXY_PORT}"
fi

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | INSTALLER_NO_MODIFY_PATH=1 BINDIR=$HOME/.local/bin sh
fi

export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

export TRAIN_SCRIPT=heuron/train_gpt_heuron_5_2.py
export RUN_ID=heuron_5_2_xsi_layerwise

export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"

echo "--- BEGIN SOURCE CODE: ${TRAIN_SCRIPT} ---"
cat "${TRAIN_SCRIPT}"
echo "--- END SOURCE CODE: ${TRAIN_SCRIPT} ---"

srun uv run torchrun --standalone --nproc_per_node=4 "${TRAIN_SCRIPT}"
