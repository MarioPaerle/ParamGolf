#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_YENDRI
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:30:00

set -euo pipefail

cd /leonardo_work/IscrC_YENDRI/paerle/parameter-golf

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

export TRAIN_SCRIPT="${TRAIN_SCRIPT:-heuron/train_gpt_heron2.py}"
SCRIPT_BASENAME=$(basename "${TRAIN_SCRIPT}" .py)
export RUN_ID="${RUN_ID:-${SCRIPT_BASENAME}_4x}"

export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"

LOGFILE="./slurm/${RUN_ID}-$(date +%Y%m%d-%H%M%S).log"
echo "Logging to: ${LOGFILE}"

echo "--- BEGIN SOURCE CODE: ${TRAIN_SCRIPT} ---"
cat "${TRAIN_SCRIPT}"
echo "--- END SOURCE CODE: ${TRAIN_SCRIPT} ---"

srun uv run torchrun --standalone --nproc_per_node=4 "${TRAIN_SCRIPT}" 2>&1 | tee -a "${LOGFILE}"
