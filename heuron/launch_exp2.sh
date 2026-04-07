#!/bin/bash
# Launch Exp 2: MLP gate
cd /leonardo_work/IscrC_YENDRI/paerle/parameter-golf
TRAIN_SCRIPT=heuron/train_gpt_heron2.py RUN_ID=heuron_mlp_gate \
  salloc --partition=boost_usr_prod --account=IscrC_YENDRI --qos=boost_qos_dbg \
  --gres=gpu:4 --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 \
  --mem=120G --time=00:30:00 bash scripts/train_4xA100_interactive.sh
