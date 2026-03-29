# Program Log for GPT Skips Agents Training

## Objective
Beat the validation baseline of 1.26 bpb on 4xA100 by performing an automated hyperparameter search on the `train_gpt_skips_agents.py` script.
We start with `NUM_PROJECTIONS=3` and will explore up to 30 experiments.

## Constraints
- **Concurrency**: Up to 4 jobs at a time.
- **Hardware**: 4xA100 per job (`train_4xA100.sh`).
- **Duration**: 15-19 minutes.
- **Goal**: Minimize `val_bpb` < 1.26 bpb.

## Current Phase: Loop Setup
We have created `train_gpt_skips_agents.py` that supports `NUM_PROJECTIONS`. We will test it over ~30 combinations using a background python orchestrator.
