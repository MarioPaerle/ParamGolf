import os
import subprocess
import time
import re
import random
import itertools

NUM_CONCURRENT = 4
MAX_EXPERIMENTS = 30
SCRIPT = "train_gpt_skips_agents.py"
OUT_DIR = "slurm"
RESULT_FILE = "tested and result.md"

grid = {
    "NUM_PROJECTIONS": [3, 4, 5],
    "MATRIX_LR": ["0.025", "0.030", "0.035"],
    "SCALAR_LR": ["0.025", "0.030"],
    "TIED_EMBED_LR": ["0.035", "0.045"],
}

keys, values = zip(*grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
random.shuffle(experiments)
experiments = experiments[:MAX_EXPERIMENTS]

def submit_job(env_vars, run_idx):
    run_id = f"skips_{run_idx}"
    export_str = ",".join([f"{k}={v}" for k, v in env_vars.items()])
    export_str += f",TRAIN_SCRIPT={SCRIPT},RUN_ID={run_id}"

    cmd = ["sbatch", f"--export=ALL,{export_str}", "scripts/train_4xA100.sh"]
    print(f"Submitting: {cmd}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = res.stdout.strip()
    match = re.search(r"Submitted batch job (\d+)", out)
    if match:
        jobid = match.group(1)
        print(f"-> Submitted Job ID: {jobid}")
        return jobid
    else:
        print(f"Failed to submit: {res.stderr}")
        return None

def get_active_jobs():
    res = subprocess.run(["squeue", "--me", "-h", "-o", "%i %t"], stdout=subprocess.PIPE, text=True)
    jobs = []
    for line in res.stdout.strip().split("\n"):
        if not line: continue
        parts = line.split()
        jobs.append(parts[0])
    return jobs

def try_parse_result(jobid):
    path = f"slurm/pgolf-train-4x-{jobid}.out"
    if not os.path.exists(path):
        return None, None
    
    with open(path, "r", errors="ignore") as f:
        content = f.read()
    code_end = content.find("END SOURCE CODE")
    if code_end != -1:
        runtime_log = content[code_end:]
    else:
        runtime_log = content

    # Check if finished
    if "DIAGNOSTIC post_ema" not in runtime_log and "Error" not in runtime_log and "Traceback" not in runtime_log and "OOM" not in runtime_log:
        # Check if slurm killed it maybe
        if "CANCELLED" in runtime_log or "DUE TO TIME LIMIT" in runtime_log:
            return "KILLED", None
        return None, None # Still running or partially complete without error
    
    if "Error" in runtime_log or "Traceback" in runtime_log or "OOM" in runtime_log or "CUDA out of memory" in runtime_log:
        return "FAILED", None

    # Parse bpb out
    # Format: DIAGNOSTIC post_ema val_loss:X val_bpb:1.1234
    match = re.search(r"DIAGNOSTIC post_ema val_loss:([0-9\.]+) val_bpb:([0-9\.]+)", runtime_log)
    if match:
        return "COMPLETED", (float(match.group(1)), float(match.group(2)))
    
    return "UNKNOWN", None

def log_result(jobid, run_id, env_vars, status, val_loss, val_bpb):
    config_str = ", ".join([f"{k}={v}" for k,v in env_vars.items()])
    if val_loss is None: val_loss = "-"
    if val_bpb is None: val_bpb = "-"
    
    line = f"| {run_id} | {jobid} | `{config_str}` | {status} | {val_loss} | {val_bpb} |\n"
    with open(RESULT_FILE, "a") as f:
        f.write(line)

def run_loop():
    active_tracking = {}
    pending_experiments = list(enumerate(experiments))
    
    while pending_experiments or active_tracking:
        current_jobs = get_active_jobs()
        
        # Check tracking
        for jobid in list(active_tracking.keys()):
            if jobid not in current_jobs:
                # Job is done, parse out
                time.sleep(5) # Wait for slurm logs to flush
                status, metrics = try_parse_result(jobid)
                if status is None:
                    status = "FAILED (No End Log)"
                
                loss, bpb = metrics if metrics else (None, None)
                log_result(jobid, active_tracking[jobid]["run_id"], active_tracking[jobid]["env"], status, loss, bpb)
                print(f"Job {jobid} completed with status {status} bpb {bpb}")
                del active_tracking[jobid]
        
        # Submit new if capacity
        while len(active_tracking) < NUM_CONCURRENT and pending_experiments:
            exp_idx, exp_env = pending_experiments.pop(0)
            jobid = submit_job(exp_env, exp_idx)
            if jobid:
                active_tracking[jobid] = {"run_id": f"skips_{exp_idx}", "env": exp_env}
            else:
                # Retry maybe later? Or just fail
                pending_experiments.insert(0, (exp_idx, exp_env))
                time.sleep(30)
                break
                
        time.sleep(60)

if __name__ == "__main__":
    run_loop()
