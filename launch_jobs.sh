#!/bin/bash
# Dispatch simulator jobs over several GPUs: one run_policy_sim.py per GPU at a time, from a jobs file where each line is
#   <name> <args...>        (e.g. "white_ae150 --ckpt ckpt/my_ckpt --auto-exposure 150"; see examples/jobs_example.txt)
# Usage: ./launch_jobs.sh jobs.txt ["0 1 2 3"]     (GPU list; default: every GPU nvidia-smi reports)
# Env:   N (episodes per job, 30), DUR (seconds per episode, 45), PYTHON (.venv-policy/bin/python), OUT (runs/batch)
# Each job writes OUT/<name>/ (ep*/summary.json + run.npz, aggregate.json) and OUT/<name>.log. Sim only; never touches hardware.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
JOBS=$1
[ -f "$JOBS" ] || { echo "usage: ./launch_jobs.sh jobs.txt [\"gpu ids\"]"; exit 1; }
GPUS=${2:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
PYTHON=${PYTHON:-.venv-policy/bin/python}
OUT=${OUT:-runs/batch}
export PYTHON OUT
mkdir -p "$OUT"
run_job() { gpu=$1; shift; name=$1; shift
  echo "[$(date +%H:%M)] gpu$gpu start $name: $*"
  CUDA_VISIBLE_DEVICES=$gpu $PYTHON run_policy_sim.py --episodes ${N:-30} --vary 1 --duration ${DUR:-45} --dynamics mujoco_contact --out $OUT/$name "$@" > $OUT/$name.log 2>&1
  echo "[$(date +%H:%M)] gpu$gpu done  $name: $(grep -o '"success_rate": [0-9.]*' $OUT/$name.log | head -1)"
}
export -f run_job
# round-robin assign jobs to GPU slots, each slot processes its jobs sequentially; slots run in parallel
i=0; declare -A slot
for g in $GPUS; do slot[$g]=""; done
arr=($GPUS); n=${#arr[@]}
while IFS= read -r line; do [ -z "$line" ] && continue; [ "${line:0:1}" = "#" ] && continue; g=${arr[$((i % n))]}; slot[$g]+="$line"$'\n'; i=$((i+1)); done < "$JOBS"
for g in $GPUS; do
  ( while IFS= read -r line; do [ -z "$line" ] && continue; run_job $g $line; done <<< "${slot[$g]}" ) &
done
wait
echo "[$(date +%H:%M)] all jobs done"
