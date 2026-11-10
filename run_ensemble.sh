#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/out"
ENS_DIR="$OUT_DIR/ensemble"
mkdir -p "$ENS_DIR"

TRIALS=${1:-1}          # default = exact replication
BASE_OFFSET=${2:-10000} # spacing between trials

# Professor's original seeds
PROF_BUYSEED=2137
PROF_BIDSEED=3915
PROF_SEED=147

TEMPLATE="$ROOT_DIR/multauct.conf"

for ((t=0; t<TRIALS; t++)); do
   buyseed=$PROF_BUYSEED
   seed=$PROF_SEED
   bidseed=$PROF_BIDSEED

  echo "trial=$t buyseed=$buyseed bidseed=$bidseed seed=$seed" >&2

  # Write fixed-seed config ONCE for this trial
  awk -v b="$buyseed" -v d="$bidseed" -v s="$seed" '
    /^[[:space:]]*buyseed[[:space:]]*=/ { print "buyseed=" b; next }
    /^[[:space:]]*bidseed[[:space:]]*=/ { print "bidseed=" d; next }
    /^[[:space:]]*seed[[:space:]]*=/    { print "seed=" s; next }
    { print }
  ' "$TEMPLATE" > "$OUT_DIR/multauct_trial.conf"

  cp "$OUT_DIR/multauct_trial.conf" "$ROOT_DIR/multauct.conf"

  csv="$ENS_DIR/results_trial_${t}.csv"
  path=$(./run_sweep.sh 0 160 1 | tail -n 1)
  cp "$path" "$csv"

done

