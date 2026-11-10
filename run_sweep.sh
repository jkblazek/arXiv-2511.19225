#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_sweep.sh [start end step]
# Example:
#   ./run_sweep.sh 0 160 10

START=${1:-0}
END=${2:-160}
STEP=${3:-10}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT_DIR/out"
CFG_DIR="$OUT_DIR/configs"
mkdir -p "$OUT_DIR" "$CFG_DIR"

BIN="$ROOT_DIR/multauct"
TEMPLATE="$ROOT_DIR/multauct.conf"

# Build once if needed
if [[ ! -x "$BIN" ]]; then
  ( cd "$ROOT_DIR" && make -s )
fi

# Read number of sellers
NSEL=$(grep -E '^[[:space:]]*nsel[[:space:]]*=' "$TEMPLATE" | cut -d= -f2 | tr -d '[:space:]')

RESULTS="$OUT_DIR/results_mauct.csv"
echo "conn,j,pavg,pstd,rtot,qres" > "$RESULTS"

TMP_OUT=$(mktemp)
trap 'rm -f "$TMP_OUT"' EXIT

for ((xp=START; xp<=END; xp+=STEP)); do
  cfg="$CFG_DIR/multauct_${xp}.conf"
  echo "xp=$xp mtype_line=$(grep -E '^[[:space:]]*mtype[[:space:]]*=' "$cfg" | tail -n 1)" >&2


  # ONLY rewrite mtype — leave seeds untouched
  awk -v xp="$xp" '
    BEGIN{done=0}
    /^[[:space:]]*mtype[[:space:]]*=/ {
        print "mtype=mauct:" xp
        done=1
        next
    }
    { print }
    END{
        if(!done) print "mtype=mauct:" xp
    }
  ' "$TEMPLATE" > "$cfg"

  # Run
  ( cd "$ROOT_DIR" && cp "$cfg" multauct.conf && ./multauct > "$TMP_OUT" )

  # Scrape final stats table
  total=$(wc -l < "$TMP_OUT")
  from_bottom=$(
    tac "$TMP_OUT" | grep -n -m1 -E '^[[:space:]]*#j[[:space:]]+pavg\b' | cut -d: -f1 || true
  )
  header=$(( total - from_bottom + 1 ))
  start=$(( header + 1 ))
  end=$(( start + NSEL - 1 ))

  sed -n "${start},${end}p" "$TMP_OUT" | \
    awk -v c="$xp" 'NF==5 {printf "%d,%d,%.15g,%.15g,%.15g,%.15g\n", c,$1,$2,$3,$4,$5}' \
    >> "$RESULTS"

  echo "xp=$xp done" >&2
done

# IMPORTANT: only print the CSV path
echo "$RESULTS"

