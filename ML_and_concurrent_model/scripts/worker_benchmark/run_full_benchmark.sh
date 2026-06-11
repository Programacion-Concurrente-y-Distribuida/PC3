#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/.mplconfig"
export MPLCONFIGDIR="$REPO_ROOT/.mplconfig"

python3 scripts/worker_benchmark/benchmark_workers.py \
  --repo-root "$REPO_ROOT" \
  --input aqs_final_3M.csv \
  --train-year-end 2020 \
  --max-rows 0 \
  --num-features 'Latitude,Longitude,Year,Observation Count,Observation Percent,Valid Day Count,Required Day Count,Exceptional Data Count,Null Data Count,Num Obs Below MDL,Primary Exceedance Count,Secondary Exceedance Count' \
  --cat-features 'pollutant,Sample Duration,State Code' \
  --parser-workers '2,4,6,8,10,12' \
  --encoder-workers '2,4,6,8,10' \
  --solver ridge \
  --fit-workers 4 \
  --ridge-lambda 1
