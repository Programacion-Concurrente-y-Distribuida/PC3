#!/usr/bin/env bash
# Entrena y evalua el modelo con la mejor configuracion obtenida en
# scripts/worker_benchmark/results/ (fit-workers=28, ridge lambda=1, sin PCA).
# Imprime metricas de train y test (validacion temporal, anios > 2020).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

NUM_FEATURES='Latitude,Longitude,Year,Observation Count,Observation Percent,Valid Day Count,Required Day Count,Null Data Count,Num Obs Below MDL,Primary Exceedance Count,Secondary Exceedance Count'
CAT_FEATURES='pollutant,Sample Duration,State Code'

go run ./cmd/train-linear \
  -input aqs_final_3M.csv \
  -max-rows 0 \
  -train-year-end 2020 \
  -num-features "$NUM_FEATURES" \
  -cat-features "$CAT_FEATURES" \
  -solver ridge \
  -ridge-lambda 1 \
  -fit-workers 28 \
  -profile
