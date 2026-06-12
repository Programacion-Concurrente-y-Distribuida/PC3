#!/usr/bin/env bash
# Entrena sobre el dataset completo con 28 goroutines en el fit y muestra
# solo el resumen: filas validas cargadas, split train/test y metricas.
set -euo pipefail

cd "$(dirname "$0")/.."

go run ./cmd/train-linear -max-rows 0 -fit-workers 28 2>&1 |
  grep -E "Filas cargadas|Train rows|--- Metricas|Train +->|Test +->"
