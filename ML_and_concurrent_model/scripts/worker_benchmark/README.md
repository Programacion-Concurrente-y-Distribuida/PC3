# Benchmark de Workers

Esta carpeta contiene scripts para encontrar configuraciones optimas de concurrencia en `train-linear`.

## Archivos

- `benchmark_workers.py`: ejecuta una grilla de `(parser_workers, encoder_workers)` usando el solver actual (`ridge` por defecto), mide tiempo y RAM pico, y genera graficos.
- `benchmark_fit_workers.py`: evalua `fit-workers`, que es la concurrencia principal del entrenamiento Ridge.
- `run_full_benchmark.sh`: benchmark de parser/encoder sobre dataset completo.
- `run_fit_benchmark.sh`: benchmark de `fit-workers` sobre dataset completo.
- `results/`: salidas (`.csv`, `.png`, logs por corrida).

## Requisitos

- Python 3
- `matplotlib` y `numpy`

Instalacion:

```bash
pip3 install matplotlib numpy
```

## Ejecucion rapida

```bash
bash scripts/worker_benchmark/run_full_benchmark.sh
```

Benchmark del entrenamiento concurrente:

```bash
bash scripts/worker_benchmark/run_fit_benchmark.sh
```

## Salidas esperadas

- `worker_benchmark.csv`: tabla con tiempo, RAM pico y metricas por combinacion.
- `time_heatmap.png`: mapa de calor de tiempo (menor es mejor).
- `ram_heatmap.png`: mapa de calor de RAM pico (menor es mejor).
- `pareto_time_vs_ram.png`: dispersion tiempo vs RAM para elegir compromiso.
- `fit_workers_benchmark_raw.csv`: todas las repeticiones crudas por `fit-workers`.
- `fit_workers_benchmark.csv`: resumen por `fit-workers` con promedio, mediana, desviacion estandar y minimo.
- `fit_workers_time_ram.png`: grafico de tiempo total, tiempo de entrenamiento y RAM pico promedio vs `fit-workers`.

## Como leer el grafico final

- **`time_heatmap.png`**:
  - eje Y: parser workers
  - eje X: encoder workers
  - color/numero de celda: segundos totales
  - mejor zona: celdas con menor valor

- **`ram_heatmap.png`**:
  - mismo layout, pero con MB de RAM pico
  - mejor zona: menor RAM sin degradar tiempo de forma extrema

- **`pareto_time_vs_ram.png`**:
  - cada punto es una combinacion `P#-E#`
  - eje X: tiempo
  - eje Y: RAM
  - las combinaciones ideales quedan en la esquina inferior izquierda

- **`fit_workers_time_ram.png`**:
  - eje X: `fit-workers`
  - eje Y izquierdo: segundos promedio (`tiempo total` y `fit_linear_regression`)
  - eje Y derecho: RAM pico promedio (MB)
  - barras de error: desviacion estandar entre repeticiones
  - la mejor configuracion es donde el tiempo deja de bajar sin aumentar RAM innecesariamente
