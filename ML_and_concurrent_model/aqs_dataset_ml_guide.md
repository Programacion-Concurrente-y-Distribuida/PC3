# Guia ML en Go para `aqs_clean.csv`

Documento actualizado para un enfoque de implementacion en Go (concurrencia como plus, no como prioridad inicial).

## 1) Prioridades del proyecto

Orden recomendado:

1. Elegir librerias en Go.
2. Definir modelo base (regresion lineal).
3. Construir pipeline de lectura/procesamiento CSV.
4. Validar distribuciones y calidad de variables.
5. Probar PCA y comparar contra baseline.
6. Reci en una segunda fase, optimizar con concurrencia.

## 2) Librerias recomendadas (Go)

### Stack minimo recomendado

- `gonum` (`gonum.org/v1/gonum`): algebra lineal, estadistica, operaciones matriciales.
- `gota` (`github.com/go-gota/gota`): manipulacion tipo DataFrame y lectura de CSV.

Con solo estas dos se puede construir un baseline serio.

### Alternativa sin DataFrame

- `encoding/csv` (stdlib) + structs propios + `gonum`.

Es mas verboso, pero te da control total y facilita paralelizar lectura/procesamiento despues.

## 3) Modelo base recomendado

Para arrancar en Go, lo mas simple y defendible es:

- **Regresion lineal multiple** con target `Arithmetic Mean`.

Razon:

- Es facil de implementar y depurar.
- Da interpretabilidad inicial.
- Sirve como baseline para comparar mejoras (PCA u otros modelos).

## 4) Variables: cuales usar primero

### Aclaracion de columnas clave

En tu CSV existen ambas:

- `pollutant` (etiqueta corta: O3, PM2.5, NO2, CO)
- `Pollutant Standard` (estandar regulatorio textual)

No son equivalentes.

### Set inicial de features (v1)

Features numericas:

- `Latitude`, `Longitude`, `Year`
- `Observation Count`, `Observation Percent`
- `Valid Day Count`, `Required Day Count`
- `Exceptional Data Count`, `Null Data Count`
- `Num Obs Below MDL`
- `Primary Exceedance Count`, `Secondary Exceedance Count`

Features categoricas (codificadas):

- `pollutant` (o `Parameter Code`, usar una de las dos para evitar redundancia)
- `Sample Duration`
- `State Code`

Target:

- `Arithmetic Mean`

### Variables a excluir al inicio (riesgo de leakage)

Si target = `Arithmetic Mean`, no usar al principio:

- `1st Max Value`, `2nd Max Value`, `3rd Max Value`, `4th Max Value`
- `99th Percentile`, `98th Percentile`, `95th Percentile`, `90th Percentile`, `75th Percentile`, `50th Percentile`, `10th Percentile`

Estas columnas resumen la misma distribucion anual y pueden inflar metricas.

### Sobre `Units of Measure`

`Units of Measure` puede ser util, pero en este dataset suele ser redundante respecto a `pollutant`/`Parameter Code`.

- En modelos por contaminante: casi no aporta.
- En modelo unico multcontaminante: se puede probar, pero no es prioridad en v1.

## 5) PCA: cuando y como usarlo

Tu idea de PCA es buena, pero conviene aplicarlo despues del baseline:

1. Entrenar baseline sin PCA.
2. Estandarizar features numericas.
3. Aplicar PCA solo a bloque numerico.
4. Elegir componentes para explicar ~90%-95% varianza.
5. Reentrenar y comparar `MAE`/`RMSE` contra baseline.

Si mejora o mantiene metrica con menos dimensiones, se adopta.

## 6) Analisis de distribucion (antes de entrenar)

Validar al menos:

- distribucion de `Arithmetic Mean` global y por `pollutant`,
- outliers en `Arithmetic Mean`, `Observation Count`, `Null Data Count`,
- correlaciones entre numericas (para multicolinealidad),
- nulos por columna y estrategia de imputacion.

Esto define si necesitas transformaciones (log, clipping, winsorizacion).

## 7) Pipeline tecnico propuesto (Go)

1. Leer CSV (`gota` o `encoding/csv`).
2. Limpiar nulos y tipar columnas.
3. Construir matriz `X` y vector `y`.
4. Split temporal (ejemplo: train <= 2020, test >= 2021).
5. Entrenar regresion lineal.
6. Evaluar `MAE` y `RMSE`.
7. Repetir con PCA y comparar.

## 8) Concurrencia: donde si aporta (fase 2)

No es prioridad para el primer modelo, pero luego sirve en:

- parseo concurrente por chunks de CSV,
- calculo paralelo de estadisticas por columna,
- entrenamiento/evaluacion por contaminante en goroutines separadas,
- grid search simple (distintas configuraciones) en paralelo.

## 9) Plan de ejecucion concreto

- **Semana 1 / v1**
  - stack: `gonum` + `gota`,
  - regresion lineal sin PCA,
  - metrica base en test temporal.
- **Semana 2 / v2**
  - analisis distribucional mas fino,
  - PCA + comparacion formal.
- **Semana 3 / v3**
  - paralelizacion de partes pesadas en Go.

## 10) Decision recomendada hoy

Si hay que decidir ahora mismo:

- Librerias: `gonum` + `gota`.
- Modelo inicial: regresion lineal multiple con target `Arithmetic Mean`.
- CSV pipeline: limpieza + encoding + split temporal.
- PCA: si, pero como experimento posterior al baseline.
