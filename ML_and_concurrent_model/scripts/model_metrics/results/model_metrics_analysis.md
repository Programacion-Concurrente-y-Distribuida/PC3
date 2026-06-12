# Analisis de metricas: PCA vs sin PCA

## Criterio de lectura

- En MAE y RMSE, menor es mejor.
- En R2, mayor es mejor.
- Si PCA mantiene metricas similares con menos componentes, puede ser util para reducir dimensionalidad.
- Si PCA empeora MAE/RMSE o baja R2 de forma clara, conviene conservar el baseline sin PCA.

## Resumen

| escenario | runs | MAE test | RMSE test | R2 test | segundos | RAM MB | componentes PCA | varianza PCA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sin_pca | 3 | 0.775546 | 1.052191 | 0.900432 | 103.669866 | 4055.411458 |  |  |
| pca_90 | 3 | 1.017426 | 1.298668 | 0.848321 | 88.609058 | 4876.015625 | 53.000000 | 0.907900 |
| pca_95 | 3 | 1.024417 | 1.294136 | 0.849377 | 46.143705 | 4913.833333 | 57.000000 | 0.962600 |
| pca_99 | 3 | 0.850972 | 1.153229 | 0.880392 | 67.953119 | 4700.661458 | 60.000000 | 0.995000 |

## Comparacion contra baseline sin PCA

- `pca_90`: delta MAE=+0.241880, delta RMSE=+0.246477, delta R2=-0.052111.
- `pca_95`: delta MAE=+0.248871, delta RMSE=+0.241945, delta R2=-0.051055.
- `pca_99`: delta MAE=+0.075426, delta RMSE=+0.101038, delta R2=-0.020040.

## Lectura rapida

- Mejor MAE test: `sin_pca` (0.775546).
- Mejor RMSE test: `sin_pca` (1.052191).
- Mejor R2 test: `sin_pca` (0.900432).
