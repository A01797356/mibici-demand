# Seguimiento de experimentos, visualización y gestión de modelos

Herramienta: **MLflow 3.16.0** (backend SQLite `mlflow.db`, artefactos en `./mlartifacts`).
Problema: demanda horaria por estación (salidas) con 24 h de anticipación, datos MiBici 2024 (versión `v1`).

`mlflow.db` y `mlartifacts/` no se versionan en Git. Los resultados de los runs se exportaron a
[`reports/experiments/runs_mibici-training.csv`](../reports/experiments/runs_mibici-training.csv)
para poder consultarlos sin levantar MLflow.

## 1. Organización en MLflow

MLflow se usa tanto para versionar los datos como para los experimentos de modelado; cada versión
de datos queda registrada como run y como dataset de MLflow con un digest único, por lo que no se
usó DVC.

| Experimento | Etapa | Runs | Lo genera |
|---|---|---|---|
| `mibici-data-versioning` | Datos y features | `trips_clean_v1`, `features_v1` | `make data`, `make features` |
| `mibici-training` | Modelado | `baseline_lag168_v1`, `ridge_v1`, `hgb_poisson_v1` | `make train` |

Convenciones:

- **Nombre del run:** `<modelo>_<versión de datos>`, p. ej. `hgb_poisson_v1`.
- **Tags propios:** `stage` (`training` / `features`), `data_version` y `model`.
- **Tags automáticos de Git:** `mlflow.source.git.commit`, `mlflow.source.git.branch` y `mlflow.source.name`.
- **Parámetros centralizados:** todo lo registrado sale de `mibici_demand/config.py`
  (`MODEL_PARAMS`, `SEARCH_SPACES`, `FEATURE_PARAMS`, `CLEANING_PARAMS`).

## 2. Qué se registra en cada run de entrenamiento

| Tipo | Contenido |
|---|---|
| Parámetros | `model`, `data_version`, `horizon_h`, `features` (12 variables), `train_end`, `test_start`, mejores hiperparámetros, `tune_rows`, `cv_splits`, `n_iter` |
| Métricas (test) | `mae`, `rmse`, `r2`, `mae_improvement_vs_baseline_pct` |
| Métrica (validación) | `cv_mae`: MAE promedio de `TimeSeriesSplit` con los mejores hiperparámetros |
| Dataset de entrada | `features_v1` (digest `5f00da09`), contexto `training` |
| Artefacto | `tuning/cv_results.csv`: todas las combinaciones probadas en el ajuste |
| Modelo | Pipeline completo (preprocesador + modelo), sabores `sklearn` y `python_function`, firma y ejemplo de entrada |

El baseline (misma hora de la semana anterior, `lag_168`) se registra como run aparte, sin modelo,
como referencia para todas las comparaciones.

## 3. Comparación de configuraciones y resultados

Periodo de test: 21-oct-2024 04:00 → 31-dic-2024 23:00 (625,551 observaciones estación-hora).

| Run | Hiperparámetros | MAE | RMSE | R² | cv_mae | Mejora vs baseline |
|---|---|---|---|---|---|---|
| `baseline_lag168_v1` | — | 0.974 | 1.773 | 0.467 | — | — |
| `ridge_v1` | `alpha=100` | 0.922 | 1.480 | 0.629 | 0.935 | 5.4 % |
| **`hgb_poisson_v1`** | `learning_rate=0.1`, `max_leaf_nodes=127`, `min_samples_leaf=500`, `l2_regularization=0` | **0.802** | **1.344** | **0.694** | **0.819** | **17.7 %** |

Configuraciones probadas en `hgb_poisson` (8 de 54, `RandomizedSearchCV` + `TimeSeriesSplit` de 3 folds):

| # | min_samples_leaf | max_leaf_nodes | learning_rate | l2_regularization | cv_mae | desv. est. |
|---|---|---|---|---|---|---|
| **1** | **500** | **127** | **0.10** | **0.0** | **0.8187** | 0.0106 |
| 2 | 500 | 127 | 0.10 | 1.0 | 0.8189 | 0.0102 |
| 3 | 100 | 127 | 0.20 | 1.0 | 0.8194 | 0.0099 |
| 4 | 100 | 63 | 0.20 | 1.0 | 0.8206 | 0.0105 |
| 5 | 20 | 63 | 0.10 | 0.0 | 0.8225 | 0.0098 |
| 6 | 100 | 31 | 0.20 | 0.0 | 0.8227 | 0.0084 |
| 7 | 20 | 63 | 0.20 | 1.0 | 0.8240 | 0.0117 |
| 8 | 500 | 63 | 0.05 | 0.0 | 0.8320 | 0.0102 |

Configuraciones probadas en `ridge` (4 de 4):

| alpha | cv_mae | desv. est. |
|---|---|---|
| **100** | **0.9348** | 0.0188 |
| 10 | 0.9351 | 0.0189 |
| 1 | 0.9351 | 0.0189 |
| 0.1 | 0.9351 | 0.0189 |

Hallazgos:

- `hgb_poisson` es el mejor en todas las métricas: reduce el MAE 17.7 % frente al baseline y 13 % frente a Ridge.
- Ridge apenas mejora al baseline y casi no cambia con `alpha`: la relación no es lineal.
- Las tres mejores configuraciones de `hgb_poisson` presentan resultados de validación muy similares:
  la diferencia máxima en `cv_mae` es aproximadamente 0.00062 (0.0007 con los valores redondeados
  de la tabla), menor que la desviación estándar entre folds (aproximadamente 0.01).
  La configuración seleccionada obtuvo el menor MAE promedio entre las evaluadas, sin que esta
  búsqueda permita aislar el efecto de cada hiperparámetro ni demostrar equivalencia estadística.
- `cv_mae` (0.819) y MAE de test (0.802) son cercanos: el modelo generaliza al periodo futuro.

## 4. Control de versiones (linaje)

| Eslabón | Identificador en MLflow |
|---|---|
| Código | commit `4651a17`, rama `feat/modeling` (tag automático en todos los runs) |
| Datos crudos → limpios | run `trips_clean_v1` · dataset `44f0fefa` · `datos_abiertos_2024_*.csv` (12 archivos) |
| Features y split | run `features_v1` · features `5f00da09` · train `f86ba0ea` · test `04ed7e39` |
| Entrenamiento | runs `baseline_lag168_v1`, `ridge_v1`, `hgb_poisson_v1` |
| Modelos registrados | `mibici-demand-24h` v1 ← `ridge_v1` · v2 ← `hgb_poisson_v1` |

Una nueva versión de datos genera runs y versiones de modelo nuevas sin sobrescribir las anteriores:

```bash
make data features train PATTERN="datos_abiertos_2023_*.csv" VERSION=v2
```

## 5. Visualización en MLflow

En la interfaz de MLflow (`make mlflow-ui`):

| Vista | Dónde |
|---|---|
| Tabla de runs con parámetros y métricas | Experiments → `mibici-training`; agregar columnas `mae`, `rmse`, `r2`, `cv_mae`, `mae_improvement_vs_baseline_pct` |
| Coordenadas paralelas (hiperparámetros vs MAE) | Seleccionar los 3 runs → **Compare** → Parallel Coordinates Plot |
| Gráfica de barras de métricas | Pestaña **Chart view** del experimento |
| Detalle de un run | Clic en `hgb_poisson_v1` → Overview y Artifacts (`tuning/cv_results.csv`) |
| Linaje de datos | Experiments → `mibici-data-versioning` |
| Model Registry | Models → `mibici-demand-24h` |

## 6. Model Registry

`make train` registra cada modelo en `mibici-demand-24h`. Cada versión queda ligada a su run, de
donde hereda hiperparámetros, métricas, dataset y firma.

| Versión | Algoritmo | Hiperparámetros | MAE | RMSE | R² | Alias | Run |
|---|---|---|---|---|---|---|---|
| 1 | Ridge | `alpha=100` | 0.922 | 1.480 | 0.629 | — | `ridge_v1` |
| **2** | HistGradientBoosting (Poisson) | `learning_rate=0.1`, `max_leaf_nodes=127`, `min_samples_leaf=500`, `l2_regularization=0` | **0.802** | **1.344** | **0.694** | **`@champion`** | `hgb_poisson_v1` |

- **Promoción automática:** `train.py` asigna `@champion` a la versión con menor MAE en test y le agrega
  la descripción `hgb_poisson | features_v1 | horizonte 24 h | MAE test 0.8020 (baseline 0.9743)`.
- **Firma:** entrada de 12 columnas (`station_id`, `month`, `is_weekend`, `is_holiday`, `hour_sin`,
  `hour_cos`, `dow_sin`, `dow_cos`, `lag_24`, `lag_168`, `latitude`, `longitude`); salida `float64`.
- **Consumo por alias:** `predict.py` carga `models:/mibici-demand-24h@champion`; promover otro modelo
  solo requiere mover el alias.
- **Verificación:** `make predict` generó 625,551 predicciones con `@champion`, sin nulos ni negativos,
  con MAE 0.802 (igual al registrado).

## 7. Cómo replicarlo

```bash
# Terminal 2 (en macOS el puerto 5000 lo ocupa AirPlay; usar 5001)
make mlflow-ui MLFLOW_PORT=5001

# Terminal 1 (.env: MLFLOW_TRACKING_URI=http://127.0.0.1:5001)
make data features   # versiones de datos en mibici-data-versioning
make train           # runs en mibici-training + Model Registry + @champion
make predict         # inferencia con models:/mibici-demand-24h@champion
```
