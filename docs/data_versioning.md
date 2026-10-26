# Registro de versiones de datos

Cada versión se genera con el pipeline y se registra en MLflow (servidor local con SQLite).
En cada run: *Parameters* (umbrales y periodo), *Metrics* (filas eliminadas por regla),
*Datasets* (nombre, digest, esquema) y en *Artifacts → dataset/* el perfil, el esquema
y una muestra de 1,000 filas.

```powershell
# Windows (PowerShell). En Linux/WSL: make mlflow-ui / make pipeline
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
python -m mibici_demand.dataset clean --pattern "datos_abiertos_2024_*.csv" --version v1   # v1: 2024
python -m mibici_demand.features --version v1
python -m mibici_demand.dataset clean --pattern "datos_abiertos_*.csv" --version v2        # v2: todo el histórico
python -m mibici_demand.features --version v2
python -m mibici_demand.plots
```

Para reproducir una versión: los parámetros del run indican el periodo y los umbrales; si se ejecutó desde un clon de Git, el tag `mlflow.source.git.commit` indica el código usado.
Nota: los runs se ejecutaron desde una carpeta descargada (no un clon de Git), por lo que MLflow no registró el tag `mlflow.source.git.commit`.

## Historial

| Versión | Experimento | Run ID | Entrada | Salida | Filas | Cambios |
|---|---|---|---|---|---|---|
| `trips_clean_v1` | `mibici-data-versioning` | `9fc67e29c6dc422b9acaba8aefff7897` | 12 CSV de 2024 (4,645,921 viajes) | `data/interim/trips_clean.parquet` | 4,478,552 | Limpieza base (reglas abajo) |
| `features_v1` | `mibici-data-versioning` | `887af1754757468588cd9790dc91f380` | `trips_clean_v1` | `features.parquet`, `train.parquet`, `test.parquet`, `preprocessor.joblib` | 3,108,454 (train 2,482,903 / test 625,551) | Demanda horaria, calendario, lags 1/24/168, split 80/20, PCA 5 componentes |
| `trips_clean_v2` | `mibici-data-versioning` | `[Run ID]` | 142 CSV, dic-2014 a sep-2026 (39,078,984 viajes) | `data/interim/trips_clean.parquet` | 37,643,853 | Mismas reglas que v1; periodo completo. Digest `c594e16e` |
| `features_v2` | `mibici-data-versioning` | `ec2d6c8952d2450290313ed4c761f188` | `trips_clean_v2` | `features.parquet`, `train.parquet`, `test.parquet`, `preprocessor.joblib` | 29,214,351 (train 21,120,492 / test 8,093,859) | Mismas features que v1; 483 estaciones; test desde 2024-05-21. Digest `fc90632d` |
| `eda_figures` | `mibici-eda` | `af95594fd18946548769275082d05aad` | `trips_clean_v1`, `features_v1` | 9 figuras en `reports/figures/` | — | Figuras del EDA |

## Reglas de limpieza y su efecto en cada versión

| Regla (`TripCleaner`) | Parámetro | v1 (2024) | v2 (2014-2026) | Justificación |
|---|---|---|---|---|
| `standardize` | — | 0 | 0 | Dos codificaciones, dos formatos de fecha y un nombre de columna roto |
| `mark_invalid_ids` | `INVALID_ID_VALUES` (ID 0) | 0 | 0 | Protección: un ID 0 no corresponde a ninguna estación |
| `drop_missing` | — | 0 | 0 | Sin fecha o estación no se puede calcular la demanda |
| `drop_duplicates` | — | 0 | 758 | Registros repetidos inflan la demanda |
| `drop_invalid_durations` | 1 min a 24 h | 143,892 | 1,271,240 | < 1 min: anclaje fallido; fin ≤ inicio o > 24 h: error |
| `drop_false_round_trips` | misma estación y < 2 min | 22,981 | 146,718 | La bicicleta se sacó y se regresó; no hubo viaje |
| `remove_duration_outliers` | IQR sobre log(duración), k = 3 | 496 | 12,524 | Cola larga; en logaritmo la distribución es casi simétrica |
| `clean_demographics` | edad 16-90 | 0 (1,100 → NaN) | 0 (8,210 → NaN) | La demanda no depende de la edad; no se pierde el viaje |
| `filter_unknown_stations` | catálogo de 484 estaciones | 0 | 3,891 | Descarta IDs de estación inexistentes |
| **Total eliminado** | | **167,369 (3.6 %)** | **1,435,131 (3.7 %)** | v1: 4,645,921 → 4,478,552 · v2: 39,078,984 → 37,643,853 |

Digests de MLflow (huella del contenido): `trips_clean_v1` 44f0fefa · `features_v1` 5f00da09 ·
`trips_clean_v2` c594e16e · `features_v2` fc90632d.
