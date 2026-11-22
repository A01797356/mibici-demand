# Reproducibilidad en un entorno limpio (Entrega Final, Punto 3)

**Objetivo:** demostrar que otra persona, a partir del repositorio, obtiene el mismo modelo y
las mismas predicciones que la corrida oficial.

## Controles de reproducibilidad

| Control | Implementación |
|---|---|
| Dependencias fijas | `requirements.txt` con versión exacta en todas las librerías (incluido `pyarrow==25.0.1`) |
| Semillas | `RANDOM_STATE = 42` en `config.py`: muestreo del tuning, `RandomizedSearchCV`, HGB, `TargetEncoder` (KFold) y PCA |
| Datos versionados | Cada versión de datos (`trips_clean_v1`, `features_v1`) se registra en MLflow con esquema, perfil y muestra |
| Referencias fijas | Catálogo de estaciones versionado en `references/nomenclatura_2026_09.csv` (484 estaciones). MiBici solo publica el catálogo más reciente: durante esta prueba ya era `nomenclatura_2026_10.csv` |
| Modelo versionado | MLflow Model Registry: `models:/mibici-demand-24h@champion` → **v4** (run `3df12bf7c6ee4bfb821aeb139875b0e9`) |

## Procedimiento

1. Clon nuevo desde GitHub (rama `feat/reproducibility`, commit `074c750`) en otra carpeta
   (`~/repro`, sistema de archivos de WSL), con un `.venv` nuevo e instalación solo desde `requirements.txt`.
2. `pytest -q` → 16 passed.
3. Servidor MLflow independiente y vacío (puerto 5001) para no alterar el Registry oficial.
4. Pipeline completo según el README: `make download`, catálogo fijo, `make data features plots train predict`.
5. Comparación contra la corrida oficial (Registry en el puerto 5000, modelo v4) y llamada a la misma petición en ambas APIs.

## Resultados

| Métrica | Oficial (v4) | Entorno limpio |
|---|---|---|
| Viajes limpios | 4,478,552 | 4,478,552 |
| Filas train / test | 2,482,903 / 625,551 | 2,482,903 / 625,551 |
| MAE baseline `lag_168` | 0.9743 | 0.9743 |
| MAE / RMSE / R² (HGB Poisson) | 0.8020 / 1.3443 / 0.6936 | 0.8020 / 1.3443 / 0.6936 |
| Predicciones de test idénticas | — | **625,551 de 625,551 (diferencia máxima 0.0)** |
| `POST /predict` (estaciones 2 y 3, 2-dic-2024 08:00) | 5.860 / 3.416 | 5.860 / 3.416 |

Detalle en [`reports/reproducibility_check.csv`](../reports/reproducibility_check.csv).

La API reporta `model_version` 4 en el Registry oficial y 2 en el de la prueba: cada Registry
numera sus versiones por separado. Las predicciones idénticas confirman que es el mismo modelo.

## Hallazgo

Antes de esta prueba, el `@champion` del Registry oficial era la v2 (MAE 0.8028), entrenada con el
catálogo de 470 estaciones antes de fijarlo, y no coincidía con el resultado documentado (0.802).
Se reentrenó con el catálogo fijo y `@champion` pasó a la v4, que es la que sirve la API.


## Prueba en otra máquina (imagen Docker)

La imagen publicada se descargó y ejecutó en una computadora con Windows distinta a la de entrenamiento. 
El digest coincide (`sha256:02aea128…b356`) y `POST /predict` con el ejemplo del README devuelve `model_version` 4 y 5.86, igual que en el entorno oficial.

![Predicción desde el contenedor](../reports/figures/evidencias/06_swagger_predict_200.png)

Capturas completas en [`reports/figures/evidencias/`](../reports/figures/evidencias/).
