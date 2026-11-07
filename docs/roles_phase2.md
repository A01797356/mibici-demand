# Roles e interacciones de la Fase 2

## Objetivo

Durante la Fase 2 del proyecto MiBici Demand se aplicaron prácticas de ingeniería de Machine Learning orientadas a la reproducibilidad, trazabilidad, seguimiento de experimentos y gestión del ciclo de vida de modelos.

---

## Data Engineer

### Responsabilidades

- Descarga y administración de datos.
- Limpieza y validación de registros.
- Generación de datasets versionados.
- Construcción de features para entrenamiento.

### Entregables

- dataset.py
- download.py
- features.py
- Versionado de datos mediante MLflow.

---

## Data Scientist

### Responsabilidades

- Diseño de variables predictoras.
- Selección de algoritmos.
- Evaluación de métricas.
- Ajuste de hiperparámetros.

### Entregables

- Baseline Lag168.
- Modelo Ridge.
- HistGradientBoosting Poisson.
- Análisis de métricas MAE, RMSE y R².

---

## ML Engineer

### Responsabilidades

- Integración del pipeline de entrenamiento.
- Automatización con Scikit-Learn Pipeline.
- Registro y seguimiento en MLflow.
- Gestión del Model Registry.

### Entregables

- train.py
- predict.py
- Registro de modelos.
- Alias @champion.

---

## Software Engineer

### Responsabilidades

- Refactorización de código.
- Organización bajo Cookiecutter.
- Centralización de parámetros.
- Desarrollo de pruebas automatizadas.

### Entregables

- config.py
- Makefile
- Tests automatizados.

---

## MLOps / SRE

### Responsabilidades

- Asegurar reproducibilidad.
- Gestión del entorno.
- Validación de ejecuciones.
- Seguimiento de dependencias.

### Entregables

- Configuración del entorno.
- Integración con MLflow.
- Documentación de ejecución.

---

# Interacción entre roles

El Data Engineer prepara y versiona los datos utilizados por el Data Scientist. El Data Scientist desarrolla y evalúa modelos utilizando los datasets generados. El ML Engineer integra el flujo completo en pipelines reproducibles y registra los resultados en MLflow. El Software Engineer mantiene la calidad y estructura del código mediante prácticas de refactorización y pruebas. Finalmente, el rol de MLOps/SRE garantiza que todo el proceso pueda reproducirse y ejecutarse de forma consistente en distintos entornos.

Esta colaboración permitió construir un flujo completo de Machine Learning con trazabilidad desde los datos hasta los modelos registrados en MLflow.