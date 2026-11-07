# ML Canvas · Predicción de demanda de MiBici

Basado en el *Machine Learning Canvas* (Louis Dorard), según la metodología de [Ignacio G. R. Gavilán](https://ignaciogavilan.com/metodologia-para-machine-learning-ii-machine-learning-canvas/).
Fuente de datos: [datos abiertos de MiBici](https://mibici.net/en/open-data/).

## 1. Propuesta de valor
- **Usuario final:** equipo de operación de MiBici que las redistribuye entre estaciones con camiones de rebalanceo.
- **Problema:** en horas pico algunas estaciones se vacían (por lo que el usuario no encuentra bicicleta) y otras se llenan (por lo que el usuario no puede dejarla). Hoy el rebalanceo es reactivo.
- **Propuesta:** pronosticar cuántas bicicletas saldrán de cada estación en las próximas horas, para reabastecer antes del pico. Menos viajes perdidos.

## 2. Decisiones
- Cuáles estaciones reabastecer y con cuántas bicicletas antes de los picos de 8h y 18h.
- Orden de rutas de los camiones según el déficit esperado.
- A mediano plazo ampliar anclajes en estaciones con demanda alta

## 3. Tarea de ML
- **Tipo:** regresión supervisada sobre series de tiempo (pronóstico).
- **Entrada:** estación, fecha-hora, calendario e historial reciente de la estación.
- **Salida:** viajes que inician en la estación en la hora (`departures`).
- **Horizonte:** 24h, para planear el día siguiente. La versión inicial de la Fase 1 planteaba 1 h, pero con ese horizonte no hay tiempo para planear la redistribución.

## 4. Fuentes de datos
- 142 CSV mensuales (dic-2014 a sep-2026, 39,078,984 viajes): ID de viaje y usuario, género, año-nacimiento, inicio y fin, estaciones origen y destino.
- Catálogo de estaciones `nomenclatura_2026_09.csv` (484 estaciones, con coordenadas).
- Posibles fuentes a agregar: clima, calendario escolar y eventos masivos.

## 5. Recolección de datos
- `mibici_demand/download.py` obtiene los enlaces de la página de datos abiertos y descarga los CSV nuevos sin sobrescribir los existentes; publica un archivo al mes.
- Cada versión limpia se registra en MLflow con parámetros, métricas de calidad, esquema y digest.

## 6. Features
- **Calendario:** hora, día de la semana, mes, fin de semana y festivo oficial; hora y día con codificación cíclica (seno y coseno).
- **Historial:** demanda de la misma estación 1h, 24h y 168h antes, y media de las 24h previas.
- **Estación:** latitud, longitud y demanda media histórica (*target encoding*).
- **Perfil de estación:** componentes PCA del perfil semanal (168 horas).

## 7. Construcción de modelos (Fase 2)
- *Baseline* estacional: la demanda de la misma hora de la semana anterior (`lag_168`, la variable más correlacionada: 0.74).
- Candidatos evaluados en la Fase 2: Ridge e HistGradientBoosting con pérdida Poisson. Random Forest, previsto inicialmente, no se evaluó.
- Experimentos en MLflow, ligados a los datasets versionados de la Fase 1.

## 8. Predicciones
- *Batch* cada hora (o diario) para todas las estaciones activas.
- Entrega a operación como tabla con el déficit esperado por estación.

## 9. Evaluación offline
- Split temporal sin barajar: último 20% del periodo como prueba (versión v2: 21,120,492 filas de entrenamiento hasta 2024-05-20 y 8,093,859 de prueba).
- MAE y RMSE globales y por franja horaria, comparados contra el baseline.

## 10. Monitoreo en producción (Fase 3)
- Error real vs. predicho por semana; *data drift* en la distribución horaria y estaciones nuevas.
- Reentrenamiento cuando el MAE supere un umbral acordado con operación.
