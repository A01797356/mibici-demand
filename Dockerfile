# Servicio de predicción MiBici: FastAPI + modelo mibici-demand-24h v4 (exportado del Registry)
FROM python:3.12.7-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_URI=/app/models/champion \
    MODEL_VERSION=4

WORKDIR /app

# Dependencias con las mismas versiones fijas del proyecto (incluye el paquete con -e .)
COPY requirements.txt pyproject.toml README.md ./
COPY mibici_demand/ mibici_demand/
RUN pip install -r requirements.txt

# Catálogo de estaciones versionado y modelo
COPY references/nomenclatura_2026_09.csv references/
COPY models/champion/ models/champion/

EXPOSE 8000
CMD ["uvicorn", "mibici_demand.api:app", "--host", "0.0.0.0", "--port", "8000"]
