FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY shared/ /app/shared/
RUN pip install --no-cache-dir /app/shared

COPY services/gateway/pyproject.toml /app/
COPY services/gateway/src/ /app/src/
COPY services/gateway/public/ /app/public/
RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=5s \
    CMD curl -f http://localhost:8080/healthz || exit 1
