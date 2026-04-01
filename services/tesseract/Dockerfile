# syntax=docker/dockerfile:1

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install shared schema first (changes less often)
COPY shared/ /app/shared/
RUN pip install --no-cache-dir /app/shared

# Copy service source + install
COPY services/tesseract/pyproject.toml /app/
COPY services/tesseract/src/ /app/src/
RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]

HEALTHCHECK --interval=15s --timeout=10s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:8080/healthz || exit 1
