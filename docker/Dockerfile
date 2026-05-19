# VisionMind API server (Phase 7)
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV MODEL_CONFIG_PATH=/app/config/model_config.yaml
ENV HF_HOME=/model_cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "src.serving.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
