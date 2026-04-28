FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy full project
COPY . /app

# Generate seed data + train model at build time so the image is self-contained.
# Skip if artefacts already exist (speeds up rebuilds).
RUN python /app/scripts/generate_data.py && \
    python /app/ml/train_numpy.py

# Create persistent-data directory (mounted as a volume on cloud deployments)
RUN mkdir -p /data/logs

EXPOSE 8000

# DB_PATH and LOG_DIR are overridden by the container orchestrator (docker-compose / Render)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
