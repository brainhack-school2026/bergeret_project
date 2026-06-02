FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code only — data is provided at runtime via bind mounts
COPY analysis/ analysis/
COPY notebooks/ notebooks/
COPY tasks.py .
COPY container_run.sh .

RUN chmod +x container_run.sh \
 && mkdir -p /data/source_data /data/output_data

ENV PYTHONUNBUFFERED=1 \
    PYTHONNOUSERSITE=1

ENTRYPOINT ["/app/container_run.sh"]
