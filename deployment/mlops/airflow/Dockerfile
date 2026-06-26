FROM apache/airflow:3.1.7-python3.11
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-onnx-exporter.txt /requirements-onnx-exporter.txt
RUN python -m venv /opt/onnx-exporter \
    && /opt/onnx-exporter/bin/pip install --no-cache-dir -r /requirements-onnx-exporter.txt
USER airflow
COPY requirements-mlops.txt /requirements-mlops.txt
RUN pip install --no-cache-dir -r /requirements-mlops.txt
