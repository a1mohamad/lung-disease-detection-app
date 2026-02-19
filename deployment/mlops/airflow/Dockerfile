FROM apache/airflow:2.8.4-python3.11
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER airflow
COPY requirements-mlops.txt /requirements-mlops.txt
RUN pip install --no-cache-dir -r /requirements-mlops.txt
