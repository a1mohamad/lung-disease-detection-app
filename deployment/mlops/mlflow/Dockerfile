FROM python:3.11-slim
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=20 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg unixodbc-dev ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /etc/apt/keyrings/microsoft-prod.gpg \
    && chmod 644 /etc/apt/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip \
    && pip install --prefer-binary --retries ${PIP_RETRIES} --timeout ${PIP_DEFAULT_TIMEOUT} mlflow==3.9.0 pyyaml==6.0.3 pyodbc==5.3.0
EXPOSE 5000
CMD mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "${MLFLOW_ALLOWED_HOSTS:-localhost,127.0.0.1,mlflow,localhost:5000,127.0.0.1:5000,mlflow:5000}" --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
