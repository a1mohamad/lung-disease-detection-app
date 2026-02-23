FROM python:3.11-slim
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
RUN pip install --no-cache-dir mlflow pyyaml pyodbc
EXPOSE 5000
CMD mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "${MLFLOW_ALLOWED_HOSTS:-localhost,127.0.0.1,mlflow,localhost:5000,127.0.0.1:5000,mlflow:5000}" --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
