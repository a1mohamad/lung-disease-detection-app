FROM python:3.11-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gnupg2 unixodbc-dev \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mlflow pyyaml pyodbc
EXPOSE 5000
CMD mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" --default-artifact-root "${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
