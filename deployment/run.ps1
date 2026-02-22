$ErrorActionPreference = "Stop"

py -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env
