ENV_FILE ?= .env.compose
MLOPS_FILE ?= docker-compose.mlops.yml
FINAL_FILE ?= docker-compose.final.yml

.PHONY: mlops-build mlops-up mlops-down mlops-logs backfill backfill-eval mlops-ps
.PHONY: final-build final-up final-down final-logs final-ps

mlops-build:
	docker compose --progress=plain --env-file $(ENV_FILE) -f $(MLOPS_FILE) build

mlops-up:
	docker compose --env-file $(ENV_FILE) -f $(MLOPS_FILE) up -d db db-init mlflow airflow

mlops-down:
	docker compose --env-file $(ENV_FILE) -f $(MLOPS_FILE) down --remove-orphans

mlops-logs:
	docker compose --env-file $(ENV_FILE) -f $(MLOPS_FILE) logs -f

mlops-ps:
	docker compose --env-file $(ENV_FILE) -f $(MLOPS_FILE) ps

BACKFILL_ARGS ?=

backfill:
	docker compose --env-file $(ENV_FILE) -f $(MLOPS_FILE) --profile manual run --rm post_hoc_backfill bash -c "python /opt/project/mlops/jobs/post_hoc_backfill.py --register-model $(BACKFILL_ARGS)"

backfill-eval:
	$(MAKE) backfill BACKFILL_ARGS="--with-eval"

final-build:
	docker compose --progress=plain --env-file $(ENV_FILE) -f $(FINAL_FILE) build

final-up:
	docker compose --env-file $(ENV_FILE) -f $(FINAL_FILE) up -d

final-down:
	docker compose --env-file $(ENV_FILE) -f $(FINAL_FILE) down --remove-orphans

final-logs:
	docker compose --env-file $(ENV_FILE) -f $(FINAL_FILE) logs -f

final-ps:
	docker compose --env-file $(ENV_FILE) -f $(FINAL_FILE) ps
