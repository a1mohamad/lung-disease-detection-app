"""MLOps configuration and model specification package."""

# Re-export settings/specs for Airflow DAGs and CLI jobs without making callers
# know the individual config module layout.
from mlops.config.model_specs import MODEL_SPECS, POST_HOC_SPECS, ModelSpec, PostHocSpec
from mlops.config.settings import MLOpsSettings
