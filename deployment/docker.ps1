param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "mlops-build",
        "mlops-up",
        "mlops-down",
        "mlops-logs",
        "mlops-ps",
        "backfill",
        "runtime-build",
        "runtime-up",
        "runtime-down",
        "runtime-logs",
        "runtime-ps",
        "final-build",
        "final-up",
        "final-down",
        "final-logs",
        "final-ps"
    )]
    [string]$Task
)

$ErrorActionPreference = "Stop"

$envFile = ".env.compose"
$runtimeEnvFile = ".env.runtime"
$mlopsFile = "docker-compose.mlops.yml"
$finalFile = "docker-compose.final.yml"
$runtimeFile = "docker-compose.runtime.yml"

switch ($Task) {
    "mlops-build" { docker compose --progress=plain --env-file $envFile -f $mlopsFile build }
    "mlops-up" { docker compose --env-file $envFile -f $mlopsFile up -d db db-init mlflow airflow }
    "mlops-down" { docker compose --env-file $envFile -f $mlopsFile down --remove-orphans }
    "mlops-logs" { docker compose --env-file $envFile -f $mlopsFile logs -f }
    "mlops-ps" { docker compose --env-file $envFile -f $mlopsFile ps }
    "backfill" { docker compose --env-file $envFile -f $mlopsFile --profile manual run --rm post_hoc_backfill }
    "runtime-build" { docker compose --progress=plain --env-file $runtimeEnvFile -f $runtimeFile build }
    "runtime-up" { docker compose --env-file $runtimeEnvFile -f $runtimeFile up -d }
    "runtime-down" { docker compose --env-file $runtimeEnvFile -f $runtimeFile down --remove-orphans }
    "runtime-logs" { docker compose --env-file $runtimeEnvFile -f $runtimeFile logs -f }
    "runtime-ps" { docker compose --env-file $runtimeEnvFile -f $runtimeFile ps }
    "final-build" { docker compose --progress=plain --env-file $envFile -f $finalFile build }
    "final-up" { docker compose --env-file $envFile -f $finalFile up -d }
    "final-down" { docker compose --env-file $envFile -f $finalFile down --remove-orphans }
    "final-logs" { docker compose --env-file $envFile -f $finalFile logs -f }
    "final-ps" { docker compose --env-file $envFile -f $finalFile ps }
}
