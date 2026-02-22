param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(
        "mlops-build",
        "mlops-up",
        "mlops-down",
        "mlops-logs",
        "mlops-ps",
        "backfill",
        "final-build",
        "final-up",
        "final-down",
        "final-logs",
        "final-ps"
    )]
    [string]$Task
)

$envFile = ".env.compose"
$mlopsFile = "docker-compose.mlops.yml"
$finalFile = "docker-compose.final.yml"

switch ($Task) {
    "mlops-build" { docker compose --progress=plain --env-file $envFile -f $mlopsFile build }
    "mlops-up" { docker compose --env-file $envFile -f $mlopsFile up -d db db-init mlflow airflow }
    "mlops-down" { docker compose --env-file $envFile -f $mlopsFile down --remove-orphans }
    "mlops-logs" { docker compose --env-file $envFile -f $mlopsFile logs -f }
    "mlops-ps" { docker compose --env-file $envFile -f $mlopsFile ps }
    "backfill" { docker compose --env-file $envFile -f $mlopsFile --profile manual run --rm post_hoc_backfill }
    "final-build" { docker compose --progress=plain --env-file $envFile -f $finalFile build }
    "final-up" { docker compose --env-file $envFile -f $finalFile up -d }
    "final-down" { docker compose --env-file $envFile -f $finalFile down --remove-orphans }
    "final-logs" { docker compose --env-file $envFile -f $finalFile logs -f }
    "final-ps" { docker compose --env-file $envFile -f $finalFile ps }
}

