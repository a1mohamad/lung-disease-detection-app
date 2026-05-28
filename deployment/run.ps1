$ErrorActionPreference = "Stop"

Write-Warning "run.ps1 is kept for compatibility. Prefer .\run-local.ps1."
& "$PSScriptRoot\run-local.ps1"
