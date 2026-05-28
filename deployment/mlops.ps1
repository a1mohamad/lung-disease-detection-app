param(
    [Parameter(Mandatory = $true)]
    [string]$Task
)

$ErrorActionPreference = "Stop"

Write-Warning "mlops.ps1 is kept for compatibility. Prefer .\docker.ps1 <task>."
& "$PSScriptRoot\docker.ps1" $Task
