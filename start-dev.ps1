param(
    [string]$BackendCommand = "python main.py",
    [string]$FrontendCommand = "npm.cmd run dev",
    [int]$BackendStartupTimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WebRoot = Join-Path $ProjectRoot "web"
$RuntimeConfigPath = Join-Path $ProjectRoot ".runtime\backend.json"
$EnvPath = Join-Path $ProjectRoot ".env"
$ExampleEnvPath = Join-Path $ProjectRoot "example.env"

function Write-DevLine {
    param(
        [string]$Prefix,
        [object]$Line
    )
    if ($null -ne $Line -and "$Line".Length -gt 0) {
        Write-Host "[$Prefix] $Line"
    }
}

function Receive-DevJob {
    param(
        [System.Management.Automation.Job]$Job,
        [string]$Prefix
    )
    $receivedErrors = @()
    Receive-Job -Job $Job -ErrorAction SilentlyContinue -ErrorVariable receivedErrors | ForEach-Object {
        Write-DevLine -Prefix $Prefix -Line $_
    }
    foreach ($errorRecord in $receivedErrors) {
        Write-DevLine -Prefix $Prefix -Line $errorRecord.Exception.Message
    }
}

if (-not (Test-Path -LiteralPath $EnvPath) -and (Test-Path -LiteralPath $ExampleEnvPath)) {
    Copy-Item -LiteralPath $ExampleEnvPath -Destination $EnvPath
    Write-Host "[dev] Created .env from example.env. Fill API keys if you have not done so."
}

if (Test-Path -LiteralPath $RuntimeConfigPath) {
    Remove-Item -LiteralPath $RuntimeConfigPath -Force
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found. Activate the Python 3.12 environment first."
}

if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "npm.cmd was not found. Install Node.js or add npm to PATH first."
}

Write-Host "[dev] Starting backend: $BackendCommand"
$backendJob = Start-Job -Name "paper-agent-backend" -ScriptBlock {
    param($Root, $Command)
    Set-Location -LiteralPath $Root
    Invoke-Expression $Command
} -ArgumentList $ProjectRoot, $BackendCommand

$frontendJob = $null

try {
    $deadline = (Get-Date).AddSeconds($BackendStartupTimeoutSeconds)
    while (-not (Test-Path -LiteralPath $RuntimeConfigPath)) {
        Receive-DevJob -Job $backendJob -Prefix "backend"

        if ($backendJob.State -ne "Running") {
            Receive-DevJob -Job $backendJob -Prefix "backend"
            throw "Backend exited before writing .runtime\backend.json."
        }

        if ((Get-Date) -gt $deadline) {
            throw "Timed out waiting for backend runtime config: $RuntimeConfigPath"
        }

        Start-Sleep -Milliseconds 500
    }

    $runtimeConfig = Get-Content -LiteralPath $RuntimeConfigPath -Raw | ConvertFrom-Json
    Write-Host "[dev] Backend URL: $($runtimeConfig.url)"
    Write-Host "[dev] Starting frontend: $FrontendCommand"

    $frontendJob = Start-Job -Name "paper-agent-frontend" -ScriptBlock {
        param($Root, $Command)
        Set-Location -LiteralPath $Root
        Invoke-Expression $Command
    } -ArgumentList $WebRoot, $FrontendCommand

    Write-Host "[dev] Frontend will usually be available at http://localhost:5173"
    Write-Host "[dev] Press Ctrl+C to stop backend and frontend."

    while ($true) {
        Receive-DevJob -Job $backendJob -Prefix "backend"
        Receive-DevJob -Job $frontendJob -Prefix "frontend"

        if ($backendJob.State -ne "Running") {
            Receive-DevJob -Job $backendJob -Prefix "backend"
            throw "Backend process exited."
        }

        if ($frontendJob.State -ne "Running") {
            Receive-DevJob -Job $frontendJob -Prefix "frontend"
            throw "Frontend process exited."
        }

        Start-Sleep -Milliseconds 500
    }
}
finally {
    Write-Host "[dev] Stopping dev processes..."
    foreach ($job in @($frontendJob, $backendJob)) {
        if ($null -ne $job) {
            Stop-Job -Job $job -ErrorAction SilentlyContinue
            Receive-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
            Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
        }
    }
}
