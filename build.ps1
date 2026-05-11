param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Invoke-Checked {
    param(
        [string]$Command,
        [string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install it first, then rerun build.ps1."
}

if (-not $env:VIRTUAL_ENV) {
    if (-not (Test-Path -LiteralPath ".venv")) {
        if ($Python) {
            Invoke-Checked uv @("venv", "--python", $Python, ".venv")
        } else {
            Invoke-Checked uv @("venv", ".venv")
        }
    }
    $env:VIRTUAL_ENV = (Resolve-Path -LiteralPath ".venv").Path
    $venvScripts = Join-Path $env:VIRTUAL_ENV "Scripts"
    $env:PATH = $venvScripts + [IO.Path]::PathSeparator + $env:PATH
}

if (-not $env:CMAKE_BUILD_PARALLEL_LEVEL) {
    $env:CMAKE_BUILD_PARALLEL_LEVEL = "1"
}

Write-Host "Using Python environment: $env:VIRTUAL_ENV"
Write-Host "CMAKE_BUILD_PARALLEL_LEVEL=$env:CMAKE_BUILD_PARALLEL_LEVEL"

Invoke-Checked uv @("pip", "install", "--upgrade", "pip", "setuptools", "wheel", "scikit-build-core", "cmake", "ninja", "numpy", "scipy", "matplotlib")
Invoke-Checked uv @("pip", "install", "--reinstall-package", "pytexgen", "--no-build-isolation", ".")

Invoke-Checked python @("-c", "import pytexgen; print('pytexgen', pytexgen.__version__, 'installed at', pytexgen.__file__)")
