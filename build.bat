@echo off
setlocal

cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv is required. Install it first, then rerun build.bat.
    exit /b 1
)

if not defined VIRTUAL_ENV (
    if not exist ".venv\Scripts\python.exe" (
        uv venv .venv
        if errorlevel 1 exit /b 1
    )
    set "VIRTUAL_ENV=%CD%\.venv"
    set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
)

if not defined CMAKE_BUILD_PARALLEL_LEVEL set "CMAKE_BUILD_PARALLEL_LEVEL=1"

echo Using Python environment: %VIRTUAL_ENV%
echo CMAKE_BUILD_PARALLEL_LEVEL=%CMAKE_BUILD_PARALLEL_LEVEL%

uv pip install --upgrade pip setuptools wheel scikit-build-core cmake ninja numpy scipy matplotlib
if errorlevel 1 exit /b 1

uv pip install --reinstall-package pytexgen --no-build-isolation .
if errorlevel 1 exit /b 1

python -c "import pytexgen; print('pytexgen', pytexgen.__version__, 'installed at', pytexgen.__file__)"
if errorlevel 1 exit /b 1

endlocal
