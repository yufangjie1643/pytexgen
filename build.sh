#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required. Install it first, then rerun build.sh." >&2
    exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -x ".venv/bin/python" ]; then
        if [ -n "${PYTHON:-}" ]; then
            uv venv --python "$PYTHON" .venv
        else
            uv venv .venv
        fi
    fi
    VIRTUAL_ENV="$(pwd)/.venv"
    export VIRTUAL_ENV
    PATH="$VIRTUAL_ENV/bin:$PATH"
    export PATH
fi

: "${CMAKE_BUILD_PARALLEL_LEVEL:=1}"
export CMAKE_BUILD_PARALLEL_LEVEL

echo "Using Python environment: $VIRTUAL_ENV"
echo "CMAKE_BUILD_PARALLEL_LEVEL=$CMAKE_BUILD_PARALLEL_LEVEL"

uv pip install --upgrade pip setuptools wheel scikit-build-core cmake ninja numpy scipy matplotlib
uv pip install --reinstall-package pytexgen --no-build-isolation .

python -c "import pytexgen; print('pytexgen', pytexgen.__version__, 'installed at', pytexgen.__file__)"
