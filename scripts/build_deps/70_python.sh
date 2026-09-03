#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_ROOT="${PROJECT_ROOT}/runtime/python"
PYTHON_PROJECT="${PROJECT_ROOT}/tools/daemonflux"

PYTHON_VERSION="3.12"

fatal() {
    echo "[fatal] $*" >&2
    exit 1
}

command -v uv >/dev/null 2>&1 ||
    fatal "uv is required"

[[ -f "${PYTHON_PROJECT}/pyproject.toml" ]] ||
    fatal "pyproject.toml not found: ${PYTHON_PROJECT}/pyproject.toml"

# ============================================================
# Keep all uv-managed runtime files inside the project
# ============================================================

export UV_PYTHON_INSTALL_DIR="${PYTHON_ROOT}/installations"
export UV_PROJECT_ENVIRONMENT="${PYTHON_ROOT}/venv"
export UV_CACHE_DIR="${PYTHON_ROOT}/cache"

# Never fall back to system Python.
export UV_MANAGED_PYTHON=1

# Do not install Python executable links into ~/.local/bin.
export UV_PYTHON_INSTALL_BIN=0

mkdir -p "${PYTHON_ROOT}"

echo
echo "============================================================"
echo "Preparing Python environment"
echo "============================================================"
echo
echo "Python root : ${PYTHON_ROOT}"
echo "Environment : ${UV_PROJECT_ENVIRONMENT}"
echo "Cache       : ${UV_CACHE_DIR}"

# ============================================================
# Install project-local Python
# ============================================================

uv python install "${PYTHON_VERSION}" --no-bin

# ============================================================
# Lock dependencies
# ============================================================

cd "${PYTHON_PROJECT}"

if [[ ! -f "uv.lock" ]]; then
    echo
    echo "[python] uv.lock not found; generating it"

    uv lock \
        --managed-python \
        --python "${PYTHON_VERSION}"
fi

# ============================================================
# Create / synchronize project-local environment
# ============================================================
#
# --locked:
#   normal rebuilds never modify the committed uv.lock.
#
# --no-install-project:
#   tools/daemonflux is a collection of scripts, not a Python
#   package that needs to be installed into the environment.
# ============================================================

uv sync \
    --locked \
    --managed-python \
    --python "${PYTHON_VERSION}" \
    --no-install-project

# ============================================================
# Verify
# ============================================================

PYTHON="${UV_PROJECT_ENVIRONMENT}/bin/python"

[[ -x "${PYTHON}" ]] ||
    fatal "project Python environment was not created"

"${PYTHON}" - <<'PY'
import sys
import daemonflux
import numpy
import h5py

print()
print("Python executable :", sys.executable)
print("Python version    :", sys.version.split()[0])
print("DaemonFlux        :", getattr(daemonflux, "__version__", "unknown"))
print("NumPy             :", numpy.__version__)
print("h5py              :", h5py.__version__)
PY

echo
echo "============================================================"
echo " Python environment ready"
echo "============================================================"
