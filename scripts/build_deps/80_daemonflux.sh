#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Project paths
# ============================================================

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

PROJECT_ROOT="$(
    cd -- "${SCRIPT_DIR}/../.."
    pwd
)"

# ============================================================
# Project-local Python environment
# ============================================================

PYTHON_ROOT="${PROJECT_ROOT}/runtime/python"
PYTHON_VENV="${PYTHON_ROOT}/venv"
PYTHON="${PYTHON_VENV}/bin/python"

GENERATOR="${PROJECT_ROOT}/tools/daemonflux/generate.py"
OUTPUT="${PROJECT_ROOT}/data/generated/daemonflux/daemonflux_0.8.2.h5"

# ============================================================
# Validate required files
# ============================================================

if [[ ! -x "${PYTHON}" ]]; then
    echo "[fatal] project Python environment not found:" >&2
    echo "        ${PYTHON}" >&2
    echo >&2
    echo "        Run 70_python.sh first." >&2
    exit 1
fi

if [[ ! -f "${GENERATOR}" ]]; then
    echo "[fatal] DaemonFlux generator not found:" >&2
    echo "        ${GENERATOR}" >&2
    exit 1
fi

# ============================================================
# Keep Python runtime state inside the project
# ============================================================

#
# Temporary files.
#
export TMPDIR="${PYTHON_ROOT}/tmp"

#
# Generic XDG caches/config/data used by Python packages.
#
export XDG_CACHE_HOME="${PYTHON_ROOT}/xdg/cache"
export XDG_CONFIG_HOME="${PYTHON_ROOT}/xdg/config"
export XDG_DATA_HOME="${PYTHON_ROOT}/xdg/data"

mkdir -p \
    "${TMPDIR}" \
    "${XDG_CACHE_HOME}" \
    "${XDG_CONFIG_HOME}" \
    "${XDG_DATA_HOME}"

# Do not create __pycache__ directories next to project
# Python source files.
export PYTHONDONTWRITEBYTECODE=1

# Never import packages from the user's Python user-site.
export PYTHONNOUSERSITE=1

# Prevent an externally configured Python environment from
# contaminating this project-local interpreter.
unset PYTHONPATH || true
unset PYTHONHOME || true

# ============================================================
# Environment verification
# ============================================================

echo
echo "============================================================"
echo "Preparing DaemonFlux data"
echo "============================================================"
echo
echo "[project] ${PROJECT_ROOT}"
echo "[python]  ${PYTHON}"
echo

export NT_EXPECTED_PYTHON_VENV="${PYTHON_VENV}"

"${PYTHON}" - <<'PY'
from __future__ import annotations

import os
import sys
from importlib.metadata import version
from pathlib import Path

import daemonflux


expected_venv = Path(
    os.environ["NT_EXPECTED_PYTHON_VENV"]
).resolve()

actual_prefix = Path(
    sys.prefix
).resolve()

executable = Path(
    sys.executable
).resolve()

daemonflux_file = Path(
    daemonflux.__file__
).resolve()


print(
    "[python] executable :",
    sys.executable,
)

print(
    "[python] prefix     :",
    sys.prefix,
)

print(
    "[python] version    :",
    sys.version.split()[0],
)

print(
    "[python] daemonflux :",
    version("daemonflux"),
)

print(
    "[python] package    :",
    daemonflux.__file__,
)


if actual_prefix != expected_venv:
    raise RuntimeError(
        "Python interpreter is not using the expected "
        "project-local virtual environment:\n"
        f"  expected: {expected_venv}\n"
        f"  actual:   {actual_prefix}"
    )


if (
    daemonflux_file != expected_venv
    and expected_venv not in daemonflux_file.parents
):
    raise RuntimeError(
        "DaemonFlux was imported from outside the "
        "project-local virtual environment:\n"
        f"  venv:     {expected_venv}\n"
        f"  package:  {daemonflux_file}"
    )


installed_version = version(
    "daemonflux"
)

if installed_version != "0.8.2":
    raise RuntimeError(
        "Unexpected DaemonFlux version: "
        f"{installed_version}; expected 0.8.2"
    )


print(
    "[OK] project-local Python environment verified"
)
PY

echo

# ============================================================
# Generate data
# ============================================================

if [[ "${FORCE:-0}" == "1" ]]; then
    "${PYTHON}" \
        "${GENERATOR}" \
        --output "${OUTPUT}" \
        --force
else
    "${PYTHON}" \
        "${GENERATOR}" \
        --output "${OUTPUT}"
fi
