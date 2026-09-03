#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo
echo "============================================================"
echo "Building native dependencies"
echo "============================================================"

"${SCRIPT_DIR}/10_zlib.sh"
"${SCRIPT_DIR}/20_gsl.sh"
"${SCRIPT_DIR}/30_hdf5.sh"
"${SCRIPT_DIR}/40_squids.sh"
"${SCRIPT_DIR}/50_nusquids.sh"
"${SCRIPT_DIR}/60_multinest.sh"

echo
echo "============================================================"
echo "Preparing Python dependencies"
echo "============================================================"

"${SCRIPT_DIR}/70_python.sh"

echo
echo "============================================================"
echo "Get Daemonflux Data"
echo "============================================================"
"${SCRIPT_DIR}/80_daemonflux.sh"

echo
echo "============================================================"
echo " All dependencies are ready"
echo "============================================================"
