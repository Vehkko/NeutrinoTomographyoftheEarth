#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

echo "============================================================"
echo "Cleaning build/runtime products"
echo "============================================================"
echo "Project root: ${ROOT}"
echo

TARGETS=(
    "${ROOT}/build"
    "${ROOT}/third_party/build"
    "${ROOT}/third_party/local"
    "${ROOT}/runtime/python"
)

for path in "${TARGETS[@]}"; do
    if [[ -e "${path}" ]]; then
        echo "[remove] ${path}"
        rm -rf -- "${path}"
    else
        echo "[skip]   ${path} (not present)"
    fi
done

echo
echo "Checking..."

failed=0

for path in "${TARGETS[@]}"; do
    if [[ -e "${path}" ]]; then
        echo "[error] still exists: ${path}"
        failed=1
    fi
done

if ((failed)); then
    echo
    echo "[fatal] clean incomplete"
    exit 1
fi

echo
echo "[done] all build/runtime products removed"
