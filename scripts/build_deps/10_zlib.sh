#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

section "Building zlib 1.3.2"

SOURCE="${SRC}/zlib"
BDIR="${BUILD}/zlib"

require_dir "${SOURCE}"

rm -rf "${BDIR}"

cmake \
    -S "${SOURCE}" \
    -B "${BDIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_C_FLAGS_RELEASE="${CFLAGS}" \
    -DBUILD_SHARED_LIBS=ON

cmake --build "${BDIR}" --parallel "${JOBS}"
cmake --install "${BDIR}"

echo "[done] zlib"
