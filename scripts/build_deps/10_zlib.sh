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
    -DCMAKE_AR="${AR}" \
    -DCMAKE_RANLIB="${RANLIB}" \
    -DCMAKE_NM="${NM}" \
    -DCMAKE_C_FLAGS_RELEASE="${CFLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS_RELEASE="${LDFLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS_RELEASE="${LDFLAGS}" \
    -DBUILD_SHARED_LIBS=ON

cmake --build "${BDIR}" --parallel "${JOBS}"
cmake --install "${BDIR}"

echo "[done] zlib"
