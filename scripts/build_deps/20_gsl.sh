#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

section "Building GSL 2.8"

SOURCE="${SRC}/gsl"
BDIR="${BUILD}/gsl"

require_dir "${SOURCE}"

rm -rf "${BDIR}"
mkdir -p "${BDIR}"

cd "${BDIR}"

"${SOURCE}/configure" \
    --prefix="${PREFIX}" \
    --libdir="${PREFIX}/lib" \
    --enable-shared \
    --enable-static \
    CC="${CC}" \
    CFLAGS="${CFLAGS}"

make -j"${JOBS}"
make install

echo "[done] GSL"
