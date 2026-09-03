#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

section "Building nuSQuIDS 1.13.3"

SOURCE="${SRC}/nusquids"
WORK="${BUILD}/nusquids"

require_dir "${SOURCE}"

# nuSQuIDS' build system assumes src/, include/, resources/, data/
# are relative to the current working directory.
rm -rf "${WORK}"
mkdir -p "${WORK}"

cp -a "${SOURCE}/." "${WORK}/"
rm -rf "${WORK}/.git"

cd "${WORK}"

# CC/CXX come from the environment exported by common.sh.
# Do NOT pass CC=... or CXX=... as configure arguments.
./configure \
    --prefix="${PREFIX}" \
    --libdir=lib \
    --bindir=bin \
    --with-gsl="${PREFIX}" \
    --with-hdf5="${PREFIX}" \
    --with-squids="${PREFIX}"

# Sanity check: make sure the generated Makefile really uses icpx.
grep -q "^CXX=${CXX}$" Makefile ||
    fatal "nuSQuIDS configure did not select the expected compiler: ${CXX}"

# nuSQuIDS overwrites CXXFLAGS in its generated Makefile.
# Override it at make time so the actual C++ compilation receives
# our native optimization flags.
NUSQUIDS_CXXFLAGS="-std=c++11 -O3 -march=native"

make -j"${JOBS}" \
    CXXFLAGS="${NUSQUIDS_CXXFLAGS}"

make install \
    CXXFLAGS="${NUSQUIDS_CXXFLAGS}"

echo
echo "[done] nuSQuIDS"
