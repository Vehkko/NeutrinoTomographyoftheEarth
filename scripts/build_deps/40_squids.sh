#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

section "Building SQuIDS"

SOURCE="${SRC}/squids"
BDIR="${BUILD}/squids"

require_dir "${SOURCE}"

rm -rf "${BDIR}"
mkdir -p "${BDIR}"

# SQuIDS' configure script assumes resources/, src/, include/, etc.
# are present relative to the working directory.
cp -a "${SOURCE}/." "${BDIR}/"
rm -rf "${BDIR}/.git"

cd "${BDIR}"

./configure \
    --prefix="${PREFIX}" \
    --libdir=lib \
    --with-gsl="${PREFIX}"

# Verify that our LTO toolchain reached the generated Makefile.
grep -Fq -- "-flto=full" settings.mk ||
    fatal "SQuIDS did not inherit -flto=full"

grep -Fq "AR:=${AR}" settings.mk ||
    fatal "SQuIDS did not select llvm-ar"

make -j"${JOBS}"
make install

echo "[done] SQuIDS"
