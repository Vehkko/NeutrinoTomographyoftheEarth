#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

section "Building MultiNest 3.12"

SOURCE="${SRC}/multinest/MultiNest_v3.12_CMake/multinest"
WORK="${BUILD}/multinest/source"
BDIR="${BUILD}/multinest/build"

require_dir "${SOURCE}"

rm -rf "${BUILD}/multinest"
mkdir -p "${WORK}" "${BDIR}"

# MultiNest 的 CMake 会往源码目录写东西，所以在副本里构建
cp -a "${SOURCE}/." "${WORK}/"

# ------------------------------------------------------------
# MultiNest 3.12 太老，只识别 __INTEL_COMPILER。
# icx/icpx 使用 __INTEL_LLVM_COMPILER，同时又定义 __GNUC__，
# 因而原版 header 会误判成 gfortran ABI。
#
# 源码树里不止一份 multinest.h，所以全部修改。
# ------------------------------------------------------------

while IFS= read -r -d '' header; do
    echo "[patch] ${header}"

    sed -i \
        's/^#ifdef __INTEL_COMPILER/#if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)/' \
        "${header}"
done < <(find "${WORK}" -type f -name 'multinest.h' -print0)

# 确认至少主 header 已经修改
grep -q '__INTEL_LLVM_COMPILER' "${WORK}/include/multinest.h" ||
    fatal "Failed to patch MultiNest headers"

# ------------------------------------------------------------
# Configure
# ------------------------------------------------------------

cmake \
    -S "${WORK}" \
    -B "${BDIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    \
    -DCMAKE_C_COMPILER="${MPI_CC}" \
    -DCMAKE_CXX_COMPILER="${MPI_CXX}" \
    -DCMAKE_Fortran_COMPILER="${MPI_FC}" \
    \
    -DMPI_C_COMPILER="${MPI_CC}" \
    -DMPI_CXX_COMPILER="${MPI_CXX}" \
    -DMPI_Fortran_COMPILER="${MPI_FC}" \
    \
    -DCMAKE_C_FLAGS_RELEASE="${CFLAGS}" \
    -DCMAKE_CXX_FLAGS_RELEASE="${CXXFLAGS}" \
    -DCMAKE_Fortran_FLAGS_RELEASE="${FCFLAGS}" \
    \
    -DBLA_VENDOR=Intel10_64lp_seq \
    -DBLA_STATIC=OFF \
    -DCMAKE_PREFIX_PATH="${MKLROOT};${PREFIX}"

# MultiNest 老 Fortran 工程的 module 依赖不适合并行构建
cmake --build "${BDIR}" --parallel 1

cmake --install "${BDIR}"

# ------------------------------------------------------------
# Basic verification
# ------------------------------------------------------------

MN_LIB="${PREFIX}/lib/libmultinest_mpi.so"

[[ -f "${MN_LIB}" ]] ||
    fatal "libmultinest_mpi.so was not installed"

echo
echo "[info] MultiNest MPI linkage:"
ldd "${MN_LIB}"

if ldd "${MN_LIB}" | grep -Eqi 'openmpi|open-rte|open-pal'; then
    fatal "OpenMPI contamination detected"
fi

echo
echo "[done] MultiNest 3.12"
