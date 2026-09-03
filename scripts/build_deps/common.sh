#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

THIRD_PARTY="${PROJECT_ROOT}/third_party"
SRC="${THIRD_PARTY}/src"
BUILD="${THIRD_PARTY}/build"
PREFIX="${THIRD_PARTY}/local"

JOBS="${JOBS:-$(nproc)}"

# ============================================================
# Helpers
# ============================================================

fatal() {
	echo "[fatal] $*" >&2
	exit 1
}

require_command() {
	command -v "$1" >/dev/null 2>&1 ||
		fatal "required command not found: $1"
}

require_executable() {
	[[ -x "$1" ]] ||
		fatal "required executable not found: $1"
}

require_dir() {
	[[ -d "$1" ]] ||
		fatal "source directory not found: $1"
}

section() {
	echo
	echo "============================================================"
	echo "$1"
	echo "============================================================"
}

# ============================================================
# Intel oneAPI must already have been activated by the user
# ============================================================

require_command icx
require_command icpx
require_command ifx
require_command cmake
require_command make
require_command pkg-config

[[ -n "${MKLROOT:-}" ]] ||
	fatal "MKLROOT is not set. Activate Intel oneAPI first."

[[ -n "${I_MPI_ROOT:-}" ]] ||
	fatal "I_MPI_ROOT is not set. Activate Intel oneAPI MPI first."

# Resolve compiler paths now, before changing PATH.
CC="$(command -v icx)"
CXX="$(command -v icpx)"
FC="$(command -v ifx)"

# ============================================================
# Intel MPI
#
# NEVER use mpicc/mpicxx/mpifort/mpirun from PATH.
# On e.g. Arch Linux they may belong to system OpenMPI.
# ============================================================

MPI_ROOT="${I_MPI_ROOT}"

MPI_CC="${MPI_ROOT}/bin/mpiicx"
MPI_CXX="${MPI_ROOT}/bin/mpiicpx"
MPI_FC="${MPI_ROOT}/bin/mpiifx"

require_executable "${MPI_CC}"
require_executable "${MPI_CXX}"
require_executable "${MPI_FC}"

if [[ -x "${MPI_ROOT}/bin/mpiexec.hydra" ]]; then
	MPIEXEC="${MPI_ROOT}/bin/mpiexec.hydra"
elif [[ -x "${MPI_ROOT}/bin/mpiexec" ]]; then
	MPIEXEC="${MPI_ROOT}/bin/mpiexec"
else
	fatal "Intel MPI launcher not found below ${MPI_ROOT}/bin"
fi

# Explicitly tell Intel MPI wrappers which Intel compiler to invoke.
export I_MPI_CC="${CC}"
export I_MPI_CXX="${CXX}"
export I_MPI_FC="${FC}"
export I_MPI_F90="${FC}"

# ============================================================
# Optimization
# ============================================================

CFLAGS="-O3 -march=native"
CXXFLAGS="-O3 -march=native"
FCFLAGS="-O3 -march=native"

export CC CXX FC
export CFLAGS CXXFLAGS FCFLAGS

# ============================================================
# Project-local dependencies
# ============================================================

mkdir -p "${BUILD}" "${PREFIX}/bin" "${PREFIX}/include" "${PREFIX}/lib"

# Put our own installed tools first, then Intel MPI, then original PATH.
export PATH="${PREFIX}/bin:${MPI_ROOT}/bin:${PATH}"

export CPATH="${PREFIX}/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${PREFIX}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PREFIX}/share/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}"

# ============================================================
# Guard against accidental OpenMPI use
# ============================================================

check_intel_mpi_wrapper() {
	local output

	output="$("${MPI_FC}" -show 2>&1)" || {
		echo "${output}" >&2
		fatal "Intel MPI compiler wrapper test failed."
	}

	if grep -Eqi 'openmpi|open-mpi|open-rte|open-pal' <<<"${output}"; then
		echo "${output}" >&2
		fatal "OpenMPI contamination detected in Intel MPI wrapper."
	fi
}

check_intel_mpi_wrapper
