# Neutrino Tomography of the Earth

A numerical framework for neutrino tomography of the Earth using
atmospheric neutrinos, nuSQuIDS propagation, detector response,
and Bayesian inference with MultiNest.

## Status

This repository is under active development and refactoring.

The numerical workflows and analysis interfaces are not yet considered stable.
This README currently focuses on reproducing the build environment.

## Requirements

Host tools:

- Linux x86-64
- Intel oneAPI
  - `llvm-ar`
  - Intel C/C++ Compiler (`icx`, `icpx`)
  - Intel Fortran Compiler (`ifx`)
  - Intel MPI
  - oneMKL
- Xmake
- uv
- Git
- CMake
- Make
- pkg-config

Project-managed dependencies include:

- zlib
- GSL
- HDF5
- SQuIDS
- nuSQuIDS
- MultiNest
- Python 3.12
- DaemonFlux

## Dependency setup

Activate Intel oneAPI first:

```bash
source /path/to/intel/oneapi/setvars.sh
```

The build scripts expect the Intel compiler, MKL, and Intel MPI environments
to be available before dependency configuration begins.

Prepare all project-managed dependencies:

```bash
./scripts/build_deps/build_all.sh
```

The dependency build is project-local.

Native libraries and headers are installed under:

```text
third_party/local/
```

Build trees are kept under:

```text
third_party/build/
```

The Python runtime is managed by `uv` and kept entirely inside:

```text
runtime/python/
```

The project does not rely on the system Python environment.

The dependency scripts explicitly use the Intel compiler toolchain and Intel
MPI wrappers. Generic `mpicc`, `mpicxx`, and `mpirun` from the system
environment are intentionally avoided to prevent accidental OpenMPI mixing.

The number of parallel build jobs can be overridden with:

```bash
JOBS=16 ./scripts/build_deps/build_all.sh
```

## Build

Configure a release build:

```bash
xmake f -m release
```

Build all targets:

```bash
xmake
```

A debug build can be configured with:

```bash
xmake f -m debug
xmake
```

The release configuration uses native CPU optimization and link-time
optimization.

## Environment tests

After building, the numerical toolchain can be checked independently with:

```bash
xmake run test_stack
xmake run test_mkl
xmake run test_mpi
xmake run test_multinest
xmake run test_multinest_mpi
```

These tests are useful for distinguishing compiler, MPI, MultiNest, and native
dependency problems from issues in the tomography code itself.

## Clean rebuild

To remove generated build products, locally installed dependencies, and the
project-local Python runtime:

```bash
./scripts/build_deps/clean_build.sh
```

Then rebuild from a clean state with:

```bash
source /path/to/intel/oneapi/setvars.sh
./scripts/build_deps/build_all.sh
xmake f -m release
xmake
```

## License

Project license to be determined.

Third-party components retain their respective licenses.
See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details.

---

## **USE XMAKE!!!**

**STOP BUILDING THE BUILD! BUILD THE DAMN PROJECT!**
