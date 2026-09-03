# Neutrino Tomography of the Earth

A numerical framework for neutrino tomography of the Earth using
atmospheric neutrinos, nuSQuIDS propagation, detector response,
and Bayesian inference with MultiNest.

## Status

This repository is under active development and refactoring.

## Requirements

Host tools:

- Linux x86-64
- Intel oneAPI
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

## Build

Activate Intel oneAPI first:

```bash
source /path/to/intel/oneapi/setvars.sh
```

Prepare project dependencies:

```
./scripts/build_deps/build_all.sh
```

Configure and build:

```
xmake f -m release
xmake
```

Run environment tests:

```
xmake run test_stack
xmake run test_mkl
xmake run test_mpi
xmake run test_multinest
xmake run test_multinest_mpi
```

## License

To be determined.
