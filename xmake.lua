set_project("NeutrinoTomography")
set_version("0.1.0")

set_languages("cxx17")

add_rules("mode.debug", "mode.release")
set_defaultmode("release")
add_rules("plugin.compile_commands.autoupdate", { outputdir = "." })

-- ============================================================
-- Fixed compiler toolchain
-- ============================================================

set_toolset("cc", "icx")
set_toolset("cxx", "icpx")
set_toolset("ld", "icpx")
set_toolset("sh", "icpx")

-- ============================================================
-- Paths
-- ============================================================

local root = os.projectdir()

local prefix = path.join(root, "third_party", "local")
local include_dir = path.join(prefix, "include")
local lib_dir = path.join(prefix, "lib")

local nt_include = path.join(root, "include")
local vndarray_include = path.join(nt_include, "VNdArray", "include")

local mpi_root = os.getenv("I_MPI_ROOT")
local mkl_root = os.getenv("MKLROOT")

if not os.isdir(prefix) then
    raise("third_party/local not found. Run scripts/build_deps/build_all.sh first.")
end

if not mpi_root then
    raise("I_MPI_ROOT is not set. Activate Intel oneAPI first.")
end

if not mkl_root then
    raise("MKLROOT is not set. Activate Intel oneAPI first.")
end

local mpi_cc = path.join(mpi_root, "bin", "mpiicx")
local mpi_cxx = path.join(mpi_root, "bin", "mpiicpx")

if not os.isfile(mpi_cc) or not os.isfile(mpi_cxx) then
    raise("Intel MPI compiler wrappers not found under I_MPI_ROOT.")
end

-- ============================================================
-- Common helpers
-- ============================================================

function apply_common()
    add_includedirs(include_dir)
    add_includedirs(nt_include)
    add_includedirs(vndarray_include)

    add_linkdirs(lib_dir)

    -- Build products are machine-local and disposable, so embedding the
    -- resolved project-local library directory as RPATH is intentional.
    add_rpathdirs(lib_dir)

    add_syslinks("m", "pthread", "dl")

    set_rundir("$(projectdir)")

    if is_mode("release") then
        add_cxxflags("-O3", "-march=native", { force = true })
    else
        add_cxxflags("-O0", "-g", { force = true })
    end
end

function apply_native_stack()
    apply_common()

    add_links("nuSQuIDS", "SQuIDS", "hdf5_hl_cpp", "hdf5_cpp", "hdf5_hl", "hdf5", "gsl", "gslcblas", "z")
end

function apply_mkl()
    add_includedirs(path.join(mkl_root, "include"))
    add_ldflags("-qmkl=sequential", { force = true })
end

function apply_intel_mpi()
    -- Never use generic mpicc/mpicxx from PATH.
    -- On Arch they may be OpenMPI.
    set_toolset("cc", "icx@" .. mpi_cc)
    set_toolset("cxx", "icpx@" .. mpi_cxx)
    set_toolset("ld", "icpx@" .. mpi_cxx)
    set_toolset("sh", "icpx@" .. mpi_cxx)
end

function apply_mpi_run(np)
    on_run(function(target)
        local mpiexec = path.join(mpi_root, "bin", "mpiexec")

        if not os.isfile(mpiexec) then
            raise("Intel MPI launcher not found: " .. mpiexec)
        end

        os.execv(mpiexec, {
            "-n",
            tostring(np),
            target:targetfile(),
        }, {
            curdir = os.projectdir(),
        })
    end)
end

-- ============================================================
-- Native scientific stack
-- ============================================================

target("test_stack")
set_kind("binary")

apply_native_stack()

add_files("tests/env/test_stack.cpp")

-- ============================================================
-- MKL
-- ============================================================
target("test_mkl")
set_kind("binary")

apply_common()
apply_mkl()

add_files("tests/env/test_mkl.cpp")

-- ============================================================
-- Intel MPI
-- ============================================================

target("test_mpi")
set_kind("binary")

apply_common()
apply_intel_mpi()
apply_mpi_run(4)

add_files("tests/env/test_mpi.cpp")

-- ============================================================
-- Serial MultiNest
-- ============================================================

target("test_multinest")
set_kind("binary")

apply_common()
apply_mkl()

add_links("multinest")

add_files("tests/env/test_multinest.cpp")

-- ============================================================
-- MPI MultiNest
-- ============================================================

target("test_multinest_mpi")
set_kind("binary")

apply_common()
apply_mkl()
apply_intel_mpi()
apply_mpi_run(4)

add_links("multinest_mpi")
add_defines("NT_TEST_MULTINEST_MPI")

add_files("tests/env/test_multinest.cpp")

-- ============================================================
-- Neutrino tomography core
-- ============================================================

target("ntcore")
set_kind("static")

apply_native_stack()

add_includedirs(path.join(root, "include"), vndarray_include, { public = true })

add_files("src/flux.cpp", "src/response.cpp", "src/events.cpp", "src/earth.cpp")

-- ============================================================
-- Flux
-- ============================================================

target("test_flux")
set_kind("binary")

apply_native_stack()

add_includedirs(path.join(root, "include"), vndarray_include)

add_deps("ntcore")
add_files("tests/core/test_flux.cpp")

-- ============================================================
-- Response & Events
-- ============================================================
target("test_response_events")
set_kind("binary")

apply_native_stack()

add_includedirs(path.join(root, "include"), vndarray_include)

add_deps("ntcore")
add_files("tests/core/test_response_events.cpp")

-- ============================================================
-- Earth & propagation
-- ============================================================

target("test_earth")
set_kind("binary")

apply_native_stack()

add_includedirs(path.join(root, "include"), vndarray_include)

add_deps("ntcore")
add_files("tests/core/test_earth.cpp")
