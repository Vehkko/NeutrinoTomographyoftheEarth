set_project("NeutrinoTomography")
set_version("0.1.0")

set_languages("cxx17")

add_rules("mode.debug", "mode.release")
set_defaultmode("release")
add_rules("plugin.compile_commands.autoupdate", { outputdir = "." })

-- ============================================================
-- Fixed compiler toolchain
-- ============================================================
local cmplr_root = os.getenv("CMPLR_ROOT")

if not cmplr_root then
	raise("CMPLR_ROOT is not set; activate Intel oneAPI setvars.sh first")
end

local intel_llvm_bin = path.join(cmplr_root, "bin", "compiler")
local llvm_ar = path.join(intel_llvm_bin, "llvm-ar")

-- Intel oneAPI icx/icpx use a Clang-compatible driver interface.
-- The generic@actual form avoids depending on whether a particular
-- Xmake version recognizes Intel LLVM compiler names directly.
set_toolset("cc", "clang@icx")
set_toolset("cxx", "clang++@icpx")
set_toolset("ld", "clang++@icpx")
set_toolset("sh", "clang++@icpx")
set_toolset("ar", llvm_ar)

-- ============================================================
-- Paths
-- ============================================================

-- Keep paths as Xmake builtin-variable strings in description scope.
-- Do not use script-scope error handling here.
local prefix = "$(projectdir)/third_party/local"
local include_dir = "$(projectdir)/third_party/local/include"
local lib_dir = "$(projectdir)/third_party/local/lib"

local nt_include = "$(projectdir)/include"
local vndarray_include = "$(projectdir)/include/VNdArray/include"

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
		add_defines("NDEBUG")

		add_cxxflags("-O3", "-march=native", "-mtune=native", "-flto=full", { force = true })

		add_ldflags("-flto=full", { force = true })
	else
		add_cxxflags("-O0", "-g", { force = true })
	end

	-- before_build() runs in script scope, where os/path/raise are valid.
	before_build(function(target)
		local local_prefix = path.join(os.projectdir(), "third_party", "local")

		if not os.isdir(local_prefix) then
			raise("third_party/local not found. " .. "Run scripts/build_deps/build_all.sh first.")
		end
	end)
end

function apply_native_stack()
	apply_common()

	-- Hot numerical stack.
	--
	-- Use the LTO static archives explicitly so the final icpx link can
	-- optimize across:
	--
	--     project -> nuSQuIDS -> SQuIDS -> GSL
	--
	-- Keep HDF5/zlib dynamic because they are startup/I/O dependencies,
	-- not likelihood hot-path libraries.
	add_links(
		"$(projectdir)/third_party/local/lib/libnuSQuIDS.a",
		"$(projectdir)/third_party/local/lib/libSQuIDS.a",
		"$(projectdir)/third_party/local/lib/libgsl.a",
		"$(projectdir)/third_party/local/lib/libgslcblas.a",

		"hdf5_hl_cpp",
		"hdf5_cpp",
		"hdf5_hl",
		"hdf5",
		"z"
	)
end

function apply_mkl()
	-- MKLROOT is expanded by Xmake itself.
	add_includedirs("$(env MKLROOT)/include")
	add_ldflags("-qmkl=sequential", { force = true })
end

function apply_intel_mpi()
	-- Intel-specific wrapper names are used deliberately.
	-- We never fall back to generic mpicc/mpicxx, which may be OpenMPI.
	set_toolset("cc", "clang@mpiicx")
	set_toolset("cxx", "clang++@mpiicpx")
	set_toolset("ld", "clang++@mpiicpx")
	set_toolset("sh", "clang++@mpiicpx")
end

function apply_mpi_run(np)
	on_run(function(target)
		local mpi_root = os.getenv("I_MPI_ROOT")

		if not mpi_root or mpi_root == "" then
			raise("I_MPI_ROOT is not set. " .. "Activate Intel oneAPI before running an MPI target.")
		end

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

add_includedirs(nt_include, vndarray_include, { public = true })

add_files("src/flux.cpp", "src/response.cpp", "src/events.cpp", "src/earth.cpp", "src/propagation.cpp")

-- ============================================================
-- Flux
-- ============================================================

target("test_flux")
set_kind("binary")

apply_native_stack()

add_includedirs(nt_include, vndarray_include)

add_deps("ntcore")
add_files("tests/core/test_flux.cpp")

-- ============================================================
-- Response & Events
-- ============================================================

target("test_response_events")
set_kind("binary")

apply_native_stack()

add_includedirs(nt_include, vndarray_include)

add_deps("ntcore")
add_files("tests/core/test_response_events.cpp")

-- ============================================================
-- Earth & propagation
-- ============================================================

target("test_earth")
set_kind("binary")

apply_native_stack()

add_includedirs(nt_include, vndarray_include)

add_deps("ntcore")
add_files("tests/core/test_earth.cpp")

-- ============================================================
-- Earth model profiles
-- ============================================================

target("case_earth_models")
set_kind("binary")

apply_native_stack()

add_includedirs(nt_include, vndarray_include)

add_deps("ntcore")
add_files("cases/earth_models/main.cpp")

-- ============================================================
-- TRIDENT event distribution
-- ============================================================

target("case_trident_events")
set_kind("binary")

apply_native_stack()

add_includedirs(nt_include, vndarray_include)

add_deps("ntcore")
add_files("cases/trident_events/main.cpp")

-- ============================================================
-- Five-layer constant-density tomography with MultiNest
-- ============================================================

target("case_tomography_mnest")
set_kind("binary")

apply_native_stack()
apply_mkl()
apply_intel_mpi()
apply_mpi_run(24)

add_links("multinest_mpi")

add_deps("ntcore")
add_files("cases/tomography_mnest/main.cpp")

-- ============================================================
-- Five-layer constant-density MultiNest settings scan
-- ============================================================

target("case_tomography_mnest_settings")
set_kind("binary")

apply_native_stack()
apply_mkl()
apply_intel_mpi()

add_links("multinest_mpi")

add_deps("ntcore")
add_files("cases/tomography_mnest_settings/main.cpp")

on_run(function(target)
	local mpi_root = os.getenv("I_MPI_ROOT")

	if not mpi_root or mpi_root == "" then
		raise("I_MPI_ROOT is not set. Activate Intel oneAPI before running an MPI target.")
	end

	local mpiexec = path.join(mpi_root, "bin", "mpiexec")

	if not os.isfile(mpiexec) then
		raise("Intel MPI launcher not found: " .. mpiexec)
	end

	local runs = {
		"baseline",
		"nlive_2000",
		"nlive_3000",
		"tol_0p01",
		"prior_3",
		"prior_5",
	}

	for _, run in ipairs(runs) do
		os.execv(mpiexec, {
			"-n",
			"24",
			target:targetfile(),
			"--run",
			run,
		}, {
			curdir = os.projectdir(),
		})
	end

	os.execv(target:targetfile(), {
		"--summarize",
	}, {
		curdir = os.projectdir(),
	})
end)

-- ============================================================
-- MultiNest model comparison
-- ============================================================

target("case_tomography_mnest_model_comparison_unconstrained")
set_kind("binary")

apply_native_stack()
apply_mkl()
apply_intel_mpi()

add_links("multinest_mpi")

add_deps("ntcore")
add_files("cases/tomography_mnest_model_comparison_unconstrained/main.cpp")

on_run(function(target)
	local mpi_root = os.getenv("I_MPI_ROOT")

	if not mpi_root or mpi_root == "" then
		raise("I_MPI_ROOT is not set. Activate Intel oneAPI before running an MPI target.")
	end

	local mpiexec = path.join(mpi_root, "bin", "mpiexec")

	if not os.isfile(mpiexec) then
		raise("Intel MPI launcher not found: " .. mpiexec)
	end

	local ranks = tonumber(os.getenv("NT_MPI_RANKS") or "24")

	os.execv(target:targetfile(), {
		"--prepare",
	}, {
		curdir = os.projectdir(),
	})

	for run = 0, 7 do
		os.execv(mpiexec, {
			"-n",
			tostring(ranks),
			target:targetfile(),
			"--run",
			tostring(run),
		}, {
			curdir = os.projectdir(),
		})
	end
end)

-- ============================================================
-- Constrained Five-layer constant-density MultiNest settings scan
-- ============================================================

target("case_tomography_mnest_constrained")
set_kind("binary")

apply_native_stack()
apply_mkl()
apply_intel_mpi()
apply_mpi_run(24)

add_links("multinest_mpi")
add_deps("ntcore")
add_files("cases/tomography_mnest_constrained/main.cpp")

-- ============================================================
-- Five-layer constant density + five daemonflux nuisances
-- ============================================================

target("case_tomography_mnest_5unc")
set_kind("binary")

apply_native_stack()
apply_mkl()
apply_intel_mpi()
apply_mpi_run(24)

add_links("multinest_mpi")
add_deps("ntcore")
add_files("cases/tomography_mnest_5unc/main.cpp")
