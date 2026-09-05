set_project("NeutrinoTomography")
set_version("0.1.0")

set_languages("cxx17")

add_rules("mode.debug", "mode.release")
set_defaultmode("release")
add_rules("plugin.compile_commands.autoupdate", { outputdir = "." })

-- ============================================================
-- Fixed compiler toolchain
-- ============================================================

-- Intel oneAPI icx/icpx use a Clang-compatible driver interface.
-- The generic@actual form avoids depending on whether a particular
-- Xmake version recognizes Intel LLVM compiler names directly.
set_toolset("cc", "clang@icx")
set_toolset("cxx", "clang++@icpx")
set_toolset("ld", "clang++@icpx")
set_toolset("sh", "clang++@icpx")

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
		add_cxxflags("-O3", "-march=native", { force = true })
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

	add_links("nuSQuIDS", "SQuIDS", "hdf5_hl_cpp", "hdf5_cpp", "hdf5_hl", "hdf5", "gsl", "gslcblas", "z")
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

add_files("src/flux.cpp", "src/response.cpp", "src/events.cpp", "src/earth.cpp")

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
