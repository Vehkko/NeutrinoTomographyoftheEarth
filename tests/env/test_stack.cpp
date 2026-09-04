#include <H5Cpp.h>
#include <gsl/gsl_integration.h>
#include <nuSQuIDS/nuSQuIDS.h>
#include <zlib.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace fs = std::filesystem;

// ============================================================
// zlib
// ============================================================

void test_zlib() {
    std::cout << "[TEST] zlib\n";
    std::cout << "       version = " << zlibVersion() << '\n';
}

// ============================================================
// GSL
// ============================================================

double square_integrand(double x, void*) { return x * x; }

void test_gsl() {
    std::cout << "[TEST] GSL\n";

    gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(1000);

    if (!workspace) {
        throw std::runtime_error("gsl_integration_workspace_alloc failed");
    }

    gsl_function F{};
    F.function = &square_integrand;
    F.params   = nullptr;

    double result = 0.0;
    double error  = 0.0;

    const int status = gsl_integration_qags(&F, 0.0, 1.0, 0.0, 1e-12, 1000, workspace, &result, &error);

    gsl_integration_workspace_free(workspace);

    if (status != GSL_SUCCESS) {
        throw std::runtime_error("GSL integration failed");
    }

    if (std::abs(result - 1.0 / 3.0) > 1e-10) {
        throw std::runtime_error("GSL returned an incorrect result");
    }

    std::cout << "       integral(x^2, 0, 1) = " << result << '\n';
}

// ============================================================
// HDF5 + zlib compression
// ============================================================

void test_hdf5() {
    std::cout << "[TEST] HDF5 + zlib filter\n";

    const fs::path dir       = "build/test_tmp";
    const fs::path file_path = dir / "environment_test.h5";

    fs::create_directories(dir);

    {
        H5::H5File file(file_path.string(), H5F_ACC_TRUNC);

        hsize_t       dims[1] = {4};
        H5::DataSpace space(1, dims);

        H5::DSetCreatPropList plist;

        hsize_t chunk[1] = {4};
        plist.setChunk(1, chunk);
        plist.setDeflate(6);

        H5::DataSet dataset = file.createDataSet("values", H5::PredType::NATIVE_DOUBLE, space, plist);

        const double input[4] = {1.0, 2.0, 3.0, 4.0};

        dataset.write(input, H5::PredType::NATIVE_DOUBLE);
    }

    {
        H5::H5File  file(file_path.string(), H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("values");

        double output[4] = {};

        dataset.read(output, H5::PredType::NATIVE_DOUBLE);

        for (int i = 0; i < 4; ++i) {
            if (output[i] != static_cast<double>(i + 1)) {
                throw std::runtime_error("HDF5 readback mismatch");
            }
        }
    }

    fs::remove(file_path);

    std::cout << "       compressed write/read OK\n";
}

// ============================================================
// SQuIDS + nuSQuIDS
// ============================================================

void test_nusquids() {
    std::cout << "[TEST] SQuIDS + nuSQuIDS\n";

    const squids::Const units;

    nusquids::marray<double, 1> cosz{2};
    cosz[0] = -1.0;
    cosz[1] = -0.5;

    nusquids::marray<double, 1> energy{3};
    energy[0] = 100.0;
    energy[1] = 1000.0;
    energy[2] = 10000.0;

    nusquids::nuSQUIDSAtm<> nus(cosz, energy * units.GeV, 3, nusquids::both, false);

    auto earth = std::make_shared<nusquids::EarthAtm>();
    nus.Set_EarthModel(earth);

    nus.Set_MixingAngle(0, 1, 0.563942);
    nus.Set_MixingAngle(0, 2, 0.154085);
    nus.Set_MixingAngle(1, 2, 0.785398);

    nus.Set_SquareMassDifference(1, 7.65e-5);
    nus.Set_SquareMassDifference(2, 2.47e-3);

    nus.Set_rel_error(1e-5);
    nus.Set_abs_error(1e-5);

    nusquids::marray<double, 4> state{2, 3, 2, 3};

    std::fill(state.begin(), state.end(), 0.0);

    for (std::size_t iz = 0; iz < 2; ++iz) {
        for (std::size_t ie = 0; ie < 3; ++ie) {
            state[iz][ie][0][1] = 1.0;
            state[iz][ie][1][1] = 1.0;
        }
    }

    nus.Set_initial_state(state);
    nus.EvolveState();

    const double value = nus.EvalFlavor(1, -1.0, 1000.0 * units.GeV, 0);

    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error("nuSQuIDS returned an invalid flux");
    }

    std::cout << "       propagated flux = " << value << '\n';
}

// ============================================================

int main() {
    try {
        test_zlib();
        test_gsl();
        test_hdf5();
        test_nusquids();

        std::cout << "\n[OK] native scientific stack works\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] " << e.what() << '\n';

        return 1;
    }
}
