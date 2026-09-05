#include <nt/flux.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

#include <H5Dpublic.h>
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ipublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <H5public.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <hdf5.h>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

    using nt::Flavor;
    using nt::Flux;
    using nt::Index_t;
    using nt::Particle;
    using nt::Real_t;
    namespace nda = nt::nda;

    constexpr Real_t eps = 1e-12;

    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    void require_close(Real_t actual, Real_t expected, const char* message) {
        const Real_t scale = std::max<Real_t>({1.0, std::abs(actual), std::abs(expected)});
        if (std::abs(actual - expected) > eps * scale) {
            throw std::runtime_error(std::string(message) + ": actual=" + std::to_string(actual) +
                                     ", expected=" + std::to_string(expected));
        }
    }

    // -----------------------------------------------------------------------------
    // Test 1: NDA component views must alias the native nuSQuIDS state.
    // -----------------------------------------------------------------------------

    void test_component_views() {
        Flux flux(2, 3);

        auto numu     = flux.numu();
        auto antinumu = flux.antinumu();

        require(numu.extent(0) == 2 && numu.extent(1) == 3, "numu shape is wrong");
        require(antinumu.extent(0) == 2 && antinumu.extent(1) == 3, "antinumu shape is wrong");

        // Write through NDA view -> native marray must change immediately.
        numu(1, 2) = 42.0;
        require_close(flux.native_state()[1][2][0][1], 42.0, "numu view does not alias native state");

        antinumu(0, 1) = 17.0;
        require_close(flux.native_state()[0][1][1][1], 17.0, "antinumu view does not alias native state");

        // Write through native marray -> NDA view must see the same value.
        flux.native_state()[1][0][0][1] = 23.0;
        require_close(numu(1, 0), 23.0, "native state does not alias numu view");

        // Different components must not alias each other.
        require_close(flux.native_state()[1][2][1][1], 0.0, "numu write corrupted antinumu component");

        std::cout << "[PASS] component views\n";
    }

    // -----------------------------------------------------------------------------
    // Minimal synthetic DaemonFlux HDF5 file.
    // -----------------------------------------------------------------------------

    void write_dataset_1d(hid_t file, const char* path, const Real_t* data, hsize_t n) {
        const hsize_t dims[1] = {n};
        const hid_t   space   = H5Screate_simple(1, dims, nullptr);
        require(space >= 0, "H5Screate_simple failed");

        const hid_t dataset = H5Dcreate2(file, path, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        require(dataset >= 0, "H5Dcreate2 failed");

        require(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0, "H5Dwrite failed");

        H5Dclose(dataset);
        H5Sclose(space);
    }

    void write_dataset_2d(hid_t file, const char* path, const Real_t* data, hsize_t n0, hsize_t n1) {
        const hsize_t dims[2] = {n0, n1};
        const hid_t   space   = H5Screate_simple(2, dims, nullptr);
        require(space >= 0, "H5Screate_simple failed");

        const hid_t dataset = H5Dcreate2(file, path, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        require(dataset >= 0, "H5Dcreate2 failed");

        require(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0, "H5Dwrite failed");

        H5Dclose(dataset);
        H5Sclose(space);
    }

    std::filesystem::path make_test_daemonflux_file() {
        const auto path = std::filesystem::temp_directory_path() / "neutrino_tomography_test_flux.h5";

        std::filesystem::remove(path);

        const hid_t file = H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        require(file >= 0, "H5Fcreate failed");

        const hid_t axes = H5Gcreate2(file, "/axes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        require(axes >= 0, "cannot create /axes");
        H5Gclose(axes);

        const hid_t flux = H5Gcreate2(file, "/flux", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        require(flux >= 0, "cannot create /flux");
        H5Gclose(flux);

        const hid_t location = H5Gcreate2(file, "/flux/Test", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        require(location >= 0, "cannot create location");
        H5Gclose(location);

        const hid_t calibrated = H5Gcreate2(file, "/flux/Test/calibrated", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        require(calibrated >= 0, "cannot create calibrated group");
        H5Gclose(calibrated);

        const Real_t coszenith[] = {-1.0, 0.0};
        const Real_t energy[]    = {100.0, 200.0, 400.0};

        // Stored HDF5 layout is [coszenith, energy].
        const Real_t numu[] = {
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        };

        const Real_t antinumu[] = {
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        };

        write_dataset_1d(file, "/axes/coszenith", coszenith, 2);
        write_dataset_1d(file, "/axes/energy_GeV", energy, 3);
        write_dataset_2d(file, "/flux/Test/calibrated/numu", numu, 2, 3);
        write_dataset_2d(file, "/flux/Test/calibrated/antinumu", antinumu, 2, 3);

        H5Fclose(file);
        return path;
    }

    // -----------------------------------------------------------------------------
    // Test 2: HDF5 [z,E] must land directly in the correct native-state slices.
    // -----------------------------------------------------------------------------

    void test_daemonflux_loader() {
        const auto path = make_test_daemonflux_file();
        auto       flux = nt::load_daemonflux(path, "Test");

        require(flux.n_coszenith() == 2, "loaded coszenith size is wrong");
        require(flux.n_energy() == 3, "loaded energy size is wrong");

        require_close(flux.coszenith()(0), -1.0, "wrong coszenith[0]");
        require_close(flux.coszenith()(1), 0.0, "wrong coszenith[1]");
        require_close(flux.energy_gev()(0), 100.0, "wrong energy[0]");
        require_close(flux.energy_gev()(2), 400.0, "wrong energy[2]");

        const auto numu     = flux.numu();
        const auto antinumu = flux.antinumu();

        const Real_t expected_numu[2][3] = {
            {1, 2, 3},
            {4, 5, 6}
        };
        const Real_t expected_antinumu[2][3] = {
            {11, 12, 13},
            {14, 15, 16}
        };

        for (Index_t z = 0; z < 2; ++z) {
            for (Index_t e = 0; e < 3; ++e) {
                require_close(numu(z, e), expected_numu[z][e], "wrong loaded numu");
                require_close(antinumu(z, e), expected_antinumu[z][e], "wrong loaded antinumu");
            }
        }

        // DaemonFlux currently supplies only νμ and ν̄μ. All other components
        // must remain zero after the HDF5 hyperslab reads.
        for (Index_t z = 0; z < 2; ++z) {
            for (Index_t e = 0; e < 3; ++e) {
                require_close(flux.component(Particle::neutrino, Flavor::electron)(z, e), 0.0,
                              "nue should remain zero");
                require_close(flux.component(Particle::neutrino, Flavor::tau)(z, e), 0.0, "nutau should remain zero");
                require_close(flux.component(Particle::antineutrino, Flavor::electron)(z, e), 0.0,
                              "antinue should remain zero");
                require_close(flux.component(Particle::antineutrino, Flavor::tau)(z, e), 0.0,
                              "antinutau should remain zero");
            }
        }

        std::filesystem::remove(path);
        std::cout << "[PASS] DaemonFlux loader\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: bilinear interpolation and boundary extrapolation.
    //
    // A bilinear function
    //
    //     f(z,E) = A + 2z + 3E + 0.25 zE
    //
    // must be reproduced exactly by bilinear interpolation, including extrapolation.
    // Give every particle/flavor component a different A so all six components are
    // checked in one test.
    // -----------------------------------------------------------------------------

    Real_t reference_flux(Particle particle, Flavor flavor, Real_t z, Real_t energy) {
        const Real_t base = 1000.0 * static_cast<Index_t>(particle) + 100.0 * static_cast<Index_t>(flavor);

        return base + 2.0 * z + 3.0 * energy + 0.25 * z * energy;
    }

    void test_resample_flux() {
        Flux source(3, 3);

        const Real_t source_z[] = {-1.0, -0.5, 0.0};
        const Real_t source_e[] = {100.0, 200.0, 400.0};

        for (Index_t i = 0; i < 3; ++i) {
            source.coszenith()(i)  = source_z[i];
            source.energy_gev()(i) = source_e[i];
        }

        for (Index_t p = 0; p < 2; ++p) {
            for (Index_t f = 0; f < 3; ++f) {
                const auto particle  = static_cast<Particle>(p);
                const auto flavor    = static_cast<Flavor>(f);
                auto       component = source.component(particle, flavor);

                for (Index_t z = 0; z < 3; ++z)
                    for (Index_t e = 0; e < 3; ++e)
                        component(z, e) = reference_flux(particle, flavor, source_z[z], source_e[e]);
            }
        }

        // Contains both interpolation points and points outside the source range,
        // so legacy linear-extrapolation behavior is tested as well.
        const std::array<Real_t, 3> target_z = {-1.2, -0.75, 0.2};
        const std::array<Real_t, 3> target_e = {50.0, 150.0, 500.0};

        const auto z_view = nda::make_view1d(static_cast<const Real_t*>(target_z.data()), target_z.size());
        const auto e_view = nda::make_view1d(static_cast<const Real_t*>(target_e.data()), target_e.size());

        auto result = nt::resample_flux(source, z_view, e_view);

        for (Index_t p = 0; p < 2; ++p) {
            for (Index_t f = 0; f < 3; ++f) {
                const auto particle  = static_cast<Particle>(p);
                const auto flavor    = static_cast<Flavor>(f);
                const auto component = result.component(particle, flavor);

                for (Index_t z = 0; z < target_z.size(); ++z) {
                    for (Index_t e = 0; e < target_e.size(); ++e) {
                        require_close(component(z, e), reference_flux(particle, flavor, target_z[z], target_e[e]),
                                      "resampled flux is wrong");
                    }
                }
            }
        }

        std::cout << "[PASS] flux resampling\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4: legacy midpoint sampling inside nonuniform coszenith bins.
    // -----------------------------------------------------------------------------

    void test_coszenith_bin_midpoints() {
        const std::array<Real_t, 4> edges = {-1.0, -0.9, -0.4, 0.0};
        const auto edge_view              = nda::make_view1d(static_cast<const Real_t*>(edges.data()), edges.size());

        constexpr Index_t samples_per_bin = 3;
        const auto        samples         = nt::sample_coszenith_bin_midpoints(edge_view, samples_per_bin);

        require(samples.extent(0) == 9, "wrong number of coszenith midpoint samples");

        Index_t index = 0;
        for (Index_t j = 0; j + 1 < edges.size(); ++j) {
            for (Index_t k = 0; k < samples_per_bin; ++k) {
                const Real_t u        = (static_cast<Real_t>(k) + Real_t{0.5}) / static_cast<Real_t>(samples_per_bin);
                const Real_t expected = edges[j] + u * (edges[j + 1] - edges[j]);
                require_close(samples(index++), expected, "wrong coszenith midpoint sample");
            }
        }

        std::cout << "[PASS] coszenith bin midpoint sampling\n";
    }

    // -----------------------------------------------------------------------------
    // Test 5: legacy arithmetic averaging from fine samples back to bins.
    // -----------------------------------------------------------------------------

    void test_average_flux_to_bins() {
        const std::array<Real_t, 4> edges = {-1.0, -0.9, -0.4, 0.0};
        const auto edge_view              = nda::make_view1d(static_cast<const Real_t*>(edges.data()), edges.size());

        constexpr Index_t samples_per_bin = 3;
        const auto        samples         = nt::sample_coszenith_bin_midpoints(edge_view, samples_per_bin);

        Flux fine(samples.extent(0), 2);

        for (Index_t z = 0; z < samples.extent(0); ++z)
            fine.coszenith()(z) = samples(z);

        fine.energy_gev()(0) = 100.0;
        fine.energy_gev()(1) = 200.0;

        for (Index_t p = 0; p < 2; ++p) {
            for (Index_t f = 0; f < 3; ++f) {
                const auto particle  = static_cast<Particle>(p);
                const auto flavor    = static_cast<Flavor>(f);
                auto       component = fine.component(particle, flavor);

                for (Index_t z = 0; z < fine.n_coszenith(); ++z) {
                    for (Index_t e = 0; e < fine.n_energy(); ++e)
                        component(z, e) = reference_flux(particle, flavor, fine.coszenith()(z), fine.energy_gev()(e));
                }
            }
        }

        const auto averaged = nt::average_flux_to_coszenith_bins(fine, edge_view, samples_per_bin);

        require(averaged.n_coszenith() == 3, "wrong averaged coszenith dimension");
        require(averaged.n_energy() == 2, "wrong averaged energy dimension");

        require_close(averaged.energy_gev()(0), 100.0, "averaging changed energy axis");
        require_close(averaged.energy_gev()(1), 200.0, "averaging changed energy axis");

        for (Index_t z = 0; z < averaged.n_coszenith(); ++z) {
            const Real_t center = Real_t{0.5} * (edges[z] + edges[z + 1]);
            require_close(averaged.coszenith()(z), center, "wrong averaged coszenith coordinate");

            for (Index_t p = 0; p < 2; ++p) {
                for (Index_t f = 0; f < 3; ++f) {
                    const auto particle  = static_cast<Particle>(p);
                    const auto flavor    = static_cast<Flavor>(f);
                    const auto component = averaged.component(particle, flavor);

                    for (Index_t e = 0; e < averaged.n_energy(); ++e) {
                        const Real_t expected = reference_flux(particle, flavor, center, averaged.energy_gev()(e));
                        require_close(component(z, e), expected, "wrong flux after averaging back to bins");
                    }
                }
            }
        }

        std::cout << "[PASS] flux averaging back to coszenith bins\n";
    }

} // namespace

int main() {
    try {
        test_component_views();
        test_daemonflux_loader();
        test_resample_flux();
        test_coszenith_bin_midpoints();
        test_average_flux_to_bins();

        std::cout << "[PASS] all flux tests\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
