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

        numu(1, 2) = 42.0;
        require_close(flux.native_state()[1][2][0][1], 42.0, "numu view does not alias native state");

        antinumu(0, 1) = 17.0;
        require_close(flux.native_state()[0][1][1][1], 17.0, "antinumu view does not alias native state");

        flux.native_state()[1][0][0][1] = 23.0;
        require_close(numu(1, 0), 23.0, "native state does not alias numu view");
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
    // -----------------------------------------------------------------------------

    void test_resample_flux() {
        Flux source(3, 3);

        const Real_t source_z[] = {-1.0, -0.5, 0.0};
        const Real_t source_e[] = {100.0, 200.0, 400.0};

        for (Index_t i = 0; i < 3; ++i) {
            source.coszenith()(i)  = source_z[i];
            source.energy_gev()(i) = source_e[i];
        }

        const auto value = [](Particle particle, Flavor flavor, Real_t z, Real_t energy) {
            const Real_t base = 1000.0 * static_cast<Index_t>(particle) + 100.0 * static_cast<Index_t>(flavor);
            return base + 2.0 * z + 3.0 * energy + 0.25 * z * energy;
        };

        for (Index_t p = 0; p < 2; ++p) {
            for (Index_t f = 0; f < 3; ++f) {
                const auto particle  = static_cast<Particle>(p);
                const auto flavor    = static_cast<Flavor>(f);
                auto       component = source.component(particle, flavor);

                for (Index_t z = 0; z < 3; ++z)
                    for (Index_t e = 0; e < 3; ++e)
                        component(z, e) = value(particle, flavor, source_z[z], source_e[e]);
            }
        }

        // Contains interpolation points and points outside the source range, so
        // the established linear-extrapolation behavior is also tested.
        const std::array<Real_t, 3> target_z = {-1.2, -0.75, 0.2};
        const std::array<Real_t, 3> target_e = {50.0, 150.0, 500.0};

        auto result =
            nt::resample_flux(source, nda::make_view1d(static_cast<const Real_t*>(target_z.data()), target_z.size()),
                              nda::make_view1d(static_cast<const Real_t*>(target_e.data()), target_e.size()));

        for (Index_t p = 0; p < 2; ++p) {
            for (Index_t f = 0; f < 3; ++f) {
                const auto particle  = static_cast<Particle>(p);
                const auto flavor    = static_cast<Flavor>(f);
                const auto component = result.component(particle, flavor);

                for (Index_t z = 0; z < target_z.size(); ++z)
                    for (Index_t e = 0; e < target_e.size(); ++e)
                        require_close(component(z, e), value(particle, flavor, target_z[z], target_e[e]),
                                      "resampled flux is wrong");
            }
        }

        std::cout << "[PASS] flux resampling\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4: midpoint sampling in coszenith and log10(E/GeV).
    // -----------------------------------------------------------------------------

    void test_bin_midpoints() {
        const std::array<Real_t, 4> z_edges           = {-1.0, -0.9, -0.4, 0.0};
        constexpr Index_t           z_samples_per_bin = 3;
        const auto                  z_samples         = nt::sample_coszenith_bin_midpoints(
            nda::make_view1d(static_cast<const Real_t*>(z_edges.data()), z_edges.size()), z_samples_per_bin);

        require(z_samples.extent(0) == 9, "wrong number of coszenith midpoint samples");

        Index_t index = 0;
        for (Index_t bin = 0; bin + 1 < z_edges.size(); ++bin) {
            for (Index_t k = 0; k < z_samples_per_bin; ++k) {
                const Real_t u = (static_cast<Real_t>(k) + Real_t{0.5}) / static_cast<Real_t>(z_samples_per_bin);
                require_close(z_samples(index++), z_edges[bin] + u * (z_edges[bin + 1] - z_edges[bin]),
                              "wrong coszenith midpoint sample");
            }
        }

        const std::array<Real_t, 3> e_edges           = {100.0, 1000.0, 10000.0};
        constexpr Index_t           e_samples_per_bin = 2;
        const auto                  e_samples         = nt::sample_log_energy_bin_midpoints(
            nda::make_view1d(static_cast<const Real_t*>(e_edges.data()), e_edges.size()), e_samples_per_bin);

        require(e_samples.extent(0) == 4, "wrong number of energy midpoint samples");

        index = 0;
        for (Index_t bin = 0; bin + 1 < e_edges.size(); ++bin) {
            const Real_t loge0 = std::log10(e_edges[bin]);
            const Real_t loge1 = std::log10(e_edges[bin + 1]);

            for (Index_t k = 0; k < e_samples_per_bin; ++k) {
                const Real_t u = (static_cast<Real_t>(k) + Real_t{0.5}) / static_cast<Real_t>(e_samples_per_bin);
                require_close(e_samples(index++), std::pow(Real_t{10}, loge0 + u * (loge1 - loge0)),
                              "wrong log-energy midpoint sample");
            }
        }

        std::cout << "[PASS] bin midpoint sampling\n";
    }

    // -----------------------------------------------------------------------------
    // Test 5: both axes independently support averaging or interpolation.
    // -----------------------------------------------------------------------------

    void test_rebin_flux() {
        const std::array<Real_t, 3> z_edges         = {-1.0, -0.5, 0.0};
        const std::array<Real_t, 3> e_edges         = {100.0, 1000.0, 10000.0};
        constexpr Index_t           samples_per_bin = 2;

        const auto fine_z = nt::sample_coszenith_bin_midpoints(
            nda::make_view1d(static_cast<const Real_t*>(z_edges.data()), z_edges.size()), samples_per_bin);
        const auto fine_e = nt::sample_log_energy_bin_midpoints(
            nda::make_view1d(static_cast<const Real_t*>(e_edges.data()), e_edges.size()), samples_per_bin);

        const std::array<Real_t, 2> target_z = {-0.75, -0.25};
        const std::array<Real_t, 2> target_e = {std::sqrt(100.0 * 1000.0), std::sqrt(1000.0 * 10000.0)};
        const auto target_z_view = nda::make_view1d(static_cast<const Real_t*>(target_z.data()), target_z.size());
        const auto target_e_view = nda::make_view1d(static_cast<const Real_t*>(target_e.data()), target_e.size());

        Flux fine(fine_z.extent(0), fine_e.extent(0));

        for (Index_t z = 0; z < fine.n_coszenith(); ++z)
            fine.coszenith()(z) = fine_z(z);
        for (Index_t e = 0; e < fine.n_energy(); ++e)
            fine.energy_gev()(e) = fine_e(e);

        const auto value = [](Particle particle, Flavor flavor, Real_t z, Real_t energy) {
            const Real_t base = 1000.0 * static_cast<Index_t>(particle) + 100.0 * static_cast<Index_t>(flavor);
            return base + 7.0 * z + 0.002 * energy + 3.0 * z * z + 1.0e-7 * energy * energy + 0.001 * z * energy;
        };

        for (Index_t p = 0; p < 2; ++p) {
            for (Index_t f = 0; f < 3; ++f) {
                const auto particle  = static_cast<Particle>(p);
                const auto flavor    = static_cast<Flavor>(f);
                auto       component = fine.component(particle, flavor);

                for (Index_t z = 0; z < fine.n_coszenith(); ++z)
                    for (Index_t e = 0; e < fine.n_energy(); ++e)
                        component(z, e) = value(particle, flavor, fine_z(z), fine_e(e));
            }
        }

        const auto average_average =
            nt::rebin_flux(fine, target_z_view, target_e_view, samples_per_bin, samples_per_bin);

        nt::FluxRebinOptions average_interpolate_options;
        average_interpolate_options.interpolate_energy = true;
        const auto average_interpolate = nt::rebin_flux(fine, target_z_view, target_e_view, samples_per_bin,
                                                        samples_per_bin, average_interpolate_options);

        nt::FluxRebinOptions interpolate_average_options;
        interpolate_average_options.interpolate_coszenith = true;
        const auto interpolate_average = nt::rebin_flux(fine, target_z_view, target_e_view, samples_per_bin,
                                                        samples_per_bin, interpolate_average_options);

        nt::FluxRebinOptions interpolate_interpolate_options;
        interpolate_interpolate_options.interpolate_coszenith = true;
        interpolate_interpolate_options.interpolate_energy    = true;
        const auto interpolate_interpolate = nt::rebin_flux(fine, target_z_view, target_e_view, samples_per_bin,
                                                            samples_per_bin, interpolate_interpolate_options);

        for (Index_t p = 0; p < 2; ++p) {
            for (Index_t f = 0; f < 3; ++f) {
                const auto particle = static_cast<Particle>(p);
                const auto flavor   = static_cast<Flavor>(f);
                const auto aa       = average_average.component(particle, flavor);
                const auto ai       = average_interpolate.component(particle, flavor);
                const auto ia       = interpolate_average.component(particle, flavor);
                const auto ii       = interpolate_interpolate.component(particle, flavor);

                for (Index_t z = 0; z < target_z.size(); ++z) {
                    const Index_t z0 = z * samples_per_bin;
                    const Index_t z1 = z0 + 1;
                    const Real_t  wz = (target_z[z] - fine_z(z0)) / (fine_z(z1) - fine_z(z0));

                    for (Index_t e = 0; e < target_e.size(); ++e) {
                        const Index_t e0 = e * samples_per_bin;
                        const Index_t e1 = e0 + 1;
                        const Real_t  we = (target_e[e] - fine_e(e0)) / (fine_e(e1) - fine_e(e0));

                        const Real_t v00 = value(particle, flavor, fine_z(z0), fine_e(e0));
                        const Real_t v01 = value(particle, flavor, fine_z(z0), fine_e(e1));
                        const Real_t v10 = value(particle, flavor, fine_z(z1), fine_e(e0));
                        const Real_t v11 = value(particle, flavor, fine_z(z1), fine_e(e1));

                        const Real_t expected_aa = Real_t{0.25} * (v00 + v01 + v10 + v11);
                        const Real_t expected_ai =
                            Real_t{0.5} * ((Real_t{1} - we) * v00 + we * v01 + (Real_t{1} - we) * v10 + we * v11);
                        const Real_t expected_ia =
                            Real_t{0.5} * ((Real_t{1} - wz) * v00 + wz * v10 + (Real_t{1} - wz) * v01 + wz * v11);
                        const Real_t expected_ii = (Real_t{1} - wz) * (Real_t{1} - we) * v00 +
                                                   (Real_t{1} - wz) * we * v01 + wz * (Real_t{1} - we) * v10 +
                                                   wz * we * v11;

                        require_close(aa(z, e), expected_aa, "average/average rebin is wrong");
                        require_close(ai(z, e), expected_ai, "average/interpolate rebin is wrong");
                        require_close(ia(z, e), expected_ia, "interpolate/average rebin is wrong");
                        require_close(ii(z, e), expected_ii, "interpolate/interpolate rebin is wrong");
                    }
                }
            }
        }

        for (Index_t z = 0; z < target_z.size(); ++z)
            require_close(interpolate_interpolate.coszenith()(z), target_z[z], "rebin changed target coszenith axis");
        for (Index_t e = 0; e < target_e.size(); ++e)
            require_close(interpolate_interpolate.energy_gev()(e), target_e[e], "rebin changed target energy axis");

        std::cout << "[PASS] flux rebinning\n";
    }

} // namespace

int main() {
    try {
        test_component_views();
        test_daemonflux_loader();
        test_resample_flux();
        test_bin_midpoints();
        test_rebin_flux();

        std::cout << "[PASS] all flux tests\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
