#include <nt/events.hpp>
#include <nt/flux.hpp>
#include <nt/types.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

    using nt::EventDistribution;
    using nt::Flux;
    using nt::Index_t;
    using nt::Real_t;
    using nt::ResponseArray;

    constexpr Index_t n_true = 20;
    constexpr Index_t n_cz   = 34;
    constexpr Index_t n_reco = 20;

    constexpr Real_t eps = 1e-11;

    constexpr std::array<Real_t, n_cz> trident_coszenith_centers = {
        -0.9988, -0.9963, -0.9937, -0.9912, -0.9855, -0.9765, -0.9675, -0.9585, -0.9495, -0.9405, -0.9315, -0.9225,
        -0.9135, -0.9045, -0.8775, -0.8325, -0.7875, -0.7425, -0.6975, -0.6525, -0.6075, -0.5625, -0.5175, -0.4725,
        -0.4275, -0.3825, -0.3375, -0.2925, -0.2475, -0.2025, -0.1575, -0.1125, -0.0675, -0.0225,
    };

    constexpr std::array<Real_t, n_cz + 1> trident_coszenith_edges = {
        -1.0,   -0.99755, -0.995, -0.99245, -0.98835, -0.981, -0.972, -0.963, -0.954, -0.945, -0.936, -0.927,
        -0.918, -0.909,   -0.891, -0.855,   -0.81,    -0.765, -0.72,  -0.675, -0.63,  -0.585, -0.54,  -0.495,
        -0.45,  -0.405,   -0.36,  -0.315,   -0.27,    -0.225, -0.18,  -0.135, -0.09,  -0.045, 0.0,
    };

    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    void require_close(Real_t actual, Real_t expected, const char* message) {
        const Real_t scale = std::max({Real_t{1}, std::abs(actual), std::abs(expected)});
        if (std::abs(actual - expected) > eps * scale) {
            throw std::runtime_error(std::string(message) + ": actual=" + std::to_string(actual) +
                                     ", expected=" + std::to_string(expected));
        }
    }

    Real_t true_loge(Index_t t) { return Real_t{3.05} + Real_t{0.1} * static_cast<Real_t>(t); }

    Real_t true_energy(Index_t t) { return std::pow(Real_t{10}, true_loge(t)); }

    Real_t energy_edge(Index_t i) { return std::pow(Real_t{10}, Real_t{3.0} + Real_t{0.1} * static_cast<Real_t>(i)); }

    Real_t coszenith(Index_t z) { return trident_coszenith_centers[z]; }

    Real_t detector_value(Index_t t, Index_t z) {
        return Real_t{1} + Real_t{0.1} * static_cast<Real_t>(t) + Real_t{0.01} * static_cast<Real_t>(z);
    }

    // -------------------------------------------------------------------------
    // Synthetic TRIDENT CSV files
    // -------------------------------------------------------------------------

    std::filesystem::path make_test_response_files() {
        const auto directory = std::filesystem::temp_directory_path() / "neutrino_tomography_test_trident";

        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);

        {
            std::ofstream out(directory / "TRIDENT_response_array_20x34.csv");
            require(static_cast<bool>(out), "cannot create synthetic detector response CSV");

            out << std::setprecision(17);
            out << "logE";

            for (Index_t z = 0; z < n_cz; ++z)
                out << ",cos_" << coszenith(z);

            out << '\n';

            for (Index_t t = 0; t < n_true; ++t) {
                out << "logE_" << true_loge(t);

                for (Index_t z = 0; z < n_cz; ++z)
                    out << ',' << detector_value(t, z);

                out << '\n';
            }
        }

        {
            std::ofstream out(directory / "energy_response_20x20_v2.csv");
            require(static_cast<bool>(out), "cannot create synthetic energy migration CSV");

            out << std::setprecision(17);
            out << "E_true_center_GeV";

            for (Index_t r = 0; r < n_reco; ++r)
                out << ',' << true_energy(r);

            out << '\n';

            // Identity migration:
            //
            //     P(E_rec=r | E_true=t) = delta_rt
            //
            // This makes the expected event result analytically trivial and
            // catches an accidental transpose immediately.
            for (Index_t t = 0; t < n_true; ++t) {
                out << true_energy(t);

                for (Index_t r = 0; r < n_reco; ++r)
                    out << ',' << (t == r ? Real_t{1} : Real_t{0});

                out << '\n';
            }
        }

        return directory;
    }

    // -------------------------------------------------------------------------
    // Response loader
    // -------------------------------------------------------------------------

    void test_response_loader(const ResponseArray& response) {
        require(response.true_energy_gev.extent(0) == n_true, "wrong true-energy dimension");
        require(response.coszenith.extent(0) == n_cz, "wrong coszenith dimension");
        require(response.reco_energy_gev.extent(0) == n_reco, "wrong reconstructed-energy dimension");

        require(response.true_energy_edges_gev.extent(0) == n_true + 1, "wrong true-energy edge dimension");
        require(response.coszenith_edges.extent(0) == n_cz + 1, "wrong coszenith edge dimension");
        require(response.reco_energy_edges_gev.extent(0) == n_reco + 1, "wrong reconstructed-energy edge dimension");

        require(response.detector_response.extent(0) == n_true && response.detector_response.extent(1) == n_cz,
                "wrong detector response shape");

        require(response.energy_migration.extent(0) == n_true && response.energy_migration.extent(1) == n_reco,
                "wrong energy migration shape");

        for (Index_t t = 0; t < n_true; ++t)
            require_close(response.true_energy_gev(t), true_energy(t), "wrong true-energy axis");

        for (Index_t z = 0; z < n_cz; ++z)
            require_close(response.coszenith(z), trident_coszenith_centers[z], "wrong coszenith axis");

        for (Index_t r = 0; r < n_reco; ++r)
            require_close(response.reco_energy_gev(r), true_energy(r), "wrong reconstructed-energy axis");

        for (Index_t i = 0; i <= n_true; ++i)
            require_close(response.true_energy_edges_gev(i), energy_edge(i), "wrong true-energy bin edge");

        for (Index_t i = 0; i <= n_cz; ++i)
            require_close(response.coszenith_edges(i), trident_coszenith_edges[i], "wrong coszenith bin edge");

        for (Index_t i = 0; i <= n_reco; ++i)
            require_close(response.reco_energy_edges_gev(i), energy_edge(i), "wrong reconstructed-energy bin edge");

        require(std::abs(response.coszenith(0) -
                         Real_t{0.5} * (response.coszenith_edges(0) + response.coszenith_edges(1))) > 1e-6,
                "canonical coszenith center was reconstructed from bin edges");

        // Selected asymmetric positions make a transpose hard to hide.
        require_close(response.detector_response(7, 11), detector_value(7, 11),
                      "detector response orientation is wrong");

        require_close(response.detector_response(11, 7), detector_value(11, 7),
                      "detector response orientation is wrong");

        for (Index_t t = 0; t < n_true; ++t) {
            for (Index_t r = 0; r < n_reco; ++r) {
                require_close(response.energy_migration(t, r), t == r ? Real_t{1} : Real_t{0},
                              "energy migration orientation is wrong");
            }
        }

        std::cout << "[PASS] TRIDENT response loader\n";
    }

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    Real_t numu_value(Index_t z, Index_t t) {
        return Real_t{10} + Real_t{0.2} * static_cast<Real_t>(z) + Real_t{1.5} * static_cast<Real_t>(t);
    }

    Real_t antinumu_value(Index_t z, Index_t t) {
        return Real_t{2} + Real_t{0.03} * static_cast<Real_t>(z) + Real_t{0.4} * static_cast<Real_t>(t);
    }

    void test_events(const ResponseArray& response) {
        Flux flux(n_cz, n_true);

        for (Index_t z = 0; z < n_cz; ++z)
            flux.coszenith()(z) = response.coszenith(z);

        for (Index_t t = 0; t < n_true; ++t)
            flux.energy_gev()(t) = response.true_energy_gev(t);

        auto numu     = flux.numu();
        auto antinumu = flux.antinumu();

        for (Index_t z = 0; z < n_cz; ++z) {
            for (Index_t t = 0; t < n_true; ++t) {
                numu(z, t)     = numu_value(z, t);
                antinumu(z, t) = antinumu_value(z, t);
            }
        }

        const EventDistribution events = nt::predict_events(flux, response);

        require(events.counts.extent(0) == n_cz && events.counts.extent(1) == n_reco, "wrong event-array shape");
        require(events.coszenith_edges.extent(0) == n_cz + 1, "wrong event coszenith-edge dimension");
        require(events.reco_energy_edges_gev.extent(0) == n_reco + 1,
                "wrong event reconstructed-energy edge dimension");

        for (Index_t z = 0; z < n_cz; ++z)
            require_close(events.coszenith(z), response.coszenith(z), "wrong event coszenith axis");

        for (Index_t r = 0; r < n_reco; ++r)
            require_close(events.reco_energy_gev(r), response.reco_energy_gev(r),
                          "wrong event reconstructed-energy axis");

        for (Index_t z = 0; z <= n_cz; ++z)
            require_close(events.coszenith_edges(z), response.coszenith_edges(z), "wrong event coszenith bin edge");

        for (Index_t r = 0; r <= n_reco; ++r)
            require_close(events.reco_energy_edges_gev(r), response.reco_energy_edges_gev(r),
                          "wrong event reconstructed-energy bin edge");

        // The synthetic migration matrix is identity, therefore only t=r
        // contributes:
        //
        // N[z,r] = (numu[z,r] + antinumu[z,r]) * R[r,z].
        for (Index_t z = 0; z < n_cz; ++z) {
            for (Index_t r = 0; r < n_reco; ++r) {
                const Real_t expected = (numu_value(z, r) + antinumu_value(z, r)) * detector_value(r, z);

                require_close(events.counts(z, r), expected, "predicted event count is wrong");
            }
        }

        std::cout << "[PASS] event prediction\n";
    }

    // `predict_events()` deliberately does not interpolate mismatched grids.
    void test_grid_mismatch(const ResponseArray& response) {
        Flux flux(n_cz, n_true);

        for (Index_t z = 0; z < n_cz; ++z)
            flux.coszenith()(z) = response.coszenith(z);

        for (Index_t t = 0; t < n_true; ++t)
            flux.energy_gev()(t) = response.true_energy_gev(t);

        flux.energy_gev()(5) *= 1.01;

        bool threw = false;
        try {
            [[maybe_unused]] auto events = nt::predict_events(flux, response);
        } catch (const std::invalid_argument&) {
            threw = true;
        }

        require(threw, "predict_events accepted a mismatched Flux grid");
        std::cout << "[PASS] event grid mismatch detection\n";
    }

    // -------------------------------------------------------------------------
    // Legacy nonuniform-bin event path
    // -------------------------------------------------------------------------

    Real_t fine_numu_value(Real_t z, Index_t t) {
        return Real_t{10} + Real_t{4} * z + Real_t{0.5} * static_cast<Real_t>(t);
    }

    Real_t fine_antinumu_value(Real_t z, Index_t t) {
        return Real_t{2} - Real_t{0.7} * z + Real_t{0.2} * static_cast<Real_t>(t);
    }

    void test_legacy_binned_events(const ResponseArray& response) {
        constexpr Index_t samples_per_bin = 3;

        const auto sample_z = nt::sample_coszenith_bin_midpoints(response.coszenith_edges.view(), samples_per_bin);

        Flux fine(sample_z.extent(0), n_true);

        for (Index_t z = 0; z < sample_z.extent(0); ++z)
            fine.coszenith()(z) = sample_z(z);

        for (Index_t t = 0; t < n_true; ++t)
            fine.energy_gev()(t) = response.true_energy_gev(t);

        auto numu     = fine.numu();
        auto antinumu = fine.antinumu();

        for (Index_t z = 0; z < fine.n_coszenith(); ++z) {
            for (Index_t t = 0; t < n_true; ++t) {
                numu(z, t)     = fine_numu_value(fine.coszenith()(z), t);
                antinumu(z, t) = fine_antinumu_value(fine.coszenith()(z), t);
            }
        }

        const auto averaged =
            nt::average_flux_to_coszenith_bins(fine, response.coszenith_edges.view(), samples_per_bin);

        for (Index_t z = 0; z < n_cz; ++z) {
            const Real_t center = Real_t{0.5} * (response.coszenith_edges(z) + response.coszenith_edges(z + 1));
            require_close(averaged.coszenith()(z), center, "legacy bin averaging produced wrong center");
        }

        const auto response_flux =
            nt::resample_flux(averaged, response.coszenith.view(), response.true_energy_gev.view());

        const auto events = nt::predict_events(response_flux, response);

        for (Index_t z = 0; z < n_cz; ++z) {
            for (Index_t r = 0; r < n_reco; ++r) {
                const Real_t expected_flux =
                    fine_numu_value(response.coszenith(z), r) + fine_antinumu_value(response.coszenith(z), r);

                const Real_t expected = expected_flux * detector_value(r, z);
                require_close(events.counts(z, r), expected, "legacy nonuniform-bin event path is wrong");
            }
        }

        std::cout << "[PASS] legacy nonuniform-bin event path\n";
    }

    // -------------------------------------------------------------------------
    // Optional integration test against the local internal TRIDENT files.
    // -------------------------------------------------------------------------

    void test_real_trident_files() {
        const std::filesystem::path directory      = "data/trident";
        const auto                  response_file  = directory / "TRIDENT_response_array_20x34.csv";
        const auto                  migration_file = directory / "energy_response_20x20_v2.csv";

        if (!std::filesystem::is_regular_file(response_file) || !std::filesystem::is_regular_file(migration_file)) {
            std::cout << "[SKIP] local internal TRIDENT files not available\n";
            return;
        }

        const auto response = nt::load_trident_response(directory);

        require(response.true_energy_gev.extent(0) == n_true, "real TRIDENT true-energy dimension is wrong");
        require(response.coszenith.extent(0) == n_cz, "real TRIDENT coszenith dimension is wrong");
        require(response.reco_energy_gev.extent(0) == n_reco, "real TRIDENT reco-energy dimension is wrong");

        Real_t min_row_sum = std::numeric_limits<Real_t>::infinity();
        Real_t max_row_sum = Real_t{0};

        for (Index_t t = 0; t < n_true; ++t) {
            Real_t row_sum = 0;

            for (Index_t r = 0; r < n_reco; ++r) {
                const Real_t value = response.energy_migration(t, r);
                require(std::isfinite(value) && value >= 0, "real TRIDENT migration contains invalid values");
                row_sum += value;
            }

            require(std::isfinite(row_sum) && row_sum > 0, "real TRIDENT migration contains an empty row");

            min_row_sum = std::min(min_row_sum, row_sum);
            max_row_sum = std::max(max_row_sum, row_sum);
        }

        for (Index_t t = 0; t < n_true; ++t) {
            for (Index_t z = 0; z < n_cz; ++z) {
                const Real_t value = response.detector_response(t, z);
                require(std::isfinite(value) && value >= 0, "real TRIDENT detector response contains invalid values");
            }
        }

        std::cout << "[PASS] local internal TRIDENT files\n";
        std::cout << "[INFO] migration row-sum range = [" << min_row_sum << ", " << max_row_sum << "]\n";
    }

} // namespace

int main() {
    try {
        const auto directory = make_test_response_files();
        const auto response  = nt::load_trident_response(directory);

        test_response_loader(response);
        test_events(response);
        test_grid_mismatch(response);
        test_legacy_binned_events(response);

        std::filesystem::remove_all(directory);

        test_real_trident_files();

        std::cout << "[PASS] all response/event tests\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
