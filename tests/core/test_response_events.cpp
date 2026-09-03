#include <nt/events.hpp>
#include <nt/flux.hpp>
#include <nt/response.hpp>
#include <nt/types.hpp>

#include <algorithm>
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

    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    void require_close(Real_t actual, Real_t expected, const char* message) {
        const Real_t scale = std::max({Real_t{1}, std::abs(actual), std::abs(expected)});
        if (std::abs(actual - expected) > eps * scale) {
            throw std::runtime_error(
                std::string(message) + ": actual=" + std::to_string(actual) +
                ", expected=" + std::to_string(expected));
        }
    }

    Real_t true_loge(Index_t t) {
        return Real_t{3.05} + Real_t{0.1} * static_cast<Real_t>(t);
    }

    Real_t true_energy(Index_t t) {
        return std::pow(Real_t{10}, true_loge(t));
    }

    Real_t coszenith(Index_t z) {
        return Real_t{-0.99} + Real_t{0.02} * static_cast<Real_t>(z);
    }

    Real_t detector_value(Index_t t, Index_t z) {
        return Real_t{1} +
               Real_t{0.1} * static_cast<Real_t>(t) +
               Real_t{0.01} * static_cast<Real_t>(z);
    }

    // -------------------------------------------------------------------------
    // Synthetic TRIDENT CSV files
    // -------------------------------------------------------------------------

    std::filesystem::path make_test_response_files() {
        const auto directory =
            std::filesystem::temp_directory_path() /
            "neutrino_tomography_test_trident";

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
        require(response.true_energy_gev.extent(0) == n_true,
                "wrong true-energy dimension");
        require(response.coszenith.extent(0) == n_cz,
                "wrong coszenith dimension");
        require(response.reco_energy_gev.extent(0) == n_reco,
                "wrong reconstructed-energy dimension");

        require(response.detector_response.extent(0) == n_true &&
                    response.detector_response.extent(1) == n_cz,
                "wrong detector response shape");

        require(response.energy_migration.extent(0) == n_true &&
                    response.energy_migration.extent(1) == n_reco,
                "wrong energy migration shape");

        for (Index_t t = 0; t < n_true; ++t)
            require_close(response.true_energy_gev(t), true_energy(t),
                          "wrong true-energy axis");

        for (Index_t z = 0; z < n_cz; ++z)
            require_close(response.coszenith(z), coszenith(z),
                          "wrong coszenith axis");

        for (Index_t r = 0; r < n_reco; ++r)
            require_close(response.reco_energy_gev(r), true_energy(r),
                          "wrong reconstructed-energy axis");

        // Selected asymmetric positions make a transpose hard to hide.
        require_close(response.detector_response(7, 11),
                      detector_value(7, 11),
                      "detector response orientation is wrong");

        require_close(response.detector_response(11, 7),
                      detector_value(11, 7),
                      "detector response orientation is wrong");

        for (Index_t t = 0; t < n_true; ++t) {
            for (Index_t r = 0; r < n_reco; ++r) {
                require_close(response.energy_migration(t, r),
                              t == r ? Real_t{1} : Real_t{0},
                              "energy migration orientation is wrong");
            }
        }

        std::cout << "[PASS] TRIDENT response loader\n";
    }

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    Real_t numu_value(Index_t z, Index_t t) {
        return Real_t{10} +
               Real_t{0.2} * static_cast<Real_t>(z) +
               Real_t{1.5} * static_cast<Real_t>(t);
    }

    Real_t antinumu_value(Index_t z, Index_t t) {
        return Real_t{2} +
               Real_t{0.03} * static_cast<Real_t>(z) +
               Real_t{0.4} * static_cast<Real_t>(t);
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

        require(events.counts.extent(0) == n_cz &&
                    events.counts.extent(1) == n_reco,
                "wrong event-array shape");

        for (Index_t z = 0; z < n_cz; ++z)
            require_close(events.coszenith(z), response.coszenith(z),
                          "wrong event coszenith axis");

        for (Index_t r = 0; r < n_reco; ++r)
            require_close(events.reco_energy_gev(r), response.reco_energy_gev(r),
                          "wrong event reconstructed-energy axis");

        // The synthetic migration matrix is identity, therefore only t=r
        // contributes:
        //
        // N[z,r] = (numu[z,r] + antinumu[z,r]) * R[r,z].
        for (Index_t z = 0; z < n_cz; ++z) {
            for (Index_t r = 0; r < n_reco; ++r) {
                const Real_t expected =
                    (numu_value(z, r) + antinumu_value(z, r)) *
                    detector_value(r, z);

                require_close(events.counts(z, r), expected,
                              "predicted event count is wrong");
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
    // Optional integration test against the local internal TRIDENT files.
    // -------------------------------------------------------------------------

    void test_real_trident_files() {
        const std::filesystem::path directory = "data/trident";
        const auto                  response_file =
            directory / "TRIDENT_response_array_20x34.csv";
        const auto migration_file =
            directory / "energy_response_20x20_v2.csv";

        if (!std::filesystem::is_regular_file(response_file) ||
            !std::filesystem::is_regular_file(migration_file)) {
            std::cout << "[SKIP] local internal TRIDENT files not available\n";
            return;
        }

        const auto response = nt::load_trident_response(directory);

        require(response.true_energy_gev.extent(0) == n_true,
                "real TRIDENT true-energy dimension is wrong");
        require(response.coszenith.extent(0) == n_cz,
                "real TRIDENT coszenith dimension is wrong");
        require(response.reco_energy_gev.extent(0) == n_reco,
                "real TRIDENT reco-energy dimension is wrong");

        Real_t min_row_sum = std::numeric_limits<Real_t>::infinity();
        Real_t max_row_sum = Real_t{0};

        for (Index_t t = 0; t < n_true; ++t) {
            Real_t row_sum = 0;

            for (Index_t r = 0; r < n_reco; ++r) {
                const Real_t value = response.energy_migration(t, r);
                require(std::isfinite(value) && value >= 0,
                        "real TRIDENT migration contains invalid values");
                row_sum += value;
            }

            require(std::isfinite(row_sum) && row_sum > 0,
                    "real TRIDENT migration contains an empty row");

            min_row_sum = std::min(min_row_sum, row_sum);
            max_row_sum = std::max(max_row_sum, row_sum);
        }

        for (Index_t t = 0; t < n_true; ++t) {
            for (Index_t z = 0; z < n_cz; ++z) {
                const Real_t value = response.detector_response(t, z);
                require(std::isfinite(value) && value >= 0,
                        "real TRIDENT detector response contains invalid values");
            }
        }

        std::cout << "[PASS] local internal TRIDENT files\n";
        std::cout << "[INFO] migration row-sum range = ["
                  << min_row_sum << ", " << max_row_sum << "]\n";
    }

} // namespace

int main() {
    try {
        const auto directory = make_test_response_files();
        const auto response  = nt::load_trident_response(directory);

        test_response_loader(response);
        test_events(response);
        test_grid_mismatch(response);

        std::filesystem::remove_all(directory);

        test_real_trident_files();

        std::cout << "[PASS] all response/event tests\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
