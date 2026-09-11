#include <nt/events.hpp>
#include <nt/flux.hpp>
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
        if (std::abs(actual - expected) > eps * scale)
            throw std::runtime_error(std::string(message) + ": actual=" + std::to_string(actual) +
                                     ", expected=" + std::to_string(expected));
    }

    Real_t coszenith_edge(Index_t i) {
        if (i <= 4)
            return Real_t{-1.0} + Real_t{0.0025} * static_cast<Real_t>(i);
        if (i <= 14)
            return Real_t{-0.99} + Real_t{0.009} * static_cast<Real_t>(i - 4);
        return Real_t{-0.9} + Real_t{0.045} * static_cast<Real_t>(i - 14);
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

            out << std::setprecision(17) << "logE";

            for (Index_t z = 0; z < n_cz; ++z) {
                const Real_t center = Real_t{0.5} * (coszenith_edge(z) + coszenith_edge(z + 1));
                out << ",cos_" << center;
            }

            out << '\n';

            for (Index_t t = 0; t < n_true; ++t) {
                out << "logE_" << Real_t{3.05} + Real_t{0.1} * static_cast<Real_t>(t);

                for (Index_t z = 0; z < n_cz; ++z)
                    out << ','
                        << Real_t{1} + Real_t{0.1} * static_cast<Real_t>(t) + Real_t{0.01} * static_cast<Real_t>(z);

                out << '\n';
            }
        }

        {
            std::ofstream out(directory / "energy_response_20x20_v2.csv");
            require(static_cast<bool>(out), "cannot create synthetic energy migration CSV");

            out << std::setprecision(17) << "E_true_center_GeV";

            for (Index_t r = 0; r < n_reco; ++r)
                out << ',' << std::pow(Real_t{10}, Real_t{3.05} + Real_t{0.1} * static_cast<Real_t>(r));

            out << '\n';

            // Identity migration:
            //
            //     P(E_rec=r | E_true=t) = delta_rt
            //
            // This makes the expected event result analytically trivial and
            // catches an accidental transpose immediately.
            for (Index_t t = 0; t < n_true; ++t) {
                out << std::pow(Real_t{10}, Real_t{3.05} + Real_t{0.1} * static_cast<Real_t>(t));

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
        require(response.detector_response.extent(0) == n_true && response.detector_response.extent(1) == n_cz,
                "wrong detector response shape");
        require(response.energy_migration.extent(0) == n_true && response.energy_migration.extent(1) == n_reco,
                "wrong energy migration shape");

        for (Index_t t = 0; t < n_true; ++t) {
            const Real_t expected = std::pow(Real_t{10}, Real_t{3.05} + Real_t{0.1} * static_cast<Real_t>(t));
            require_close(response.true_energy_gev(t), expected, "wrong true-energy axis");
            require_close(response.reco_energy_gev(t), expected, "wrong reconstructed-energy axis");
        }

        for (Index_t i = 0; i <= n_true; ++i) {
            const Real_t expected = std::pow(Real_t{10}, Real_t{3.0} + Real_t{0.1} * static_cast<Real_t>(i));
            require_close(response.true_energy_edges_gev(i), expected, "wrong true-energy bin edge");
            require_close(response.reco_energy_edges_gev(i), expected, "wrong reconstructed-energy bin edge");
        }

        for (Index_t z = 0; z < n_cz; ++z) {
            const Real_t low    = coszenith_edge(z);
            const Real_t high   = coszenith_edge(z + 1);
            const Real_t center = Real_t{0.5} * (low + high);
            require_close(response.coszenith_edges(z), low, "wrong coszenith bin edge");
            require_close(response.coszenith(z), center, "wrong coszenith axis");
        }
        require_close(response.coszenith_edges(n_cz), coszenith_edge(n_cz), "wrong final coszenith bin edge");

        // Selected asymmetric positions make a transpose hard to hide.
        require_close(response.detector_response(7, 11), Real_t{1} + Real_t{0.7} + Real_t{0.11},
                      "detector response orientation is wrong");
        require_close(response.detector_response(11, 7), Real_t{1} + Real_t{1.1} + Real_t{0.07},
                      "detector response orientation is wrong");

        for (Index_t t = 0; t < n_true; ++t)
            for (Index_t r = 0; r < n_reco; ++r)
                require_close(response.energy_migration(t, r), t == r ? Real_t{1} : Real_t{0},
                              "energy migration orientation is wrong");

        std::cout << "[PASS] TRIDENT response loader\n";
    }

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

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
                numu(z, t) = Real_t{10} + Real_t{0.2} * static_cast<Real_t>(z) + Real_t{1.5} * static_cast<Real_t>(t);
                antinumu(z, t) =
                    Real_t{2} + Real_t{0.03} * static_cast<Real_t>(z) + Real_t{0.4} * static_cast<Real_t>(t);
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

        for (Index_t z = 0; z < n_cz; ++z) {
            for (Index_t r = 0; r < n_reco; ++r) {
                const Real_t phi =
                    Real_t{12} + Real_t{0.23} * static_cast<Real_t>(z) + Real_t{1.9} * static_cast<Real_t>(r);
                const Real_t detector =
                    Real_t{1} + Real_t{0.1} * static_cast<Real_t>(r) + Real_t{0.01} * static_cast<Real_t>(z);
                require_close(events.counts(z, r), phi * detector, "predicted event count is wrong");
            }
        }

        std::cout << "[PASS] event prediction\n";
    }

    // -------------------------------------------------------------------------
    // Rebin -> events integration path
    // -------------------------------------------------------------------------

    void test_rebinned_events(const ResponseArray& response) {
        constexpr Index_t samples_per_bin = 2;

        const auto fine_z = nt::sample_coszenith_bin_midpoints(response.coszenith_edges.view(), samples_per_bin);
        const auto fine_e = nt::sample_log_energy_bin_midpoints(response.true_energy_edges_gev.view(), samples_per_bin);

        Flux fine(fine_z.extent(0), fine_e.extent(0));

        for (Index_t z = 0; z < fine_z.extent(0); ++z)
            fine.coszenith()(z) = fine_z(z);
        for (Index_t e = 0; e < fine_e.extent(0); ++e)
            fine.energy_gev()(e) = fine_e(e);

        auto numu     = fine.numu();
        auto antinumu = fine.antinumu();

        for (Index_t z = 0; z < fine.n_coszenith(); ++z) {
            for (Index_t e = 0; e < fine.n_energy(); ++e) {
                const Real_t cz     = fine.coszenith()(z);
                const Real_t energy = fine.energy_gev()(e);
                numu(z, e)     = Real_t{10} + Real_t{4} * cz + Real_t{0.002} * energy + Real_t{0.001} * cz * energy;
                antinumu(z, e) = Real_t{2} - Real_t{0.7} * cz + Real_t{0.001} * energy;
            }
        }

        nt::FluxRebinOptions options;
        options.interpolate_coszenith = true;
        options.interpolate_energy    = true;

        const auto rebinned = nt::rebin_flux(fine, response.coszenith.view(), response.true_energy_gev.view(),
                                             samples_per_bin, samples_per_bin, options);
        const auto events   = nt::predict_events(rebinned, response);

        for (Index_t z = 0; z < n_cz; ++z) {
            for (Index_t r = 0; r < n_reco; ++r) {
                const Real_t cz     = response.coszenith(z);
                const Real_t energy = response.true_energy_gev(r);
                const Real_t phi = Real_t{12} + Real_t{3.3} * cz + Real_t{0.003} * energy + Real_t{0.001} * cz * energy;
                const Real_t detector =
                    Real_t{1} + Real_t{0.1} * static_cast<Real_t>(r) + Real_t{0.01} * static_cast<Real_t>(z);
                require_close(events.counts(z, r), phi * detector, "rebinned event path is wrong");
            }
        }

        std::cout << "[PASS] rebin -> event prediction\n";
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

        for (Index_t t = 0; t < n_true; ++t)
            for (Index_t z = 0; z < n_cz; ++z)
                require(std::isfinite(response.detector_response(t, z)) && response.detector_response(t, z) >= 0,
                        "real TRIDENT detector response contains invalid values");

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
        test_rebinned_events(response);

        std::filesystem::remove_all(directory);
        test_real_trident_files();

        std::cout << "[PASS] all response/event tests\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
