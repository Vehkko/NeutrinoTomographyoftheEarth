#include <nt/earth.hpp>
#include <nt/events.hpp>
#include <nt/flux.hpp>
#include <nt/propagation.hpp>
#include <nt/response.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

    using nt::EventDistribution;
    using nt::Index_t;
    using nt::Real_t;
    using nt::ResponseArray;

    namespace fs  = std::filesystem;
    namespace nda = nt::nda;

    constexpr Real_t pi = 3.141592653589793238462643383279502884;

    constexpr std::string_view reset = "\033[0m";
    constexpr std::string_view bold  = "\033[1m";
    constexpr std::string_view dim   = "\033[2m";
    constexpr std::string_view cyan  = "\033[1;36m";
    constexpr std::string_view green = "\033[1;32m";

    template <std::size_t N> auto view(const std::array<Real_t, N>& values) {
        return nda::make_view1d(static_cast<const Real_t*>(values.data()), values.size());
    }

    Real_t detector_zenith_deg(Real_t coszenith) { return std::acos(coszenith) * Real_t{180} / pi; }

    Index_t parse_positive_index(std::string_view text, const char* option) {
        std::size_t consumed = 0;
        const auto  value    = std::stoull(std::string(text), &consumed);

        if (consumed != text.size() || value == 0)
            throw std::invalid_argument(std::string(option) + " must be a positive integer");

        return static_cast<Index_t>(value);
    }

    void check_energy_grids(const ResponseArray& response) {
        if (response.true_energy_gev.extent(0) != response.reco_energy_gev.extent(0) ||
            response.true_energy_edges_gev.extent(0) != response.reco_energy_edges_gev.extent(0)) {
            throw std::runtime_error("TRIDENT true-energy and proxy-energy binning differ");
        }

        for (Index_t i = 0; i < response.true_energy_gev.extent(0); ++i) {
            const Real_t a = response.true_energy_gev(i);
            const Real_t b = response.reco_energy_gev(i);

            if (std::abs(a - b) > Real_t{1e-12} * std::abs(a))
                throw std::runtime_error("TRIDENT true-energy and proxy-energy bin centers differ");
        }

        for (Index_t i = 0; i < response.true_energy_edges_gev.extent(0); ++i) {
            const Real_t a = response.true_energy_edges_gev(i);
            const Real_t b = response.reco_energy_edges_gev(i);

            if (std::abs(a - b) > Real_t{1e-12} * std::abs(a))
                throw std::runtime_error("TRIDENT true-energy and proxy-energy bin edges differ");
        }
    }

    nda::Array<Real_t, 1> sample_log_energy_bin_midpoints(nda::View<const Real_t, 1> edges, Index_t samples_per_bin) {
        if (edges.extent(0) < 2)
            throw std::invalid_argument("Energy bin edges must contain at least two points");

        if (samples_per_bin == 0)
            throw std::invalid_argument("Energy samples per bin must be positive");

        const Index_t bins = edges.extent(0) - 1;

        nda::Array<Real_t, 1> samples({bins * samples_per_bin});

        for (Index_t bin = 0; bin < bins; ++bin) {
            const Real_t e0 = edges(bin);
            const Real_t e1 = edges(bin + 1);

            if (!(e0 > 0.0 && e1 > e0))
                throw std::invalid_argument("Energy bin edges must be positive and strictly increasing");

            const Real_t loge0 = std::log10(e0);
            const Real_t loge1 = std::log10(e1);

            for (Index_t k = 0; k < samples_per_bin; ++k) {
                const Real_t u = (static_cast<Real_t>(k) + Real_t{0.5}) / static_cast<Real_t>(samples_per_bin);

                samples(bin * samples_per_bin + k) = std::pow(Real_t{10}, loge0 + u * (loge1 - loge0));
            }
        }

        return samples;
    }

    nt::Flux average_flux_to_response_bins(const nt::Flux& fine_flux, const ResponseArray& response,
                                           Index_t coszenith_samples_per_bin, Index_t energy_samples_per_bin) {
        if (coszenith_samples_per_bin == 0 || energy_samples_per_bin == 0)
            throw std::invalid_argument("Samples per bin must be positive");

        const Index_t ncz   = response.coszenith.extent(0);
        const Index_t ntrue = response.true_energy_gev.extent(0);

        if (fine_flux.n_coszenith() != ncz * coszenith_samples_per_bin)
            throw std::invalid_argument("Fine Flux coszenith dimension does not match requested refinement");

        if (fine_flux.n_energy() != ntrue * energy_samples_per_bin)
            throw std::invalid_argument("Fine Flux energy dimension does not match requested refinement");

        nt::Flux coarse(ncz, ntrue);

        auto coarse_z = coarse.coszenith();
        auto coarse_e = coarse.energy_gev();

        for (Index_t z = 0; z < ncz; ++z)
            coarse_z(z) = response.coszenith(z);

        for (Index_t e = 0; e < ntrue; ++e)
            coarse_e(e) = response.true_energy_gev(e);

        const Real_t inv_samples = Real_t{1} / static_cast<Real_t>(coszenith_samples_per_bin * energy_samples_per_bin);

        for (Index_t p = 0; p < 2; ++p) {
            for (Index_t f = 0; f < 3; ++f) {
                const auto src = fine_flux.component(static_cast<nt::Particle>(p), static_cast<nt::Flavor>(f));

                auto dst = coarse.component(static_cast<nt::Particle>(p), static_cast<nt::Flavor>(f));

                for (Index_t z = 0; z < ncz; ++z) {
                    for (Index_t e = 0; e < ntrue; ++e) {
                        Real_t sum = 0.0;

                        for (Index_t iz = 0; iz < coszenith_samples_per_bin; ++iz) {
                            const Index_t fine_z = z * coszenith_samples_per_bin + iz;

                            for (Index_t ie = 0; ie < energy_samples_per_bin; ++ie) {
                                const Index_t fine_e = e * energy_samples_per_bin + ie;
                                sum += src(fine_z, fine_e);
                            }
                        }

                        dst(z, e) = sum * inv_samples;
                    }
                }
            }
        }

        return coarse;
    }

    Real_t total_events(const EventDistribution& events) {
        Real_t total = 0.0;

        for (Index_t z = 0; z < events.counts.extent(0); ++z) {
            for (Index_t e = 0; e < events.counts.extent(1); ++e) {
                const Real_t value = events.counts(z, e);

                if (!std::isfinite(value) || value < 0.0)
                    throw std::runtime_error("Event distribution contains a non-finite or negative value");

                total += value;
            }
        }

        return total;
    }

    void write_events_csv(const fs::path& filename, const EventDistribution& events) {
        std::ofstream file(filename);

        if (!file)
            throw std::runtime_error("Cannot create event CSV: " + filename.string());

        file << "coszenith_bin,proxy_energy_bin,"
                "coszenith_low,coszenith_center,coszenith_high,"
                "zenith_low_deg,zenith_center_deg,zenith_high_deg,"
                "proxy_energy_low_gev,proxy_energy_center_gev,proxy_energy_high_gev,events\n";

        file << std::setprecision(17);

        for (Index_t z = 0; z < events.coszenith.extent(0); ++z) {
            const Real_t cos_low    = events.coszenith_edges(z);
            const Real_t cos_center = events.coszenith(z);
            const Real_t cos_high   = events.coszenith_edges(z + 1);

            const Real_t zenith_low    = detector_zenith_deg(cos_high);
            const Real_t zenith_center = detector_zenith_deg(cos_center);
            const Real_t zenith_high   = detector_zenith_deg(cos_low);

            for (Index_t e = 0; e < events.reco_energy_gev.extent(0); ++e) {
                file << z << ',' << e << ',' << cos_low << ',' << cos_center << ',' << cos_high << ',' << zenith_low
                     << ',' << zenith_center << ',' << zenith_high << ',' << events.reco_energy_edges_gev(e) << ','
                     << events.reco_energy_gev(e) << ',' << events.reco_energy_edges_gev(e + 1) << ','
                     << events.counts(z, e) << '\n';
            }
        }
    }

    void run_process(const std::vector<std::string>& arguments) {
        if (arguments.empty())
            throw std::runtime_error("Cannot run an empty command");

        const pid_t pid = fork();

        if (pid < 0)
            throw std::runtime_error("fork() failed");

        if (pid == 0) {
            std::vector<char*> argv;
            argv.reserve(arguments.size() + 1);

            for (const auto& argument : arguments)
                argv.push_back(const_cast<char*>(argument.c_str()));

            argv.push_back(nullptr);

            execv(arguments.front().c_str(), argv.data());
            _exit(127);
        }

        int status = 0;

        if (waitpid(pid, &status, 0) < 0)
            throw std::runtime_error("waitpid() failed");

        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("Plotting process failed");
    }

    void plot_events(const fs::path& csv_file, const fs::path& figure_dir) {
        const fs::path python = "runtime/python/venv/bin/python";
        const fs::path script = "cases/trident_events/plot_events.py";

        if (!fs::exists(python))
            throw std::runtime_error("Project Python not found: " + python.string());

        if (!fs::exists(script))
            throw std::runtime_error("Plotting script not found: " + script.string());

        run_process({
            python.string(),
            script.string(),
            csv_file.string(),
            figure_dir.string(),
        });
    }

    void print_model(const nt::LayeredEarth& earth) {
        std::cout << '\n';
        std::cout << cyan << bold << "TRIDENT event case" << reset << '\n';
        std::cout << dim << "────────────────────────────────────────────────────────────" << reset << '\n';
        std::cout << "Earth model     : 5-layer constant density\n";
        std::cout << "Electron fraction: PREM Y_e(r)\n";
        std::cout << "Flux location   : IceCube\n";
        std::cout << "Detector zenith : 0 deg overhead, 90--180 deg up-going\n";
        std::cout << '\n';

        std::cout << cyan << "Layer densities" << reset << '\n';

        Real_t inner = 0.0;

        for (Index_t i = 0; i < earth.layers; ++i) {
            std::cout << "  L" << i + 1 << "  " << std::fixed << std::setprecision(0) << std::setw(4) << inner << " -- "
                      << std::setw(4) << earth.outer_radius_km[i] << " km"
                      << "    rho = " << std::setprecision(6) << earth.density_g_cm3[i] << " g/cm^3\n";

            inner = earth.outer_radius_km[i];
        }
    }

} // namespace

int main(int argc, char** argv) {
    try {
        bool    plot                      = true;
        Index_t energy_samples_per_bin    = 1;
        Index_t coszenith_samples_per_bin = 1;

        for (int i = 1; i < argc; ++i) {
            const std::string_view argument = argv[i];

            if (argument == "--no-plot") {
                plot = false;
            } else if (argument == "--energy-samples-per-bin") {
                if (++i >= argc)
                    throw std::invalid_argument("--energy-samples-per-bin requires a value");

                energy_samples_per_bin = parse_positive_index(argv[i], "--energy-samples-per-bin");
            } else if (argument == "--coszenith-samples-per-bin") {
                if (++i >= argc)
                    throw std::invalid_argument("--coszenith-samples-per-bin requires a value");

                coszenith_samples_per_bin = parse_positive_index(argv[i], "--coszenith-samples-per-bin");
            } else {
                throw std::invalid_argument("Unknown argument: " + std::string(argument));
            }
        }

        const fs::path daemonflux_file     = "data/generated/daemonflux/daemonflux_0.8.2.h5";
        const fs::path result_dir          = "result/trident_events";
        const fs::path figure_dir          = result_dir / "figures";
        const fs::path initial_csv_file    = result_dir / "initial_events.csv";
        const fs::path propagated_csv_file = result_dir / "propagated_events.csv";

        if (!fs::is_regular_file(daemonflux_file)) {
            throw std::runtime_error("DaemonFlux table not found: " + daemonflux_file.string() +
                                     "\nRun scripts/build_deps/80_daemonflux.sh first.");
        }

        fs::create_directories(result_dir);
        fs::create_directories(figure_dir);

        const auto response = nt::load_trident_response();

        check_energy_grids(response);

        // {  // print cosZ bins
        //     std::cout << '\n';
        //     std::cout << cyan << "cosZenith binning" << reset << '\n';
        //
        //     for (Index_t z = 0; z < response.coszenith.extent(0); ++z) {
        //         std::cout << "  bin " << std::setw(2) << z << "  [" << std::fixed << std::setprecision(8)
        //                   << response.coszenith_edges(z) << ", " << response.coszenith_edges(z + 1) << "]"
        //                   << "  center = " << response.coszenith(z) << '\n';
        //     }
        //
        //     auto response = nt::load_trident_response();
        //
        //     check_energy_grids(response);
        //
        //     for (Index_t t = 0; t < response.true_energy_gev.extent(0); ++t) {
        //         for (Index_t r = 0; r < response.reco_energy_gev.extent(0); ++r)
        //             response.energy_migration(t, r) = t == r ? Real_t{1} : Real_t{0};
        //     }
        // }

        const auto prem = nt::load_prem();

        const std::array<Real_t, 5> unity = {
            1.0, 1.0, 1.0, 1.0, 1.0,
        };

        const auto earth = nt::make_layered_constant_5(prem, view(unity));

        print_model(earth);

        const auto fine_coszenith =
            nt::sample_coszenith_bin_midpoints(response.coszenith_edges.view(), coszenith_samples_per_bin);

        const auto fine_energy =
            sample_log_energy_bin_midpoints(response.true_energy_edges_gev.view(), energy_samples_per_bin);

        // {  // print proxy energy bins
        //     std::cout << '\n';
        //     std::cout << cyan << "Fine true-energy binning" << reset << '\n';
        //
        //     for (Index_t bin = 0; bin < response.true_energy_gev.extent(0); ++bin) {
        //         const Real_t loge0 = std::log10(response.true_energy_edges_gev(bin));
        //         const Real_t loge1 = std::log10(response.true_energy_edges_gev(bin + 1));
        //
        //         for (Index_t k = 0; k < energy_samples_per_bin; ++k) {
        //             const Real_t u0 = static_cast<Real_t>(k) / static_cast<Real_t>(energy_samples_per_bin);
        //
        //             const Real_t u1 = static_cast<Real_t>(k + 1) / static_cast<Real_t>(energy_samples_per_bin);
        //
        //             const Real_t fine_loge0 = loge0 + u0 * (loge1 - loge0);
        //             const Real_t fine_loge1 = loge0 + u1 * (loge1 - loge0);
        //
        //             const Real_t fine_e0 = std::pow(Real_t{10}, fine_loge0);
        //             const Real_t fine_e1 = std::pow(Real_t{10}, fine_loge1);
        //
        //             const Index_t index  = bin * energy_samples_per_bin + k;
        //             const Real_t  center = fine_energy(index);
        //
        //             std::cout << "  coarse " << std::setw(2) << bin << "  fine " << std::setw(3) << index
        //                       << "  log10(E/GeV) = [" << std::fixed << std::setprecision(6) << fine_loge0 << ", "
        //                       << fine_loge1 << "]"
        //                       << "  center = " << std::log10(center) << "  E = " << std::scientific
        //                       << std::setprecision(8) << center << " GeV" << '\n';
        //         }
        //     }
        // }

        std::cout << '\n';
        std::cout << cyan << "Grid" << reset << '\n';
        std::cout << "  cosZenith bins        : " << response.coszenith.extent(0) << '\n';
        std::cout << "  cosZenith samples/bin : " << coszenith_samples_per_bin << '\n';
        std::cout << "  cosZenith grid points : " << fine_coszenith.extent(0) << '\n';
        std::cout << "  energy bins           : " << response.true_energy_gev.extent(0) << '\n';
        std::cout << "  energy samples/bin    : " << energy_samples_per_bin << '\n';
        std::cout << "  energy grid points    : " << fine_energy.extent(0) << '\n';
        std::cout << "  fine grid points      : " << fine_coszenith.extent(0) * fine_energy.extent(0) << '\n';

        const Real_t theta_first = detector_zenith_deg(response.coszenith(response.coszenith.extent(0) - 1));
        const Real_t theta_last  = detector_zenith_deg(response.coszenith(0));

        std::cout << "  zenith range          : " << std::fixed << std::setprecision(4) << theta_first << " -- "
                  << theta_last << " deg\n";

        std::cout << '\n' << cyan << "Loading atmospheric flux..." << reset << '\n';

        const auto daemonflux = nt::load_daemonflux(daemonflux_file, "IceCube");

        // Fine quadrature grid:
        //   - uniform midpoint sampling in cos(zenith);
        //   - uniform midpoint sampling in log10(E/GeV).
        //
        // One sample per bin reproduces the original center-only treatment.
        const auto initial_fine = nt::resample_flux(daemonflux, fine_coszenith.view(), fine_energy.view());

        const auto initial =
            average_flux_to_response_bins(initial_fine, response, coszenith_samples_per_bin, energy_samples_per_bin);

        std::cout << cyan << "Applying detector response to averaged initial flux..." << reset << '\n';

        const auto initial_events = nt::predict_events(initial, response);

        write_events_csv(initial_csv_file, initial_events);

        const Real_t initial_total = total_events(initial_events);

        std::cout << cyan << "Propagating fine grid through Earth with interactions enabled..." << reset << '\n';

        nt::PropagationOptions propagation_options;
        propagation_options.interactions = true;
        propagation_options.threads      = 24;

        nt::EarthPropagator solver(initial_fine, propagation_options);

        const auto propagated_fine = solver.propagate(initial_fine, prem, earth);

        const auto propagated =
            average_flux_to_response_bins(propagated_fine, response, coszenith_samples_per_bin, energy_samples_per_bin);

        std::cout << cyan << "Applying detector response to averaged propagated flux..." << reset << '\n';

        const auto propagated_events = nt::predict_events(propagated, response);

        write_events_csv(propagated_csv_file, propagated_events);

        const Real_t propagated_total = total_events(propagated_events);

        std::cout << '\n';
        std::cout << dim << "────────────────────────────────────────────────────────────" << reset << '\n';
        std::cout << green << bold << "Initial total events    : " << std::fixed << std::setprecision(6)
                  << initial_total << reset << '\n';
        std::cout << green << bold << "Propagated total events : " << std::fixed << std::setprecision(6)
                  << propagated_total << reset << '\n';
        std::cout << "Initial event table     : " << initial_csv_file << '\n';
        std::cout << "Propagated event table  : " << propagated_csv_file << '\n';

        if (plot) {
            plot_events(initial_csv_file, figure_dir / "initial");
            plot_events(propagated_csv_file, figure_dir / "propagated");

            std::cout << "Initial figures         : " << figure_dir / "initial" << '\n';
            std::cout << "Propagated figures      : " << figure_dir / "propagated" << '\n';
        }

        std::cout << dim << "────────────────────────────────────────────────────────────" << reset << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\033[1;31m[fatal]\033[0m " << error.what() << '\n';
        return 1;
    }
}
