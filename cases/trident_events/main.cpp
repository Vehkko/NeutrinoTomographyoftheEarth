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

    namespace fs  = std::filesystem;
    namespace nda = nt::nda;

    constexpr Real_t pi = 3.141592653589793238462643383279502884;

    constexpr std::string_view reset = "\033[0m";
    constexpr std::string_view bold  = "\033[1m";
    constexpr std::string_view dim   = "\033[2m";
    constexpr std::string_view cyan  = "\033[1;36m";
    constexpr std::string_view green = "\033[1;32m";

    Real_t detector_zenith_deg(Real_t coszenith) { return std::acos(coszenith) * Real_t{180} / pi; }

    Index_t parse_positive_index(std::string_view text, const char* option) {
        std::size_t consumed = 0;
        const auto  value    = std::stoull(std::string(text), &consumed);

        if (consumed != text.size() || value == 0)
            throw std::invalid_argument(std::string(option) + " must be a positive integer");
        return static_cast<Index_t>(value);
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

        run_process({python.string(), script.string(), csv_file.string(), figure_dir.string()});
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
        bool                 plot                      = true;
        Index_t              energy_samples_per_bin    = 1;
        Index_t              coszenith_samples_per_bin = 1;
        nt::FluxRebinOptions rebin_options{.interpolate_coszenith = false, .interpolate_energy = true};

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
            } else if (argument == "--interpolate-energy") {
                rebin_options.interpolate_energy = true;
            } else if (argument == "--interpolate-coszenith") {
                rebin_options.interpolate_coszenith = true;
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
        const auto prem     = nt::load_prem();

        const std::array<Real_t, 5> unity = {
            1.0, 1.0, 1.0, 1.0, 1.0,
        };

        const auto earth =
            nt::make_layered_constant_5(prem, nda::make_view1d(static_cast<const Real_t*>(unity.data()), unity.size()));

        print_model(earth);

        // { // debug density and Ye around the core-mantle boundary
        //     constexpr Real_t earth_radius_km = 6371.0;
        //
        //     std::vector<double>   prem_x(prem.radius_fraction.data(),
        //                                  prem.radius_fraction.data() + prem.radius_fraction.extent(0));
        //     std::vector<double>   prem_ye(prem.ye.data(), prem.ye.data() + prem.ye.extent(0));
        //     nusquids::AkimaSpline ye_interp(prem_x, prem_ye);
        //
        //     const std::array<Real_t, 18> samples = {
        //         0.519465, 0.522810, 0.526155, 0.529500, 0.532845, 0.536190, 0.539535, 0.542880, 0.546225,
        //         0.549541, 0.552857, 0.556173, 0.559490, 0.562806, 0.566122, 0.569438, 0.572754, 0.576070,
        //     };
        //
        //     std::cout << '\n' << std::fixed << std::setprecision(6);
        //
        //     for (const Real_t x : samples) {
        //         const Real_t radius_km = x * earth_radius_km;
        //         const Real_t rho       = nt::density_g_cm3(earth, radius_km);
        //         const Real_t ye        = ye_interp(x);
        //         std::cout << x << ' ' << rho << ' ' << ye << '\n';
        //     }
        // }

        const auto fine_coszenith =
            nt::sample_coszenith_bin_midpoints(response.coszenith_edges.view(), coszenith_samples_per_bin);
        const auto fine_energy =
            nt::sample_log_energy_bin_midpoints(response.true_energy_edges_gev.view(), energy_samples_per_bin);

        std::cout << '\n';
        std::cout << cyan << "Grid" << reset << '\n';
        std::cout << "  cosZenith bins        : " << response.coszenith.extent(0) << '\n';
        std::cout << "  cosZenith samples/bin : " << coszenith_samples_per_bin << '\n';
        std::cout << "  cosZenith grid points : " << fine_coszenith.extent(0) << '\n';
        std::cout << "  cosZenith rebin       : " << (rebin_options.interpolate_coszenith ? "interpolate" : "average")
                  << '\n';
        std::cout << "  energy bins           : " << response.true_energy_gev.extent(0) << '\n';
        std::cout << "  energy samples/bin    : " << energy_samples_per_bin << '\n';
        std::cout << "  energy grid points    : " << fine_energy.extent(0) << '\n';
        std::cout << "  energy rebin          : " << (rebin_options.interpolate_energy ? "interpolate" : "average")
                  << '\n';
        std::cout << "  fine grid points      : " << fine_coszenith.extent(0) * fine_energy.extent(0) << '\n';

        const Real_t theta_first = detector_zenith_deg(response.coszenith(response.coszenith.extent(0) - 1));
        const Real_t theta_last  = detector_zenith_deg(response.coszenith(0));

        std::cout << "  zenith range          : " << std::fixed << std::setprecision(4) << theta_first << " -- "
                  << theta_last << " deg\n";

        std::cout << '\n' << cyan << "Loading atmospheric flux..." << reset << '\n';

        const auto daemonflux = nt::load_daemonflux(daemonflux_file, "IceCube");

        // Fine propagation grid:
        //   - uniform midpoint sampling in cos(zenith);
        //   - uniform midpoint sampling in log10(E/GeV).
        const auto initial_fine = nt::resample_flux(daemonflux, fine_coszenith.view(), fine_energy.view());
        const auto initial = nt::rebin_flux(initial_fine, response.coszenith.view(), response.true_energy_gev.view(),
                                            coszenith_samples_per_bin, energy_samples_per_bin, rebin_options);

        std::cout << cyan << "Applying detector response to rebinned initial flux..." << reset << '\n';

        const auto initial_events = nt::predict_events(initial, response);
        write_events_csv(initial_csv_file, initial_events);

        Real_t initial_total = 0.0;
        for (Index_t z = 0; z < initial_events.counts.extent(0); ++z)
            for (Index_t e = 0; e < initial_events.counts.extent(1); ++e)
                initial_total += initial_events.counts(z, e);

        std::cout << cyan << "Propagating fine grid through Earth with interactions enabled..." << reset << '\n';

        nt::PropagationOptions propagation_options;
        propagation_options.interactions = true;
        propagation_options.threads      = 24;

        nt::EarthPropagator solver(initial_fine, propagation_options);
        const auto          propagated_fine = solver.propagate(initial_fine, prem, earth);

        // { // debug: save propagated nu_mu + anti-nu_mu flux matrix to project root
        //     const auto numu     = propagated_fine.numu();
        //     const auto antinumu = propagated_fine.antinumu();
        //     std::ofstream file("propagated_numu_total_34x100.csv");
        //
        //     if (!file)
        //         throw std::runtime_error("Cannot create propagated_numu_total_34x100.csv");
        //
        //     file << std::scientific << std::setprecision(17);
        //
        //     for (Index_t z = 0; z < propagated_fine.n_coszenith(); ++z) {
        //         for (Index_t e = 0; e < propagated_fine.n_energy(); ++e) {
        //             if (e != 0)
        //                 file << ',';
        //             file << numu(z, e) + antinumu(z, e);
        //         }
        //         file << '\n';
        //     }
        // }

        const auto propagated =
            nt::rebin_flux(propagated_fine, response.coszenith.view(), response.true_energy_gev.view(),
                           coszenith_samples_per_bin, energy_samples_per_bin, rebin_options);

        // { // debug: save 34x20 propagated nu_mu + anti-nu_mu flux
        //     const auto    numu     = propagated.numu();
        //     const auto    antinumu = propagated.antinumu();
        //     std::ofstream file("propagated_flux_34x20.csv");
        //
        //     if (!file)
        //         throw std::runtime_error("Cannot create propagated_flux_34x20.csv");
        //
        //     file << std::scientific << std::setprecision(17);
        //
        //     for (Index_t z = 0; z < propagated.n_coszenith(); ++z) {
        //         for (Index_t e = 0; e < propagated.n_energy(); ++e) {
        //             if (e != 0)
        //                 file << ',';
        //             file << numu(z, e) + antinumu(z, e);
        //         }
        //         file << '\n';
        //     }
        // }

        std::cout << cyan << "Applying detector response to rebinned propagated flux..." << reset << '\n';

        const auto propagated_events = nt::predict_events(propagated, response);
        write_events_csv(propagated_csv_file, propagated_events);

        Real_t propagated_total = 0.0;
        for (Index_t z = 0; z < propagated_events.counts.extent(0); ++z)
            for (Index_t e = 0; e < propagated_events.counts.extent(1); ++e)
                propagated_total += propagated_events.counts(z, e);

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
