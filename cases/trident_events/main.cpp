#include <nt/earth.hpp>
#include <nt/events.hpp>
#include <nt/flux.hpp>
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

    void check_energy_grids(const ResponseArray& response) {
        if (response.true_energy_gev.extent(0) != response.reco_energy_gev.extent(0) ||
            response.true_energy_edges_gev.extent(0) != response.reco_energy_edges_gev.extent(0)) {
            throw std::runtime_error("TRIDENT true-energy and proxy-energy binning differ");
        }

        for (Index_t i = 0; i < response.true_energy_gev.extent(0); ++i) {
            const Real_t a = response.true_energy_gev(i);
            const Real_t b = response.reco_energy_gev(i);

            if (std::abs(a - b) > Real_t{1e-12} * std::abs(a)) {
                throw std::runtime_error("TRIDENT true-energy and proxy-energy bin centers differ");
            }
        }

        for (Index_t i = 0; i < response.true_energy_edges_gev.extent(0); ++i) {
            const Real_t a = response.true_energy_edges_gev(i);
            const Real_t b = response.reco_energy_edges_gev(i);

            if (std::abs(a - b) > Real_t{1e-12} * std::abs(a)) {
                throw std::runtime_error("TRIDENT true-energy and proxy-energy bin edges differ");
            }
        }
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
        bool plot = true;

        for (int i = 1; i < argc; ++i) {
            const std::string_view argument = argv[i];

            if (argument == "--no-plot") {
                plot = false;
            } else {
                throw std::invalid_argument("Unknown argument: " + std::string(argument));
            }
        }

        const fs::path daemonflux_file = "data/generated/daemonflux/daemonflux_0.8.2.h5";
        const fs::path result_dir      = "result/trident_events";
        const fs::path figure_dir      = result_dir / "figures";
        const fs::path csv_file        = result_dir / "events.csv";

        if (!fs::is_regular_file(daemonflux_file)) {
            throw std::runtime_error("DaemonFlux table not found: " + daemonflux_file.string() +
                                     "\nRun scripts/build_deps/80_daemonflux.sh first.");
        }

        fs::create_directories(result_dir);
        fs::create_directories(figure_dir);

        const auto response = nt::load_trident_response();

        check_energy_grids(response);

        const auto prem = nt::load_prem();

        const std::array<Real_t, 5> unity = {
            1.0, 1.0, 1.0, 1.0, 1.0,
        };

        const auto earth = nt::make_layered_constant_5(prem, view(unity));

        print_model(earth);

        std::cout << '\n';
        std::cout << cyan << "Grid" << reset << '\n';
        std::cout << "  cosZenith bins : " << response.coszenith.extent(0) << '\n';
        std::cout << "  energy bins    : " << response.true_energy_gev.extent(0) << '\n';
        std::cout << "  grid points    : " << response.coszenith.extent(0) * response.true_energy_gev.extent(0) << '\n';

        const Real_t theta_first = detector_zenith_deg(response.coszenith(response.coszenith.extent(0) - 1));
        const Real_t theta_last  = detector_zenith_deg(response.coszenith(0));

        std::cout << "  zenith range   : " << std::fixed << std::setprecision(4) << theta_first << " -- " << theta_last
                  << " deg\n";

        std::cout << '\n' << cyan << "Loading atmospheric flux..." << reset << '\n';

        const auto daemonflux = nt::load_daemonflux(daemonflux_file, "IceCube");

        // Propagation is physically performed in true neutrino energy.
        // The current TRIDENT true-energy and proxy-energy grids are identical,
        // which was checked above.
        const auto initial = nt::resample_flux(daemonflux, response.coszenith.view(), response.true_energy_gev.view());

        std::cout << cyan << "Propagating through Earth..." << reset << '\n';

        const auto propagated = nt::propagate_flux(initial, prem, earth);

        std::cout << cyan << "Applying detector response..." << reset << '\n';

        const auto events = nt::predict_events(propagated, response);

        write_events_csv(csv_file, events);

        const Real_t total = total_events(events);

        std::cout << '\n';
        std::cout << dim << "────────────────────────────────────────────────────────────" << reset << '\n';
        std::cout << green << bold << "Total expected events : " << std::fixed << std::setprecision(6) << total << reset
                  << '\n';
        std::cout << "Event table           : " << csv_file << '\n';

        if (plot) {
            plot_events(csv_file, figure_dir);
            std::cout << "Figures               : " << figure_dir << '\n';
        }

        std::cout << dim << "────────────────────────────────────────────────────────────" << reset << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\033[1;31m[fatal]\033[0m " << error.what() << '\n';
        return 1;
    }
}
