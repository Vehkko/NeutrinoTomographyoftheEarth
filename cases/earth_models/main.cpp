#include <nt/earth.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

#include <array>
#include <cstdlib>
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

    using nt::Real_t;

    namespace fs  = std::filesystem;
    namespace nda = nt::nda;

    constexpr Real_t earth_radius_km = 6371.0;
    constexpr Real_t sample_step_km  = 1.0;

    struct Profile {
        std::string name;
        fs::path    data_file;
    };

    struct Summary {
        std::string name;
        Real_t      mass_kg;
        Real_t      inertia_kg_m2;
        Real_t      mean_density_g_cm3;
    };

    template <std::size_t N> auto view(const std::array<Real_t, N>& values) {
        return nda::make_view1d(static_cast<const Real_t*>(values.data()), values.size());
    }

    template <typename DensityFunction>
    void write_density_profile(const fs::path& filename, DensityFunction&& density) {
        std::ofstream file(filename);

        if (!file) {
            throw std::runtime_error("Cannot create density file: " + filename.string());
        }

        // Keep the file format deliberately simple and portable. Duplicate or
        // closely spaced radii are permitted, so discontinuous layered models
        // can also be represented without inventing a model-specific format.
        file << "radius_km,density_g_cm3\n" << std::setprecision(17);

        for (Real_t r = 0.0; r < earth_radius_km; r += sample_step_km) {
            file << r << ',' << density(r) << '\n';
        }

        // The final point is written explicitly because the Earth radius is not
        // required to be an integer multiple of the sampling step.
        file << earth_radius_km << ',' << density(earth_radius_km) << '\n';
    }

    // Run the project-local Python interpreter directly rather than through a
    // shell. This keeps argument passing unambiguous and leaves Python entirely
    // outside the native tomography library.
    void run_process(const std::vector<std::string>& arguments) {
        if (arguments.empty())
            throw std::runtime_error("Cannot run an empty command");

        const pid_t pid = fork();

        if (pid < 0)
            throw std::runtime_error("fork() failed");

        if (pid == 0) {
            std::vector<char*> argv;
            argv.reserve(arguments.size() + 1);

            for (const auto& argument : arguments) {
                argv.push_back(const_cast<char*>(argument.c_str()));
            }

            argv.push_back(nullptr);

            execv(arguments.front().c_str(), argv.data());

            _exit(127);
        }

        int status = 0;

        if (waitpid(pid, &status, 0) < 0) {
            throw std::runtime_error("waitpid() failed");
        }

        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            throw std::runtime_error("Plotting process failed");
        }
    }

    void plot_profiles(const std::vector<Profile>& profiles, const fs::path& output, std::string_view title) {
        const fs::path python = "runtime/python/venv/bin/python";

        const fs::path script = "cases/earth_models/plot_density.py";

        if (!fs::exists(python)) {
            throw std::runtime_error("Project Python not found: " + python.string());
        }

        if (!fs::exists(script)) {
            throw std::runtime_error("Plotting script not found: " + script.string());
        }

        std::vector<std::string> arguments;

        arguments.emplace_back(python.string());

        arguments.emplace_back(script.string());

        for (const auto& profile : profiles) {
            arguments.emplace_back(profile.data_file.string());
        }

        arguments.emplace_back("--output");
        arguments.emplace_back(output.string());

        arguments.emplace_back("--title");
        arguments.emplace_back(title);

        for (const auto& profile : profiles) {
            arguments.emplace_back("--label");
            arguments.emplace_back(profile.name);
        }

        run_process(arguments);
    }

    void print_summary_table(const std::vector<Summary>& summaries) {
        constexpr int name_width   = 26;
        constexpr int number_width = 20;

        std::cout << '\n';

        std::cout << std::left << std::setw(name_width) << "Model"

                  << std::right << std::setw(number_width) << "Mass [1e24 kg]"

                  << std::setw(number_width) << "I [1e37 kg m^2]"

                  << std::setw(number_width) << "Mean rho [g/cm^3]" << '\n';

        std::cout << std::string(name_width + 3 * number_width, '-') << '\n';

        std::cout << std::fixed << std::setprecision(6);

        for (const auto& summary : summaries) {
            std::cout << std::left << std::setw(name_width) << summary.name

                      << std::right << std::setw(number_width) << summary.mass_kg / 1.0e24

                      << std::setw(number_width) << summary.inertia_kg_m2 / 1.0e37

                      << std::setw(number_width) << summary.mean_density_g_cm3 << '\n';
        }

        std::cout << '\n';
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

        const fs::path result_dir = "result/earth_models";

        const fs::path data_dir = result_dir / "data";

        const fs::path figure_dir = result_dir / "figures";

        fs::create_directories(data_dir);
        fs::create_directories(figure_dir);

        const auto prem = nt::load_prem();

        // These non-unity factors are deliberately chosen only to make the
        // different parameterizations visually distinguishable in this case.
        // They are not intended to represent fitted or preferred Earth models.
        const std::array<Real_t, 3> constant_q3 = {
            1.12,
            0.92,
            1.05,
        };

        const std::array<Real_t, 5> constant_q5 = {
            1.10, 0.95, 1.08, 0.90, 1.03,
        };

        const std::array<Real_t, 3> scaled_q3 = {
            1.08,
            0.94,
            1.04,
        };

        const std::array<Real_t, 5> scaled_q5 = {
            1.06, 0.96, 1.07, 0.92, 1.03,
        };

        const auto constant3 = nt::make_layered_constant_3(prem, view(constant_q3));

        const auto constant5 = nt::make_layered_constant_5(prem, view(constant_q5));

        const auto scaled3 = nt::make_prem_scaled_3(view(scaled_q3));

        const auto scaled5 = nt::make_prem_scaled_5(view(scaled_q5));

        const fs::path prem_file = data_dir / "prem.csv";

        const fs::path constant3_file = data_dir / "layered_constant_3.csv";

        const fs::path constant5_file = data_dir / "layered_constant_5.csv";

        const fs::path scaled3_file = data_dir / "prem_scaled_3.csv";

        const fs::path scaled5_file = data_dir / "prem_scaled_5.csv";

        write_density_profile(prem_file, [&](Real_t r) { return nt::density_g_cm3(prem, r); });

        write_density_profile(constant3_file, [&](Real_t r) { return nt::density_g_cm3(constant3, r); });

        write_density_profile(constant5_file, [&](Real_t r) { return nt::density_g_cm3(constant5, r); });

        write_density_profile(scaled3_file, [&](Real_t r) { return nt::density_g_cm3(prem, scaled3, r); });

        write_density_profile(scaled5_file, [&](Real_t r) { return nt::density_g_cm3(prem, scaled5, r); });

        std::vector<Summary> summaries;

        summaries.push_back({
            "PREM",
            nt::mass_kg(prem, 0.0, earth_radius_km),
            nt::moment_of_inertia_kg_m2(prem, 0.0, earth_radius_km),
            nt::mean_density_g_cm3(prem, 0.0, earth_radius_km),
        });

        summaries.push_back({
            "Layered constant 3",
            nt::mass_kg(constant3, 0.0, earth_radius_km),
            nt::moment_of_inertia_kg_m2(constant3, 0.0, earth_radius_km),
            nt::mean_density_g_cm3(constant3, 0.0, earth_radius_km),
        });

        summaries.push_back({
            "Layered constant 5",
            nt::mass_kg(constant5, 0.0, earth_radius_km),
            nt::moment_of_inertia_kg_m2(constant5, 0.0, earth_radius_km),
            nt::mean_density_g_cm3(constant5, 0.0, earth_radius_km),
        });

        summaries.push_back({
            "PREM scaled 3",
            nt::mass_kg(prem, scaled3, 0.0, earth_radius_km),
            nt::moment_of_inertia_kg_m2(prem, scaled3, 0.0, earth_radius_km),
            nt::mean_density_g_cm3(prem, scaled3, 0.0, earth_radius_km),
        });

        summaries.push_back({
            "PREM scaled 5",
            nt::mass_kg(prem, scaled5, 0.0, earth_radius_km),
            nt::moment_of_inertia_kg_m2(prem, scaled5, 0.0, earth_radius_km),
            nt::mean_density_g_cm3(prem, scaled5, 0.0, earth_radius_km),
        });

        print_summary_table(summaries);

        const std::vector<Profile> profiles = {
            {"PREM",               prem_file     },
            {"Layered constant 3", constant3_file},
            {"Layered constant 5", constant5_file},
            {"PREM scaled 3",      scaled3_file  },
            {"PREM scaled 5",      scaled5_file  },
        };

        if (plot) {
            for (const auto& profile : profiles) {
                const fs::path output = figure_dir / (profile.data_file.stem().string() + ".png");

                plot_profiles({profile}, output, profile.name);
            }

            plot_profiles(profiles, figure_dir / "comparison.png", "Earth density profiles");
        }

        std::cout << "Density data : " << data_dir << '\n';

        if (plot) {
            std::cout << "Figures      : " << figure_dir << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << '\n';

        return 1;
    }
}
