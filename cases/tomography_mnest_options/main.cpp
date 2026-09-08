#include <mpi.h>
#include <multinest.h>

#include <nt/earth.hpp>
#include <nt/events.hpp>
#include <nt/flux.hpp>
#include <nt/propagation.hpp>
#include <nt/response.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

    using nt::EventDistribution;
    using nt::Index_t;
    using nt::Real_t;

    namespace nda = nt::nda;

    constexpr Real_t Q_MIN = 0.0;

    constexpr Real_t EXPOSURE_YEARS = 1.0;

    constexpr int    MN_IS        = 0;
    constexpr int    MN_MMODAL    = 1;
    constexpr int    MN_CEFF      = 0;
    constexpr double MN_EFR       = 0.3;
    constexpr int    MN_MAX_MODES = 100;
    constexpr int    MN_UPD_INT   = 100;
    constexpr double MN_ZTOL      = -1.0e90;
    constexpr int    MN_SEED      = 12345;
    constexpr int    MN_FB        = 1;
    constexpr int    MN_RESUME    = 0;
    constexpr int    MN_OUTFILE   = 1;
    constexpr int    MN_INIT_MPI  = 0;
    constexpr double MN_LOG_ZERO  = -1.0e90;
    constexpr int    MN_MAXITER   = 0;

    constexpr std::uint64_t LOG_EVERY_N_EVAL = 1000;

    constexpr std::array<Real_t, 6> LAYER_EDGES_KM = {
        0.0, 1221.0, 3480.0, 4811.0, 5700.0, 6371.0,
    };

    constexpr std::array<const char*, 5> LAYER_NAMES = {
        "Inner core", "Outer core", "Lower mantle", "Middle mantle", "Upper mantle",
    };

    struct RunConfig {
        const char* name;
        int         nlive;
        double      tol;
        Real_t      q_max;
    };

    constexpr std::array<RunConfig, 6> RUNS = {
        {
         {"baseline", 1000, 0.1, 4.0},
         {"nlive_2000", 2000, 0.1, 4.0},
         {"nlive_3000", 3000, 0.1, 4.0},
         {"tol_0p01", 1000, 0.01, 4.0},
         {"prior_3", 1000, 0.1, 3.0},
         {"prior_5", 1000, 0.1, 5.0},
         }
    };

    struct Context {
        const nt::EarthProfile*      prem     = nullptr;
        const nt::ResponseArray*     response = nullptr;
        const nt::Flux*              initial  = nullptr;
        const nt::EventDistribution* data     = nullptr;
        nt::EarthPropagator*         solver   = nullptr;

        Real_t      q_max       = 4.0;
        const char* run_name    = nullptr;
        bool        console     = false;
        bool        fatal_error = false;
        std::string fatal_message;

        std::uint64_t evaluations = 0;

        bool   has_dumper = false;
        double logz       = std::numeric_limits<double>::quiet_NaN();
        double logz_error = std::numeric_limits<double>::quiet_NaN();
    };

    struct ParameterSummary {
        double q16       = 0.0;
        double q50       = 0.0;
        double q84       = 0.0;
        double rho16     = 0.0;
        double rho50     = 0.0;
        double rho84     = 0.0;
        double precision = 0.0;
    };

    struct RunSummary {
        RunConfig                       config{};
        std::array<ParameterSummary, 5> parameters{};
        double                          logz       = std::numeric_limits<double>::quiet_NaN();
        double                          logz_error = std::numeric_limits<double>::quiet_NaN();
    };

    template <std::size_t N> auto view(const std::array<Real_t, N>& values) {
        return nda::make_view1d(static_cast<const Real_t*>(values.data()), values.size());
    }

    Real_t total_events(const EventDistribution& events) {
        Real_t total = 0.0;

        for (Index_t z = 0; z < events.counts.extent(0); ++z) {
            for (Index_t e = 0; e < events.counts.extent(1); ++e)
                total += events.counts(z, e);
        }

        return total;
    }

    std::array<Real_t, 5> mean_layer_densities(const nt::EarthProfile& prem) {
        std::array<Real_t, 5> density{};

        for (Index_t i = 0; i < 5; ++i)
            density[i] = nt::mean_density_g_cm3(prem, LAYER_EDGES_KM[i], LAYER_EDGES_KM[i + 1]);

        return density;
    }

    double percentile(const std::vector<double>& sorted, double p) {
        if (sorted.empty())
            throw std::runtime_error("cannot calculate percentile of an empty posterior");

        if (sorted.size() == 1)
            return sorted.front();

        const double position = p * static_cast<double>(sorted.size() - 1);
        const auto   lower    = static_cast<std::size_t>(std::floor(position));
        const auto   upper    = static_cast<std::size_t>(std::ceil(position));
        const double weight   = position - static_cast<double>(lower);

        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
    }

    std::array<ParameterSummary, 5> summarize_posterior(const fs::path&              filename,
                                                        const std::array<Real_t, 5>& prem_density) {
        std::ifstream file(filename);

        if (!file)
            throw std::runtime_error("cannot open posterior file: " + filename.string());

        std::array<std::vector<double>, 5> samples;

        std::string line;

        while (std::getline(file, line)) {
            if (line.empty())
                continue;

            std::istringstream row(line);

            std::array<double, 5> q{};
            double                loglike = 0.0;

            for (double& value : q) {
                if (!(row >> value))
                    throw std::runtime_error("invalid posterior row in: " + filename.string());
            }

            if (!(row >> loglike))
                throw std::runtime_error("posterior row is missing log-likelihood in: " + filename.string());

            for (Index_t i = 0; i < 5; ++i)
                samples[i].push_back(q[i]);
        }

        std::array<ParameterSummary, 5> summary{};

        for (Index_t i = 0; i < 5; ++i) {
            auto& values = samples[i];

            std::sort(values.begin(), values.end());

            auto& out = summary[i];

            out.q16 = percentile(values, 0.16);
            out.q50 = percentile(values, 0.50);
            out.q84 = percentile(values, 0.84);

            out.rho16 = out.q16 * prem_density[i];
            out.rho50 = out.q50 * prem_density[i];
            out.rho84 = out.q84 * prem_density[i];

            if (out.q50 > 0.0)
                out.precision = (out.q84 - out.q16) / (2.0 * out.q50);
            else
                out.precision = std::numeric_limits<double>::quiet_NaN();
        }

        return summary;
    }

    void loglike(double* cube, int& ndim, int& npars, double& loglike_value, void* context) {
        (void)npars;

        auto&               ctx        = *static_cast<Context*>(context);
        const std::uint64_t evaluation = ++ctx.evaluations;

        try {
            if (ndim != 5)
                throw std::runtime_error("MultiNest ndim must be 5");

            std::array<Real_t, 5> q{};

            for (int i = 0; i < 5; ++i) {
                q[i]    = Q_MIN + (ctx.q_max - Q_MIN) * cube[i];
                cube[i] = q[i];
            }

            const auto earth      = nt::make_layered_constant_5(*ctx.prem, view(q));
            const auto propagated = ctx.solver->propagate(*ctx.initial, *ctx.prem, earth);
            const auto prediction = nt::predict_events(propagated, *ctx.response);

            double value = 0.0;

            for (Index_t z = 0; z < prediction.counts.extent(0); ++z) {
                for (Index_t e = 0; e < prediction.counts.extent(1); ++e) {
                    const double nexp = prediction.counts(z, e) * EXPOSURE_YEARS;
                    const double nobs = ctx.data->counts(z, e) * EXPOSURE_YEARS;

                    if (!std::isfinite(nexp) || !std::isfinite(nobs) || nexp < 0.0 || nobs < 0.0)
                        throw std::runtime_error("non-finite or negative event count in likelihood");

                    if (nobs > 1.0e-12) {
                        if (!(nexp > 0.0)) {
                            loglike_value = MN_LOG_ZERO;
                            return;
                        }

                        value += -nexp + nobs - nobs * std::log(nobs / nexp);
                    } else {
                        value += -nexp;
                    }
                }
            }

            if (!std::isfinite(value))
                throw std::runtime_error("non-finite total log-likelihood");

            loglike_value = value;

            if (ctx.console && LOG_EVERY_N_EVAL > 0 && evaluation % LOG_EVERY_N_EVAL == 0) {
                std::cout << "[LIKE][" << ctx.run_name << "] eval=" << evaluation << " logL=" << std::setprecision(12)
                          << loglike_value << " | q=";

                for (int i = 0; i < 5; ++i)
                    std::cout << ' ' << std::setprecision(7) << q[i];

                std::cout << '\n';
            }
        } catch (const std::exception& error) {
            ctx.fatal_error   = true;
            ctx.fatal_message = error.what();

            std::cerr << "[FATAL] likelihood: " << error.what() << '\n';

            loglike_value = std::numeric_limits<double>::max();
        }
    }

    void dumper(int& nSamples, int& nlive, int& nPar, double** physLive, double** posterior, double** paramConstr,
                double& maxLogLike, double& logZ, double& INSlogZ, double& logZerr, void* context) {
        (void)nSamples;
        (void)nlive;
        (void)nPar;
        (void)physLive;
        (void)posterior;
        (void)paramConstr;
        (void)maxLogLike;
        (void)INSlogZ;

        auto& ctx      = *static_cast<Context*>(context);
        ctx.has_dumper = true;
        ctx.logz       = logZ;
        ctx.logz_error = logZerr;
    }

    RunSummary run_multinest(const RunConfig& config, const fs::path& result_dir, const nt::EarthProfile& prem,
                             const nt::ResponseArray& response, const nt::Flux& initial,
                             const nt::EventDistribution& asimov_data, nt::EarthPropagator& solver,
                             const std::array<Real_t, 5>& prem_density, int rank) {
        const bool root_process = rank == 0;

        const fs::path run_dir        = result_dir / config.name;
        const fs::path raw_output_dir = run_dir / "raw_output";

        fs::create_directories(raw_output_dir);

        if (root_process) {
            std::cout << '\n';
            std::cout << "============================================================\n";
            std::cout << "Run       : " << config.name << '\n';
            std::cout << "nlive     : " << config.nlive << '\n';
            std::cout << "tol       : " << config.tol << '\n';
            std::cout << "prior     : [" << Q_MIN << ", " << config.q_max << "]\n";
            std::cout << "============================================================\n";
        }

        Context context;
        context.prem     = &prem;
        context.response = &response;
        context.initial  = &initial;
        context.data     = &asimov_data;
        context.solver   = &solver;
        context.q_max    = config.q_max;
        context.run_name = config.name;
        context.console  = root_process;

        int ndims   = 5;
        int nPar    = 5;
        int nClsPar = 5;

        int pWrap[5] = {0, 0, 0, 0, 0};

        std::string multinest_root = raw_output_dir.string();

        if (multinest_root.back() != '/')
            multinest_root.push_back('/');

        nested::run(MN_IS, MN_MMODAL, MN_CEFF, config.nlive, config.tol, MN_EFR, ndims, nPar, nClsPar, MN_MAX_MODES,
                    MN_UPD_INT, MN_ZTOL, multinest_root.c_str(), MN_SEED, pWrap, MN_FB, MN_RESUME, MN_OUTFILE,
                    MN_INIT_MPI, MN_LOG_ZERO, MN_MAXITER, loglike, dumper, &context);

        int local_fatal  = context.fatal_error ? 1 : 0;
        int global_fatal = 0;

        MPI_Allreduce(&local_fatal, &global_fatal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_fatal)
            throw std::runtime_error("MultiNest stopped because a likelihood evaluation failed");

        RunSummary summary;
        summary.config = config;

        int         summary_ok = 1;
        std::string summary_error;

        if (root_process) {
            try {
                if (!context.has_dumper)
                    throw std::runtime_error("MultiNest did not provide final evidence");

                summary.logz       = context.logz;
                summary.logz_error = context.logz_error;
                summary.parameters = summarize_posterior(raw_output_dir / "post_equal_weights.dat", prem_density);

                std::cout << std::setprecision(12);
                std::cout << "log Z      : " << summary.logz << " +/- " << summary.logz_error << '\n';
                std::cout << "evaluations: " << context.evaluations << '\n';
            } catch (const std::exception& error) {
                summary_ok    = 0;
                summary_error = error.what();
            }
        }

        MPI_Bcast(&summary_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!summary_ok) {
            if (root_process)
                throw std::runtime_error(summary_error);
            throw std::runtime_error("root process failed while summarizing posterior");
        }

        return summary;
    }

    void write_summary(const fs::path& filename, const std::vector<RunSummary>& summaries) {
        std::ofstream file(filename);

        if (!file)
            throw std::runtime_error("cannot open summary file: " + filename.string());

        file << "# Five-layer constant-density MultiNest parameter scan\n\n";
        file << "Baseline: `nlive = 1000`, `tol = 0.1`, `q_i ~ Uniform[0,4]`.\n\n";
        file << "All runs use the same PREM Asimov data, propagation grid, likelihood, MultiNest settings and seed "
                "except for the parameter explicitly varied.\n\n";
        file << "The central value is the posterior median. The central 68% interval is `[q16, q84]` and\n\n";
        file << "`precision = (q84 - q16) / (2 * q50)`.\n\n";

        file << "## Bayesian evidence\n\n";
        file << "| Run | nlive | tol | prior | ln Z | ln Z error | Delta ln Z vs baseline |\n";
        file << "|:---|---:|---:|:---:|---:|---:|---:|\n";

        const double baseline_logz = summaries.front().logz;

        file << std::fixed << std::setprecision(8);

        for (const auto& run : summaries) {
            file << "| " << run.config.name << " | " << run.config.nlive << " | " << run.config.tol << " | [0, "
                 << run.config.q_max << "] | " << run.logz << " | " << run.logz_error << " | "
                 << run.logz - baseline_logz << " |\n";
        }

        file << "\n";
        file << "## Posterior constraints\n\n";
        file << "| Run | Layer | q16 | q50 | q84 | rho16 [g/cm^3] | rho50 [g/cm^3] | rho84 [g/cm^3] | Precision |\n";
        file << "|:---|:---|---:|---:|---:|---:|---:|---:|---:|\n";

        for (const auto& run : summaries) {
            for (Index_t i = 0; i < 5; ++i) {
                const auto& p = run.parameters[i];

                file << "| " << run.config.name << " | " << LAYER_NAMES[i] << " | " << p.q16 << " | " << p.q50 << " | "
                     << p.q84 << " | " << p.rho16 << " | " << p.rho50 << " | " << p.rho84 << " | "
                     << 100.0 * p.precision << "% |\n";
            }
        }
    }

} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "[fatal] MPI_Init failed\n";
        return 1;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const bool root_process = rank == 0;

    int exit_code = 0;

    try {
        const fs::path daemonflux_file = "data/generated/daemonflux/daemonflux_0.8.2.h5";
        const fs::path result_dir      = "result/tomography_mnest_settings";

        if (!fs::is_regular_file(daemonflux_file))
            throw std::runtime_error("DaemonFlux table not found: " + daemonflux_file.string());

        fs::create_directories(result_dir);

        const auto response   = nt::load_trident_response();
        const auto prem       = nt::load_prem();
        const auto daemonflux = nt::load_daemonflux(daemonflux_file, "IceCube");

        // Keep exactly the same energy treatment as the baseline tomography case.
        const auto initial = nt::resample_flux(daemonflux, response.coszenith.view(), response.true_energy_gev.view());

        nt::PropagationOptions options;
        options.interactions = true;
        options.threads      = 1;

        nt::EarthPropagator solver(initial, options);

        // The same PREM Asimov data are used by every MultiNest run.
        const auto asimov_flux = solver.propagate(initial, prem);
        const auto asimov_data = nt::predict_events(asimov_flux, response);

        const auto prem_density = mean_layer_densities(prem);

        if (root_process) {
            std::cout << '\n';
            std::cout << "TRIDENT five-layer constant-density MultiNest settings scan\n";
            std::cout << "------------------------------------------------------------\n";
            std::cout << "Asimov truth     : full PREM\n";
            std::cout << "Fit model        : five-layer constant density\n";
            std::cout << "Exposure         : " << EXPOSURE_YEARS << " year\n";
            std::cout << "Interactions     : enabled\n";
            std::cout << "Energy treatment : response true-energy centers\n";
            std::cout << "Random seed      : " << MN_SEED << '\n';
            std::cout << "Asimov events    : " << std::fixed << std::setprecision(6)
                      << total_events(asimov_data) * EXPOSURE_YEARS << '\n';
        }

        std::vector<RunSummary> summaries;

        if (root_process)
            summaries.reserve(RUNS.size());

        for (const auto& config : RUNS) {
            const auto summary =
                run_multinest(config, result_dir, prem, response, initial, asimov_data, solver, prem_density, rank);

            if (root_process)
                summaries.push_back(summary);
        }

        if (root_process) {
            const fs::path summary_file = result_dir / "summary.md";

            write_summary(summary_file, summaries);

            std::cout << '\n';
            std::cout << "============================================================\n";
            std::cout << "All runs finished\n";
            std::cout << "Summary: " << summary_file << '\n';
            std::cout << "============================================================\n";
        }
    } catch (const std::exception& error) {
        if (root_process)
            std::cerr << "[fatal] " << error.what() << '\n';

        exit_code = 1;
    }

    int global_exit_code = 0;
    MPI_Allreduce(&exit_code, &global_exit_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Finalize();

    return global_exit_code;
}
