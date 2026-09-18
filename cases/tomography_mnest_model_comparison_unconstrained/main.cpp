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
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

    using nt::Index_t;
    using nt::Real_t;

    namespace nda = nt::nda;

    constexpr int RUN_SCHEMA_VERSION = 1;

    constexpr Real_t Q_MIN = 0.0;
    constexpr Real_t Q_MAX = 4.0;

    constexpr int    MN_IS        = 0;
    constexpr int    MN_MMODAL    = 1;
    constexpr int    MN_CEFF      = 0;
    constexpr int    MN_NLIVE     = 2000;
    constexpr double MN_TOL       = 0.1;
    constexpr double MN_EFR       = 0.3;
    constexpr int    MN_MAX_MODES = 100;
    constexpr int    MN_UPD_INT   = 100;
    constexpr double MN_ZTOL      = -1.0e90;
    constexpr int    MN_SEED      = -1;
    constexpr int    MN_FB        = 1;
    constexpr int    MN_RESUME    = 1;
    constexpr int    MN_OUTFILE   = 1;
    constexpr int    MN_INIT_MPI  = 0;
    constexpr double MN_LOG_ZERO  = -1.0e90;
    constexpr int    MN_MAXITER   = 0;

    constexpr std::uint64_t LOG_EVERY_N_EVAL = 500;

    constexpr Index_t COSZENITH_SAMPLES_PER_BIN = 1;
    constexpr Index_t ENERGY_SAMPLES_PER_BIN    = 5;

    // Keep the current project baseline exactly: one angular center trajectory
    // per response bin and five log-energy trajectories per energy bin, followed
    // by linear interpolation back to the response energy centers.
    constexpr nt::FluxRebinOptions REBIN_OPTIONS{false, true};

    constexpr const char* DAEMONFLUX_FILE = "data/generated/daemonflux/daemonflux_0.8.2.h5";
    constexpr const char* PREM_FILE       = "data/PREM/EARTH_MODEL_PREM.dat";
    constexpr const char* RESPONSE_DIR    = "data/trident";

    const fs::path RESULT_ROOT = "result/tomography_mnest_model_comparison_unconstrained";

    enum class DensityProfile {
        constant,
        prem_scaled,
    };

    struct RunConfig {
        Index_t        layers;
        DensityProfile profile;
        int            exposure_years;
    };

    // Keep the old model-comparison ordering after removing the constraint axis:
    // layer -> profile -> exposure.
    constexpr std::array<RunConfig, 8> RUNS = {
        {
         {5, DensityProfile::constant, 1},
         {5, DensityProfile::constant, 10},
         {5, DensityProfile::prem_scaled, 1},
         {5, DensityProfile::prem_scaled, 10},
         {3, DensityProfile::constant, 1},
         {3, DensityProfile::constant, 10},
         {3, DensityProfile::prem_scaled, 1},
         {3, DensityProfile::prem_scaled, 10},
         }
    };

    constexpr std::array<Real_t, 6> LAYER_EDGES_5_KM = {
        0.0, 1221.0, 3480.0, 4811.0, 5700.0, 6371.0,
    };

    constexpr std::array<Real_t, 4> LAYER_EDGES_3_KM = {
        0.0,
        3480.0,
        5700.0,
        6371.0,
    };

    constexpr std::array<const char*, 5> LAYER_NAMES_5 = {
        "Inner core", "Outer core", "Lower mantle", "Middle mantle", "Upper mantle",
    };

    constexpr std::array<const char*, 3> LAYER_NAMES_3 = {
        "Core",
        "Lower mantle",
        "Upper mantle",
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

    const char* profile_name(DensityProfile profile) noexcept {
        return profile == DensityProfile::constant ? "constant" : "prem_scaled";
    }

    fs::path run_directory(const RunConfig& config) {
        return RESULT_ROOT / (std::to_string(config.layers) + "layer") / profile_name(config.profile) /
               (std::to_string(config.exposure_years) + "year");
    }

    std::string build_run_info(const RunConfig& config) {
        std::ostringstream out;
        out << std::setprecision(17);
        out << "run_schema_version=" << RUN_SCHEMA_VERSION << '\n';
        out << "layers=" << config.layers << '\n';
        out << "profile=" << profile_name(config.profile) << '\n';
        out << "exposure_years=" << config.exposure_years << '\n';
        out << "asimov_truth=full_prem\n";
        out << "daemonflux_file=" << DAEMONFLUX_FILE << '\n';
        out << "prem_file=" << PREM_FILE << '\n';
        out << "response_dir=" << RESPONSE_DIR << '\n';
        out << "q_min=" << Q_MIN << '\n';
        out << "q_max=" << Q_MAX << '\n';
        out << "coszenith_samples_per_bin=" << COSZENITH_SAMPLES_PER_BIN << '\n';
        out << "energy_samples_per_bin=" << ENERGY_SAMPLES_PER_BIN << '\n';
        out << "interpolate_coszenith=" << REBIN_OPTIONS.interpolate_coszenith << '\n';
        out << "interpolate_energy=" << REBIN_OPTIONS.interpolate_energy << '\n';
        out << "propagation_interactions=1\n";
        out << "propagation_threads=1\n";
        out << "mn_is=" << MN_IS << '\n';
        out << "mn_mmodal=" << MN_MMODAL << '\n';
        out << "mn_ceff=" << MN_CEFF << '\n';
        out << "mn_nlive=" << MN_NLIVE << '\n';
        out << "mn_tol=" << MN_TOL << '\n';
        out << "mn_efr=" << MN_EFR << '\n';
        out << "mn_max_modes=" << MN_MAX_MODES << '\n';
        out << "mn_upd_int=" << MN_UPD_INT << '\n';
        out << "mn_ztol=" << MN_ZTOL << '\n';
        out << "mn_seed=" << MN_SEED << '\n';
        out << "mn_fb=" << MN_FB << '\n';
        out << "mn_resume=" << MN_RESUME << '\n';
        out << "mn_outfile=" << MN_OUTFILE << '\n';
        out << "mn_init_mpi=" << MN_INIT_MPI << '\n';
        out << "mn_log_zero=" << MN_LOG_ZERO << '\n';
        out << "mn_maxiter=" << MN_MAXITER << '\n';
        return out.str();
    }

    std::string read_text_file(const fs::path& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
            throw std::runtime_error("Cannot read " + filename.string());

        std::ostringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    void ensure_run_info(const RunConfig& config) {
        const fs::path dir       = run_directory(config);
        const fs::path raw_dir   = dir / "raw_output";
        const fs::path info_file = dir / "run.info";
        const fs::path completed = dir / ".completed";

        const std::string expected = build_run_info(config);

        if (fs::exists(info_file)) {
            if (read_text_file(info_file) != expected)
                throw std::runtime_error("Existing run.info does not match the current configuration: " +
                                         info_file.string() + "\nRefusing to overwrite or resume this directory.");
            fs::create_directories(raw_dir);
            return;
        }

        if (fs::exists(completed))
            throw std::runtime_error("Found .completed without run.info: " + dir.string());

        if (fs::exists(raw_dir) && !fs::is_empty(raw_dir))
            throw std::runtime_error("Found non-empty raw_output without run.info: " + raw_dir.string());

        fs::create_directories(raw_dir);

        std::ofstream file(info_file, std::ios::trunc);
        if (!file)
            throw std::runtime_error("Cannot write " + info_file.string());
        file << expected;
    }

    void validate_run_info(const RunConfig& config) {
        const fs::path dir       = run_directory(config);
        const fs::path raw_dir   = dir / "raw_output";
        const fs::path info_file = dir / "run.info";

        if (!fs::is_regular_file(info_file))
            throw std::runtime_error("run.info is missing for " + dir.string() +
                                     ". Run --prepare before launching MPI tasks.");

        if (read_text_file(info_file) != build_run_info(config))
            throw std::runtime_error("Existing run.info does not match the current configuration: " +
                                     info_file.string() + "\nRefusing to run or resume this directory.");

        if (!fs::is_directory(raw_dir))
            throw std::runtime_error("Prepared raw_output directory is missing: " + raw_dir.string());
    }

    bool is_completed(const RunConfig& config) { return fs::is_regular_file(run_directory(config) / ".completed"); }

    void check_input_files() {
        for (const fs::path file : {
                 fs::path(DAEMONFLUX_FILE),
                 fs::path(PREM_FILE),
                 fs::path(RESPONSE_DIR) / "TRIDENT_response_array_20x34.csv",
                 fs::path(RESPONSE_DIR) / "energy_response_20x20_v2.csv",
             }) {
            if (!fs::is_regular_file(file))
                throw std::runtime_error("Required input file not found: " + file.string());
        }
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

    Real_t prem_layer_mean_density(const nt::EarthProfile& prem, const RunConfig& config, Index_t layer) {
        if (config.layers == 5)
            return nt::mean_density_g_cm3(prem, LAYER_EDGES_5_KM[layer], LAYER_EDGES_5_KM[layer + 1]);

        return nt::mean_density_g_cm3(prem, LAYER_EDGES_3_KM[layer], LAYER_EDGES_3_KM[layer + 1]);
    }

    const char* layer_name(const RunConfig& config, Index_t layer) noexcept {
        return config.layers == 5 ? LAYER_NAMES_5[layer] : LAYER_NAMES_3[layer];
    }

    std::array<ParameterSummary, 5> summarize_posterior(const fs::path& filename, const RunConfig& config,
                                                        const nt::EarthProfile& prem) {
        std::ifstream file(filename);

        if (!file)
            throw std::runtime_error("cannot open posterior file: " + filename.string());

        std::array<std::vector<double>, 5> samples;
        std::string                        line;

        while (std::getline(file, line)) {
            if (line.empty())
                continue;

            std::istringstream row(line);

            for (Index_t i = 0; i < config.layers; ++i) {
                double q = 0.0;
                if (!(row >> q))
                    throw std::runtime_error("invalid posterior row in: " + filename.string());
                samples[i].push_back(q);
            }

            double loglike = 0.0;
            if (!(row >> loglike))
                throw std::runtime_error("posterior row is missing log-likelihood in: " + filename.string());
        }

        std::array<ParameterSummary, 5> summary{};

        for (Index_t i = 0; i < config.layers; ++i) {
            auto& values = samples[i];
            std::sort(values.begin(), values.end());

            auto& out = summary[i];
            out.q16   = percentile(values, 0.16);
            out.q50   = percentile(values, 0.50);
            out.q84   = percentile(values, 0.84);

            const double prem_density = prem_layer_mean_density(prem, config, i);
            out.rho16                 = out.q16 * prem_density;
            out.rho50                 = out.q50 * prem_density;
            out.rho84                 = out.q84 * prem_density;

            if (out.q50 > 0.0)
                out.precision = (out.q84 - out.q16) / (2.0 * out.q50);
            else
                out.precision = std::numeric_limits<double>::quiet_NaN();
        }

        return summary;
    }

    void write_run_summary(const fs::path& filename, const RunSummary& summary) {
        std::ofstream file(filename, std::ios::trunc);

        if (!file)
            throw std::runtime_error("cannot write run summary: " + filename.string());

        file << std::setprecision(17);
        file << summary.logz << ' ' << summary.logz_error << '\n';

        for (Index_t i = 0; i < summary.config.layers; ++i) {
            const auto& p = summary.parameters[i];
            file << p.q16 << ' ' << p.q50 << ' ' << p.q84 << ' ' << p.rho16 << ' ' << p.rho50 << ' ' << p.rho84 << ' '
                 << p.precision << '\n';
        }
    }

    RunSummary read_run_summary(const RunConfig& config) {
        const fs::path filename = run_directory(config) / "summary.dat";
        std::ifstream  file(filename);

        if (!file)
            throw std::runtime_error("cannot open run summary: " + filename.string());

        RunSummary summary;
        summary.config = config;

        if (!(file >> summary.logz >> summary.logz_error))
            throw std::runtime_error("invalid run summary header: " + filename.string());

        for (Index_t i = 0; i < config.layers; ++i) {
            auto& p = summary.parameters[i];
            if (!(file >> p.q16 >> p.q50 >> p.q84 >> p.rho16 >> p.rho50 >> p.rho84 >> p.precision))
                throw std::runtime_error("invalid run summary data: " + filename.string());
        }

        return summary;
    }

    bool all_runs_completed() {
        for (const auto& config : RUNS) {
            if (!is_completed(config))
                return false;
        }
        return true;
    }

    void write_markdown_summary(const fs::path& filename, const std::array<RunSummary, RUNS.size()>& summaries) {
        std::ofstream file(filename, std::ios::trunc);

        if (!file)
            throw std::runtime_error("cannot write markdown summary: " + filename.string());

        file << "# Unconstrained MultiNest model comparison\n\n";
        file << "Baseline: `nlive = " << MN_NLIVE << "`, `tol = " << MN_TOL << "`, `q_i ~ Uniform[0,4]`.\n\n";
        file << "All runs use the same full-PREM Asimov construction, propagation grid, likelihood and MultiNest "
                "settings. The varied dimensions are layer count, density parameterization and exposure.\n\n";
        file << "Propagation uses one cosZenith-center trajectory per response bin and five log-energy trajectories "
                "per true-energy bin, followed by linear interpolation to the response energy centers.\n\n";
        file << "The central value is the posterior median. The central 68% interval is `[q16, q84]` and\n\n";
        file << "`precision = (q84 - q16) / (2 * q50)`.\n\n";
        file << "`rho` is the volume-weighted mean density of the corresponding layer. For both parameterizations, "
                "its posterior summary is `q_i` times the PREM mean density of that layer.\n\n";

        file << "## Bayesian evidence\n\n";
        file << "| Layers | Density parameterization | Exposure | ln Z | ln Z error |\n";
        file << "|---:|:---|---:|---:|---:|\n";

        file << std::fixed << std::setprecision(8);

        for (const auto& run : summaries) {
            file << "| " << run.config.layers << " | "
                 << (run.config.profile == DensityProfile::constant ? "Layered constant-density" : "PREM-scaled")
                 << " | " << run.config.exposure_years << " year" << (run.config.exposure_years == 1 ? "" : "s")
                 << " | " << run.logz << " | " << run.logz_error << " |\n";
        }

        file << "\n## Posterior constraints\n\n";
        file << "| Layers | Density parameterization | Exposure | Layer | q16 | q50 | q84 | rho16 [g/cm^3] | "
                "rho50 [g/cm^3] | rho84 [g/cm^3] | Precision |\n";
        file << "|---:|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|\n";

        for (const auto& run : summaries) {
            const char* profile =
                run.config.profile == DensityProfile::constant ? "Layered constant-density" : "PREM-scaled";

            for (Index_t i = 0; i < run.config.layers; ++i) {
                const auto& p = run.parameters[i];
                file << "| " << run.config.layers << " | " << profile << " | " << run.config.exposure_years << " year"
                     << (run.config.exposure_years == 1 ? "" : "s") << " | " << layer_name(run.config, i) << " | "
                     << p.q16 << " | " << p.q50 << " | " << p.q84 << " | " << p.rho16 << " | " << p.rho50 << " | "
                     << p.rho84 << " | " << 100.0 * p.precision << "% |\n";
            }
        }
    }

    int summarize_all() {
        if (!all_runs_completed())
            throw std::runtime_error("cannot summarize: not all eight runs are complete");

        std::array<RunSummary, RUNS.size()> summaries{};

        for (std::size_t i = 0; i < RUNS.size(); ++i)
            summaries[i] = read_run_summary(RUNS[i]);

        const fs::path summary_file = RESULT_ROOT / "summary.md";
        write_markdown_summary(summary_file, summaries);

        std::cout << "Summary: " << summary_file << '\n';
        return 0;
    }

    struct Context {
        const RunConfig*             config       = nullptr;
        const nt::EarthProfile*      prem         = nullptr;
        const nt::ResponseArray*     response     = nullptr;
        const nt::Flux*              initial_fine = nullptr;
        const nt::EventDistribution* data         = nullptr;
        nt::EarthPropagator*         solver       = nullptr;

        std::uint64_t evaluations = 0;
        bool          console     = false;
        bool          fatal_error = false;
        std::string   fatal_message;

        bool   has_dumper  = false;
        double max_loglike = std::numeric_limits<double>::quiet_NaN();
        double logz        = std::numeric_limits<double>::quiet_NaN();
        double ins_logz    = std::numeric_limits<double>::quiet_NaN();
        double logz_error  = std::numeric_limits<double>::quiet_NaN();
    };

    void loglike(double* cube, int& ndim, int& npars, double& loglike_value, void* context) {
        (void)npars;

        auto&               ctx        = *static_cast<Context*>(context);
        const std::uint64_t evaluation = ++ctx.evaluations;

        try {
            if (ndim != static_cast<int>(ctx.config->layers))
                throw std::runtime_error("MultiNest dimension does not match the configured Earth model");

            std::array<Real_t, 5> q{};

            for (int i = 0; i < ndim; ++i) {
                q[i]    = Q_MIN + (Q_MAX - Q_MIN) * cube[i];
                cube[i] = q[i];
            }

            const auto q_view =
                nda::make_view1d(static_cast<const Real_t*>(q.data()), static_cast<std::size_t>(ctx.config->layers));

            const auto propagated_fine = [&]() -> nt::Flux {
                if (ctx.config->profile == DensityProfile::constant) {
                    if (ctx.config->layers == 5) {
                        const auto earth = nt::make_layered_constant_5(*ctx.prem, q_view);
                        return ctx.solver->propagate(*ctx.initial_fine, *ctx.prem, earth);
                    }

                    const auto earth = nt::make_layered_constant_3(*ctx.prem, q_view);
                    return ctx.solver->propagate(*ctx.initial_fine, *ctx.prem, earth);
                }

                if (ctx.config->layers == 5) {
                    const auto earth = nt::make_prem_scaled_5(q_view);
                    return ctx.solver->propagate(*ctx.initial_fine, *ctx.prem, earth);
                }

                const auto earth = nt::make_prem_scaled_3(q_view);
                return ctx.solver->propagate(*ctx.initial_fine, *ctx.prem, earth);
            }();

            const auto propagated =
                nt::rebin_flux(propagated_fine, ctx.response->coszenith.view(), ctx.response->true_energy_gev.view(),
                               COSZENITH_SAMPLES_PER_BIN, ENERGY_SAMPLES_PER_BIN, REBIN_OPTIONS);
            const auto prediction = nt::predict_events(propagated, *ctx.response);

            double value = 0.0;

            for (Index_t z = 0; z < prediction.counts.extent(0); ++z) {
                for (Index_t e = 0; e < prediction.counts.extent(1); ++e) {
                    const double nexp = prediction.counts(z, e) * static_cast<double>(ctx.config->exposure_years);
                    const double nobs = ctx.data->counts(z, e) * static_cast<double>(ctx.config->exposure_years);

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
                std::cout << "[LIKE] eval=" << evaluation << " logL=" << std::setprecision(12) << loglike_value
                          << " | q=";

                for (Index_t i = 0; i < ctx.config->layers; ++i)
                    std::cout << ' ' << std::setprecision(7) << q[i];

                std::cout << '\n';
            }
        } catch (const std::exception& error) {
            ctx.fatal_error   = true;
            ctx.fatal_message = error.what();

            std::cerr << "[FATAL] likelihood: " << error.what() << '\n';

            // MultiNest treats HUGE(1d0) as an exit signal in the likelihood callback.
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

        auto& ctx       = *static_cast<Context*>(context);
        ctx.has_dumper  = true;
        ctx.max_loglike = maxLogLike;
        ctx.logz        = logZ;
        ctx.ins_logz    = INSlogZ;
        ctx.logz_error  = logZerr;
    }

    void write_completed_file(const RunConfig& config, const Context& context, std::uint64_t total_evaluations,
                              double elapsed_seconds) {
        const fs::path filename = run_directory(config) / ".completed";
        std::ofstream  file(filename, std::ios::trunc);

        if (!file)
            throw std::runtime_error("Cannot write completion marker: " + filename.string());

        file << std::setprecision(17);
        file << "status=completed\n";
        file << "root_evaluations=" << context.evaluations << '\n';
        file << "total_evaluations=" << total_evaluations << '\n';
        file << "elapsed_seconds=" << elapsed_seconds << '\n';
        file << "max_loglike=" << context.max_loglike << '\n';
        file << "logZ=" << context.logz << '\n';
        file << "INS_logZ=" << context.ins_logz << '\n';
        file << "logZ_error=" << context.logz_error << '\n';
    }

    void plot_run(const RunConfig& config) {
        const std::string python  = "runtime/python/venv/bin/python";
        const std::string script  = "cases/tomography_mnest_model_comparison_unconstrained/plot_results.py";
        const std::string command = python + " " + script + " --run-dir " + run_directory(config).string();

        std::cout << "[PLOT] " << run_directory(config) << '\n';

        const int status = std::system(command.c_str());
        if (status != 0)
            std::cerr << "[WARN] plotting failed for " << run_directory(config) << " (status=" << status << ")\n";
    }

    int run_atomic(const RunConfig& config, std::size_t run_index, int rank) {
        const bool root_process = rank == 0;

        validate_run_info(config);

        if (is_completed(config)) {
            if (root_process) {
                std::cout << "[SKIP] " << run_directory(config) << " is already complete\n";
                plot_run(config);
            }
            return 0;
        }

        check_input_files();

        const auto response   = nt::load_trident_response();
        const auto prem       = nt::load_prem();
        const auto daemonflux = nt::load_daemonflux(DAEMONFLUX_FILE, "IceCube");

        const auto fine_coszenith =
            nt::sample_coszenith_bin_midpoints(response.coszenith_edges.view(), COSZENITH_SAMPLES_PER_BIN);
        const auto fine_energy =
            nt::sample_log_energy_bin_midpoints(response.true_energy_edges_gev.view(), ENERGY_SAMPLES_PER_BIN);
        const auto initial_fine = nt::resample_flux(daemonflux, fine_coszenith.view(), fine_energy.view());

        nt::PropagationOptions options;
        options.interactions = true;
        options.threads      = 1;

        nt::EarthPropagator solver(initial_fine, options);

        // Every atomic run independently regenerates the same full-PREM Asimov data
        // on the same fine grid used by all likelihood evaluations.
        const auto asimov_fine = solver.propagate(initial_fine, prem);
        const auto asimov_flux = nt::rebin_flux(asimov_fine, response.coszenith.view(), response.true_energy_gev.view(),
                                                COSZENITH_SAMPLES_PER_BIN, ENERGY_SAMPLES_PER_BIN, REBIN_OPTIONS);
        const auto asimov_data = nt::predict_events(asimov_flux, response);

        if (root_process) {
            Real_t asimov_events = 0.0;
            for (Index_t z = 0; z < asimov_data.counts.extent(0); ++z)
                for (Index_t e = 0; e < asimov_data.counts.extent(1); ++e)
                    asimov_events += asimov_data.counts(z, e);

            std::cout << '\n';
            std::cout << "============================================================\n";
            std::cout << "[RUN " << run_index + 1 << '/' << RUNS.size() << "]\n";
            std::cout << "Layers           : " << config.layers << '\n';
            std::cout << "Profile          : " << profile_name(config.profile) << '\n';
            std::cout << "Exposure         : " << config.exposure_years << " year"
                      << (config.exposure_years == 1 ? "" : "s") << '\n';
            std::cout << "q prior          : Uniform[" << Q_MIN << ", " << Q_MAX << "]\n";
            std::cout << "Asimov truth     : full PREM\n";
            std::cout << "Interactions     : enabled\n";
            std::cout << "nuSQuIDS threads : 1\n";
            std::cout << "cosZenith samples: 1 center/bin\n";
            std::cout << "Energy samples   : 5 log-midpoints/bin\n";
            std::cout << "Energy rebin     : linear interpolation\n";
            std::cout << "Asimov events    : " << std::fixed << std::setprecision(6)
                      << asimov_events * static_cast<Real_t>(config.exposure_years) << '\n';
            std::cout << "MultiNest        : nlive=" << MN_NLIVE << " tol=" << MN_TOL << " efr=" << MN_EFR << '\n';
            std::cout << "Resume           : enabled\n";
            std::cout << "Output           : " << run_directory(config) << '\n';
            std::cout << "============================================================\n";
        }

        Context context;
        context.config       = &config;
        context.prem         = &prem;
        context.response     = &response;
        context.initial_fine = &initial_fine;
        context.data         = &asimov_data;
        context.solver       = &solver;
        context.console      = root_process;

        int ndims   = static_cast<int>(config.layers);
        int nPar    = ndims;
        int nClsPar = ndims;

        std::array<int, 5> pWrap{};

        std::string multinest_root = (run_directory(config) / "raw_output").string();
        if (multinest_root.back() != '/')
            multinest_root.push_back('/');

        const auto start = std::chrono::steady_clock::now();

        nested::run(MN_IS, MN_MMODAL, MN_CEFF, MN_NLIVE, MN_TOL, MN_EFR, ndims, nPar, nClsPar, MN_MAX_MODES, MN_UPD_INT,
                    MN_ZTOL, multinest_root.c_str(), MN_SEED, pWrap.data(), MN_FB, MN_RESUME, MN_OUTFILE, MN_INIT_MPI,
                    MN_LOG_ZERO, MN_MAXITER, loglike, dumper, &context);

        const double local_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        int local_fatal  = context.fatal_error ? 1 : 0;
        int global_fatal = 0;
        MPI_Allreduce(&local_fatal, &global_fatal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        if (global_fatal)
            throw std::runtime_error("MultiNest stopped because a likelihood evaluation failed");

        const unsigned long long local_evaluations = static_cast<unsigned long long>(context.evaluations);
        unsigned long long       total_evaluations = 0;
        double                   elapsed_seconds   = 0.0;

        MPI_Reduce(&local_evaluations, &total_evaluations, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_elapsed, &elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        int         completion_ok = 1;
        std::string completion_error;

        if (root_process) {
            try {
                if (!context.has_dumper || !std::isfinite(context.logz) || !std::isfinite(context.logz_error) ||
                    !std::isfinite(context.max_loglike))
                    throw std::runtime_error("MultiNest returned without a valid final evidence state");

                RunSummary summary;
                summary.config     = config;
                summary.logz       = context.logz;
                summary.logz_error = context.logz_error;
                summary.parameters =
                    summarize_posterior(run_directory(config) / "raw_output" / "post_equal_weights.dat", config, prem);

                const fs::path summary_file = run_directory(config) / "summary.dat";
                write_run_summary(summary_file, summary);

                // Only mark the atomic run complete after its posterior summary has
                // also been generated successfully.
                write_completed_file(config, context, static_cast<std::uint64_t>(total_evaluations), elapsed_seconds);

                std::cout << std::setprecision(12);
                std::cout << "[DONE] wall time       : " << elapsed_seconds << " s\n";
                std::cout << "       total evals     : " << total_evaluations << '\n';
                std::cout << "       log Z           : " << context.logz << " +/- " << context.logz_error << '\n';
                std::cout << "       INS log Z       : " << context.ins_logz << '\n';
                std::cout << "       max log L       : " << context.max_loglike << '\n';
                std::cout << "       run summary     : " << summary_file << '\n';
                std::cout << "       completion file : " << run_directory(config) / ".completed" << '\n';

                plot_run(config);

                if (all_runs_completed()) {
                    try {
                        summarize_all();
                    } catch (const std::exception& error) {
                        std::cerr << "[WARN] failed to write aggregate summary: " << error.what() << '\n';
                        std::cerr << "       rerun with --summarize after fixing the output files\n";
                    }
                }
            } catch (const std::exception& error) {
                completion_ok    = 0;
                completion_error = error.what();
            }
        }

        MPI_Bcast(&completion_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!completion_ok) {
            if (root_process)
                throw std::runtime_error(completion_error);
            throw std::runtime_error("root process failed while finalizing the run");
        }

        return 0;
    }

    int prepare() {
        check_input_files();
        fs::create_directories(RESULT_ROOT);

        std::cout << "TRIDENT MultiNest model comparison\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << "Atomic runs       : " << RUNS.size() << '\n';
        std::cout << "q prior           : Uniform[" << Q_MIN << ", " << Q_MAX << "]\n";
        std::cout << "MultiNest         : nlive=" << MN_NLIVE << " tol=" << MN_TOL << " efr=" << MN_EFR << '\n';
        std::cout << "cosZenith samples : 1 center/bin\n";
        std::cout << "Energy samples    : 5 log-midpoints/bin\n";
        std::cout << "Energy rebin      : linear interpolation\n";
        std::cout << "Constraints       : none\n";
        std::cout << '\n';

        for (std::size_t i = 0; i < RUNS.size(); ++i) {
            ensure_run_info(RUNS[i]);
            std::cout << "  " << i << " : " << RUNS[i].layers << "layer / " << profile_name(RUNS[i].profile) << " / "
                      << RUNS[i].exposure_years << "year";
            if (is_completed(RUNS[i]))
                std::cout << " [completed]";
            std::cout << '\n';
        }

        std::cout << "\nPrepared/validated " << RUNS.size() << " run directories.\n";
        return 0;
    }

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--prepare") {
        try {
            return prepare();
        } catch (const std::exception& error) {
            std::cerr << "[fatal] " << error.what() << '\n';
            return 1;
        }
    }

    if (argc == 2 && std::string(argv[1]) == "--summarize") {
        try {
            return summarize_all();
        } catch (const std::exception& error) {
            std::cerr << "[fatal] " << error.what() << '\n';
            return 1;
        }
    }

    if (argc != 3 || std::string(argv[1]) != "--run") {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " --prepare\n";
        std::cerr << "  " << argv[0] << " --run <index>\n";
        std::cerr << "  " << argv[0] << " --summarize\n";
        return 1;
    }

    std::size_t run_index = 0;

    try {
        const std::string text  = argv[2];
        std::size_t       pos   = 0;
        const auto        value = std::stoull(text, &pos);

        if (pos != text.size() || value >= RUNS.size())
            throw std::runtime_error("run index out of range");

        run_index = static_cast<std::size_t>(value);
    } catch (const std::exception&) {
        std::cerr << "[fatal] invalid run index: " << argv[2] << '\n';
        return 1;
    }

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "[fatal] MPI_Init failed\n";
        return 1;
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const bool root_process = rank == 0;
    int        exit_code    = 0;

    try {
        run_atomic(RUNS[run_index], run_index, rank);
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
