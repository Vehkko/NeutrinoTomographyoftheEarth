#include <multinest.h>

#include <nt/earth.hpp>
#include <nt/events.hpp>
#include <nt/flux.hpp>
#include <nt/response.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

    using nt::EventDistribution;
    using nt::Index_t;
    using nt::Real_t;

    namespace nda = nt::nda;

    constexpr Real_t Q_MIN = 0.0;
    constexpr Real_t Q_MAX = 4.0;

    constexpr Real_t EXPOSURE_YEARS = 1.0;

    constexpr int    MN_IS        = 0;
    constexpr int    MN_MMODAL    = 1;
    constexpr int    MN_CEFF      = 0;
    constexpr int    MN_NLIVE     = 1000;
    constexpr double MN_TOL       = 0.2;
    constexpr double MN_EFR       = 0.3;
    constexpr int    MN_MAX_MODES = 100;
    constexpr int    MN_UPD_INT   = 100;
    constexpr double MN_ZTOL      = -1.0e90;
    constexpr int    MN_SEED      = -1;
    constexpr int    MN_FB        = 1;
    constexpr int    MN_RESUME    = 0;
    constexpr int    MN_OUTFILE   = 1;
    constexpr int    MN_INIT_MPI  = 1;
    constexpr double MN_LOG_ZERO  = -1.0e90;
    constexpr int    MN_MAXITER   = 0;

    constexpr std::uint64_t LOG_EVERY_N_EVAL = 500;

    template <std::size_t N> auto view(const std::array<Real_t, N>& values) {
        return nda::make_view1d(static_cast<const Real_t*>(values.data()), values.size());
    }

    int mpi_rank_from_environment() {
        for (const char* name : {"PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK"}) {
            if (const char* value = std::getenv(name))
                return std::atoi(value);
        }

        return 0;
    }

    Real_t total_events(const EventDistribution& events) {
        Real_t total = 0.0;

        for (Index_t z = 0; z < events.counts.extent(0); ++z) {
            for (Index_t e = 0; e < events.counts.extent(1); ++e)
                total += events.counts(z, e);
        }

        return total;
    }

    struct Context {
        const nt::EarthProfile*      prem     = nullptr;
        const nt::ResponseArray*     response = nullptr;
        const nt::Flux*              initial  = nullptr;
        const nt::EventDistribution* data     = nullptr;

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
            if (ndim != 5)
                throw std::runtime_error("MultiNest ndim must be 5");

            std::array<Real_t, 5> q{};

            for (int i = 0; i < 5; ++i) {
                q[i]    = Q_MIN + (Q_MAX - Q_MIN) * cube[i];
                cube[i] = q[i];
            }

            const auto earth = nt::make_layered_constant_5(*ctx.prem, view(q));

            nt::PropagationOptions options;
            options.interactions = true;
            options.threads      = 1;

            const auto propagated = nt::propagate_flux(*ctx.initial, *ctx.prem, earth, options);
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
                std::cout << "[LIKE] eval=" << evaluation << " logL=" << std::setprecision(12) << loglike_value
                          << " | q=";

                for (int i = 0; i < 5; ++i)
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

} // namespace

int main() {
    try {
        const int  rank         = mpi_rank_from_environment();
        const bool root_process = rank == 0;

        const fs::path daemonflux_file = "data/generated/daemonflux/daemonflux_0.8.2.h5";
        const fs::path result_dir      = "result/tomography_mnest";
        const fs::path raw_output_dir  = result_dir / "raw_output";

        if (!fs::is_regular_file(daemonflux_file))
            throw std::runtime_error("DaemonFlux table not found: " + daemonflux_file.string());

        fs::create_directories(raw_output_dir);

        const auto response   = nt::load_trident_response();
        const auto prem       = nt::load_prem();
        const auto daemonflux = nt::load_daemonflux(daemonflux_file, "IceCube");

        const auto initial = nt::resample_flux(daemonflux, response.coszenith.view(), response.true_energy_gev.view());

        nt::PropagationOptions options;
        options.interactions = true;
        options.threads      = 1;

        // Asimov data are generated from the full PREM profile.
        const auto asimov_flux = nt::propagate_flux(initial, prem, options);
        const auto asimov_data = nt::predict_events(asimov_flux, response);

        if (root_process) {
            std::cout << '\n';
            std::cout << "TRIDENT five-layer constant-density MultiNest scan\n";
            std::cout << "------------------------------------------------------------\n";
            std::cout << "Asimov truth     : full PREM\n";
            std::cout << "Fit model        : five-layer constant density\n";
            std::cout << "q prior          : Uniform[" << Q_MIN << ", " << Q_MAX << "]\n";
            std::cout << "Exposure         : " << EXPOSURE_YEARS << " year\n";
            std::cout << "Interactions     : enabled\n";
            std::cout << "cosZenith bins   : " << response.coszenith.extent(0) << '\n';
            std::cout << "true-energy bins : " << response.true_energy_gev.extent(0) << '\n';
            std::cout << "Asimov events    : " << std::fixed << std::setprecision(6)
                      << total_events(asimov_data) * EXPOSURE_YEARS << '\n';

            std::cout << '\n';
            std::cout << "PREM volume-weighted layer densities\n";

            constexpr std::array<Real_t, 6> edges = {
                0.0, 1221.0, 3480.0, 4811.0, 5700.0, 6371.0,
            };

            for (Index_t i = 0; i < 5; ++i) {
                std::cout << "  q" << i + 1 << " : " << std::setprecision(8)
                          << nt::mean_density_g_cm3(prem, edges[i], edges[i + 1]) << " g/cm^3\n";
            }

            std::cout << '\n';
            std::cout << "MultiNest\n";
            std::cout << "  nlive = " << MN_NLIVE << '\n';
            std::cout << "  tol   = " << MN_TOL << '\n';
            std::cout << "  efr   = " << MN_EFR << '\n';
            std::cout << "  output= " << raw_output_dir << '\n';
            std::cout << '\n';
        }

        Context context;
        context.prem     = &prem;
        context.response = &response;
        context.initial  = &initial;
        context.data     = &asimov_data;
        context.console  = root_process;

        int ndims   = 5;
        int nPar    = 5;
        int nClsPar = 5;

        int pWrap[5] = {0, 0, 0, 0, 0};

        std::string multinest_root = raw_output_dir.string();

        if (multinest_root.back() != '/')
            multinest_root.push_back('/');

        nested::run(MN_IS, MN_MMODAL, MN_CEFF, MN_NLIVE, MN_TOL, MN_EFR, ndims, nPar, nClsPar, MN_MAX_MODES, MN_UPD_INT,
                    MN_ZTOL, multinest_root.c_str(), MN_SEED, pWrap, MN_FB, MN_RESUME, MN_OUTFILE, MN_INIT_MPI,
                    MN_LOG_ZERO, MN_MAXITER, loglike, dumper, &context);

        if (!root_process)
            return 0;

        if (context.fatal_error)
            throw std::runtime_error("MultiNest stopped because the likelihood failed: " + context.fatal_message);

        std::cout << '\n';
        std::cout << "------------------------------------------------------------\n";
        std::cout << "MultiNest finished\n";
        std::cout << "evaluations(root) : " << context.evaluations << '\n';

        if (context.has_dumper) {
            std::cout << std::setprecision(12);
            std::cout << "log Z             : " << context.logz << " +/- " << context.logz_error << '\n';
            std::cout << "INS log Z         : " << context.ins_logz << '\n';
            std::cout << "max log L         : " << context.max_loglike << '\n';
        }

        std::cout << "posterior         : " << raw_output_dir / "post_equal_weights.dat" << '\n';
        std::cout << "------------------------------------------------------------\n";

        const std::string python = "runtime/python/venv/bin/python";

        for (const std::string script : {
                 "cases/tomography_mnest/plot_posterior.py",
                 "cases/tomography_mnest/plot_derived.py",
             }) {
            const std::string command = python + " " + script;
            const int         status  = std::system(command.c_str());

            if (status != 0)
                std::cerr << "[WARN] plotting failed: " << script << '\n';
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << '\n';
        return 1;
    }
}
