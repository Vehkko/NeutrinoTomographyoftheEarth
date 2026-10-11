#include <mpi.h>
#include <multinest.h>

#include <nt/earth.hpp>
#include <nt/events.hpp>
#include <nt/flux.hpp>
#include <nt/propagation.hpp>
#include <nt/response.hpp>
#include <nt/timer.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

    using nt::Index_t;
    using nt::Real_t;
    namespace nda = nt::nda;

    constexpr Real_t Q_MIN          = 0.0;
    constexpr Real_t Q_MAX          = 5.0;
    constexpr Real_t EXPOSURE_YEARS = 1.0;
    constexpr Real_t M_EARTH_KG     = 5.9722e24;
    constexpr Real_t I_EARTH_KG_M2  = 8.01736e37;

    constexpr int    MN_IS        = 0;
    constexpr int    MN_MMODAL    = 1;
    constexpr int    MN_CEFF      = 0;
    constexpr int    MN_NLIVE     = 2000;
    constexpr double MN_TOL       = 0.2;
    constexpr double MN_EFR       = 0.3;
    constexpr int    MN_MAX_MODES = 100;
    constexpr int    MN_UPD_INT   = 100;
    constexpr double MN_ZTOL      = -1.0e90;
    constexpr int    MN_SEED      = -1;
    constexpr int    MN_FB        = 1;
    constexpr int    MN_RESUME    = 0;
    constexpr int    MN_OUTFILE   = 1;
    constexpr int    MN_INIT_MPI  = 0;
    constexpr double MN_LOG_ZERO  = -1.0e90;
    constexpr int    MN_MAXITER   = 0;

    constexpr std::uint64_t        LOG_EVERY_N_EVAL          = 500;
    constexpr Index_t              COSZENITH_SAMPLES_PER_BIN = 1;
    constexpr Index_t              ENERGY_SAMPLES_PER_BIN    = 5;
    constexpr nt::FluxRebinOptions REBIN_OPTIONS{false, true};

    struct Context {
        const nt::EarthProfile*      prem         = nullptr;
        const nt::ResponseArray*     response     = nullptr;
        const nt::Flux*              initial_fine = nullptr;
        const nt::EventDistribution* data         = nullptr;
        nt::EarthPropagator*         solver       = nullptr;

        nt::LayeredEarth baseline;
        // M/M_ref = sum_i mass[i] * q_i, I/I_ref = sum_i inertia[i] * q_i.
        std::array<Real_t, 5> mass{};
        std::array<Real_t, 5> inertia{};
        Real_t                determinant = 0.0;

        std::uint64_t evaluations = 0;
        std::uint64_t rejected    = 0;
        bool          console     = false;
        bool          fatal_error = false;
        bool          has_dumper  = false;
        double        max_loglike = std::numeric_limits<double>::quiet_NaN();
        double        logz        = std::numeric_limits<double>::quiet_NaN();
        double        logz_error  = std::numeric_limits<double>::quiet_NaN();
    };

    void loglike(double* cube, int& ndim, int& npars, double& loglike_value, void* context) {
        auto&      ctx        = *static_cast<Context*>(context);
        const auto evaluation = ++ctx.evaluations;

        try {
            if (ndim != 3 || npars != 5)
                throw std::runtime_error("Expected three free parameters and five output parameters");

            std::array<Real_t, 5> q{};
            Real_t                rhs_mass = 1.0, rhs_inertia = 1.0;
            for (int i = 0; i < 3; ++i) {
                q[i] = Q_MIN + (Q_MAX - Q_MIN) * cube[i];
                rhs_mass -= ctx.mass[i] * q[i];
                rhs_inertia -= ctx.inertia[i] * q[i];
            }

            // q1, q2, q3 are independent. Solve the two exact constraints
            // for q4 and q5, with both equations normalized to O(1).
            q[3] = (rhs_mass * ctx.inertia[4] - ctx.mass[4] * rhs_inertia) / ctx.determinant;
            q[4] = (ctx.mass[3] * rhs_inertia - rhs_mass * ctx.inertia[3]) / ctx.determinant;
            for (int i = 0; i < 5; ++i)
                cube[i] = q[i];

            auto earth   = ctx.baseline;
            bool allowed = true;
            for (int i = 0; i < 5; ++i) {
                allowed = allowed && std::isfinite(q[i]) && q[i] > Q_MIN && q[i] <= Q_MAX;
                earth.density_g_cm3[i] *= q[i];
                if (i > 0)
                    allowed = allowed && earth.density_g_cm3[i - 1] > earth.density_g_cm3[i];
            }
            // EHE means rho1 > rho2 > rho3 > rho4 > rho5 > 0, not ordered q_i.
            // Reject unphysical points before the expensive propagation.
            if (!allowed) {
                ++ctx.rejected;
                loglike_value = MN_LOG_ZERO;
                return;
            }

            const auto propagated_fine = ctx.solver->propagate(*ctx.initial_fine, *ctx.prem, earth);
            const auto propagated =
                nt::rebin_flux(propagated_fine, ctx.response->coszenith.view(), ctx.response->true_energy_gev.view(),
                               COSZENITH_SAMPLES_PER_BIN, ENERGY_SAMPLES_PER_BIN, REBIN_OPTIONS);
            const auto prediction = nt::predict_events(propagated, *ctx.response);

            double value = 0.0;
            for (Index_t z = 0; z < prediction.counts.extent(0); ++z) {
                for (Index_t e = 0; e < prediction.counts.extent(1); ++e) {
                    const double nexp = prediction.counts(z, e) * EXPOSURE_YEARS;
                    const double nobs = ctx.data->counts(z, e) * EXPOSURE_YEARS;
                    if (!std::isfinite(nexp) || !std::isfinite(nobs) || nexp < 0.0 || nobs < 0.0)
                        throw std::runtime_error("Non-finite or negative event count in likelihood");
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
                throw std::runtime_error("Non-finite total log-likelihood");
            loglike_value = value;

            if (ctx.console && evaluation % LOG_EVERY_N_EVAL == 0) {
                std::cout << "[LIKE] eval=" << evaluation << " rejected=" << ctx.rejected
                          << " logL=" << std::setprecision(12) << value << " | q=";
                for (const auto qi : q)
                    std::cout << ' ' << std::setprecision(7) << qi;
                std::cout << '\n';
            }
        } catch (const std::exception& error) {
            ctx.fatal_error = true;
            std::cerr << "[FATAL] likelihood: " << error.what() << '\n';
            // MultiNest treats HUGE(1d0) as an exit signal.
            loglike_value = std::numeric_limits<double>::max();
        }
    }

    void dumper(int&, int&, int&, double**, double**, double**, double& maxLogLike, double& logZ, double&,
                double& logZerr, void* context) {
        auto& ctx       = *static_cast<Context*>(context);
        ctx.has_dumper  = true;
        ctx.max_loglike = maxLogLike;
        ctx.logz        = logZ;
        ctx.logz_error  = logZerr;
    }

} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        return 1;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool root_process = rank == 0;

    try {
        const fs::path daemonflux_file = "data/generated/daemonflux/daemonflux_0.8.2.h5";
        const fs::path result_dir      = "result/tomography_mnest_constrained";
        const fs::path raw_output_dir  = result_dir / "raw_output";
        if (!fs::is_regular_file(daemonflux_file))
            throw std::runtime_error("DaemonFlux table not found: " + daemonflux_file.string());
        fs::create_directories(raw_output_dir);

        const auto response   = nt::load_trident_response();
        const auto prem       = nt::load_prem();
        const auto daemonflux = nt::load_daemonflux(daemonflux_file, "IceCube");
        const auto fine_coszenith =
            nt::sample_coszenith_bin_midpoints(response.coszenith_edges.view(), COSZENITH_SAMPLES_PER_BIN);
        const auto fine_energy =
            nt::sample_log_energy_bin_midpoints(response.true_energy_edges_gev.view(), ENERGY_SAMPLES_PER_BIN);
        const auto initial_fine = nt::resample_flux(daemonflux, fine_coszenith.view(), fine_energy.view());

        nt::PropagationOptions options;
        options.interactions = true;
        options.threads      = 1;
        nt::EarthPropagator solver(initial_fine, options);

        Context context;
        context.prem         = &prem;
        context.response     = &response;
        context.initial_fine = &initial_fine;
        context.solver       = &solver;
        context.console      = root_process;
        const std::array<Real_t, 5> unity{1.0, 1.0, 1.0, 1.0, 1.0};
        context.baseline =
            nt::make_layered_constant_5(prem, nda::make_view1d(static_cast<const Real_t*>(unity.data()), unity.size()));
        Real_t inner = 0.0;
        for (Index_t i = 0; i < 5; ++i) {
            const Real_t outer = context.baseline.outer_radius_km[i];
            context.mass[i]    = nt::mass_kg(context.baseline, inner, outer) / M_EARTH_KG;
            context.inertia[i] = nt::moment_of_inertia_kg_m2(context.baseline, inner, outer) / I_EARTH_KG_M2;
            inner              = outer;
        }
        context.determinant = context.mass[3] * context.inertia[4] - context.mass[4] * context.inertia[3];
        if (!std::isfinite(context.determinant) || std::abs(context.determinant) < 1.0e-14)
            throw std::runtime_error("Singular mass/inertia constraint system");

        // Same full-PREM Asimov construction and propagation grid as the baseline case.
        nt::Timer  timer;
        const auto asimov_fine = solver.propagate(initial_fine, prem);
        const auto asimov_flux = nt::rebin_flux(asimov_fine, response.coszenith.view(), response.true_energy_gev.view(),
                                                COSZENITH_SAMPLES_PER_BIN, ENERGY_SAMPLES_PER_BIN, REBIN_OPTIONS);
        const auto asimov_data = nt::predict_events(asimov_flux, response);
        context.data           = &asimov_data;

        if (root_process) {
            std::cout << "[TIME] Asimov construction: " << timer.seconds() << " s\n";
            std::cout << "TRIDENT five-layer constant-density scan: hard M + I + EHE\n";
            std::cout << "Free parameters: q1, q2, q3; derived parameters: q4, q5\n";
            std::cout << "Free q prior: Uniform[0,5], restricted by all-layer support and EHE\n";
            std::cout << "Exposure: " << EXPOSURE_YEARS << " year; nlive=" << MN_NLIVE << "; tol=" << MN_TOL
                      << "; efr=" << MN_EFR << '\n';
            std::cout << "Output: " << raw_output_dir << '\n';

            // Reference densities also allow conversion of all five posterior q columns to rho.
            std::ofstream info(result_dir / "run_info.dat");
            if (!info)
                throw std::runtime_error("Cannot write run_info.dat");
            info << std::setprecision(17);
            info << "model=5layer_constant\nasimov_truth=full_prem\nfree_parameters=q1,q2,q3\n";
            info << "derived_parameters=q4,q5\nposterior_columns=q1,q2,q3,q4,q5,loglike\n";
            info << "q_min=" << Q_MIN << "\nq_max=" << Q_MAX << '\n';
            info << "ehe=rho1>rho2>rho3>rho4>rho5>0\n";
            info << "mass_kg=" << M_EARTH_KG << "\ninertia_kg_m2=" << I_EARTH_KG_M2 << '\n';
            info << "exposure_years=" << EXPOSURE_YEARS << "\nnlive=" << MN_NLIVE << "\ntol=" << MN_TOL
                 << "\nefr=" << MN_EFR << "\nresume=" << MN_RESUME << '\n';
            info << "coszenith_samples_per_bin=1\nenergy_samples_per_bin=5\n";
            info << "interpolate_coszenith=0\ninterpolate_energy=1\n";
            for (Index_t i = 0; i < 5; ++i) {
                info << "outer_radius_km_" << i + 1 << '=' << context.baseline.outer_radius_km[i] << '\n';
                info << "prem_mean_density_g_cm3_" << i + 1 << '=' << context.baseline.density_g_cm3[i] << '\n';
            }
        }

        int         ndims = 3, nPar = 5, nClsPar = 3;
        int         pWrap[3]       = {0, 0, 0};
        std::string multinest_root = raw_output_dir.string() + '/';
        timer.reset();
        nested::run(MN_IS, MN_MMODAL, MN_CEFF, MN_NLIVE, MN_TOL, MN_EFR, ndims, nPar, nClsPar, MN_MAX_MODES, MN_UPD_INT,
                    MN_ZTOL, multinest_root.c_str(), MN_SEED, pWrap, MN_FB, MN_RESUME, MN_OUTFILE, MN_INIT_MPI,
                    MN_LOG_ZERO, MN_MAXITER, loglike, dumper, &context);

        int local_fatal = context.fatal_error ? 1 : 0, global_fatal = 0;
        MPI_Allreduce(&local_fatal, &global_fatal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (global_fatal)
            throw std::runtime_error("MultiNest stopped because a likelihood evaluation failed");

        if (root_process) {
            if (!context.has_dumper || !std::isfinite(context.logz) || !std::isfinite(context.logz_error) ||
                !std::isfinite(context.max_loglike))
                throw std::runtime_error("MultiNest returned without valid final evidence");
            std::cout << std::setprecision(12);
            std::cout << "[DONE] wall time: " << timer.seconds() << " s\n";
            std::cout << "log Z: " << context.logz << " +/- " << context.logz_error << '\n';
            std::cout << "max log L: " << context.max_loglike << '\n';
            std::cout << "evaluations(root): " << context.evaluations << "; rejected(root): " << context.rejected
                      << '\n';
            std::cout << "Posterior: " << raw_output_dir / "post_equal_weights.dat" << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "[fatal][rank " << rank << "] " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    MPI_Finalize();
    return 0;
}
