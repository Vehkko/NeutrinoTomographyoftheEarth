#include <H5Cpp.h>
#include <gsl/gsl_cdf.h>
#include <multinest.h>

#include <nt/earth.hpp>
#include <nt/events.hpp>
#include <nt/flux.hpp>
#include <nt/propagation.hpp>
#include <nt/response.hpp>
#include <nt/timer.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

    using nt::Index_t;
    using nt::Real_t;

    constexpr std::array<const char*, 5> PARAMETERS = {
        "K+_2P", "K-_2P", "pi+_20T", "pi+_2P", "pi-_20T",
    };

    constexpr double EXPOSURE_YEARS = 1.0;

    constexpr double Q_MIN  = 0.0;
    constexpr double Q_MAX  = 4.0;
    constexpr double Z_CLIP = 5.0;

    constexpr double COUNT_FLOOR = 1.0e-10;
    constexpr double LOG_ZERO    = -1.0e90;

    constexpr int    N_LIVE          = 5000;
    constexpr double TOLERANCE       = 0.2;
    constexpr double EFFICIENCY      = 0.3;
    constexpr int    UPDATE_INTERVAL = 1000;

    constexpr Index_t COSZENITH_SAMPLES = 1;
    constexpr Index_t ENERGY_SAMPLES    = 5;

    constexpr nt::FluxRebinOptions REBIN_OPTIONS{
        false, // Average sub-angle samples; currently one sample/bin.
        true,  // Interpolate in E to response centers.
    };

    using Matrix5 = std::array<double, 25>;

    int mpi_rank_from_environment() {
        for (const char* name : {
                 "PMI_RANK",
                 "PMIX_RANK",
                 "OMPI_COMM_WORLD_RANK",
             }) {
            if (const char* value = std::getenv(name))
                return std::atoi(value);
        }

        return 0;
    }

    std::vector<double> read_array(const H5::H5File& file, const std::string& path,
                                   std::initializer_list<hsize_t> expected) {
        const auto dataset = file.openDataSet(path);
        const auto space   = dataset.getSpace();

        if (space.getSimpleExtentNdims() != static_cast<int>(expected.size()))
            throw std::runtime_error("Unexpected dataset rank: " + path);

        std::vector<hsize_t> shape(expected.size());
        space.getSimpleExtentDims(shape.data());

        if (!std::equal(shape.begin(), shape.end(), expected.begin()))
            throw std::runtime_error("Unexpected dataset shape: " + path);

        std::size_t size = 1;
        for (const auto n : shape)
            size *= n;

        std::vector<double> values(size);
        dataset.read(values.data(), H5::PredType::NATIVE_DOUBLE);

        return values;
    }

    nt::Flux empty_on_grid(const nt::Flux& grid) {
        nt::Flux result(grid.n_coszenith(), grid.n_energy());

        for (Index_t z = 0; z < grid.n_coszenith(); ++z)
            result.coszenith()(z) = grid.coszenith()(z);

        for (Index_t e = 0; e < grid.n_energy(); ++e)
            result.energy_gev()(e) = grid.energy_gev()(e);

        return result;
    }

    struct FluxModel {
        nt::Flux              central;
        std::vector<nt::Flux> templates;
        Matrix5               cholesky;
    };

    FluxModel load_flux_model(const fs::path& path, nt::nda::View<const Real_t, 1> coszenith,
                              nt::nda::View<const Real_t, 1> energy) {
        const auto master = nt::load_daemonflux(path, "IceCube");

        FluxModel model{
            nt::resample_flux(master, coszenith, energy),
            {},
            {},
        };

        const H5::H5File file(path.string(), H5F_ACC_RDONLY);

        const auto lower = read_array(file, "/nuisance/cholesky", {5, 5});
        std::copy(lower.begin(), lower.end(), model.cholesky.begin());

        auto buffer = empty_on_grid(master);

        const Index_t nz = master.n_coszenith();
        const Index_t ne = master.n_energy();

        model.templates.reserve(5);

        for (const char* name : PARAMETERS) {
            const std::string base = std::string("/nuisance/templates/") + name + "/";

            for (Index_t p = 0; p < 2; ++p) {
                const auto values = read_array(file, base + (p == 0 ? "numu" : "antinumu"),
                                               {
                                                   static_cast<hsize_t>(nz),
                                                   static_cast<hsize_t>(ne),
                                               });

                auto component = buffer.component(static_cast<nt::Particle>(p), nt::Flavor::muon);

                for (Index_t z = 0; z < nz; ++z)
                    for (Index_t e = 0; e < ne; ++e)
                        component(z, e) = values[z * ne + e];
            }

            model.templates.push_back(nt::resample_flux(buffer, coszenith, energy));
        }

        return model;
    }

    double normal_prior(double u) {
        static const double lower = gsl_cdf_ugaussian_P(-Z_CLIP);
        static const double upper = gsl_cdf_ugaussian_P(Z_CLIP);

        const double p = lower + std::clamp(u, 0.0, 1.0) * (upper - lower);

        return gsl_cdf_ugaussian_Pinv(p);
    }

    // Return false for the same unphysical-flux samples rejected
    // by the Python implementation.
    bool fill_initial_flux(const FluxModel& model, const double* latent, nt::Flux& initial) {
        std::array<double, 5> shifts{};

        // s = L z, with L stored in row-major order.
        for (int a = 0; a < 5; ++a)
            for (int b = 0; b <= a; ++b)
                shifts[a] += model.cholesky[5 * a + b] * latent[b];

        auto&       state   = initial.native_state();
        const auto& central = model.central.native_state();

        std::copy(central.begin(), central.end(), state.begin());

        // Phi = Phi0 + sum_a s_a T_a.
        // Electron and tau initial components remain zero.
        for (int a = 0; a < 5; ++a) {
            auto        dst    = state.begin();
            const auto& source = model.templates[a].native_state();

            for (auto src = source.begin(); src != source.end(); ++src, ++dst)
                *dst += shifts[a] * *src;
        }

        for (const auto value : state)
            if (!std::isfinite(value) || value < 0.0)
                return false;

        return true;
    }

    nt::EventDistribution events_from_fine(const nt::Flux& fine, const nt::ResponseArray& response) {
        const auto coarse = nt::rebin_flux(fine, response.coszenith.view(), response.true_energy_gev.view(),
                                           COSZENITH_SAMPLES, ENERGY_SAMPLES, REBIN_OPTIONS);

        return nt::predict_events(coarse, response);
    }

    struct Context {
        const nt::EarthProfile&      prem;
        const nt::ResponseArray&     response;
        const FluxModel&             flux;
        nt::Flux&                    initial;
        const nt::EventDistribution& data;
        nt::EarthPropagator&         solver;

        bool console;

        std::uint64_t evaluations = 0;
        bool          fatal       = false;
        std::string   fatal_message;
    };

    void loglike(double* cube, int& ndim, int& npars, double& value, void* pointer) {
        (void)ndim;
        (void)npars;

        auto& ctx = *static_cast<Context*>(pointer);
        ++ctx.evaluations;

        try {
            std::array<Real_t, 5> q{};

            // Transform the unit cube into the physical sampling coordinates.
            // The posterior stores q1...q5, z1...z5.
            for (int i = 0; i < 5; ++i) {
                cube[i] = Q_MIN + (Q_MAX - Q_MIN) * cube[i];
                q[i]    = cube[i];

                cube[5 + i] = normal_prior(cube[5 + i]);
            }

            if (!fill_initial_flux(ctx.flux, cube + 5, ctx.initial)) {
                value = -1.0e100;
                return;
            }

            // Keep the project's existing PREM-derived layer averages
            // and exact piecewise-constant Earth implementation.
            const auto earth = nt::make_layered_constant_5(
                ctx.prem, nt::nda::make_view1d(static_cast<const Real_t*>(q.data()), q.size()));

            const auto fine = ctx.solver.propagate(ctx.initial, ctx.prem, earth);

            const auto predicted = events_from_fine(fine, ctx.response);

            // Gaussian priors are already included in normal_prior().
            // Do not add another Gaussian penalty here.
            value = 0.0;

            for (Index_t z = 0; z < predicted.counts.extent(0); ++z) {
                for (Index_t e = 0; e < predicted.counts.extent(1); ++e) {
                    const double raw = predicted.counts(z, e) * EXPOSURE_YEARS;

                    if (!std::isfinite(raw))
                        throw std::runtime_error("Non-finite predicted event count");

                    const double expected = std::max(raw, COUNT_FLOOR);

                    const double observed = std::max(ctx.data.counts(z, e) * EXPOSURE_YEARS, COUNT_FLOOR);

                    value -= expected - observed + observed * std::log(observed / expected);
                }
            }

            if (!std::isfinite(value))
                throw std::runtime_error("Non-finite log-likelihood");

            if (ctx.console && ctx.evaluations % 500 == 0) {
                std::cout << "[LIKE] eval=" << ctx.evaluations << " logL=" << std::setprecision(12) << value << '\n';
            }

        } catch (const std::exception& error) {
            ctx.fatal         = true;
            ctx.fatal_message = error.what();

            std::cerr << "[FATAL] likelihood: " << error.what() << '\n';

            // Same MultiNest exit signal used by the existing case.
            value = std::numeric_limits<double>::max();
        }
    }

    void dumper(int&, int&, int&, double**, double**, double**, double& max_loglike, double& logz, double& ins_logz,
                double& logz_error, void* pointer) {
        const auto& ctx = *static_cast<Context*>(pointer);

        if (ctx.console) {
            std::cout << std::setprecision(12) << "[MN] logZ=" << logz << " +/- " << logz_error
                      << " INSlogZ=" << ins_logz << " maxLogL=" << max_loglike << '\n';
        }
    }

} // namespace

int main() {
    H5::Exception::dontPrint();

    try {
        const bool root = mpi_rank_from_environment() == 0;

        const fs::path table = "data/generated/daemonflux/daemonflux_0.8.2_5unc.h5";

        const fs::path result     = "result/tomography_mnest_5unc";
        const fs::path raw_output = result / "raw_output";

        fs::create_directories(raw_output);

        const auto response = nt::load_trident_response();
        const auto prem     = nt::load_prem();

        const auto fine_z = nt::sample_coszenith_bin_midpoints(response.coszenith_edges.view(), COSZENITH_SAMPLES);

        const auto fine_e = nt::sample_log_energy_bin_midpoints(response.true_energy_edges_gev.view(), ENERGY_SAMPLES);

        const auto flux = load_flux_model(table, fine_z.view(), fine_e.view());

        auto initial = empty_on_grid(flux.central);

        nt::PropagationOptions options;
        options.interactions = true;
        options.threads      = 1;

        nt::EarthPropagator solver(flux.central, options);

        // Full-PREM Asimov data with central atmospheric flux.
        nt::Timer timer;

        const auto truth_fine = solver.propagate(flux.central, prem);
        const auto data       = events_from_fine(truth_fine, response);

        double total = 0.0;

        for (Index_t z = 0; z < data.counts.extent(0); ++z) {
            for (Index_t e = 0; e < data.counts.extent(1); ++e) {
                if (!std::isfinite(data.counts(z, e)))
                    throw std::runtime_error("Non-finite Asimov event count");

                total += std::max(data.counts(z, e) * EXPOSURE_YEARS, COUNT_FLOOR);
            }
        }

        if (root) {
            std::cout << "Five-layer constant-density tomography"
                         " + five flux nuisances\n"
                      << "Asimov: full PREM, central flux\n"
                      << "q: Uniform[0,4]; z: independent N(0,1),"
                         " truncated at +/-5\n"
                      << "No M/I/EHE constraints\n"
                      << "Exposure: " << EXPOSURE_YEARS << " year\n"
                      << "Grid: " << flux.central.n_coszenith() << " directions x " << flux.central.n_energy()
                      << " energies\n"
                      << "Energy rebin: linear interpolation"
                         " to response centers\n"
                      << "Asimov events: " << std::setprecision(12) << total << '\n'
                      << "Asimov time: " << timer.seconds() << " s\n"
                      << "MultiNest: nlive=" << N_LIVE << " tol=" << TOLERANCE << " efr=" << EFFICIENCY
                      << "; INS=true, multimodal=false, resume=false\n";

            std::ofstream labels(result / "parameter_names.txt");

            if (!labels)
                throw std::runtime_error("Cannot write parameter_names.txt");

            for (int i = 1; i <= 5; ++i)
                labels << "q" << i << '\n';

            for (int i = 1; i <= 5; ++i)
                labels << "z" << i << '\n';
        }

        Context context{
            prem, response, flux, initial, data, solver, root,
        };

        int ndim     = 10;
        int npars    = 10;
        int ncluster = 10;
        int wrap[10] = {};

        const std::string output_root = raw_output.string() + "/";

        timer.reset();

        nested::run(1, // importance nested sampling
                    0, // multimodal
                    0, // constant efficiency
                    N_LIVE, TOLERANCE, EFFICIENCY, ndim, npars, ncluster,
                    100, // maximum modes
                    UPDATE_INTERVAL,
                    -1.0e90, // mode evidence threshold
                    output_root.c_str(),
                    -1, // random seed
                    wrap,
                    1, // feedback
                    0, // resume
                    1, // write output files
                    1, // MultiNest initializes MPI
                    LOG_ZERO,
                    0, // unlimited iterations
                    loglike, dumper, &context);

        if (context.fatal)
            throw std::runtime_error(context.fatal_message);

        if (root) {
            std::cout << "[TIME] MultiNest: " << timer.seconds() << " s\n"
                      << "Posterior: " << raw_output / "post_equal_weights.dat" << '\n';
        }

        return 0;

    } catch (const H5::Exception& error) {
        std::cerr << "[fatal] HDF5: " << error.getDetailMsg() << '\n';

    } catch (const std::exception& error) {
        std::cerr << "[fatal] " << error.what() << '\n';
    }

    return 1;
}
