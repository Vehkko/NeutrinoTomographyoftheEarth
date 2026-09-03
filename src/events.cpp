#include <nt/events.hpp>
#include <nt/flux.hpp>
#include <nt/response.hpp>
#include <nt/types.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace nt {

    namespace {

        bool nearly_equal(Real_t a, Real_t b) noexcept {
            const Real_t scale = std::max({Real_t{1}, std::abs(a), std::abs(b)});
            return std::abs(a - b) <= Real_t{1e-10} * scale;
        }

        void check_axes(const Flux& flux, const ResponseArray& response) {
            const Index_t ncz   = response.coszenith.extent(0);
            const Index_t ntrue = response.true_energy_gev.extent(0);

            if (flux.n_coszenith() != ncz || flux.n_energy() != ntrue)
                throw std::invalid_argument(
                    "predict_events: Flux grid dimensions do not match TRIDENT response");

            const auto flux_z = flux.coszenith();
            const auto flux_e = flux.energy_gev();

            for (Index_t z = 0; z < ncz; ++z) {
                if (!nearly_equal(flux_z(z), response.coszenith(z)))
                    throw std::invalid_argument(
                        "predict_events: Flux coszenith grid does not match TRIDENT response");
            }

            for (Index_t e = 0; e < ntrue; ++e) {
                if (!nearly_equal(flux_e(e), response.true_energy_gev(e)))
                    throw std::invalid_argument(
                        "predict_events: Flux energy grid does not match TRIDENT response");
            }
        }

        void check_response_shape(const ResponseArray& response) {
            const Index_t ntrue = response.true_energy_gev.extent(0);
            const Index_t ncz   = response.coszenith.extent(0);
            const Index_t nreco = response.reco_energy_gev.extent(0);

            if (response.detector_response.extent(0) != ntrue ||
                response.detector_response.extent(1) != ncz)
                throw std::invalid_argument(
                    "predict_events: detector_response shape is inconsistent with its axes");

            if (response.energy_migration.extent(0) != ntrue ||
                response.energy_migration.extent(1) != nreco)
                throw std::invalid_argument(
                    "predict_events: energy_migration shape is inconsistent with its axes");
        }

    } // namespace

    EventDistribution predict_events(const Flux& flux, const ResponseArray& response) {
        check_response_shape(response);
        check_axes(flux, response);

        const Index_t ncz   = response.coszenith.extent(0);
        const Index_t ntrue = response.true_energy_gev.extent(0);
        const Index_t nreco = response.reco_energy_gev.extent(0);

        EventDistribution events(ncz, nreco);

        // The output owns its axes. These are tiny necessary copies:
        // 34 coszenith values and 20 reconstructed-energy values for TRIDENT.
        std::copy_n(response.coszenith.data(), ncz, events.coszenith.data());
        std::copy_n(response.reco_energy_gev.data(), nreco, events.reco_energy_gev.data());

        const auto numu     = flux.numu();
        const auto antinumu = flux.antinumu();

        const Real_t* nu_data  = numu.data();
        const Real_t* anu_data = antinumu.data();

        const Index_t nu_z_stride  = numu.stride(0);
        const Index_t nu_e_stride  = numu.stride(1);
        const Index_t anu_z_stride = antinumu.stride(0);
        const Index_t anu_e_stride = antinumu.stride(1);

        // NDA owning arrays are row-major:
        //
        // detector_response [true, coszenith]
        // energy_migration  [true, reco]
        // events.counts     [coszenith, reco]
        const Real_t* detector  = response.detector_response.data();
        const Real_t* migration = response.energy_migration.data();
        Real_t*       counts    = events.counts.data();

        for (Index_t z = 0; z < ncz; ++z) {
            Real_t* out = counts + z * nreco;
            std::fill_n(out, nreco, Real_t{0});

            const Real_t* nu_row  = nu_data + z * nu_z_stride;
            const Real_t* anu_row = anu_data + z * anu_z_stride;

            for (Index_t t = 0; t < ntrue; ++t) {
                const Real_t phi =
                    nu_row[t * nu_e_stride] + anu_row[t * anu_e_stride];

                const Real_t weighted_flux =
                    phi * detector[t * ncz + z];

                const Real_t* migration_row = migration + t * nreco;

                // The reconstructed-energy dimension is contiguous, which gives
                // the compiler a simple vectorizable inner loop.
                for (Index_t r = 0; r < nreco; ++r)
                    out[r] += weighted_flux * migration_row[r];
            }
        }

        return events;
    }

} // namespace nt
