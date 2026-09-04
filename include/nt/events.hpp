#pragma once

#include <nt/flux.hpp>
#include <nt/response.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

namespace nt {

    // Predicted reconstructed event distribution.
    //
    // counts:
    //     [coszenith, reconstructed_energy]
    //
    // The axes are owned by the result so EventDistribution remains
    // self-contained after the detector response object goes out of scope.
    struct EventDistribution {
        nda::Array<Real_t, 1> coszenith;
        nda::Array<Real_t, 1> reco_energy_gev;
        nda::Array<Real_t, 2> counts;

        EventDistribution(Index_t n_coszenith, Index_t n_reco)
            : coszenith({n_coszenith}), reco_energy_gev({n_reco}), counts({n_coszenith, n_reco}) {}
    };

    // Convert propagated νμ + ν̄μ flux into reconstructed event counts:
    //
    //   N[z,r] = sum_t
    //       (Phi_numu[z,t] + Phi_antinumu[z,t])
    //       * R[t,z]
    //       * P[r | t]
    //
    // where:
    //   z = coszenith bin
    //   t = true-energy bin
    //   r = reconstructed-energy bin
    //
    // The Flux grid must already match the TRIDENT response grid. This is
    // intentional: interpolation belongs before propagation / event evaluation,
    // not inside this hot path.
    //
    // No intermediate resampled Flux or 3D response array is allocated.
    [[nodiscard]] EventDistribution predict_events(const Flux& flux, const ResponseArray& response);

} // namespace nt
