#pragma once

#include <memory>

#include <nt/earth.hpp>
#include <nt/flux.hpp>
#include <nt/types.hpp>
#include <nuSQuIDS/nuSQuIDS.h>

namespace nt {

    struct PropagationOptions {
        bool    interactions = true;
        Index_t threads      = 1;
        Real_t  h_max_km     = 500.0;
    };

    // Reusable Earth-propagation solver on one fixed coszenith/energy grid.
    //
    // The grid and solver options are fixed at construction. Initial flux
    // values and Earth models may change between propagate() calls.
    //
    // EarthPropagator is stateful and must not be used concurrently.
    class EarthPropagator {
      public:
        explicit EarthPropagator(const Flux& grid, const PropagationOptions& options = {});

        EarthPropagator(const EarthPropagator&)            = delete;
        EarthPropagator& operator=(const EarthPropagator&) = delete;

        // Plain PREM.
        [[nodiscard]] Flux propagate(const Flux& initial, const EarthProfile& prem);

        // Layered constant-density rho with PREM Ye(r).
        [[nodiscard]] Flux propagate(const Flux& initial, const EarthProfile& prem, const LayeredEarth& earth);

        // Piecewise PREM density scaling with PREM Ye(r).
        [[nodiscard]] Flux propagate(const Flux& initial, const EarthProfile& prem, const PremScaledEarth& earth);

        // Local PREM perturbation with PREM Ye(r).
        [[nodiscard]] Flux propagate(const Flux& initial, const EarthProfile& prem,
                                     const DensityPerturbation& perturbation);

      private:
        [[nodiscard]] Flux propagate_with_body(const Flux& initial, std::shared_ptr<nusquids::EarthAtm> earth);

        nusquids::nuSQUIDSAtm<> nus_;
    };

} // namespace nt
