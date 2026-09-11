#include <nt/propagation.hpp>

#include <SQuIDS/const.h>
#include <gsl/gsl_odeiv2.h>
#include <nuSQuIDS/body.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace nt {

    namespace {

        const squids::Const units;

        template <typename Array> std::vector<double> to_vector(const Array& a) {
            return std::vector<double>(a.data(), a.data() + a.extent(0));
        }

        Index_t layer_index(Index_t layers, const std::array<Real_t, 5>& outer_radius, Real_t radius_km) noexcept {
            assert(layers == 3 || layers == 5);

            if (layers == 3) {
                if (radius_km < outer_radius[0])
                    return 0;
                if (radius_km < outer_radius[1])
                    return 1;
                return 2;
            }

            if (radius_km < outer_radius[0])
                return 0;
            if (radius_km < outer_radius[1])
                return 1;
            if (radius_km < outer_radius[2])
                return 2;
            if (radius_km < outer_radius[3])
                return 3;
            return 4;
        }

        // Common PREM-backed EarthAtm implementation used only inside the
        // nuSQuIDS backend.
        //
        // Atmosphere height is zero because DaemonFlux supplies the atmospheric
        // flux at the detector/site; this propagation describes passage through
        // the Earth from surface to surface.
        class PremBackedEarthAtm : public nusquids::EarthAtm {
          public:
            explicit PremBackedEarthAtm(const EarthProfile& prem)
                : EarthAtm(to_vector(prem.radius_fraction), to_vector(prem.density_g_cm3), to_vector(prem.ye)) {
                SetAtmosphereHeight(0.0);
            }

          protected:
            // For atmosphere height zero:
            //
            //     r^2 = R^2 + s^2 - L s
            //
            // where s is distance from the start of the chord and L is the
            // complete surface-to-surface path length.
            Real_t radius_from_track_km(const nusquids::GenericTrack& track) const noexcept {
                const Real_t s  = (track.GetX() - track.GetInitialX()) / units.km;
                const Real_t L  = (track.GetFinalX() - track.GetInitialX()) / units.km;
                const Real_t r2 = radius * radius + s * s - L * s;
                const Real_t r  = std::sqrt(std::max(Real_t{0}, r2));
                return std::min(r, radius);
            }

            Real_t prem_density_at_radius(Real_t radius_km) const {
                const Real_t x = radius_km / radius;

                if (x <= x_radius_min)
                    return x_rho_min;
                if (x >= x_radius_max)
                    return x_rho_max;
                return inter_density(x);
            }
        };

        class LayeredEarthAtm final : public PremBackedEarthAtm {
          public:
            LayeredEarthAtm(const EarthProfile& prem, const LayeredEarth& earth)
                : PremBackedEarthAtm(prem), earth_(earth) {}

            double density(const nusquids::GenericTrack& track) const override {
                const Real_t  r = radius_from_track_km(track);
                const Index_t i = layer_index(earth_.layers, earth_.outer_radius_km, r);
                return earth_.density_g_cm3[i];
            }

            // Ye is deliberately inherited from EarthAtm and therefore remains
            // the complete PREM Ye(r) profile.

          private:
            LayeredEarth earth_;
        };

        class PremScaledEarthAtm final : public PremBackedEarthAtm {
          public:
            PremScaledEarthAtm(const EarthProfile& prem, const PremScaledEarth& earth)
                : PremBackedEarthAtm(prem), earth_(earth) {}

            double density(const nusquids::GenericTrack& track) const override {
                const Real_t  r = radius_from_track_km(track);
                const Index_t i = layer_index(earth_.layers, earth_.outer_radius_km, r);
                return prem_density_at_radius(r) * earth_.density_factor[i];
            }

          private:
            PremScaledEarth earth_;
        };

        class PerturbedPremEarthAtm final : public PremBackedEarthAtm {
          public:
            PerturbedPremEarthAtm(const EarthProfile& prem, const DensityPerturbation& perturbation)
                : PremBackedEarthAtm(prem), perturbation_(perturbation) {}

            double density(const nusquids::GenericTrack& track) const override {
                const Real_t r   = radius_from_track_km(track);
                const Real_t rho = prem_density_at_radius(r);
                const Real_t w   = perturbation_.weight(r);

                if (perturbation_.mode == PerturbationMode::relative)
                    return rho * (Real_t{1} + perturbation_.amplitude * w);
                return rho + perturbation_.amplitude * w;
            }

          private:
            DensityPerturbation perturbation_;
        };

        Flux::Axis make_energy_ev(const Flux& flux) {
            Flux::Axis energy_ev({flux.n_energy()});
            const auto energy_gev = flux.energy_gev();

            for (Index_t e = 0; e < flux.n_energy(); ++e)
                energy_ev[e] = energy_gev(e) * units.GeV;
            return energy_ev;
        }

        void configure_solver(nusquids::nuSQUIDSAtm<>& nus, const PropagationOptions& options) {
            // Preserve the oscillation parameters used by the established
            // analysis.
            nus.Set_MixingAngle(0, 1, 0.563942);
            nus.Set_MixingAngle(0, 2, 0.154085);
            nus.Set_MixingAngle(1, 2, 0.785398);

            nus.Set_SquareMassDifference(1, 7.65e-5);
            nus.Set_SquareMassDifference(2, 2.47e-3);
            nus.Set_CPPhase(0, 2, 0.0);

            nus.Set_h_max(options.h_max_km * units.km);
            nus.Set_GSL_step(gsl_odeiv2_step_rk4);
            nus.Set_rel_error(1e-6);
            nus.Set_abs_error(1e-6);
            nus.Set_EvalThreads(static_cast<unsigned int>(options.threads));
        }

    } // namespace

    EarthPropagator::EarthPropagator(const Flux& grid, const PropagationOptions& options)
        : nus_(grid.native_coszenith(), make_energy_ev(grid), 3, nusquids::both, options.interactions) {
        configure_solver(nus_, options);
    }

    Flux EarthPropagator::propagate_with_body(const Flux& initial, std::shared_ptr<nusquids::EarthAtm> earth) {
        nus_.Set_EarthModel(std::move(earth));

        // Flux already has nuSQuIDS' required native layout:
        //
        //     [coszenith, energy, particle, flavor]
        //
        // No project-side 4D repacking is performed.
        nus_.Set_initial_state(initial.native_state(), nusquids::flavor);
        nus_.EvolveState();

        Flux result(initial.n_coszenith(), initial.n_energy());

        auto       result_z  = result.coszenith();
        auto       result_e  = result.energy_gev();
        const auto initial_z = initial.coszenith();
        const auto initial_e = initial.energy_gev();

        for (Index_t z = 0; z < initial.n_coszenith(); ++z)
            result_z(z) = initial_z(z);
        for (Index_t e = 0; e < initial.n_energy(); ++e)
            result_e(e) = initial_e(e);

        auto& state = result.native_state();

        // Evaluate directly at nuSQuIDS' own nodes. This avoids the
        // unnecessary angular/energy interpolation performed by
        // nuSQUIDSAtm::EvalFlavor().
        for (Index_t z = 0; z < initial.n_coszenith(); ++z) {
            auto& nsq = nus_.GetnuSQuIDS(static_cast<unsigned int>(z));

            for (Index_t e = 0; e < initial.n_energy(); ++e) {
                for (Index_t p = 0; p < 2; ++p) {
                    for (Index_t f = 0; f < 3; ++f) {
                        state[z][e][p][f] = nsq.EvalFlavorAtNode(
                            static_cast<unsigned int>(f), static_cast<unsigned int>(e), static_cast<unsigned int>(p));
                    }
                }
            }
        }

        return result;
    }

    Flux EarthPropagator::propagate(const Flux& initial, const EarthProfile& prem) {
        auto earth = std::make_shared<nusquids::EarthAtm>(to_vector(prem.radius_fraction),
                                                          to_vector(prem.density_g_cm3), to_vector(prem.ye));
        earth->SetAtmosphereHeight(0.0);
        return propagate_with_body(initial, std::move(earth));
    }

    Flux EarthPropagator::propagate(const Flux& initial, const EarthProfile& prem, const LayeredEarth& earth) {
        return propagate_with_body(initial, std::make_shared<LayeredEarthAtm>(prem, earth));
    }

    Flux EarthPropagator::propagate(const Flux& initial, const EarthProfile& prem, const PremScaledEarth& earth) {
        return propagate_with_body(initial, std::make_shared<PremScaledEarthAtm>(prem, earth));
    }

    Flux EarthPropagator::propagate(const Flux& initial, const EarthProfile& prem,
                                    const DensityPerturbation& perturbation) {
        return propagate_with_body(initial, std::make_shared<PerturbedPremEarthAtm>(prem, perturbation));
    }

} // namespace nt
