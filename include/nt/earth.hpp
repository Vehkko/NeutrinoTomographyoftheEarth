#pragma once

#include <array>
#include <filesystem>

#include <nt/flux.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

namespace nt {

    // Tabulated PREM profile:
    //
    //     radius_fraction = r / R_Earth
    //     density_g_cm3   = rho [g/cm^3]
    //     ye              = electron fraction
    //
    // The project PREM file contains 201 nodes from r/R = 0 to 1.
    struct EarthProfile {
        nda::Array<Real_t, 1> radius_fraction;
        nda::Array<Real_t, 1> density_g_cm3;
        nda::Array<Real_t, 1> ye;

        explicit EarthProfile(Index_t n) : radius_fraction({n}), density_g_cm3({n}), ye({n}) {}
    };

    // Piecewise-constant density Earth.
    //
    // Only rho is piecewise constant. Ye(r) remains the PREM profile during
    // propagation.
    //
    // layers is always 3 or 5; only the first `layers` entries are used.
    struct LayeredEarth {
        Index_t               layers = 0;
        std::array<Real_t, 5> outer_radius_km{};
        std::array<Real_t, 5> density_g_cm3{};
    };

    // PREM-scaled Earth:
    //
    //     rho(r) = q_i * rho_PREM(r)
    //
    // inside layer i. Ye(r) is unchanged PREM.
    struct PremScaledEarth {
        Index_t               layers = 0;
        std::array<Real_t, 5> outer_radius_km{};
        std::array<Real_t, 5> density_factor{};
    };

    enum class PerturbationShape {
        box,
        gaussian,
    };

    enum class PerturbationMode {
        relative,
        absolute,
    };

    // Local perturbation of the PREM density.
    //
    // relative:
    //     rho(r) = rho_PREM(r) * [1 + amplitude * w(r)]
    //
    // absolute:
    //     rho(r) = rho_PREM(r) + amplitude * w(r)
    //
    // For relative mode amplitude is dimensionless.
    // For absolute mode amplitude is in g/cm^3.
    //
    // Box:
    //     half_width_km is used.
    //     edge_width_km = 0 gives an exact top-hat.
    //     edge_width_km > 0 gives tanh-smoothed edges.
    //
    // Gaussian:
    //     sigma_km is used.
    //     gaussian_cutoff_sigma <= 0 means an untruncated Gaussian.
    //     gaussian_cutoff_sigma > 0 truncates at that many sigma.
    //
    // Ye(r) remains PREM for every perturbation.
    struct DensityPerturbation {
        PerturbationShape shape = PerturbationShape::box;
        PerturbationMode  mode  = PerturbationMode::relative;

        Real_t center_radius_km = 0.0;
        Real_t half_width_km    = 0.0;
        Real_t sigma_km         = 0.0;
        Real_t amplitude        = 0.0;

        Real_t edge_width_km         = 0.0;
        Real_t gaussian_cutoff_sigma = 0.0;

        [[nodiscard]] Real_t weight(Real_t radius_km) const noexcept;
    };

    struct PropagationOptions {
        bool    interactions = true;
        Index_t threads      = 1;
        Real_t  h_max_km     = 500.0;
    };

    [[nodiscard]] EarthProfile load_prem(const std::filesystem::path& filename = "data/PREM/EARTH_MODEL_PREM.dat");

    // Physical model parameterization used by the analysis:
    //
    //     rho_i = q_i * <rho_PREM>_i
    //
    // where <rho_PREM>_i is the volume-weighted PREM mean density in the
    // corresponding spherical shell.
    //
    // 3 layers:
    //     [0, 3480], [3480, 5700], [5700, 6371] km
    //
    // 5 layers:
    //     [0, 1221], [1221, 3480], [3480, 4811],
    //     [4811, 5700], [5700, 6371] km
    [[nodiscard]] LayeredEarth make_layered_constant_3(const EarthProfile&        prem,
                                                       nda::View<const Real_t, 1> density_factor);

    [[nodiscard]] LayeredEarth make_layered_constant_5(const EarthProfile&        prem,
                                                       nda::View<const Real_t, 1> density_factor);

    [[nodiscard]] PremScaledEarth make_prem_scaled_3(nda::View<const Real_t, 1> density_factor);

    [[nodiscard]] PremScaledEarth make_prem_scaled_5(nda::View<const Real_t, 1> density_factor);

    [[nodiscard]] DensityPerturbation make_relative_box_perturbation(Real_t center_radius_km, Real_t width_km,
                                                                     Real_t relative_change,
                                                                     Real_t edge_width_km = 0.0);

    [[nodiscard]] DensityPerturbation make_absolute_box_perturbation(Real_t center_radius_km, Real_t width_km,
                                                                     Real_t density_change_g_cm3,
                                                                     Real_t edge_width_km = 0.0);

    [[nodiscard]] DensityPerturbation make_relative_gaussian_perturbation(Real_t center_radius_km, Real_t sigma_km,
                                                                          Real_t relative_peak,
                                                                          Real_t cutoff_sigma = 0.0);

    [[nodiscard]] DensityPerturbation make_absolute_gaussian_perturbation(Real_t center_radius_km, Real_t sigma_km,
                                                                          Real_t density_peak_g_cm3,
                                                                          Real_t cutoff_sigma = 0.0);

    // Spherically symmetric bulk properties over a radial shell:
    //
    //     inner_radius_km <= r <= outer_radius_km
    //
    // Mass is returned in kg. Moment of inertia is taken about any diameter
    // through the Earth center and returned in kg m^2. Mean density is returned
    // in g/cm^3.
    //
    // PREM-based quantities use the same piecewise-linear interpolation between
    // tabulated PREM nodes as the layered-constant model construction.
    //
    // Plain PREM.
    [[nodiscard]] Real_t mass_kg(const EarthProfile& prem, Real_t inner_radius_km, Real_t outer_radius_km);

    [[nodiscard]] Real_t moment_of_inertia_kg_m2(const EarthProfile& prem, Real_t inner_radius_km,
                                                 Real_t outer_radius_km);

    [[nodiscard]] Real_t mean_density_g_cm3(const EarthProfile& prem, Real_t inner_radius_km, Real_t outer_radius_km);

    // Layered constant-density Earth.
    [[nodiscard]] Real_t mass_kg(const LayeredEarth& earth, Real_t inner_radius_km, Real_t outer_radius_km);

    [[nodiscard]] Real_t moment_of_inertia_kg_m2(const LayeredEarth& earth, Real_t inner_radius_km,
                                                 Real_t outer_radius_km);

    [[nodiscard]] Real_t mean_density_g_cm3(const LayeredEarth& earth, Real_t inner_radius_km, Real_t outer_radius_km);

    // Piecewise PREM density scaling.
    [[nodiscard]] Real_t mass_kg(const EarthProfile& prem, const PremScaledEarth& earth, Real_t inner_radius_km,
                                 Real_t outer_radius_km);

    [[nodiscard]] Real_t moment_of_inertia_kg_m2(const EarthProfile& prem, const PremScaledEarth& earth,
                                                 Real_t inner_radius_km, Real_t outer_radius_km);

    [[nodiscard]] Real_t mean_density_g_cm3(const EarthProfile& prem, const PremScaledEarth& earth,
                                            Real_t inner_radius_km, Real_t outer_radius_km);

    // Plain PREM.
    [[nodiscard]] Flux propagate_flux(const Flux& initial, const EarthProfile& prem,
                                      const PropagationOptions& options = {});

    // Layered constant-density rho with PREM Ye(r).
    [[nodiscard]] Flux propagate_flux(const Flux& initial, const EarthProfile& prem, const LayeredEarth& earth,
                                      const PropagationOptions& options = {});

    // Piecewise PREM density scaling with PREM Ye(r).
    [[nodiscard]] Flux propagate_flux(const Flux& initial, const EarthProfile& prem, const PremScaledEarth& earth,
                                      const PropagationOptions& options = {});

    // Local PREM perturbation with PREM Ye(r).
    [[nodiscard]] Flux propagate_flux(const Flux& initial, const EarthProfile& prem,
                                      const DensityPerturbation& perturbation, const PropagationOptions& options = {});

} // namespace nt
