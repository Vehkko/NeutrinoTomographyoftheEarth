#include <nt/earth.hpp>
#include <nt/flux.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

#include <SQuIDS/const.h>
#include <gsl/gsl_odeiv2.h>
#include <nuSQuIDS/body.h>
#include <nuSQuIDS/nuSQuIDS.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nt {

    namespace {

        constexpr Index_t     prem_points     = 201;
        constexpr Real_t      earth_radius_km = 6371.0;
        constexpr long double pi              = 3.141592653589793238462643383279502884L;

        constexpr long double km_to_m        = 1000.0L;
        constexpr long double g_cm3_to_kg_m3 = 1000.0L;

        constexpr std::array<Real_t, 3> outer_radius_3 = {
            3480.0,
            5700.0,
            6371.0,
        };

        constexpr std::array<Real_t, 5> outer_radius_5 = {
            1221.0, 3480.0, 4811.0, 5700.0, 6371.0,
        };

        const squids::Const units;

        template <typename Array> std::vector<double> to_vector(const Array& a) {
            return std::vector<double>(a.data(), a.data() + a.extent(0));
        }

        long double cube(long double x) noexcept { return x * x * x; }

        long double fourth(long double x) noexcept {
            const long double x2 = x * x;
            return x2 * x2;
        }

        long double fifth(long double x) noexcept {
            const long double x2 = x * x;
            return x2 * x2 * x;
        }

        long double sixth(long double x) noexcept {
            const long double x3 = x * x * x;
            return x3 * x3;
        }

        // PREM is treated as piecewise linear between file nodes, matching the
        // definition used previously when computing model-comparison mass
        // coefficients.
        //
        // Returns:
        //
        //     integral_a^b rho(x) x^2 dx
        //
        // with x = r / R_Earth.
        long double integrate_prem_rho_x2(const EarthProfile& prem, long double a, long double b) {
            long double total = 0.0L;

            for (Index_t i = 0; i + 1 < prem.radius_fraction.extent(0); ++i) {
                const long double x0 = prem.radius_fraction(i);
                const long double x1 = prem.radius_fraction(i + 1);

                const long double u = std::max(a, x0);
                const long double v = std::min(b, x1);

                if (!(v > u))
                    continue;

                const long double rho0 = prem.density_g_cm3(i);
                const long double rho1 = prem.density_g_cm3(i + 1);

                const long double slope     = (rho1 - rho0) / (x1 - x0);
                const long double intercept = rho0 - slope * x0;

                total += intercept * (cube(v) - cube(u)) / 3.0L;
                total += slope * (fourth(v) - fourth(u)) / 4.0L;
            }

            return total;
        }

        // Same piecewise-linear PREM convention as
        // integrate_prem_rho_x2().
        //
        // Returns:
        //
        //     integral_a^b rho(x) x^4 dx
        //
        // with x = r / R_Earth.
        long double integrate_prem_rho_x4(const EarthProfile& prem, long double a, long double b) {
            long double total = 0.0L;

            for (Index_t i = 0; i + 1 < prem.radius_fraction.extent(0); ++i) {
                const long double x0 = prem.radius_fraction(i);
                const long double x1 = prem.radius_fraction(i + 1);

                const long double u = std::max(a, x0);
                const long double v = std::min(b, x1);

                if (!(v > u))
                    continue;

                const long double rho0 = prem.density_g_cm3(i);
                const long double rho1 = prem.density_g_cm3(i + 1);

                const long double slope = (rho1 - rho0) / (x1 - x0);

                const long double intercept = rho0 - slope * x0;

                total += intercept * (fifth(v) - fifth(u)) / 5.0L;

                total += slope * (sixth(v) - sixth(u)) / 6.0L;
            }

            return total;
        }

        void validate_radius_range(Real_t inner_radius_km, Real_t outer_radius_km) {
            if (!(inner_radius_km >= 0.0 && inner_radius_km < outer_radius_km && outer_radius_km <= earth_radius_km)) {
                throw std::invalid_argument(
                    "Earth radial range must satisfy 0 <= inner_radius_km < outer_radius_km <= 6371");
            }
        }

        long double shell_volume_m3(Real_t inner_radius_km, Real_t outer_radius_km) noexcept {
            const long double inner_m = static_cast<long double>(inner_radius_km) * km_to_m;

            const long double outer_m = static_cast<long double>(outer_radius_km) * km_to_m;

            return 4.0L * pi / 3.0L * (cube(outer_m) - cube(inner_m));
        }

        Real_t average_prem_density(const EarthProfile& prem, Real_t inner_radius_km, Real_t outer_radius_km) {
            const long double a = static_cast<long double>(inner_radius_km / earth_radius_km);
            const long double b = static_cast<long double>(outer_radius_km / earth_radius_km);

            const long double integral      = integrate_prem_rho_x2(prem, a, b);
            const long double volume_weight = (cube(b) - cube(a)) / 3.0L;

            return static_cast<Real_t>(integral / volume_weight);
        }

        template <std::size_t N>
        LayeredEarth make_layered_constant_impl(const EarthProfile& prem, nda::View<const Real_t, 1> factor,
                                                const std::array<Real_t, N>& outer_radius) {
            assert(factor.extent(0) == N);

            LayeredEarth earth;
            earth.layers = N;

            Real_t inner_radius = 0.0;

            for (Index_t i = 0; i < N; ++i) {
                const Real_t outer   = outer_radius[i];
                const Real_t avg_rho = average_prem_density(prem, inner_radius, outer);

                earth.outer_radius_km[i] = outer;
                earth.density_g_cm3[i]   = factor(i) * avg_rho;

                inner_radius = outer;
            }

            return earth;
        }

        template <std::size_t N>
        PremScaledEarth make_prem_scaled_impl(nda::View<const Real_t, 1>   factor,
                                              const std::array<Real_t, N>& outer_radius) {
            assert(factor.extent(0) == N);

            PremScaledEarth earth;
            earth.layers = N;

            for (Index_t i = 0; i < N; ++i) {
                earth.outer_radius_km[i] = outer_radius[i];
                earth.density_factor[i]  = factor(i);
            }

            return earth;
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

        void validate_radius(Real_t radius_km) {
            if (!(radius_km >= 0.0 && radius_km <= earth_radius_km)) {
                throw std::invalid_argument("Earth radius must satisfy 0 <= radius_km <= 6371");
            }
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
                const Real_t s = (track.GetX() - track.GetInitialX()) / units.km;
                const Real_t L = (track.GetFinalX() - track.GetInitialX()) / units.km;

                const Real_t r2 = radius * radius + s * s - L * s;

                const Real_t r = std::sqrt(std::max(Real_t{0}, r2));

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

        std::shared_ptr<nusquids::EarthAtm> make_prem_body(const EarthProfile& prem) {
            auto earth = std::make_shared<nusquids::EarthAtm>(to_vector(prem.radius_fraction),
                                                              to_vector(prem.density_g_cm3), to_vector(prem.ye));

            earth->SetAtmosphereHeight(0.0);
            return earth;
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

        Flux propagate_with_body(const Flux& initial, std::shared_ptr<nusquids::EarthAtm> earth,
                                 const PropagationOptions& options) {
            Flux::Axis energy_ev({initial.n_energy()});

            const auto energy_gev = initial.energy_gev();

            for (Index_t e = 0; e < initial.n_energy(); ++e)
                energy_ev[e] = energy_gev(e) * units.GeV;

            nusquids::nuSQUIDSAtm<> nus(initial.native_coszenith(), energy_ev, 3, nusquids::both, options.interactions);

            nus.Set_EarthModel(std::move(earth));
            configure_solver(nus, options);

            // Flux already has nuSQuIDS' required native layout:
            //
            //     [coszenith, energy, particle, flavor]
            //
            // No project-side 4D repacking is performed.
            nus.Set_initial_state(initial.native_state(), nusquids::flavor);

            nus.EvolveState();

            Flux result(initial.n_coszenith(), initial.n_energy());

            auto result_z = result.coszenith();
            auto result_e = result.energy_gev();

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
                auto& nsq = nus.GetnuSQuIDS(static_cast<unsigned int>(z));

                for (Index_t e = 0; e < initial.n_energy(); ++e) {
                    for (Index_t p = 0; p < 2; ++p) {
                        for (Index_t f = 0; f < 3; ++f) {
                            state[z][e][p][f] =
                                nsq.EvalFlavorAtNode(static_cast<unsigned int>(f), static_cast<unsigned int>(e),
                                                     static_cast<unsigned int>(p));
                        }
                    }
                }
            }

            return result;
        }

    } // namespace

    Real_t DensityPerturbation::weight(Real_t radius_km) const noexcept {
        const Real_t dr = radius_km - center_radius_km;

        if (shape == PerturbationShape::box) {
            const Real_t left  = center_radius_km - half_width_km;
            const Real_t right = center_radius_km + half_width_km;

            if (edge_width_km <= 0.0)
                return radius_km >= left && radius_km <= right ? Real_t{1} : Real_t{0};

            const Real_t w = Real_t{0.5} * (std::tanh((radius_km - left) / edge_width_km) -
                                            std::tanh((radius_km - right) / edge_width_km));

            return std::clamp(w, Real_t{0}, Real_t{1});
        }

        const Real_t z = dr / sigma_km;

        if (gaussian_cutoff_sigma > 0.0 && std::abs(z) > gaussian_cutoff_sigma)
            return 0.0;

        return std::exp(-Real_t{0.5} * z * z);
    }

    EarthProfile load_prem(const std::filesystem::path& filename) {
        std::ifstream file(filename);

        if (!file)
            throw std::runtime_error("Cannot open PREM file: " + filename.string());

        EarthProfile prem(prem_points);

        for (Index_t i = 0; i < prem_points; ++i) {
            if (!(file >> prem.radius_fraction(i) >> prem.density_g_cm3(i) >> prem.ye(i))) {
                throw std::runtime_error("Cannot read PREM file: " + filename.string());
            }
        }

        return prem;
    }

    LayeredEarth make_layered_constant_3(const EarthProfile& prem, nda::View<const Real_t, 1> density_factor) {
        return make_layered_constant_impl(prem, density_factor, outer_radius_3);
    }

    LayeredEarth make_layered_constant_5(const EarthProfile& prem, nda::View<const Real_t, 1> density_factor) {
        return make_layered_constant_impl(prem, density_factor, outer_radius_5);
    }

    PremScaledEarth make_prem_scaled_3(nda::View<const Real_t, 1> density_factor) {
        return make_prem_scaled_impl(density_factor, outer_radius_3);
    }

    PremScaledEarth make_prem_scaled_5(nda::View<const Real_t, 1> density_factor) {
        return make_prem_scaled_impl(density_factor, outer_radius_5);
    }

    DensityPerturbation make_relative_box_perturbation(Real_t center_radius_km, Real_t width_km, Real_t relative_change,
                                                       Real_t edge_width_km) {
        DensityPerturbation p;
        p.shape            = PerturbationShape::box;
        p.mode             = PerturbationMode::relative;
        p.center_radius_km = center_radius_km;
        p.half_width_km    = Real_t{0.5} * width_km;
        p.amplitude        = relative_change;
        p.edge_width_km    = edge_width_km;
        return p;
    }

    DensityPerturbation make_absolute_box_perturbation(Real_t center_radius_km, Real_t width_km,
                                                       Real_t density_change_g_cm3, Real_t edge_width_km) {
        DensityPerturbation p;
        p.shape            = PerturbationShape::box;
        p.mode             = PerturbationMode::absolute;
        p.center_radius_km = center_radius_km;
        p.half_width_km    = Real_t{0.5} * width_km;
        p.amplitude        = density_change_g_cm3;
        p.edge_width_km    = edge_width_km;
        return p;
    }

    DensityPerturbation make_relative_gaussian_perturbation(Real_t center_radius_km, Real_t sigma_km,
                                                            Real_t relative_peak, Real_t cutoff_sigma) {
        DensityPerturbation p;
        p.shape                 = PerturbationShape::gaussian;
        p.mode                  = PerturbationMode::relative;
        p.center_radius_km      = center_radius_km;
        p.sigma_km              = sigma_km;
        p.amplitude             = relative_peak;
        p.gaussian_cutoff_sigma = cutoff_sigma;
        return p;
    }

    DensityPerturbation make_absolute_gaussian_perturbation(Real_t center_radius_km, Real_t sigma_km,
                                                            Real_t density_peak_g_cm3, Real_t cutoff_sigma) {
        DensityPerturbation p;
        p.shape                 = PerturbationShape::gaussian;
        p.mode                  = PerturbationMode::absolute;
        p.center_radius_km      = center_radius_km;
        p.sigma_km              = sigma_km;
        p.amplitude             = density_peak_g_cm3;
        p.gaussian_cutoff_sigma = cutoff_sigma;
        return p;
    }

    Real_t density_g_cm3(const EarthProfile& prem, Real_t radius_km) {
        validate_radius(radius_km);

        const Real_t x = radius_km / earth_radius_km;

        const Index_t n = prem.radius_fraction.extent(0);

        if (x <= prem.radius_fraction(0))
            return prem.density_g_cm3(0);

        if (x >= prem.radius_fraction(n - 1))
            return prem.density_g_cm3(n - 1);

        for (Index_t i = 0; i + 1 < n; ++i) {
            const Real_t x0 = prem.radius_fraction(i);

            const Real_t x1 = prem.radius_fraction(i + 1);

            if (x > x1)
                continue;

            const Real_t rho0 = prem.density_g_cm3(i);

            const Real_t rho1 = prem.density_g_cm3(i + 1);

            const Real_t t = (x - x0) / (x1 - x0);

            return rho0 + t * (rho1 - rho0);
        }

        return prem.density_g_cm3(n - 1);
    }

    Real_t density_g_cm3(const LayeredEarth& earth, Real_t radius_km) {
        validate_radius(radius_km);

        assert(earth.layers == 3 || earth.layers == 5);

        const Index_t i = layer_index(earth.layers, earth.outer_radius_km, radius_km);

        return earth.density_g_cm3[i];
    }

    Real_t density_g_cm3(const EarthProfile& prem, const PremScaledEarth& earth, Real_t radius_km) {
        validate_radius(radius_km);

        assert(earth.layers == 3 || earth.layers == 5);

        const Index_t i = layer_index(earth.layers, earth.outer_radius_km, radius_km);

        return density_g_cm3(prem, radius_km) * earth.density_factor[i];
    }

    Real_t mass_kg(const EarthProfile& prem, Real_t inner_radius_km, Real_t outer_radius_km) {
        validate_radius_range(inner_radius_km, outer_radius_km);

        const long double a = static_cast<long double>(inner_radius_km / earth_radius_km);

        const long double b = static_cast<long double>(outer_radius_km / earth_radius_km);

        const long double radius_m = static_cast<long double>(earth_radius_km) * km_to_m;

        const long double mass = 4.0L * pi * cube(radius_m) * g_cm3_to_kg_m3 * integrate_prem_rho_x2(prem, a, b);

        return static_cast<Real_t>(mass);
    }

    Real_t moment_of_inertia_kg_m2(const EarthProfile& prem, Real_t inner_radius_km, Real_t outer_radius_km) {
        validate_radius_range(inner_radius_km, outer_radius_km);

        const long double a = static_cast<long double>(inner_radius_km / earth_radius_km);

        const long double b = static_cast<long double>(outer_radius_km / earth_radius_km);

        const long double radius_m = static_cast<long double>(earth_radius_km) * km_to_m;

        const long double inertia =
            8.0L * pi / 3.0L * fifth(radius_m) * g_cm3_to_kg_m3 * integrate_prem_rho_x4(prem, a, b);

        return static_cast<Real_t>(inertia);
    }

    Real_t mean_density_g_cm3(const EarthProfile& prem, Real_t inner_radius_km, Real_t outer_radius_km) {
        validate_radius_range(inner_radius_km, outer_radius_km);

        const long double density_kg_m3 = static_cast<long double>(mass_kg(prem, inner_radius_km, outer_radius_km)) /
                                          shell_volume_m3(inner_radius_km, outer_radius_km);

        return static_cast<Real_t>(density_kg_m3 / g_cm3_to_kg_m3);
    }

    Real_t mass_kg(const LayeredEarth& earth, Real_t inner_radius_km, Real_t outer_radius_km) {
        validate_radius_range(inner_radius_km, outer_radius_km);

        assert(earth.layers == 3 || earth.layers == 5);

        long double total       = 0.0L;
        Real_t      layer_inner = 0.0;

        for (Index_t i = 0; i < earth.layers; ++i) {
            const Real_t layer_outer = earth.outer_radius_km[i];

            const Real_t a = std::max(inner_radius_km, layer_inner);

            const Real_t b = std::min(outer_radius_km, layer_outer);

            if (b > a) {
                const long double a_m = static_cast<long double>(a) * km_to_m;

                const long double b_m = static_cast<long double>(b) * km_to_m;

                const long double rho = static_cast<long double>(earth.density_g_cm3[i]) * g_cm3_to_kg_m3;

                total += 4.0L * pi / 3.0L * rho * (cube(b_m) - cube(a_m));
            }

            layer_inner = layer_outer;

            if (layer_inner >= outer_radius_km)
                break;
        }

        return static_cast<Real_t>(total);
    }

    Real_t moment_of_inertia_kg_m2(const LayeredEarth& earth, Real_t inner_radius_km, Real_t outer_radius_km) {
        validate_radius_range(inner_radius_km, outer_radius_km);

        assert(earth.layers == 3 || earth.layers == 5);

        long double total       = 0.0L;
        Real_t      layer_inner = 0.0;

        for (Index_t i = 0; i < earth.layers; ++i) {
            const Real_t layer_outer = earth.outer_radius_km[i];

            const Real_t a = std::max(inner_radius_km, layer_inner);

            const Real_t b = std::min(outer_radius_km, layer_outer);

            if (b > a) {
                const long double a_m = static_cast<long double>(a) * km_to_m;

                const long double b_m = static_cast<long double>(b) * km_to_m;

                const long double rho = static_cast<long double>(earth.density_g_cm3[i]) * g_cm3_to_kg_m3;

                total += 8.0L * pi / 15.0L * rho * (fifth(b_m) - fifth(a_m));
            }

            layer_inner = layer_outer;

            if (layer_inner >= outer_radius_km)
                break;
        }

        return static_cast<Real_t>(total);
    }

    Real_t mean_density_g_cm3(const LayeredEarth& earth, Real_t inner_radius_km, Real_t outer_radius_km) {
        validate_radius_range(inner_radius_km, outer_radius_km);

        const long double density_kg_m3 = static_cast<long double>(mass_kg(earth, inner_radius_km, outer_radius_km)) /
                                          shell_volume_m3(inner_radius_km, outer_radius_km);

        return static_cast<Real_t>(density_kg_m3 / g_cm3_to_kg_m3);
    }

    Real_t mass_kg(const EarthProfile& prem, const PremScaledEarth& earth, Real_t inner_radius_km,
                   Real_t outer_radius_km) {
        validate_radius_range(inner_radius_km, outer_radius_km);

        assert(earth.layers == 3 || earth.layers == 5);

        long double total       = 0.0L;
        Real_t      layer_inner = 0.0;

        for (Index_t i = 0; i < earth.layers; ++i) {
            const Real_t layer_outer = earth.outer_radius_km[i];

            const Real_t a = std::max(inner_radius_km, layer_inner);

            const Real_t b = std::min(outer_radius_km, layer_outer);

            if (b > a) {
                total += static_cast<long double>(earth.density_factor[i]) * mass_kg(prem, a, b);
            }

            layer_inner = layer_outer;

            if (layer_inner >= outer_radius_km)
                break;
        }

        return static_cast<Real_t>(total);
    }

    Real_t moment_of_inertia_kg_m2(const EarthProfile& prem, const PremScaledEarth& earth, Real_t inner_radius_km,
                                   Real_t outer_radius_km) {
        validate_radius_range(inner_radius_km, outer_radius_km);

        assert(earth.layers == 3 || earth.layers == 5);

        long double total       = 0.0L;
        Real_t      layer_inner = 0.0;

        for (Index_t i = 0; i < earth.layers; ++i) {
            const Real_t layer_outer = earth.outer_radius_km[i];

            const Real_t a = std::max(inner_radius_km, layer_inner);

            const Real_t b = std::min(outer_radius_km, layer_outer);

            if (b > a) {
                total += static_cast<long double>(earth.density_factor[i]) * moment_of_inertia_kg_m2(prem, a, b);
            }

            layer_inner = layer_outer;

            if (layer_inner >= outer_radius_km)
                break;
        }

        return static_cast<Real_t>(total);
    }

    Real_t mean_density_g_cm3(const EarthProfile& prem, const PremScaledEarth& earth, Real_t inner_radius_km,
                              Real_t outer_radius_km) {
        validate_radius_range(inner_radius_km, outer_radius_km);

        const long double density_kg_m3 =
            static_cast<long double>(mass_kg(prem, earth, inner_radius_km, outer_radius_km)) /
            shell_volume_m3(inner_radius_km, outer_radius_km);

        return static_cast<Real_t>(density_kg_m3 / g_cm3_to_kg_m3);
    }

    Flux propagate_flux(const Flux& initial, const EarthProfile& prem, const PropagationOptions& options) {
        return propagate_with_body(initial, make_prem_body(prem), options);
    }

    Flux propagate_flux(const Flux& initial, const EarthProfile& prem, const LayeredEarth& earth,
                        const PropagationOptions& options) {
        return propagate_with_body(initial, std::make_shared<LayeredEarthAtm>(prem, earth), options);
    }

    Flux propagate_flux(const Flux& initial, const EarthProfile& prem, const PremScaledEarth& earth,
                        const PropagationOptions& options) {
        return propagate_with_body(initial, std::make_shared<PremScaledEarthAtm>(prem, earth), options);
    }

    Flux propagate_flux(const Flux& initial, const EarthProfile& prem, const DensityPerturbation& perturbation,
                        const PropagationOptions& options) {
        return propagate_with_body(initial, std::make_shared<PerturbedPremEarthAtm>(prem, perturbation), options);
    }

} // namespace nt
