#include <nt/earth.hpp>
#include <nt/flux.hpp>
#include <nt/types.hpp>

#include <SQuIDS/const.h>
#include <gsl/gsl_odeiv2.h>
#include <nuSQuIDS/body.h>
#include <nuSQuIDS/marray.h>
#include <nuSQuIDS/nuSQuIDS.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

    using nt::Flux;
    using nt::Index_t;
    using nt::LayeredEarth;
    using nt::PropagationOptions;
    using nt::Real_t;

    namespace nda = nt::nda;

    const squids::Const units;

    constexpr Real_t eps = 1e-10;

    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    void require_close(Real_t actual, Real_t expected, Real_t tolerance, const char* message) {
        const Real_t scale = std::max({Real_t{1}, std::abs(actual), std::abs(expected)});

        const Real_t error = std::abs(actual - expected) / scale;

        if (error > tolerance) {
            throw std::runtime_error(
                std::string(message) + ": actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) +
                ", scaled_error=" + std::to_string(error) + ", tolerance=" + std::to_string(tolerance));
        }
    }

    template <std::size_t N> auto view(const std::array<Real_t, N>& a) {
        return nda::make_view1d(static_cast<const Real_t*>(a.data()), a.size());
    }

    void compare_flux(const Flux& actual, const Flux& expected, Real_t tolerance, const char* message) {
        require(actual.n_coszenith() == expected.n_coszenith(), "Flux coszenith dimensions differ");

        require(actual.n_energy() == expected.n_energy(), "Flux energy dimensions differ");

        const auto& a = actual.native_state();
        const auto& b = expected.native_state();

        Real_t max_error      = 0.0;
        Real_t worst_actual   = 0.0;
        Real_t worst_expected = 0.0;

        Index_t worst_z = 0;
        Index_t worst_e = 0;
        Index_t worst_p = 0;
        Index_t worst_f = 0;

        for (Index_t z = 0; z < actual.n_coszenith(); ++z) {
            for (Index_t e = 0; e < actual.n_energy(); ++e) {
                for (Index_t p = 0; p < 2; ++p) {
                    for (Index_t f = 0; f < 3; ++f) {
                        const Real_t va = a[z][e][p][f];
                        const Real_t vb = b[z][e][p][f];

                        const Real_t scale = std::max({Real_t{1}, std::abs(va), std::abs(vb)});

                        const Real_t error = std::abs(va - vb) / scale;

                        if (error > max_error) {
                            max_error      = error;
                            worst_actual   = va;
                            worst_expected = vb;

                            worst_z = z;
                            worst_e = e;
                            worst_p = p;
                            worst_f = f;
                        }
                    }
                }
            }
        }

        if (max_error > tolerance) {
            throw std::runtime_error(
                std::string(message) + ": max_scaled_error=" + std::to_string(max_error) +
                ", tolerance=" + std::to_string(tolerance) + ", actual=" + std::to_string(worst_actual) +
                ", expected=" + std::to_string(worst_expected) + ", index=[" + std::to_string(worst_z) + "," +
                std::to_string(worst_e) + "," + std::to_string(worst_p) + "," + std::to_string(worst_f) + "]");
        }
    }

    Flux make_test_flux() {
        Flux flux(3, 5);

        flux.coszenith()(0) = -1.0;
        flux.coszenith()(1) = -0.75;
        flux.coszenith()(2) = -0.35;

        flux.energy_gev()(0) = 5.0;
        flux.energy_gev()(1) = 10.0;
        flux.energy_gev()(2) = 20.0;
        flux.energy_gev()(3) = 40.0;
        flux.energy_gev()(4) = 80.0;

        auto& state = flux.native_state();

        for (Index_t z = 0; z < 3; ++z) {
            for (Index_t e = 0; e < 5; ++e) {
                for (Index_t p = 0; p < 2; ++p) {
                    for (Index_t f = 0; f < 3; ++f) {
                        state[z][e][p][f] =
                            Real_t{0.2} + Real_t{0.3} * z + Real_t{0.1} * e + Real_t{0.4} * p + Real_t{0.25} * f;
                    }
                }
            }
        }

        return flux;
    }

    Flux make_vertical_test_flux() {
        Flux flux(1, 5);

        flux.coszenith()(0) = -1.0;

        flux.energy_gev()(0) = 5.0;
        flux.energy_gev()(1) = 10.0;
        flux.energy_gev()(2) = 20.0;
        flux.energy_gev()(3) = 40.0;
        flux.energy_gev()(4) = 80.0;

        auto& state = flux.native_state();

        for (Index_t e = 0; e < 5; ++e) {
            for (Index_t p = 0; p < 2; ++p) {
                for (Index_t f = 0; f < 3; ++f) {
                    state[0][e][p][f] = Real_t{0.3} + Real_t{0.13} * e + Real_t{0.37} * p + Real_t{0.19} * f;
                }
            }
        }

        return flux;
    }

    void configure_reference_solver(nusquids::nuSQUIDS& nus, Real_t h_max_km) {
        nus.Set_MixingAngle(0, 1, 0.563942);
        nus.Set_MixingAngle(0, 2, 0.154085);
        nus.Set_MixingAngle(1, 2, 0.785398);

        nus.Set_SquareMassDifference(1, 7.65e-5);
        nus.Set_SquareMassDifference(2, 2.47e-3);
        nus.Set_CPPhase(0, 2, 0.0);

        // Reference calculation is intentionally tighter than the production
        // path. Production settings themselves are not changed.
        nus.Set_h_max(h_max_km * units.km);
        nus.Set_GSL_step(gsl_odeiv2_step_rk4);
        nus.Set_rel_error(1e-8);
        nus.Set_abs_error(1e-8);
    }

    // Independent reference for a vertical layered Earth.
    //
    // Ye is fixed to 0.5 only in this test so the exact same matter profile
    // can be expressed with nuSQuIDS' native ConstantDensity body.
    //
    // The vertical path is explicitly split at every spherical boundary:
    //
    //     surface -> center -> surface
    //
    // This gives an implementation independent of our LayeredEarthAtm
    // density callback.
    Flux propagate_vertical_constant_reference(const Flux& initial, const LayeredEarth& earth) {
        assert(initial.n_coszenith() == 1);
        assert(initial.coszenith()(0) == -1.0);

        Flux::Axis energy_ev({initial.n_energy()});

        for (Index_t e = 0; e < initial.n_energy(); ++e) {
            energy_ev[e] = initial.energy_gev()(e) * units.GeV;
        }

        nusquids::nuSQUIDS nus(energy_ev, 3, nusquids::both, false);

        configure_reference_solver(nus, 25.0);

        nusquids::marray<double, 3> state({initial.n_energy(), 2, 3});

        for (Index_t e = 0; e < initial.n_energy(); ++e) {
            for (Index_t p = 0; p < 2; ++p) {
                for (Index_t f = 0; f < 3; ++f) {
                    state[e][p][f] = initial.native_state()[0][e][p][f];
                }
            }
        }

        bool initialized = false;

        auto evolve_segment = [&](Index_t layer, Real_t length_km) {
            auto body = std::make_shared<nusquids::ConstantDensity>(earth.density_g_cm3[layer], 0.5);

            auto track = std::make_shared<nusquids::ConstantDensity::Track>(length_km * units.km);

            nus.Set_Body(body);
            nus.Set_Track(track);

            if (!initialized) {
                nus.Set_initial_state(state, nusquids::flavor);
                initialized = true;
            }

            nus.EvolveState();
        };

        // Surface -> central layer.
        for (Index_t i = earth.layers - 1; i > 0; --i) {
            const Real_t thickness = earth.outer_radius_km[i] - earth.outer_radius_km[i - 1];

            evolve_segment(i, thickness);
        }

        // The central shell is traversed from one side to the other.
        evolve_segment(0, Real_t{2} * earth.outer_radius_km[0]);

        // Central layer -> opposite surface.
        for (Index_t i = 1; i < earth.layers; ++i) {
            const Real_t thickness = earth.outer_radius_km[i] - earth.outer_radius_km[i - 1];

            evolve_segment(i, thickness);
        }

        Flux result(1, initial.n_energy());

        result.coszenith()(0) = -1.0;

        for (Index_t e = 0; e < initial.n_energy(); ++e)
            result.energy_gev()(e) = initial.energy_gev()(e);

        auto& output = result.native_state();

        for (Index_t e = 0; e < initial.n_energy(); ++e) {
            for (Index_t p = 0; p < 2; ++p) {
                for (Index_t f = 0; f < 3; ++f) {
                    output[0][e][p][f] = nus.EvalFlavorAtNode(
                        static_cast<unsigned int>(f), static_cast<unsigned int>(e), static_cast<unsigned int>(p));
                }
            }
        }

        return result;
    }

    void test_prem_loader() {
        const auto prem = nt::load_prem();

        require(prem.radius_fraction.extent(0) == 201, "PREM point count is wrong");

        require_close(prem.radius_fraction(0), 0.0, eps, "wrong PREM center radius");

        require_close(prem.density_g_cm3(0), 13.0885, 1e-12, "wrong PREM center density");

        require_close(prem.ye(0), 0.4656, 1e-12, "wrong PREM core Ye");

        require_close(prem.radius_fraction(109), 0.545, 1e-12, "wrong PREM radius near CMB");

        require_close(prem.ye(109), 0.4656, 1e-12, "wrong PREM core Ye near CMB");

        require_close(prem.radius_fraction(110), 0.55, 1e-12, "wrong PREM mantle radius");

        require_close(prem.ye(110), 0.4957, 1e-12, "wrong PREM mantle Ye");

        require_close(prem.radius_fraction(200), 1.0, eps, "wrong PREM surface radius");

        require_close(prem.density_g_cm3(200), 1.02, 1e-12, "wrong PREM surface density");

        require_close(prem.ye(200), 0.4957, 1e-12, "wrong PREM surface Ye");

        std::cout << "[PASS] PREM loader and Ye\n";
    }

    void test_model_factories() {
        const auto prem = nt::load_prem();

        const std::array<Real_t, 3> one3 = {
            1.0,
            1.0,
            1.0,
        };

        const std::array<Real_t, 3> two3 = {
            2.0,
            2.0,
            2.0,
        };

        const auto constant_one = nt::make_layered_constant_3(prem, view(one3));

        const auto constant_two = nt::make_layered_constant_3(prem, view(two3));

        require(constant_one.layers == 3, "wrong 3-layer model size");

        require_close(constant_one.outer_radius_km[0], 3480.0, 0.0, "wrong 3-layer boundary");

        require_close(constant_one.outer_radius_km[1], 5700.0, 0.0, "wrong 3-layer boundary");

        require_close(constant_one.outer_radius_km[2], 6371.0, 0.0, "wrong 3-layer boundary");

        for (Index_t i = 0; i < 3; ++i) {
            require_close(constant_two.density_g_cm3[i], Real_t{2} * constant_one.density_g_cm3[i], 1e-13,
                          "constant-density q scaling is wrong");
        }

        const std::array<Real_t, 5> q5 = {
            1.1, 0.9, 1.2, 0.8, 1.05,
        };

        const auto constant5 = nt::make_layered_constant_5(prem, view(q5));

        require(constant5.layers == 5, "wrong 5-layer model size");

        require_close(constant5.outer_radius_km[0], 1221.0, 0.0, "wrong 5-layer boundary");

        require_close(constant5.outer_radius_km[4], 6371.0, 0.0, "wrong 5-layer boundary");

        const auto scaled5 = nt::make_prem_scaled_5(view(q5));

        for (Index_t i = 0; i < 5; ++i) {
            require_close(scaled5.density_factor[i], q5[i], 0.0, "PREM scaling factor is wrong");
        }

        std::cout << "[PASS] Earth model factories\n";
    }

    void test_bulk_properties() {
        const auto prem = nt::load_prem();

        const std::array<Real_t, 3> unity3 = {
            1.0,
            1.0,
            1.0,
        };

        const std::array<Real_t, 5> unity5 = {
            1.0, 1.0, 1.0, 1.0, 1.0,
        };

        const auto constant3 = nt::make_layered_constant_3(prem, view(unity3));

        const auto constant5 = nt::make_layered_constant_5(prem, view(unity5));

        const auto scaled3 = nt::make_prem_scaled_3(view(unity3));

        const auto scaled5 = nt::make_prem_scaled_5(view(unity5));

        const Real_t prem_mass = nt::mass_kg(prem, 0.0, 6371.0);

        const Real_t prem_inertia = nt::moment_of_inertia_kg_m2(prem, 0.0, 6371.0);

        // Unity PREM scaling must reproduce the complete PREM integrals,
        // independent of whether the 3-layer or 5-layer partition is used.
        require_close(nt::mass_kg(prem, scaled3, 0.0, 6371.0), prem_mass, 1e-12,
                      "3-layer unity PREM scaling changed Earth mass");

        require_close(nt::mass_kg(prem, scaled5, 0.0, 6371.0), prem_mass, 1e-12,
                      "5-layer unity PREM scaling changed Earth mass");

        require_close(nt::moment_of_inertia_kg_m2(prem, scaled3, 0.0, 6371.0), prem_inertia, 1e-12,
                      "3-layer unity PREM scaling changed Earth inertia");

        require_close(nt::moment_of_inertia_kg_m2(prem, scaled5, 0.0, 6371.0), prem_inertia, 1e-12,
                      "5-layer unity PREM scaling changed Earth inertia");

        // The unity layered-constant models preserve PREM mass because each
        // constant density is defined as the PREM volume-weighted shell mean.
        require_close(nt::mass_kg(constant3, 0.0, 6371.0), prem_mass, 1e-12,
                      "3-layer constant model changed Earth mass");

        require_close(nt::mass_kg(constant5, 0.0, 6371.0), prem_mass, 1e-12,
                      "5-layer constant model changed Earth mass");

        // A constant-density shell must report exactly its stored density as
        // its volume-averaged density.
        require_close(nt::mean_density_g_cm3(constant3, 0.0, 3480.0), constant3.density_g_cm3[0], 1e-13,
                      "constant shell mean density is wrong");

        // Queries may cross model boundaries; the result must remain additive.
        require_close(nt::mass_kg(constant3, 0.0, 5700.0),
                      nt::mass_kg(constant3, 0.0, 3480.0) + nt::mass_kg(constant3, 3480.0, 5700.0), 1e-13,
                      "cross-layer mass is not additive");

        std::cout << "[PASS] Earth bulk properties\n";
    }

    void test_perturbation_profiles() {
        const auto box = nt::make_relative_box_perturbation(3000.0, 400.0, 0.1);

        require_close(box.weight(3000.0), 1.0, 0.0, "box center weight is wrong");

        require_close(box.weight(3199.0), 1.0, 0.0, "box interior weight is wrong");

        require_close(box.weight(3201.0), 0.0, 0.0, "box exterior weight is wrong");

        const auto gaussian = nt::make_absolute_gaussian_perturbation(3000.0, 100.0, 0.5);

        require_close(gaussian.weight(3000.0), 1.0, 1e-14, "Gaussian center weight is wrong");

        require_close(gaussian.weight(3100.0), std::exp(-0.5), 1e-14, "Gaussian sigma weight is wrong");

        const auto truncated = nt::make_absolute_gaussian_perturbation(3000.0, 100.0, 0.5, 3.0);

        require_close(truncated.weight(3301.0), 0.0, 0.0, "Gaussian cutoff is wrong");

        std::cout << "[PASS] density perturbation profiles\n";
    }

    void test_unity_prem_scaling() {
        const auto prem    = nt::load_prem();
        const auto initial = make_test_flux();

        const std::array<Real_t, 3> unity = {
            1.0,
            1.0,
            1.0,
        };

        const auto scaled = nt::make_prem_scaled_3(view(unity));

        PropagationOptions options;
        options.interactions = false;
        options.threads      = 1;
        options.h_max_km     = 500.0;

        const auto reference = nt::propagate_flux(initial, prem, options);

        const auto result = nt::propagate_flux(initial, prem, scaled, options);

        // Both profiles are mathematically identical, but they reach the
        // PREM spline through slightly different floating-point paths and the
        // adaptive solver itself targets 1e-6 accuracy.
        compare_flux(result, reference, 2e-5, "unity PREM scaling changed propagation");

        std::cout << "[PASS] unity PREM scaling\n";
    }

    void test_prem_scaled_3_5_equivalence() {
        const auto prem    = nt::load_prem();
        const auto initial = make_test_flux();

        const std::array<Real_t, 3> q3 = {
            1.12,
            0.91,
            1.07,
        };

        // The 3-layer profile represented on the 5-layer partition:
        //
        // core         -> q0, q0
        // lower mantle -> q1, q1
        // upper mantle -> q2
        const std::array<Real_t, 5> q5 = {
            q3[0], q3[0], q3[1], q3[1], q3[2],
        };

        const auto earth3 = nt::make_prem_scaled_3(view(q3));

        const auto earth5 = nt::make_prem_scaled_5(view(q5));

        PropagationOptions options;
        options.interactions = false;
        options.threads      = 1;
        options.h_max_km     = 500.0;

        const auto result3 = nt::propagate_flux(initial, prem, earth3, options);

        const auto result5 = nt::propagate_flux(initial, prem, earth5, options);

        compare_flux(result3, result5, 2e-5, "3/5-layer PREM scaling is inconsistent");

        std::cout << "[PASS] 3/5-layer PREM-scaled equivalence\n";
    }

    void test_layered_constant_against_native_reference() {
        auto prem = nt::load_prem();

        // Only for this independent reference test.
        //
        // Setting Ye=0.5 makes our piecewise-constant rho model exactly
        // representable using nuSQuIDS' native ConstantDensity body.
        for (Index_t i = 0; i < prem.ye.extent(0); ++i)
            prem.ye(i) = 0.5;

        const auto initial = make_vertical_test_flux();

        PropagationOptions options;
        options.interactions = false;
        options.threads      = 1;
        options.h_max_km     = 500.0;

        {
            const std::array<Real_t, 3> q = {
                1.15,
                0.90,
                1.08,
            };

            const auto earth = nt::make_layered_constant_3(prem, view(q));

            const auto result = nt::propagate_flux(initial, prem, earth, options);

            const auto reference = propagate_vertical_constant_reference(initial, earth);

            compare_flux(result, reference, 5e-4,
                         "3-layer constant propagation disagrees "
                         "with native ConstantDensity reference");
        }

        {
            const std::array<Real_t, 5> q = {
                1.08, 0.93, 1.11, 0.88, 1.04,
            };

            const auto earth = nt::make_layered_constant_5(prem, view(q));

            const auto result = nt::propagate_flux(initial, prem, earth, options);

            const auto reference = propagate_vertical_constant_reference(initial, earth);

            compare_flux(result, reference, 5e-4,
                         "5-layer constant propagation disagrees "
                         "with native ConstantDensity reference");
        }

        std::cout << "[PASS] layered constant vs native "
                     "ConstantDensity reference\n";
    }

    void test_layered_step_convergence() {
        const auto prem    = nt::load_prem();
        const auto initial = make_test_flux();

        const std::array<Real_t, 5> q = {
            1.17, 0.86, 1.14, 0.89, 1.06,
        };

        const auto earth = nt::make_layered_constant_5(prem, view(q));

        PropagationOptions coarse;
        coarse.interactions = false;
        coarse.threads      = 1;
        coarse.h_max_km     = 500.0;

        PropagationOptions fine = coarse;
        fine.h_max_km           = 50.0;

        const auto result_coarse = nt::propagate_flux(initial, prem, earth, coarse);

        const auto result_fine = nt::propagate_flux(initial, prem, earth, fine);

        // This is deliberately tighter than any physics-level uncertainty, but
        // loose enough to reflect an adaptive solver configured at 1e-6.
        //
        // If this fails with a substantially larger error, do not simply loosen
        // the tolerance: inspect the reported maximum error first.
        compare_flux(result_coarse, result_fine, 5e-4,
                     "500 km h_max has not converged for "
                     "layered constant Earth");

        std::cout << "[PASS] layered boundary convergence\n";
    }

    void test_zero_perturbation() {
        const auto prem    = nt::load_prem();
        const auto initial = make_test_flux();

        PropagationOptions options;
        options.interactions = false;
        options.threads      = 1;
        options.h_max_km     = 500.0;

        const auto reference = nt::propagate_flux(initial, prem, options);

        const auto relative_box = nt::make_relative_box_perturbation(3000.0, 400.0, 0.0);

        const auto absolute_gaussian = nt::make_absolute_gaussian_perturbation(3000.0, 150.0, 0.0);

        const auto box_result = nt::propagate_flux(initial, prem, relative_box, options);

        const auto gaussian_result = nt::propagate_flux(initial, prem, absolute_gaussian, options);

        compare_flux(box_result, reference, 2e-5, "zero relative box changed PREM");

        compare_flux(gaussian_result, reference, 2e-5, "zero absolute Gaussian changed PREM");

        std::cout << "[PASS] zero density perturbation\n";
    }

} // namespace

int main() {
    try {
        test_prem_loader();
        test_model_factories();
        test_bulk_properties();
        test_perturbation_profiles();

        test_unity_prem_scaling();
        test_prem_scaled_3_5_equivalence();

        test_layered_constant_against_native_reference();
        test_layered_step_convergence();

        test_zero_perturbation();

        std::cout << "[PASS] all Earth/propagation tests\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << '\n';

        return 1;
    }
}
