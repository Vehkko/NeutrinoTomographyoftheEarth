#pragma once

#include <cstddef>
#include <filesystem>
#include <string_view>

#include <nuSQuIDS/marray.h>
#include <vndarray/ndarray.hpp>

namespace nt {

    using Real_t  = double;
    using Index_t = std::size_t;

    namespace nsq = nusquids;
    namespace nda = vehkko::ndarray;

    enum class Particle : Index_t {
        neutrino     = 0,
        antineutrino = 1,
    };

    enum class Flavor : Index_t {
        electron = 0,
        muon     = 1,
        tau      = 2,
    };

    // Flux owns one nuSQuIDS-compatible state:
    //
    //     [coszenith, energy, particle, flavor]
    //
    // particle: 0 = neutrino, 1 = antineutrino
    // flavor:   0 = electron, 1 = muon, 2 = tau
    //
    // Public component views always use:
    //
    //     [coszenith, energy]
    //
    // and refer directly to the native state without copying.
    class Flux {
      public:
        using Axis        = nsq::marray<Real_t, 1>;
        using NativeState = nsq::marray<Real_t, 4>;

        Flux(Index_t n_coszenith, Index_t n_energy);

        Flux(const Flux&)                = delete;
        Flux& operator=(const Flux&)     = delete;
        Flux(Flux&&) noexcept            = default;
        Flux& operator=(Flux&&) noexcept = default;

        [[nodiscard]] Index_t n_coszenith() const noexcept;
        [[nodiscard]] Index_t n_energy() const noexcept;

        // Detector convention: -1 = vertically up-going, 0 = horizontal.
        [[nodiscard]] nda::View<Real_t, 1>       coszenith() noexcept;
        [[nodiscard]] nda::View<const Real_t, 1> coszenith() const noexcept;

        // Energy is stored in GeV throughout the tomography code.
        [[nodiscard]] nda::View<Real_t, 1>       energy_gev() noexcept;
        [[nodiscard]] nda::View<const Real_t, 1> energy_gev() const noexcept;

        // Zero-copy strided view [coszenith, energy].
        [[nodiscard]] nda::View<Real_t, 2>       component(Particle particle, Flavor flavor) noexcept;
        [[nodiscard]] nda::View<const Real_t, 2> component(Particle particle, Flavor flavor) const noexcept;

        // Frequent components used by the current analysis.
        [[nodiscard]] nda::View<Real_t, 2>       numu() noexcept;
        [[nodiscard]] nda::View<const Real_t, 2> numu() const noexcept;
        [[nodiscard]] nda::View<Real_t, 2>       antinumu() noexcept;
        [[nodiscard]] nda::View<const Real_t, 2> antinumu() const noexcept;

        // Native representation for the nuSQuIDS propagation backend.
        [[nodiscard]] NativeState&       native_state() noexcept;
        [[nodiscard]] const NativeState& native_state() const noexcept;

        // Native axes are exposed only for backend interoperation.
        [[nodiscard]] const Axis& native_coszenith() const noexcept;
        [[nodiscard]] const Axis& native_energy_gev() const noexcept;

      private:
        Axis        coszenith_;
        Axis        energy_gev_;
        NativeState state_;
    };

    // Load the project-generated DaemonFlux HDF5 table.
    //
    // Expected datasets:
    //
    //     /axes/coszenith
    //     /axes/energy_GeV
    //     /flux/<location>/calibrated/numu
    //     /flux/<location>/calibrated/antinumu
    //
    // Flux datasets are [coszenith, energy]. HDF5 writes them directly into
    // state[:, :, particle, muon] using a memory hyperslab; no temporary 2D
    // owning array is created.
    //
    // This function intentionally does not interpret legacy IceCube flux files.
    [[nodiscard]] Flux load_daemonflux(const std::filesystem::path& filename,
                                       std::string_view             location);

    // Resample the complete six-component flux state onto a new grid.
    //
    // Interpolation matches the legacy numerical convention:
    //     - linear in coszenith;
    //     - linear in E_GeV;
    //     - linear extrapolation from the nearest boundary interval.
    //
    // Source and target axes must be strictly increasing. Interpolation intervals
    // and weights are precomputed once; all six contiguous particle/flavor values
    // at each grid point are then processed together.
    [[nodiscard]] Flux resample_flux(const Flux&                source,
                                     nda::View<const Real_t, 1> coszenith,
                                     nda::View<const Real_t, 1> energy_gev);

} // namespace nt
