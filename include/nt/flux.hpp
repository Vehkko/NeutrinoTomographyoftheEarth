#pragma once

#include <filesystem>
#include <string_view>

#include <nt/types.hpp>
#include <nuSQuIDS/marray.h>
#include <vndarray/ndarray.hpp>

namespace nt {

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
    [[nodiscard]] Flux load_daemonflux(const std::filesystem::path& filename, std::string_view location);

    // Resample the complete six-component flux state onto a new grid.
    //
    // Interpolation is linear in coszenith and E_GeV. Boundary points retain
    // the established linear-extrapolation behavior.
    [[nodiscard]] Flux resample_flux(const Flux& source, nda::View<const Real_t, 1> coszenith,
                                     nda::View<const Real_t, 1> energy_gev);

    // Generate uniformly spaced midpoint samples inside each coszenith bin.
    // Samples belonging to one bin are contiguous.
    [[nodiscard]] nda::Array<Real_t, 1> sample_coszenith_bin_midpoints(nda::View<const Real_t, 1> bin_edges,
                                                                       Index_t                    samples_per_bin);

    // Generate midpoint samples uniformly in log10(E/GeV) inside each energy bin.
    // Samples belonging to one bin are contiguous.
    [[nodiscard]] nda::Array<Real_t, 1> sample_log_energy_bin_midpoints(nda::View<const Real_t, 1> bin_edges,
                                                                        Index_t                    samples_per_bin);

    // Fine-grid -> coarse-grid treatment for each axis. The default is arithmetic
    // averaging in both directions. Interpolation uses all fine-grid nodes and is
    // piecewise linear in coszenith or E_GeV respectively.
    struct FluxRebinOptions {
        bool interpolate_coszenith = false;
        bool interpolate_energy    = false;
    };

    // Rebin a regularly refined Flux back to the target grid.
    //
    // The fine-grid ordering must be produced by the midpoint samplers above:
    // consecutive samples belong to the same target bin. Average and interpolation
    // can be selected independently for coszenith and energy. Interpolation follows
    // numpy.interp endpoint behavior and therefore clamps outside the fine-grid range.
    [[nodiscard]] Flux rebin_flux(const Flux& fine_flux, nda::View<const Real_t, 1> target_coszenith,
                                  nda::View<const Real_t, 1> target_energy_gev, Index_t coszenith_samples_per_bin,
                                  Index_t energy_samples_per_bin, const FluxRebinOptions& options = {});

} // namespace nt
