#pragma once

#include <filesystem>

#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

namespace nt {

    namespace nda = vehkko::ndarray;

    // TRIDENT detector response as supplied by the collaboration.
    //
    // detector_response:
    //     [true_energy, coszenith]
    //
    // energy_migration:
    //     [true_energy, reconstructed_energy]
    //
    // The migration matrix is stored in the same orientation as the CSV:
    //
    //     P(E_rec | E_true)
    //
    // No pre-contracted 3D [reco, coszenith, true] array is stored. The
    // contraction is performed later when event counts are evaluated.
    struct ResponseArray {
        nda::Array<Real_t, 1> true_energy_gev;
        nda::Array<Real_t, 1> coszenith;
        nda::Array<Real_t, 1> reco_energy_gev;

        nda::Array<Real_t, 2> detector_response; // [true_energy, coszenith]
        nda::Array<Real_t, 2> energy_migration;  // [true_energy, reco_energy]

        ResponseArray(Index_t n_true, Index_t n_coszenith, Index_t n_reco)
            : true_energy_gev({n_true}), coszenith({n_coszenith}), reco_energy_gev({n_reco}),
              detector_response({n_true, n_coszenith}), energy_migration({n_true, n_reco}) {}
    };

    // Read the final TRIDENT response files:
    //
    //   <directory>/TRIDENT_response_array_20x34.csv
    //   <directory>/energy_response_20x20_v2.csv
    //
    // Expected response CSV:
    //   - header: first label column + 34 "cos_<value>" columns
    //   - 20 rows
    //   - first field of each row: "logE_<log10(E/GeV)>"
    //
    // Expected migration CSV:
    //   - header: first label column + 20 reconstructed-energy centers
    //   - 20 rows
    //   - first field of each row: true-energy center in GeV
    //
    // Values are parsed directly into the final NDA arrays. No vector<vector>,
    // stringstream table, matrix transpose, or intermediate numeric array is used.
    [[nodiscard]] ResponseArray load_trident_response(const std::filesystem::path& directory = "data/trident");

} // namespace nt
