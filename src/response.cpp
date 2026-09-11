#include <nt/response.hpp>
#include <nt/types.hpp>

#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace nt {

    namespace {

        constexpr Index_t n_true = 20;
        constexpr Index_t n_cz   = 34;
        constexpr Index_t n_reco = 20;

        constexpr const char* response_filename  = "TRIDENT_response_array_20x34.csv";
        constexpr const char* migration_filename = "energy_response_20x20_v2.csv";

        constexpr Real_t trident_log_energy_edge_min   = 3.0;
        constexpr Real_t trident_log_energy_center_min = 3.05;
        constexpr Real_t trident_log_energy_step       = 0.1;

        constexpr std::array<Real_t, 4> trident_coszenith_segment_edges = {
            -1.0,
            -0.99,
            -0.9,
            0.0,
        };

        constexpr std::array<Index_t, 3> trident_coszenith_segment_bins = {
            4,
            10,
            20,
        };

        static_assert(trident_coszenith_segment_bins[0] + trident_coszenith_segment_bins[1] +
                          trident_coszenith_segment_bins[2] ==
                      n_cz);

        void set_trident_binning(ResponseArray& out) {
            for (Index_t i = 0; i <= n_true; ++i) {
                const Real_t energy = std::pow(Real_t{10}, trident_log_energy_edge_min +
                                                               trident_log_energy_step * static_cast<Real_t>(i));
                out.true_energy_edges_gev(i) = energy;
                out.reco_energy_edges_gev(i) = energy;
            }

            for (Index_t i = 0; i < n_true; ++i) {
                const Real_t energy    = std::pow(Real_t{10}, trident_log_energy_center_min +
                                                                  trident_log_energy_step * static_cast<Real_t>(i));
                out.true_energy_gev(i) = energy;
                out.reco_energy_gev(i) = energy;
            }

            Index_t bin = 0;

            for (Index_t segment = 0; segment < trident_coszenith_segment_bins.size(); ++segment) {
                const Real_t  low  = trident_coszenith_segment_edges[segment];
                const Real_t  high = trident_coszenith_segment_edges[segment + 1];
                const Index_t bins = trident_coszenith_segment_bins[segment];
                const Real_t  step = (high - low) / static_cast<Real_t>(bins);

                for (Index_t i = 0; i < bins; ++i) {
                    out.coszenith_edges(bin) = low + step * static_cast<Real_t>(i);
                    out.coszenith(bin)       = low + step * (static_cast<Real_t>(i) + Real_t{0.5});
                    ++bin;
                }
            }

            out.coszenith_edges(n_cz) = trident_coszenith_segment_edges.back();
        }

        std::string_view trim(std::string_view s) noexcept {
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r'))
                s.remove_prefix(1);
            while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
                s.remove_suffix(1);
            return s;
        }

        std::string_view next_cell(std::string_view& line) noexcept {
            const std::size_t      comma = line.find(',');
            const std::string_view cell  = comma == std::string_view::npos ? line : line.substr(0, comma);

            if (comma == std::string_view::npos)
                line = {};
            else
                line.remove_prefix(comma + 1);

            return trim(cell);
        }

        Real_t parse_real(std::string_view text, const std::filesystem::path& file, Index_t line, Index_t column) {
            text = trim(text);

            Real_t      value{};
            const char* begin    = text.data();
            const char* end      = begin + text.size();
            const auto [ptr, ec] = std::from_chars(begin, end, value, std::chars_format::general);

            if (ec != std::errc{} || ptr != end)
                throw std::runtime_error("Invalid numeric field '" + std::string(text) + "' in " + file.string() +
                                         " at line " + std::to_string(line) + ", column " + std::to_string(column));
            return value;
        }

        void read_detector_response(const std::filesystem::path& file, ResponseArray& out) {
            std::ifstream in(file);
            if (!in)
                throw std::runtime_error("Cannot open TRIDENT response file: " + file.string());

            std::string line;
            std::getline(in, line); // header

            // CSV storage convention:
            //
            //     row    = true energy
            //     column = coszenith
            for (Index_t row = 0; row < n_true; ++row) {
                std::getline(in, line);
                std::string_view cells = line;
                next_cell(cells); // logE_<...>

                for (Index_t z = 0; z < n_cz; ++z)
                    out.detector_response(row, z) = parse_real(next_cell(cells), file, row + 2, z + 2);
            }
        }

        void read_energy_migration(const std::filesystem::path& file, ResponseArray& out) {
            std::ifstream in(file);
            if (!in)
                throw std::runtime_error("Cannot open TRIDENT migration file: " + file.string());

            std::string line;
            std::getline(in, line); // header

            // CSV storage convention:
            //
            //     row    = true energy
            //     column = reconstructed energy
            //
            // therefore:
            //
            //     energy_migration(trueE, recoE) = P(E_rec | E_true)
            //
            // Do not transpose the file while loading.
            for (Index_t row = 0; row < n_true; ++row) {
                std::getline(in, line);
                std::string_view cells = line;
                next_cell(cells); // true-energy center

                for (Index_t reco = 0; reco < n_reco; ++reco)
                    out.energy_migration(row, reco) = parse_real(next_cell(cells), file, row + 2, reco + 2);
            }
        }

    } // namespace

    ResponseArray load_trident_response(const std::filesystem::path& directory) {
        ResponseArray out(n_true, n_cz, n_reco);

        set_trident_binning(out);
        read_detector_response(directory / response_filename, out);
        read_energy_migration(directory / migration_filename, out);

        return out;
    }

} // namespace nt
