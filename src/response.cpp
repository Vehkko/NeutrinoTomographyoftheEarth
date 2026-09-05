#include <nt/response.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
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

        constexpr const char* response_filename = "TRIDENT_response_array_20x34.csv";

        constexpr const char* migration_filename = "energy_response_20x20_v2.csv";

        constexpr Real_t trident_log_energy_edge_min   = 3.0;
        constexpr Real_t trident_log_energy_center_min = 3.05;
        constexpr Real_t trident_log_energy_step       = 0.1;

        constexpr std::array<Real_t, n_cz> trident_coszenith_centers = {
            -0.9988, -0.9963, -0.9937, -0.9912, -0.9855, -0.9765, -0.9675, -0.9585, -0.9495, -0.9405, -0.9315, -0.9225,
            -0.9135, -0.9045, -0.8775, -0.8325, -0.7875, -0.7425, -0.6975, -0.6525, -0.6075, -0.5625, -0.5175, -0.4725,
            -0.4275, -0.3825, -0.3375, -0.2925, -0.2475, -0.2025, -0.1575, -0.1125, -0.0675, -0.0225,
        };

        constexpr std::array<Real_t, n_cz + 1> trident_coszenith_edges = {
            -1.0,   -0.99755, -0.995, -0.99245, -0.98835, -0.981, -0.972, -0.963, -0.954, -0.945, -0.936, -0.927,
            -0.918, -0.909,   -0.891, -0.855,   -0.81,    -0.765, -0.72,  -0.675, -0.63,  -0.585, -0.54,  -0.495,
            -0.45,  -0.405,   -0.36,  -0.315,   -0.27,    -0.225, -0.18,  -0.135, -0.09,  -0.045, 0.0,
        };

        void set_trident_binning(ResponseArray& out) {
            for (Index_t i = 0; i <= n_true; ++i) {
                const Real_t energy = std::pow(Real_t{10}, trident_log_energy_edge_min +
                                                               trident_log_energy_step * static_cast<Real_t>(i));

                out.true_energy_edges_gev(i) = energy;
                out.reco_energy_edges_gev(i) = energy;
            }

            for (Index_t i = 0; i < n_true; ++i) {
                const Real_t energy = std::pow(Real_t{10}, trident_log_energy_center_min +
                                                               trident_log_energy_step * static_cast<Real_t>(i));

                out.true_energy_gev(i) = energy;
                out.reco_energy_gev(i) = energy;
            }

            for (Index_t i = 0; i < n_cz; ++i)
                out.coszenith(i) = trident_coszenith_centers[i];

            for (Index_t i = 0; i <= n_cz; ++i)
                out.coszenith_edges(i) = trident_coszenith_edges[i];
        }

        std::string_view trim(std::string_view s) noexcept {
            while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r'))
                s.remove_prefix(1);
            while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
                s.remove_suffix(1);
            return s;
        }

        Real_t parse_real(std::string_view text, const std::filesystem::path& file, Index_t line, Index_t column) {
            text = trim(text);
            if (text.empty())
                throw std::runtime_error("Empty CSV field in " + file.string() + " at line " + std::to_string(line) +
                                         ", column " + std::to_string(column));

            Real_t      value{};
            const char* begin    = text.data();
            const char* end      = begin + text.size();
            const auto [ptr, ec] = std::from_chars(begin, end, value, std::chars_format::general);

            if (ec != std::errc{} || ptr != end || !std::isfinite(value))
                throw std::runtime_error("Invalid numeric field '" + std::string(text) + "' in " + file.string() +
                                         " at line " + std::to_string(line) + ", column " + std::to_string(column));
            return value;
        }

        Real_t parse_prefixed_real(std::string_view text, std::string_view prefix, const std::filesystem::path& file,
                                   Index_t line, Index_t column) {
            text = trim(text);
            if (text.size() <= prefix.size() || text.substr(0, prefix.size()) != prefix)
                throw std::runtime_error("Expected '" + std::string(prefix) + "<number>' in " + file.string() +
                                         " at line " + std::to_string(line) + ", column " + std::to_string(column));

            return parse_real(text.substr(prefix.size()), file, line, column);
        }

        // Iterate over comma-separated fields without allocating a vector or
        // copying individual cells. The supplied callback receives
        //
        //     (column_index, field)
        //
        // for every field in the line.
        template <typename F> Index_t for_each_cell(std::string_view line, F&& f) {
            Index_t     column = 0;
            std::size_t begin  = 0;

            while (true) {
                const std::size_t comma = line.find(',', begin);
                const std::size_t end   = comma == std::string_view::npos ? line.size() : comma;

                f(column++, trim(line.substr(begin, end - begin)));

                if (comma == std::string_view::npos)
                    break;
                begin = comma + 1;
            }

            return column;
        }

        bool read_nonempty_line(std::ifstream& in, std::string& line) {
            while (std::getline(in, line)) {
                if (!trim(line).empty())
                    return true;
            }
            return false;
        }

        void require_strictly_increasing(nda::View<const Real_t, 1> axis, const char* name) {
            for (Index_t i = 1; i < axis.extent(0); ++i) {
                if (!(axis(i) > axis(i - 1)))
                    throw std::runtime_error(std::string(name) + " must be strictly increasing");
            }
        }

        void read_detector_response(const std::filesystem::path& file, ResponseArray& out) {
            std::ifstream in(file);
            if (!in)
                throw std::runtime_error("Cannot open TRIDENT response file: " + file.string());

            std::string line;

            // Header:
            //
            //     <label>,cos_<cz0>,...,cos_<cz33>
            if (!read_nonempty_line(in, line))
                throw std::runtime_error("Empty TRIDENT response file: " + file.string());

            const Index_t header_fields = for_each_cell(line, [&](Index_t col, std::string_view cell) {
                if (col == 0)
                    return;
                if (col > n_cz)
                    return;

                const Real_t value = parse_prefixed_real(cell, "cos_", file, 1, col + 1);

                if (std::abs(value - out.coszenith(col - 1)) > 1e-12)
                    throw std::runtime_error("TRIDENT response coszenith header disagrees with canonical binning in " +
                                             file.string() + " at column " + std::to_string(col + 1));
            });

            if (header_fields != n_cz + 1)
                throw std::runtime_error("TRIDENT response header must contain 35 fields: " + file.string());

            // Rows:
            //
            //     logE_<log10(E/GeV)>,r0,...,r33
            for (Index_t row = 0; row < n_true; ++row) {
                if (!read_nonempty_line(in, line))
                    throw std::runtime_error("TRIDENT response file has fewer than 20 data rows: " + file.string());

                const Index_t fields = for_each_cell(line, [&](Index_t col, std::string_view cell) {
                    if (col == 0) {
                        const Real_t loge = parse_prefixed_real(cell, "logE_", file, row + 2, 1);
                        const Real_t expected =
                            trident_log_energy_center_min + trident_log_energy_step * static_cast<Real_t>(row);

                        if (std::abs(loge - expected) > 1e-12)
                            throw std::runtime_error("TRIDENT response true-energy label disagrees with canonical "
                                                     "binning in " +
                                                     file.string() + " at line " + std::to_string(row + 2));
                    } else if (col <= n_cz) {
                        const Real_t value = parse_real(cell, file, row + 2, col + 1);
                        if (value < 0)
                            throw std::runtime_error("Negative detector response in " + file.string() + " at line " +
                                                     std::to_string(row + 2));
                        out.detector_response(row, col - 1) = value;
                    }
                });

                if (fields != n_cz + 1)
                    throw std::runtime_error("TRIDENT response row must contain 35 fields in " + file.string() +
                                             " at line " + std::to_string(row + 2));
            }

            if (read_nonempty_line(in, line))
                throw std::runtime_error("TRIDENT response file has more than 20 data rows: " + file.string());
        }

        void read_energy_migration(const std::filesystem::path& file, ResponseArray& out) {
            std::ifstream in(file);
            if (!in)
                throw std::runtime_error("Cannot open TRIDENT migration file: " + file.string());

            std::string line;

            // Header:
            //
            //     <true-energy-label>,Erec0,...,Erec19
            //
            // The first header name is deliberately not hard-coded; only its
            // meaning as the row-label column matters. Remaining headers must
            // be numeric reconstructed-energy centers.
            if (!read_nonempty_line(in, line))
                throw std::runtime_error("Empty TRIDENT migration file: " + file.string());

            const Index_t header_fields = for_each_cell(line, [&](Index_t col, std::string_view cell) {
                if (col == 0)
                    return;
                if (col > n_reco)
                    return;

                const Real_t value = parse_real(cell, file, 1, col + 1);
                const Real_t a     = std::log10(out.reco_energy_gev(col - 1));
                const Real_t b     = std::log10(value);

                if (!std::isfinite(b) || std::abs(a - b) > 1e-4)
                    throw std::runtime_error("TRIDENT migration reconstructed-energy header disagrees with canonical "
                                             "binning in " +
                                             file.string() + " at column " + std::to_string(col + 1));
            });

            if (header_fields != n_reco + 1)
                throw std::runtime_error("TRIDENT migration header must contain 21 fields: " + file.string());

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
                if (!read_nonempty_line(in, line))
                    throw std::runtime_error("TRIDENT migration file has fewer than 20 data rows: " + file.string());

                Real_t migration_true_energy = 0;

                const Index_t fields = for_each_cell(line, [&](Index_t col, std::string_view cell) {
                    if (col == 0) {
                        migration_true_energy = parse_real(cell, file, row + 2, 1);
                    } else if (col <= n_reco) {
                        const Real_t value = parse_real(cell, file, row + 2, col + 1);
                        if (value < 0)
                            throw std::runtime_error("Negative migration probability in " + file.string() +
                                                     " at line " + std::to_string(row + 2));
                        out.energy_migration(row, col - 1) = value;
                    }
                });

                if (fields != n_reco + 1)
                    throw std::runtime_error("TRIDENT migration row must contain 21 fields in " + file.string() +
                                             " at line " + std::to_string(row + 2));

                // The two collaboration files describe the same 20 true-energy
                // bins. Compare in log-space with a modest tolerance so harmless
                // decimal rounding in the CSV labels is accepted, while a shifted
                // or reordered grid is rejected.
                const Real_t a = std::log10(out.true_energy_gev(row));
                const Real_t b = std::log10(migration_true_energy);
                if (!std::isfinite(b) || std::abs(a - b) > 1e-4)
                    throw std::runtime_error("True-energy grids disagree between TRIDENT response and "
                                             "migration files at row " +
                                             std::to_string(row));
            }

            if (read_nonempty_line(in, line))
                throw std::runtime_error("TRIDENT migration file has more than 20 data rows: " + file.string());
        }

    } // namespace

    ResponseArray load_trident_response(const std::filesystem::path& directory) {
        ResponseArray out(n_true, n_cz, n_reco);

        set_trident_binning(out);

        read_detector_response(directory / response_filename, out);
        read_energy_migration(directory / migration_filename, out);

        require_strictly_increasing(out.true_energy_gev.view(), "TRIDENT true-energy axis");
        require_strictly_increasing(out.coszenith.view(), "TRIDENT coszenith axis");
        require_strictly_increasing(out.reco_energy_gev.view(), "TRIDENT reconstructed-energy axis");

        require_strictly_increasing(out.true_energy_edges_gev.view(), "TRIDENT true-energy bin edges");
        require_strictly_increasing(out.coszenith_edges.view(), "TRIDENT coszenith bin edges");
        require_strictly_increasing(out.reco_energy_edges_gev.view(), "TRIDENT reconstructed-energy bin edges");

        return out;
    }

} // namespace nt
