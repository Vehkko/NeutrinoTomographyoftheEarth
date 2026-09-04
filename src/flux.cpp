#include <nt/flux.hpp>
#include <nt/types.hpp>
#include <vndarray/ndarray.hpp>

#include <H5Dpublic.h>
#include <H5Fpublic.h>
#include <H5Ipublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <H5public.h>
#include <array>
#include <cassert>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <nuSQuIDS/marray.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace nt {

    namespace {

        constexpr Index_t particle_count  = 2;
        constexpr Index_t flavor_count    = 3;
        constexpr Index_t component_count = particle_count * flavor_count;

        // -----------------------------------------------------------------------------
        // marray access
        // -----------------------------------------------------------------------------

        template <typename T, unsigned int Rank> T* marray_data(nsq::marray<T, Rank>& a) noexcept {
            return a.empty() ? nullptr : std::addressof(*a.begin());
        }

        template <typename T, unsigned int Rank> const T* marray_data(const nsq::marray<T, Rank>& a) noexcept {
            return a.empty() ? nullptr : std::addressof(*a.begin());
        }

        // -----------------------------------------------------------------------------
        // Small local HDF5 helpers
        // -----------------------------------------------------------------------------

        class H5Handle {
          public:
            using Closer = herr_t (*)(hid_t);

            H5Handle(hid_t id, Closer closer) noexcept : id_(id), closer_(closer) {}
            H5Handle(const H5Handle&)            = delete;
            H5Handle& operator=(const H5Handle&) = delete;

            H5Handle(H5Handle&& other) noexcept : id_(other.id_), closer_(other.closer_) {
                other.id_     = -1;
                other.closer_ = nullptr;
            }

            ~H5Handle() noexcept {
                if (id_ >= 0 && closer_)
                    closer_(id_);
            }

            [[nodiscard]] hid_t get() const noexcept { return id_; }

          private:
            hid_t  id_     = -1;
            Closer closer_ = nullptr;
        };

        [[noreturn]] void hdf5_error(const std::string& message, const std::string& object) {
            throw std::runtime_error("HDF5: " + message + ": " + object);
        }

        hid_t hdf5_real_type() noexcept {
            if constexpr (std::is_same_v<Real_t, float>)
                return H5T_NATIVE_FLOAT;
            else if constexpr (std::is_same_v<Real_t, double>)
                return H5T_NATIVE_DOUBLE;
            else if constexpr (std::is_same_v<Real_t, long double>)
                return H5T_NATIVE_LDOUBLE;
            else {
                static_assert(std::is_same_v<Real_t, float> || std::is_same_v<Real_t, double> ||
                                  std::is_same_v<Real_t, long double>,
                              "Unsupported Real_t");
            }
        }

        H5Handle open_dataset(hid_t file, const std::string& path) {
            const hid_t id = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
            if (id < 0)
                hdf5_error("cannot open dataset", path);
            return H5Handle{id, H5Dclose};
        }

        template <std::size_t Rank> std::array<hsize_t, Rank> dataset_shape(hid_t dataset, const std::string& path) {
            const hid_t space_id = H5Dget_space(dataset);
            if (space_id < 0)
                hdf5_error("cannot get dataspace", path);
            H5Handle space{space_id, H5Sclose};

            if (H5Sget_simple_extent_ndims(space.get()) != static_cast<int>(Rank))
                hdf5_error("unexpected dataset rank", path);

            std::array<hsize_t, Rank> shape{};
            if (H5Sget_simple_extent_dims(space.get(), shape.data(), nullptr) < 0)
                hdf5_error("cannot read dataset shape", path);
            return shape;
        }

        Index_t to_index(hsize_t n, const std::string& path) {
            if (n > static_cast<hsize_t>(std::numeric_limits<Index_t>::max()))
                throw std::overflow_error("HDF5 dimension exceeds Index_t: " + path);
            return static_cast<Index_t>(n);
        }

        Index_t read_axis_size(hid_t file, const std::string& path) {
            auto dataset = open_dataset(file, path);
            return to_index(dataset_shape<1>(dataset.get(), path)[0], path);
        }

        void read_axis(hid_t file, const std::string& path, nda::View<Real_t, 1> dst) {
            auto       dataset = open_dataset(file, path);
            const auto shape   = dataset_shape<1>(dataset.get(), path);

            if (shape[0] != static_cast<hsize_t>(dst.extent(0)))
                hdf5_error("axis size does not match Flux", path);

            if (H5Dread(dataset.get(), hdf5_real_type(), H5S_ALL, H5S_ALL, H5P_DEFAULT, dst.data()) < 0)
                hdf5_error("cannot read dataset", path);
        }

        // Read a compact HDF5 [coszenith, energy] dataset directly into one
        // [particle, flavor] slice of the final 4D state.
        void read_flux_component(hid_t file, const std::string& path, Flux& flux, Particle particle, Flavor flavor) {
            auto       dataset = open_dataset(file, path);
            const auto shape   = dataset_shape<2>(dataset.get(), path);

            const Index_t ncz = flux.n_coszenith();
            const Index_t ne  = flux.n_energy();
            if (shape[0] != static_cast<hsize_t>(ncz) || shape[1] != static_cast<hsize_t>(ne))
                hdf5_error("flux shape does not match axes", path);

            const hid_t file_space_id = H5Dget_space(dataset.get());
            if (file_space_id < 0)
                hdf5_error("cannot get file dataspace", path);
            H5Handle file_space{file_space_id, H5Sclose};

            const hsize_t memory_dims[4]  = {ncz, ne, particle_count, flavor_count};
            const hid_t   memory_space_id = H5Screate_simple(4, memory_dims, nullptr);
            if (memory_space_id < 0)
                hdf5_error("cannot create memory dataspace", path);
            H5Handle memory_space{memory_space_id, H5Sclose};

            const hsize_t start[4] = {0, 0, static_cast<hsize_t>(particle), static_cast<hsize_t>(flavor)};
            const hsize_t count[4] = {ncz, ne, 1, 1};

            if (H5Sselect_hyperslab(memory_space.get(), H5S_SELECT_SET, start, nullptr, count, nullptr) < 0)
                hdf5_error("cannot select memory hyperslab", path);

            if (H5Dread(dataset.get(), hdf5_real_type(), memory_space.get(), file_space.get(), H5P_DEFAULT,
                        marray_data(flux.native_state())) < 0)
                hdf5_error("cannot read flux dataset", path);
        }

        // -----------------------------------------------------------------------------
        // Resampling
        // -----------------------------------------------------------------------------

        struct InterpPoint {
            Index_t lo;
            Index_t hi;
            Real_t  w0;
            Real_t  w1;
        };

        void validate_axis(nda::View<const Real_t, 1> axis, const char* name) {
            if (axis.extent(0) < 2)
                throw std::invalid_argument(std::string(name) + " must contain at least two points");

            for (Index_t i = 1; i < axis.extent(0); ++i)
                if (!(axis(i) > axis(i - 1)))
                    throw std::invalid_argument(std::string(name) + " must be strictly increasing");
        }

        // Both axes are increasing, so one forward pass is enough to determine all
        // interpolation intervals. Boundary points retain the legacy linear
        // extrapolation behavior.
        std::vector<InterpPoint> make_interp_map(nda::View<const Real_t, 1> source, nda::View<const Real_t, 1> target) {
            std::vector<InterpPoint> map(target.extent(0));
            const Index_t            n  = source.extent(0);
            Index_t                  hi = 1;

            for (Index_t i = 0; i < target.extent(0); ++i) {
                const Real_t x = target(i);

                if (x <= source(0))
                    hi = 1;
                else if (x >= source(n - 1))
                    hi = n - 1;
                else {
                    while (hi < n - 1 && source(hi) < x)
                        ++hi;
                }

                const Index_t lo = hi - 1;
                const Real_t  w1 = (x - source(lo)) / (source(hi) - source(lo));
                map[i]           = {lo, hi, Real_t{1} - w1, w1};
            }

            return map;
        }

        void copy_axis(nda::View<const Real_t, 1> src, nda::View<Real_t, 1> dst) {
            assert(src.extent(0) == dst.extent(0));
            for (Index_t i = 0; i < src.extent(0); ++i)
                dst(i) = src(i);
        }

        // The six [particle, flavor] values belonging to one (z,E) grid point are
        // contiguous in memory. Interpolate all six together so interval lookup,
        // address arithmetic and source loads are shared instead of repeating six
        // complete 2D passes.
        void resample_state(const Flux& source, Flux& target, const std::vector<InterpPoint>& zmap,
                            const std::vector<InterpPoint>& emap) {
            const Real_t* src = marray_data(source.native_state());
            Real_t*       dst = marray_data(target.native_state());

            const Index_t src_ne = source.n_energy();
            const Index_t dst_ne = target.n_energy();

            const Index_t src_z_stride = src_ne * component_count;
            const Index_t dst_z_stride = dst_ne * component_count;

            for (Index_t iz = 0; iz < target.n_coszenith(); ++iz) {
                const auto& z = zmap[iz];

                const Real_t* src_z0  = src + z.lo * src_z_stride;
                const Real_t* src_z1  = src + z.hi * src_z_stride;
                Real_t*       dst_row = dst + iz * dst_z_stride;

                for (Index_t ie = 0; ie < dst_ne; ++ie) {
                    const auto& e = emap[ie];

                    const Real_t* p00 = src_z0 + e.lo * component_count;
                    const Real_t* p01 = src_z0 + e.hi * component_count;
                    const Real_t* p10 = src_z1 + e.lo * component_count;
                    const Real_t* p11 = src_z1 + e.hi * component_count;
                    Real_t*       out = dst_row + ie * component_count;

                    const Real_t c00 = z.w0 * e.w0;
                    const Real_t c01 = z.w0 * e.w1;
                    const Real_t c10 = z.w1 * e.w0;
                    const Real_t c11 = z.w1 * e.w1;

                    for (Index_t c = 0; c < component_count; ++c)
                        out[c] = c00 * p00[c] + c01 * p01[c] + c10 * p10[c] + c11 * p11[c];
                }
            }
        }

    } // namespace

    // =============================================================================
    // Flux
    // =============================================================================

    Flux::Flux(Index_t n_coszenith, Index_t n_energy)
        : coszenith_({n_coszenith}), energy_gev_({n_energy}),
          state_({n_coszenith, n_energy, particle_count, flavor_count}) {
        // marray's ordinary construction value-initializes double elements, so the
        // initial state is already zero. Do not perform a second full-memory clear.
    }

    Index_t Flux::n_coszenith() const noexcept { return coszenith_.size(); }
    Index_t Flux::n_energy() const noexcept { return energy_gev_.size(); }

    nda::View<Real_t, 1> Flux::coszenith() noexcept {
        return nda::make_view1d(marray_data(coszenith_), coszenith_.size());
    }

    nda::View<const Real_t, 1> Flux::coszenith() const noexcept {
        return nda::make_view1d(marray_data(coszenith_), coszenith_.size());
    }

    nda::View<Real_t, 1> Flux::energy_gev() noexcept {
        return nda::make_view1d(marray_data(energy_gev_), energy_gev_.size());
    }

    nda::View<const Real_t, 1> Flux::energy_gev() const noexcept {
        return nda::make_view1d(marray_data(energy_gev_), energy_gev_.size());
    }

    nda::View<Real_t, 2> Flux::component(Particle particle, Flavor flavor) noexcept {
        assert(static_cast<Index_t>(particle) < particle_count);
        assert(static_cast<Index_t>(flavor) < flavor_count);

        const Index_t ne       = n_energy();
        const Index_t e_stride = component_count;
        const Index_t z_stride = ne * component_count;
        const Index_t offset   = static_cast<Index_t>(particle) * flavor_count + static_cast<Index_t>(flavor);

        Real_t* data = marray_data(state_);
        if (data)
            data += offset;
        return nda::make_strided_view2d(data, n_coszenith(), ne, z_stride, e_stride);
    }

    nda::View<const Real_t, 2> Flux::component(Particle particle, Flavor flavor) const noexcept {
        assert(static_cast<Index_t>(particle) < particle_count);
        assert(static_cast<Index_t>(flavor) < flavor_count);

        const Index_t ne       = n_energy();
        const Index_t e_stride = component_count;
        const Index_t z_stride = ne * component_count;
        const Index_t offset   = static_cast<Index_t>(particle) * flavor_count + static_cast<Index_t>(flavor);

        const Real_t* data = marray_data(state_);
        if (data)
            data += offset;
        return nda::make_strided_view2d(data, n_coszenith(), ne, z_stride, e_stride);
    }

    nda::View<Real_t, 2> Flux::numu() noexcept { return component(Particle::neutrino, Flavor::muon); }

    nda::View<const Real_t, 2> Flux::numu() const noexcept { return component(Particle::neutrino, Flavor::muon); }

    nda::View<Real_t, 2> Flux::antinumu() noexcept { return component(Particle::antineutrino, Flavor::muon); }

    nda::View<const Real_t, 2> Flux::antinumu() const noexcept {
        return component(Particle::antineutrino, Flavor::muon);
    }

    Flux::NativeState&       Flux::native_state() noexcept { return state_; }
    const Flux::NativeState& Flux::native_state() const noexcept { return state_; }

    const Flux::Axis& Flux::native_coszenith() const noexcept { return coszenith_; }
    const Flux::Axis& Flux::native_energy_gev() const noexcept { return energy_gev_; }

    // =============================================================================
    // DaemonFlux
    // =============================================================================

    Flux load_daemonflux(const std::filesystem::path& filename, std::string_view location) {
        const std::string name    = filename.string();
        const hid_t       file_id = H5Fopen(name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0)
            hdf5_error("cannot open file", name);
        H5Handle file{file_id, H5Fclose};

        constexpr const char* cz_path = "/axes/coszenith";
        constexpr const char* en_path = "/axes/energy_GeV";

        Flux flux(read_axis_size(file.get(), cz_path), read_axis_size(file.get(), en_path));

        read_axis(file.get(), cz_path, flux.coszenith());
        read_axis(file.get(), en_path, flux.energy_gev());

        const std::string base = "/flux/" + std::string(location) + "/calibrated/";
        read_flux_component(file.get(), base + "numu", flux, Particle::neutrino, Flavor::muon);
        read_flux_component(file.get(), base + "antinumu", flux, Particle::antineutrino, Flavor::muon);

        return flux;
    }

    // =============================================================================
    // Resampling
    // =============================================================================

    Flux resample_flux(const Flux& source, nda::View<const Real_t, 1> coszenith,
                       nda::View<const Real_t, 1> energy_gev) {
        const auto src_z = source.coszenith();
        const auto src_e = source.energy_gev();

        validate_axis(src_z, "source coszenith");
        validate_axis(src_e, "source energy");
        validate_axis(coszenith, "target coszenith");
        validate_axis(energy_gev, "target energy");

        const auto zmap = make_interp_map(src_z, coszenith);
        const auto emap = make_interp_map(src_e, energy_gev);

        Flux result(coszenith.extent(0), energy_gev.extent(0));
        copy_axis(coszenith, result.coszenith());
        copy_axis(energy_gev, result.energy_gev());
        resample_state(source, result, zmap, emap);

        return result;
    }

} // namespace nt
