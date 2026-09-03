#pragma once

#include "ndarray.hpp"

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <type_traits>

#ifndef VEHKKO_NDARRAY_RESTRICT
#if defined(__clang__) || defined(__GNUC__)
#define VEHKKO_NDARRAY_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define VEHKKO_NDARRAY_RESTRICT __restrict
#else
#define VEHKKO_NDARRAY_RESTRICT
#endif
#endif

namespace vehkko::ndarray {

    // Algorithm contracts
    // -----------------------------------------------------------------------------
    // 1. Algorithms operate only on View. Owning containers are passed explicitly
    //    through .view().
    // 2. Functions ending in _inplace modify their first argument.
    // 3. Functions ending in _to write their result to their first argument.
    // 4. Ordinary copy/_to/_inplace algorithms are alias-aware when
    //    runtime_alias_dispatch is enabled. Complex overlap may allocate a
    //    temporary Array and may therefore propagate allocation failures.
    // 5. _noalias algorithms never allocate and never perform a safety fallback.
    //    Their writable output must not overlap any input. Read-only inputs may
    //    overlap each other.
    // 6. Writable views must have injective mappings. Arbitrary internally
    //    overlapping strided views are outside the contract.

    namespace detail {

        // Shape checks
        // -----------------------------------------------------------------------------

        template <typename T, typename U, index_t Rank>
        inline void require_same_extents(View<T, Rank> a, View<U, Rank> b) noexcept {
            if constexpr (runtime_shape_check) {
                if (!same_extents(a, b)) {
                    for (index_t d = 0; d < Rank; ++d) {
                        if (a.extent(d) != b.extent(d)) {
                            std::fprintf(stderr,
                                         "[vehkko::ndarray][shape] require_same_extents: rank=%zu, "
                                         "dimension=%zu, first_extent=%zu, second_extent=%zu, "
                                         "first_data=%p, second_data=%p; %s:%d\n",
                                         Rank, d, a.extent(d), b.extent(d),
                                         const_cast<void*>(static_cast<const void*>(a.data())),
                                         const_cast<void*>(static_cast<const void*>(b.data())),
                                         __FILE__, __LINE__);
                            contract_violation();
                        }
                    }
                }
            }
        }

        template <typename T, typename U, typename V, index_t Rank>
        inline void require_same_extents(View<T, Rank> a, View<U, Rank> b,
                                         View<V, Rank> c) noexcept {
            if constexpr (runtime_shape_check) {
                if (!same_extents(a, b)) {
                    for (index_t d = 0; d < Rank; ++d) {
                        if (a.extent(d) != b.extent(d)) {
                            std::fprintf(stderr,
                                         "[vehkko::ndarray][shape] require_same_extents: rank=%zu, "
                                         "dimension=%zu, first_extent=%zu, second_extent=%zu, "
                                         "first_data=%p, second_data=%p; %s:%d\n",
                                         Rank, d, a.extent(d), b.extent(d),
                                         const_cast<void*>(static_cast<const void*>(a.data())),
                                         const_cast<void*>(static_cast<const void*>(b.data())),
                                         __FILE__, __LINE__);
                            contract_violation();
                        }
                    }
                }
                if (!same_extents(a, c)) {
                    for (index_t d = 0; d < Rank; ++d) {
                        if (a.extent(d) != c.extent(d)) {
                            std::fprintf(stderr,
                                         "[vehkko::ndarray][shape] require_same_extents: rank=%zu, "
                                         "dimension=%zu, first_extent=%zu, third_extent=%zu, "
                                         "first_data=%p, third_data=%p; %s:%d\n",
                                         Rank, d, a.extent(d), c.extent(d),
                                         const_cast<void*>(static_cast<const void*>(a.data())),
                                         const_cast<void*>(static_cast<const void*>(c.data())),
                                         __FILE__, __LINE__);
                            contract_violation();
                        }
                    }
                }
            }
        }

        // Conservative View alias classification
        // -----------------------------------------------------------------------------
        // The classifier never returns disjoint for views that can access the same
        // element. It may conservatively return may_overlap for complex layouts that
        // are actually disjoint.

        enum class AliasRelation : unsigned char {
            disjoint,
            same_mapping,
            may_overlap
        };

        struct ByteEnvelope {
            std::uintptr_t begin = 0;
            std::uintptr_t end   = 0; // exclusive
        };

        constexpr index_t gcd_index(index_t a, index_t b) noexcept {
            while (b != 0) {
                const index_t r = a % b;
                a               = b;
                b               = r;
            }
            return a;
        }

        template <typename T, index_t Rank>
        inline bool has_zero_extent(View<T, Rank> v) noexcept {
            for (index_t d = 0; d < Rank; ++d) {
                if (v.extent(d) == 0) {
                    return true;
                }
            }
            return false;
        }

        template <typename T, typename U, index_t Rank>
        inline bool same_mapping(View<T, Rank> a, View<U, Rank> b) noexcept {
            static_assert(std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>,
                          "same_mapping: element value types must match");

            if (static_cast<const void*>(a.data()) != static_cast<const void*>(b.data())) {
                return false;
            }

            for (index_t d = 0; d < Rank; ++d) {
                if (a.extent(d) != b.extent(d) || a.stride(d) != b.stride(d)) {
                    return false;
                }
            }
            return true;
        }

        template <typename T, index_t Rank>
        inline ByteEnvelope byte_envelope(View<T, Rank> v) noexcept {
            using E = std::remove_const_t<T>;

            index_t max_offset = 0;
            for (index_t d = 0; d < Rank; ++d) {
                max_offset += (v.extent(d) - 1) * v.stride(d);
            }

            const index_t        span_bytes = (max_offset + 1) * sizeof(E);
            const std::uintptr_t begin      = reinterpret_cast<std::uintptr_t>(v.data());
            const std::uintptr_t end        = begin + span_bytes;
            return ByteEnvelope{begin, end};
        }

        inline bool envelopes_disjoint(ByteEnvelope a, ByteEnvelope b) noexcept {
            return a.end <= b.begin || b.end <= a.begin;
        }

        template <typename T, index_t Rank>
        inline index_t byte_lattice_step(View<T, Rank> v) noexcept {
            using E   = std::remove_const_t<T>;
            index_t g = 0;

            for (index_t d = 0; d < Rank; ++d) {
                if (v.extent(d) <= 1) {
                    continue;
                }
                const index_t byte_stride = v.stride(d) * sizeof(E);
                g                         = gcd_index(g, byte_stride);
            }
            return g;
        }

        template <typename T, typename U, index_t Rank>
        inline bool lattice_congruence_proves_disjoint(View<T, Rank> a, View<U, Rank> b) noexcept {
            const index_t ga = byte_lattice_step(a);
            const index_t gb = byte_lattice_step(b);
            const index_t g  = gcd_index(ga, gb);
            if (g == 0) {
                return false;
            }

            const std::uintptr_t ap    = reinterpret_cast<std::uintptr_t>(a.data());
            const std::uintptr_t bp    = reinterpret_cast<std::uintptr_t>(b.data());
            const std::uintptr_t delta = ap < bp ? bp - ap : ap - bp;
            return delta % g != 0;
        }

        template <typename T, typename U, index_t Rank>
        inline AliasRelation classify_alias(View<T, Rank> a, View<U, Rank> b) noexcept {
            static_assert(std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>,
                          "classify_alias: element value types must match");

            if (has_zero_extent(a) || has_zero_extent(b)) {
                return AliasRelation::disjoint;
            }

            if (same_mapping(a, b)) {
                return AliasRelation::same_mapping;
            }

            const ByteEnvelope ae = byte_envelope(a);
            const ByteEnvelope be = byte_envelope(b);
            if (envelopes_disjoint(ae, be)) {
                return AliasRelation::disjoint;
            }

            if (lattice_congruence_proves_disjoint(a, b)) {
                return AliasRelation::disjoint;
            }

            return AliasRelation::may_overlap;
        }

        // This verifier deliberately catches only relationships that can be proven
        // cheaply. A may_overlap result is not itself a contract violation because the
        // classifier is conservative.
        template <typename T, typename U, index_t Rank>
        inline bool definitely_overlap_simple(View<T, Rank> a, View<U, Rank> b) noexcept {
            const AliasRelation relation = classify_alias(a, b);
            if (relation == AliasRelation::disjoint) {
                return false;
            }
            if (relation == AliasRelation::same_mapping) {
                return true;
            }

            if (is_row_major_contiguous(a) && is_row_major_contiguous(b)) {
                return true;
            }

            if constexpr (Rank == 1) {
                if (a.stride(0) == b.stride(0)) {
                    const index_t step = a.stride(0) * sizeof(std::remove_const_t<T>);
                    if (step == 0) {
                        return true;
                    }
                    const std::uintptr_t ap    = reinterpret_cast<std::uintptr_t>(a.data());
                    const std::uintptr_t bp    = reinterpret_cast<std::uintptr_t>(b.data());
                    const std::uintptr_t delta = ap < bp ? bp - ap : ap - bp;
                    return delta % step == 0;
                }
            }

            return false;
        }

        template <typename T, typename U, index_t Rank>
        inline void check_noalias_contract(View<T, Rank> dst, View<U, Rank> src) noexcept {
            if constexpr (runtime_noalias_contract_check) {
                if (definitely_overlap_simple(dst, src)) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][noalias] check_noalias_contract: output overlaps "
                                 "input, rank=%zu, dst_data=%p, src_data=%p, dst_size=%zu, "
                                 "src_size=%zu; %s:%d\n",
                                 Rank,
                                 const_cast<void*>(static_cast<const void*>(dst.data())),
                                 const_cast<void*>(static_cast<const void*>(src.data())), dst.size(),
                                 src.size(), __FILE__, __LINE__);
                    contract_violation();
                }
            }
        }

        // Fixed-rank strided traversal
        // -----------------------------------------------------------------------------

        template <typename T, index_t Rank, typename F>
        inline void for_each_offset(View<T, Rank> v, F&& f) noexcept {
            if (v.empty()) {
                return;
            }

            if constexpr (Rank == 1) {
                const index_t n0 = v.extent(0);
                const index_t s0 = v.stride(0);
                for (index_t i = 0; i < n0; ++i) {
                    f(i * s0);
                }
            } else if constexpr (Rank == 2) {
                const index_t n0 = v.extent(0);
                const index_t n1 = v.extent(1);
                const index_t s0 = v.stride(0);
                const index_t s1 = v.stride(1);

                if (s1 == 1) {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t base = i * s0;
                        for (index_t j = 0; j < n1; ++j) {
                            f(base + j);
                        }
                    }
                } else {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t base = i * s0;
                        for (index_t j = 0; j < n1; ++j) {
                            f(base + j * s1);
                        }
                    }
                }
            } else if constexpr (Rank == 3) {
                const index_t n0 = v.extent(0);
                const index_t n1 = v.extent(1);
                const index_t n2 = v.extent(2);
                const index_t s0 = v.stride(0);
                const index_t s1 = v.stride(1);
                const index_t s2 = v.stride(2);

                if (s2 == 1) {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t base0 = i * s0;
                        for (index_t j = 0; j < n1; ++j) {
                            const index_t base1 = base0 + j * s1;
                            for (index_t k = 0; k < n2; ++k) {
                                f(base1 + k);
                            }
                        }
                    }
                } else {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t base0 = i * s0;
                        for (index_t j = 0; j < n1; ++j) {
                            const index_t base1 = base0 + j * s1;
                            for (index_t k = 0; k < n2; ++k) {
                                f(base1 + k * s2);
                            }
                        }
                    }
                }
            } else {
                index_t idx[Rank] = {};
                index_t extent[Rank];
                index_t step[Rank];
                index_t reset[Rank];
                for (index_t d = 0; d < Rank; ++d) {
                    extent[d] = v.extent(d);
                    step[d]   = v.stride(d);
                    reset[d]  = (extent[d] - 1) * step[d];
                }

                index_t offset = 0;
                while (true) {
                    f(offset);

                    for (index_t r = Rank; r > 0; --r) {
                        const index_t d = r - 1;
                        ++idx[d];
                        if (idx[d] < extent[d]) {
                            offset += step[d];
                            break;
                        }
                        if (d == 0) {
                            return;
                        }
                        idx[d] = 0;
                        offset -= reset[d];
                    }
                }
            }
        }

        template <typename T, typename U, index_t Rank, typename F>
        inline void for_each_offset2(View<T, Rank> a, View<U, Rank> b, F&& f) noexcept {
            if (a.empty()) {
                return;
            }

            if constexpr (Rank == 1) {
                const index_t n0  = a.extent(0);
                const index_t as0 = a.stride(0);
                const index_t bs0 = b.stride(0);
                for (index_t i = 0; i < n0; ++i) {
                    f(i * as0, i * bs0);
                }
            } else if constexpr (Rank == 2) {
                const index_t n0  = a.extent(0);
                const index_t n1  = a.extent(1);
                const index_t as0 = a.stride(0);
                const index_t as1 = a.stride(1);
                const index_t bs0 = b.stride(0);
                const index_t bs1 = b.stride(1);

                if (as1 == 1 && bs1 == 1) {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t abase = i * as0;
                        const index_t bbase = i * bs0;
                        for (index_t j = 0; j < n1; ++j) {
                            f(abase + j, bbase + j);
                        }
                    }
                } else {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t abase = i * as0;
                        const index_t bbase = i * bs0;
                        for (index_t j = 0; j < n1; ++j) {
                            f(abase + j * as1, bbase + j * bs1);
                        }
                    }
                }
            } else if constexpr (Rank == 3) {
                const index_t n0  = a.extent(0);
                const index_t n1  = a.extent(1);
                const index_t n2  = a.extent(2);
                const index_t as0 = a.stride(0);
                const index_t as1 = a.stride(1);
                const index_t as2 = a.stride(2);
                const index_t bs0 = b.stride(0);
                const index_t bs1 = b.stride(1);
                const index_t bs2 = b.stride(2);

                if (as2 == 1 && bs2 == 1) {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t abase0 = i * as0;
                        const index_t bbase0 = i * bs0;
                        for (index_t j = 0; j < n1; ++j) {
                            const index_t abase1 = abase0 + j * as1;
                            const index_t bbase1 = bbase0 + j * bs1;
                            for (index_t k = 0; k < n2; ++k) {
                                f(abase1 + k, bbase1 + k);
                            }
                        }
                    }
                } else {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t abase0 = i * as0;
                        const index_t bbase0 = i * bs0;
                        for (index_t j = 0; j < n1; ++j) {
                            const index_t abase1 = abase0 + j * as1;
                            const index_t bbase1 = bbase0 + j * bs1;
                            for (index_t k = 0; k < n2; ++k) {
                                f(abase1 + k * as2, bbase1 + k * bs2);
                            }
                        }
                    }
                }
            } else {
                index_t idx[Rank] = {};
                index_t extent[Rank];
                index_t astep[Rank];
                index_t bstep[Rank];
                index_t areset[Rank];
                index_t breset[Rank];
                for (index_t d = 0; d < Rank; ++d) {
                    extent[d] = a.extent(d);
                    astep[d]  = a.stride(d);
                    bstep[d]  = b.stride(d);
                    areset[d] = (extent[d] - 1) * astep[d];
                    breset[d] = (extent[d] - 1) * bstep[d];
                }

                index_t ao = 0;
                index_t bo = 0;
                while (true) {
                    f(ao, bo);

                    for (index_t r = Rank; r > 0; --r) {
                        const index_t d = r - 1;
                        ++idx[d];
                        if (idx[d] < extent[d]) {
                            ao += astep[d];
                            bo += bstep[d];
                            break;
                        }
                        if (d == 0) {
                            return;
                        }
                        idx[d] = 0;
                        ao -= areset[d];
                        bo -= breset[d];
                    }
                }
            }
        }

        template <typename T, typename U, typename V, index_t Rank, typename F>
        inline void for_each_offset3(View<T, Rank> a, View<U, Rank> b, View<V, Rank> c,
                                     F&& f) noexcept {
            if (a.empty()) {
                return;
            }

            if constexpr (Rank == 1) {
                const index_t n0  = a.extent(0);
                const index_t as0 = a.stride(0);
                const index_t bs0 = b.stride(0);
                const index_t cs0 = c.stride(0);
                for (index_t i = 0; i < n0; ++i) {
                    f(i * as0, i * bs0, i * cs0);
                }
            } else if constexpr (Rank == 2) {
                const index_t n0  = a.extent(0);
                const index_t n1  = a.extent(1);
                const index_t as0 = a.stride(0);
                const index_t as1 = a.stride(1);
                const index_t bs0 = b.stride(0);
                const index_t bs1 = b.stride(1);
                const index_t cs0 = c.stride(0);
                const index_t cs1 = c.stride(1);

                if (as1 == 1 && bs1 == 1 && cs1 == 1) {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t abase = i * as0;
                        const index_t bbase = i * bs0;
                        const index_t cbase = i * cs0;
                        for (index_t j = 0; j < n1; ++j) {
                            f(abase + j, bbase + j, cbase + j);
                        }
                    }
                } else {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t abase = i * as0;
                        const index_t bbase = i * bs0;
                        const index_t cbase = i * cs0;
                        for (index_t j = 0; j < n1; ++j) {
                            f(abase + j * as1, bbase + j * bs1, cbase + j * cs1);
                        }
                    }
                }
            } else if constexpr (Rank == 3) {
                const index_t n0  = a.extent(0);
                const index_t n1  = a.extent(1);
                const index_t n2  = a.extent(2);
                const index_t as0 = a.stride(0);
                const index_t as1 = a.stride(1);
                const index_t as2 = a.stride(2);
                const index_t bs0 = b.stride(0);
                const index_t bs1 = b.stride(1);
                const index_t bs2 = b.stride(2);
                const index_t cs0 = c.stride(0);
                const index_t cs1 = c.stride(1);
                const index_t cs2 = c.stride(2);

                if (as2 == 1 && bs2 == 1 && cs2 == 1) {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t abase0 = i * as0;
                        const index_t bbase0 = i * bs0;
                        const index_t cbase0 = i * cs0;
                        for (index_t j = 0; j < n1; ++j) {
                            const index_t abase1 = abase0 + j * as1;
                            const index_t bbase1 = bbase0 + j * bs1;
                            const index_t cbase1 = cbase0 + j * cs1;
                            for (index_t k = 0; k < n2; ++k) {
                                f(abase1 + k, bbase1 + k, cbase1 + k);
                            }
                        }
                    }
                } else {
                    for (index_t i = 0; i < n0; ++i) {
                        const index_t abase0 = i * as0;
                        const index_t bbase0 = i * bs0;
                        const index_t cbase0 = i * cs0;
                        for (index_t j = 0; j < n1; ++j) {
                            const index_t abase1 = abase0 + j * as1;
                            const index_t bbase1 = bbase0 + j * bs1;
                            const index_t cbase1 = cbase0 + j * cs1;
                            for (index_t k = 0; k < n2; ++k) {
                                f(abase1 + k * as2, bbase1 + k * bs2, cbase1 + k * cs2);
                            }
                        }
                    }
                }
            } else {
                index_t idx[Rank] = {};
                index_t extent[Rank];
                index_t astep[Rank];
                index_t bstep[Rank];
                index_t cstep[Rank];
                index_t areset[Rank];
                index_t breset[Rank];
                index_t creset[Rank];
                for (index_t d = 0; d < Rank; ++d) {
                    extent[d] = a.extent(d);
                    astep[d]  = a.stride(d);
                    bstep[d]  = b.stride(d);
                    cstep[d]  = c.stride(d);
                    areset[d] = (extent[d] - 1) * astep[d];
                    breset[d] = (extent[d] - 1) * bstep[d];
                    creset[d] = (extent[d] - 1) * cstep[d];
                }

                index_t ao = 0;
                index_t bo = 0;
                index_t co = 0;
                while (true) {
                    f(ao, bo, co);

                    for (index_t r = Rank; r > 0; --r) {
                        const index_t d = r - 1;
                        ++idx[d];
                        if (idx[d] < extent[d]) {
                            ao += astep[d];
                            bo += bstep[d];
                            co += cstep[d];
                            break;
                        }
                        if (d == 0) {
                            return;
                        }
                        idx[d] = 0;
                        ao -= areset[d];
                        bo -= breset[d];
                        co -= creset[d];
                    }
                }
            }
        }

        // Temporary owning storage with the same shape as a View.

        template <typename T, typename U, index_t Rank>
        inline Array<T, Rank> make_array_like(View<U, Rank> v) {
            index_t extent[Rank];
            for (index_t d = 0; d < Rank; ++d) {
                extent[d] = v.extent(d);
            }
            return Array<T, Rank>(extent);
        }

        // Contiguous noalias kernels. Only writable output pointers are restrict.
        // Read-only inputs are allowed to alias each other.

        template <typename T>
        inline void copy_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT dst,
                                            const T* src) noexcept {
            std::memcpy(static_cast<void*>(dst), static_cast<const void*>(src), n * sizeof(T));
        }

        template <typename T>
        inline void add_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT dst, const T* x,
                                           const T* y) noexcept {
            for (index_t i = 0; i < n; ++i) {
                dst[i] = x[i] + y[i];
            }
        }

        template <typename T>
        inline void sub_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT dst, const T* x,
                                           const T* y) noexcept {
            for (index_t i = 0; i < n; ++i) {
                dst[i] = x[i] - y[i];
            }
        }

        template <typename T>
        inline void mul_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT dst, const T* x,
                                           const T* y) noexcept {
            for (index_t i = 0; i < n; ++i) {
                dst[i] = x[i] * y[i];
            }
        }

        template <typename T>
        inline void div_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT dst, const T* x,
                                           const T* y) noexcept {
            for (index_t i = 0; i < n; ++i) {
                dst[i] = x[i] / y[i];
            }
        }

        template <typename T>
        inline void axpby_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT dst, T alpha,
                                             const T* x, T beta, const T* y) noexcept {
            for (index_t i = 0; i < n; ++i) {
                dst[i] = alpha * x[i] + beta * y[i];
            }
        }

        template <typename T>
        inline void add_inplace_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT dst,
                                                   const T* x) noexcept {
            for (index_t i = 0; i < n; ++i) {
                dst[i] += x[i];
            }
        }

        template <typename T>
        inline void sub_inplace_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT dst,
                                                   const T* x) noexcept {
            for (index_t i = 0; i < n; ++i) {
                dst[i] -= x[i];
            }
        }

        template <typename T>
        inline void mul_inplace_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT dst,
                                                   const T* x) noexcept {
            for (index_t i = 0; i < n; ++i) {
                dst[i] *= x[i];
            }
        }

        template <typename T>
        inline void axpy_contiguous_noalias(index_t n, T* VEHKKO_NDARRAY_RESTRICT y, T alpha,
                                            const T* x) noexcept {
            for (index_t i = 0; i < n; ++i) {
                y[i] += alpha * x[i];
            }
        }

        // Absolute-value helpers without signed-integer negation overflow.

        template <typename T, bool SignedIntegral = std::is_integral_v<T> && std::is_signed_v<T>>
        struct abs_result {
            using type = T;
        };

        template <typename T>
        struct abs_result<T, true> {
            using type = std::make_unsigned_t<T>;
        };

        template <typename T>
        using abs_result_t = typename abs_result<std::remove_const_t<T>>::type;

        template <typename T>
        inline abs_result_t<T> abs_value(T x) noexcept {
            using E = std::remove_const_t<T>;
            static_assert(std::is_arithmetic_v<E>, "abs_value requires an arithmetic type");

            if constexpr (std::is_floating_point_v<E>) {
                return x < E{} ? -x : x;
            } else if constexpr (std::is_signed_v<E>) {
                using U    = std::make_unsigned_t<E>;
                const U ux = static_cast<U>(x);
                return x < E{} ? U{} - ux : ux;
            } else {
                return x;
            }
        }

        template <typename T>
        inline abs_result_t<T> abs_diff_value(T a, T b) noexcept {
            using E = std::remove_const_t<T>;
            static_assert(std::is_arithmetic_v<E>, "abs_diff_value requires an arithmetic type");

            if constexpr (std::is_floating_point_v<E>) {
                return abs_value(a - b);
            } else {
                using U    = abs_result_t<E>;
                const U ua = static_cast<U>(a);
                const U ub = static_cast<U>(b);
                return a < b ? ub - ua : ua - ub;
            }
        }

    } // namespace detail

    // Unary in-place operations
    // -----------------------------------------------------------------------------

    template <typename T, index_t Rank>
    inline void fill_inplace(View<T, Rank> dst, T value) noexcept {
        static_assert(!std::is_const_v<T>, "fill_inplace: dst must be writable");

        if (is_row_major_contiguous(dst)) {
            T*            p = dst.data();
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                p[i] = value;
            }
            return;
        }

        T* p = dst.data();
        detail::for_each_offset(dst, [&](index_t o) noexcept { p[o] = value; });
    }

    template <typename T, index_t Rank>
    inline void zero_inplace(View<T, Rank> dst) noexcept {
        fill_inplace(dst, T{});
    }

    template <typename T, typename U, index_t Rank>
    inline void set_inplace(View<T, Rank> dst, std::initializer_list<U> values) noexcept {
        static_assert(!std::is_const_v<T>, "set_inplace: dst must be writable");

        static_assert(std::is_assignable_v<T&, const U&>,
                      "set_inplace: values must be assignable to dst elements");

        if constexpr (runtime_shape_check) {
            if (dst.size() != values.size()) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][shape] set_inplace: rank=%zu, dst_size=%zu, "
                             "values_size=%zu, dst_data=%p; %s:%d\n",
                             Rank, dst.size(), values.size(),
                             const_cast<void*>(static_cast<const void*>(dst.data())), __FILE__,
                             __LINE__);
                contract_violation();
            }
        }

        T*       data = dst.data();
        const U* src  = values.begin();
        index_t  i    = 0;

        detail::for_each_offset(dst, [&](index_t offset) noexcept { data[offset] = src[i++]; });
    }

    template <typename T, index_t Rank>
    inline void scale_inplace(View<T, Rank> dst, T alpha) noexcept {
        static_assert(!std::is_const_v<T>, "scale_inplace: dst must be writable");

        if (is_row_major_contiguous(dst)) {
            T*            p = dst.data();
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                p[i] *= alpha;
            }
            return;
        }

        T* p = dst.data();
        detail::for_each_offset(dst, [&](index_t o) noexcept { p[o] *= alpha; });
    }

    template <typename T, index_t Rank>
    inline void shift_inplace(View<T, Rank> dst, T beta) noexcept {
        static_assert(!std::is_const_v<T>, "shift_inplace: dst must be writable");

        if (is_row_major_contiguous(dst)) {
            T*            p = dst.data();
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                p[i] += beta;
            }
            return;
        }

        T* p = dst.data();
        detail::for_each_offset(dst, [&](index_t o) noexcept { p[o] += beta; });
    }

    template <typename T, index_t Rank>
    inline void negate_inplace(View<T, Rank> dst) noexcept {
        static_assert(!std::is_const_v<T>, "negate_inplace: dst must be writable");

        if (is_row_major_contiguous(dst)) {
            T*            p = dst.data();
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                p[i] = -p[i];
            }
            return;
        }

        T* p = dst.data();
        detail::for_each_offset(dst, [&](index_t o) noexcept { p[o] = -p[o]; });
    }

    template <typename T, index_t Rank>
    inline void square_inplace(View<T, Rank> dst) noexcept {
        static_assert(!std::is_const_v<T>, "square_inplace: dst must be writable");

        if (is_row_major_contiguous(dst)) {
            T*            p = dst.data();
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                p[i] *= p[i];
            }
            return;
        }

        T* p = dst.data();
        detail::for_each_offset(dst, [&](index_t o) noexcept { p[o] *= p[o]; });
    }

    // Copy
    // -----------------------------------------------------------------------------

    template <typename T, typename S, index_t Rank>
    inline void copy_to_noalias(View<T, Rank> dst, View<S, Rank> src) noexcept {
        static_assert(!std::is_const_v<T>, "copy_to_noalias: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<S>>,
                      "copy_to_noalias: value types must match");

        detail::require_same_extents(dst, src);
        detail::check_noalias_contract(dst, src);

        if (dst.empty()) {
            return;
        }

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(src)) {
            detail::copy_contiguous_noalias(dst.size(), dst.data(), src.data());
            return;
        }

        T*       d = dst.data();
        const T* s = src.data();
        detail::for_each_offset2(dst, src, [&](index_t doff, index_t soff) noexcept { d[doff] = s[soff]; });
    }

    template <typename T, typename S, index_t Rank>
    inline void copy_to(View<T, Rank> dst, View<S, Rank> src) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "copy_to: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<S>>, "copy_to: value types must match");

        detail::require_same_extents(dst, src);
        if (dst.empty()) {
            return;
        }

        if constexpr (!runtime_alias_dispatch) {
            if (is_row_major_contiguous(dst) && is_row_major_contiguous(src)) {
                std::memmove(static_cast<void*>(dst.data()), static_cast<const void*>(src.data()), dst.size() * sizeof(T));
                return;
            }

            T*       d = dst.data();
            const T* s = src.data();
            detail::for_each_offset2(
                dst, src, [&](index_t doff, index_t soff) noexcept { d[doff] = s[soff]; });
            return;
        }

        const detail::AliasRelation relation = detail::classify_alias(dst, src);
        if (relation == detail::AliasRelation::same_mapping) {
            return;
        }
        if (relation == detail::AliasRelation::disjoint) {
            copy_to_noalias(dst, src);
            return;
        }

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(src)) {
            std::memmove(static_cast<void*>(dst.data()), static_cast<const void*>(src.data()), dst.size() * sizeof(T));
            return;
        }

        if constexpr (Rank == 1) {
            if (dst.stride(0) == src.stride(0)) {
                T*                   d  = dst.data();
                const T*             s  = src.data();
                const index_t        n  = dst.extent(0);
                const index_t        ds = dst.stride(0);
                const index_t        ss = src.stride(0);
                const std::uintptr_t dp = reinterpret_cast<std::uintptr_t>(dst.data());
                const std::uintptr_t sp = reinterpret_cast<std::uintptr_t>(src.data());

                if (dp < sp) {
                    for (index_t i = 0; i < n; ++i) {
                        d[i * ds] = s[i * ss];
                    }
                } else {
                    for (index_t r = n; r > 0; --r) {
                        const index_t i = r - 1;
                        d[i * ds]       = s[i * ss];
                    }
                }
                return;
            }
        }

        auto temporary = detail::make_array_like<T>(dst);
        copy_to_noalias(temporary.view(), src);
        copy_to_noalias(dst, temporary.view());
    }

    // Elementwise _to operations
    // -----------------------------------------------------------------------------

    template <typename T, typename X, typename Y, index_t Rank>
    inline void add_to_noalias(View<T, Rank> dst, View<X, Rank> x, View<Y, Rank> y) noexcept {
        static_assert(!std::is_const_v<T>, "add_to_noalias: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "add_to_noalias: value types must match");

        detail::require_same_extents(dst, x, y);
        detail::check_noalias_contract(dst, x);
        detail::check_noalias_contract(dst, y);

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            detail::add_contiguous_noalias(dst.size(), dst.data(), x.data(), y.data());
            return;
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = xp[xoff] + yp[yoff];
        });
    }

    template <typename T, typename X, typename Y, index_t Rank>
    inline void add_to(View<T, Rank> dst, View<X, Rank> x,
                       View<Y, Rank> y) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "add_to: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "add_to: value types must match");

        detail::require_same_extents(dst, x, y);

        if constexpr (runtime_alias_dispatch) {
            const auto rx = detail::classify_alias(dst, x);
            const auto ry = detail::classify_alias(dst, y);
            if (rx == detail::AliasRelation::disjoint && ry == detail::AliasRelation::disjoint) {
                add_to_noalias(dst, x, y);
                return;
            }
            if (rx == detail::AliasRelation::may_overlap ||
                ry == detail::AliasRelation::may_overlap) {
                auto temporary = detail::make_array_like<T>(dst);
                add_to_noalias(temporary.view(), x, y);
                copy_to_noalias(dst, temporary.view());
                return;
            }
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                d[i] = xp[i] + yp[i];
            }
            return;
        }

        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = xp[xoff] + yp[yoff];
        });
    }

    template <typename T, typename X, typename Y, index_t Rank>
    inline void sub_to_noalias(View<T, Rank> dst, View<X, Rank> x, View<Y, Rank> y) noexcept {
        static_assert(!std::is_const_v<T>, "sub_to_noalias: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "sub_to_noalias: value types must match");

        detail::require_same_extents(dst, x, y);
        detail::check_noalias_contract(dst, x);
        detail::check_noalias_contract(dst, y);

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            detail::sub_contiguous_noalias(dst.size(), dst.data(), x.data(), y.data());
            return;
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = xp[xoff] - yp[yoff];
        });
    }

    template <typename T, typename X, typename Y, index_t Rank>
    inline void sub_to(View<T, Rank> dst, View<X, Rank> x,
                       View<Y, Rank> y) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "sub_to: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "sub_to: value types must match");

        detail::require_same_extents(dst, x, y);

        if constexpr (runtime_alias_dispatch) {
            const auto rx = detail::classify_alias(dst, x);
            const auto ry = detail::classify_alias(dst, y);
            if (rx == detail::AliasRelation::disjoint && ry == detail::AliasRelation::disjoint) {
                sub_to_noalias(dst, x, y);
                return;
            }
            if (rx == detail::AliasRelation::may_overlap ||
                ry == detail::AliasRelation::may_overlap) {
                auto temporary = detail::make_array_like<T>(dst);
                sub_to_noalias(temporary.view(), x, y);
                copy_to_noalias(dst, temporary.view());
                return;
            }
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                d[i] = xp[i] - yp[i];
            }
            return;
        }

        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = xp[xoff] - yp[yoff];
        });
    }

    template <typename T, typename X, typename Y, index_t Rank>
    inline void mul_to_noalias(View<T, Rank> dst, View<X, Rank> x, View<Y, Rank> y) noexcept {
        static_assert(!std::is_const_v<T>, "mul_to_noalias: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "mul_to_noalias: value types must match");

        detail::require_same_extents(dst, x, y);
        detail::check_noalias_contract(dst, x);
        detail::check_noalias_contract(dst, y);

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            detail::mul_contiguous_noalias(dst.size(), dst.data(), x.data(), y.data());
            return;
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = xp[xoff] * yp[yoff];
        });
    }

    template <typename T, typename X, typename Y, index_t Rank>
    inline void mul_to(View<T, Rank> dst, View<X, Rank> x,
                       View<Y, Rank> y) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "mul_to: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "mul_to: value types must match");

        detail::require_same_extents(dst, x, y);

        if constexpr (runtime_alias_dispatch) {
            const auto rx = detail::classify_alias(dst, x);
            const auto ry = detail::classify_alias(dst, y);
            if (rx == detail::AliasRelation::disjoint && ry == detail::AliasRelation::disjoint) {
                mul_to_noalias(dst, x, y);
                return;
            }
            if (rx == detail::AliasRelation::may_overlap ||
                ry == detail::AliasRelation::may_overlap) {
                auto temporary = detail::make_array_like<T>(dst);
                mul_to_noalias(temporary.view(), x, y);
                copy_to_noalias(dst, temporary.view());
                return;
            }
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                d[i] = xp[i] * yp[i];
            }
            return;
        }

        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = xp[xoff] * yp[yoff];
        });
    }

    template <typename T, typename X, typename Y, index_t Rank>
    inline void div_to_noalias(View<T, Rank> dst, View<X, Rank> x, View<Y, Rank> y) noexcept {
        static_assert(!std::is_const_v<T>, "div_to_noalias: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "div_to_noalias: value types must match");

        detail::require_same_extents(dst, x, y);
        detail::check_noalias_contract(dst, x);
        detail::check_noalias_contract(dst, y);

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            detail::div_contiguous_noalias(dst.size(), dst.data(), x.data(), y.data());
            return;
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = xp[xoff] / yp[yoff];
        });
    }

    template <typename T, typename X, typename Y, index_t Rank>
    inline void div_to(View<T, Rank> dst, View<X, Rank> x,
                       View<Y, Rank> y) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "div_to: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "div_to: value types must match");

        detail::require_same_extents(dst, x, y);

        if constexpr (runtime_alias_dispatch) {
            const auto rx = detail::classify_alias(dst, x);
            const auto ry = detail::classify_alias(dst, y);
            if (rx == detail::AliasRelation::disjoint && ry == detail::AliasRelation::disjoint) {
                div_to_noalias(dst, x, y);
                return;
            }
            if (rx == detail::AliasRelation::may_overlap ||
                ry == detail::AliasRelation::may_overlap) {
                auto temporary = detail::make_array_like<T>(dst);
                div_to_noalias(temporary.view(), x, y);
                copy_to_noalias(dst, temporary.view());
                return;
            }
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                d[i] = xp[i] / yp[i];
            }
            return;
        }

        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = xp[xoff] / yp[yoff];
        });
    }

    template <typename T, typename X, typename Y, index_t Rank>
    inline void axpby_to_noalias(View<T, Rank> dst, T alpha, View<X, Rank> x, T beta,
                                 View<Y, Rank> y) noexcept {
        static_assert(!std::is_const_v<T>, "axpby_to_noalias: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "axpby_to_noalias: value types must match");

        detail::require_same_extents(dst, x, y);
        detail::check_noalias_contract(dst, x);
        detail::check_noalias_contract(dst, y);

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            detail::axpby_contiguous_noalias(dst.size(), dst.data(), alpha, x.data(), beta, y.data());
            return;
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = alpha * xp[xoff] + beta * yp[yoff];
        });
    }

    template <typename T, typename X, typename Y, index_t Rank>
    inline void axpby_to(View<T, Rank> dst, T alpha, View<X, Rank> x, T beta,
                         View<Y, Rank> y) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "axpby_to: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>> &&
                          std::is_same_v<T, std::remove_const_t<Y>>,
                      "axpby_to: value types must match");

        detail::require_same_extents(dst, x, y);

        if constexpr (runtime_alias_dispatch) {
            const auto rx = detail::classify_alias(dst, x);
            const auto ry = detail::classify_alias(dst, y);
            if (rx == detail::AliasRelation::disjoint && ry == detail::AliasRelation::disjoint) {
                axpby_to_noalias(dst, alpha, x, beta, y);
                return;
            }
            if (rx == detail::AliasRelation::may_overlap ||
                ry == detail::AliasRelation::may_overlap) {
                auto temporary = detail::make_array_like<T>(dst);
                axpby_to_noalias(temporary.view(), alpha, x, beta, y);
                copy_to_noalias(dst, temporary.view());
                return;
            }
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        const T* yp = y.data();
        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x) &&
            is_row_major_contiguous(y)) {
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                d[i] = alpha * xp[i] + beta * yp[i];
            }
            return;
        }

        detail::for_each_offset3(dst, x, y, [&](index_t doff, index_t xoff, index_t yoff) noexcept {
            d[doff] = alpha * xp[xoff] + beta * yp[yoff];
        });
    }

    // Binary in-place operations
    // -----------------------------------------------------------------------------

    template <typename T, typename X, index_t Rank>
    inline void add_inplace_noalias(View<T, Rank> dst, View<X, Rank> x) noexcept {
        static_assert(!std::is_const_v<T>, "add_inplace_noalias: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>>,
                      "add_inplace_noalias: value types must match");

        detail::require_same_extents(dst, x);
        detail::check_noalias_contract(dst, x);

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x)) {
            detail::add_inplace_contiguous_noalias(dst.size(), dst.data(), x.data());
            return;
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        detail::for_each_offset2(dst, x, [&](index_t doff, index_t xoff) noexcept { d[doff] += xp[xoff]; });
    }

    template <typename T, typename X, index_t Rank>
    inline void add_inplace(View<T, Rank> dst, View<X, Rank> x) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "add_inplace: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>>,
                      "add_inplace: value types must match");

        detail::require_same_extents(dst, x);

        if constexpr (runtime_alias_dispatch) {
            const auto relation = detail::classify_alias(dst, x);
            if (relation == detail::AliasRelation::disjoint) {
                add_inplace_noalias(dst, x);
                return;
            }
            if (relation == detail::AliasRelation::may_overlap) {
                auto temporary = detail::make_array_like<T>(dst);
                add_to_noalias(temporary.view(), dst, x);
                copy_to_noalias(dst, temporary.view());
                return;
            }
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x)) {
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                d[i] += xp[i];
            }
            return;
        }

        detail::for_each_offset2(dst, x, [&](index_t doff, index_t xoff) noexcept { d[doff] += xp[xoff]; });
    }

    template <typename T, typename X, index_t Rank>
    inline void sub_inplace_noalias(View<T, Rank> dst, View<X, Rank> x) noexcept {
        static_assert(!std::is_const_v<T>, "sub_inplace_noalias: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>>,
                      "sub_inplace_noalias: value types must match");

        detail::require_same_extents(dst, x);
        detail::check_noalias_contract(dst, x);

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x)) {
            detail::sub_inplace_contiguous_noalias(dst.size(), dst.data(), x.data());
            return;
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        detail::for_each_offset2(dst, x, [&](index_t doff, index_t xoff) noexcept { d[doff] -= xp[xoff]; });
    }

    template <typename T, typename X, index_t Rank>
    inline void sub_inplace(View<T, Rank> dst, View<X, Rank> x) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "sub_inplace: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>>,
                      "sub_inplace: value types must match");

        detail::require_same_extents(dst, x);

        if constexpr (runtime_alias_dispatch) {
            const auto relation = detail::classify_alias(dst, x);
            if (relation == detail::AliasRelation::disjoint) {
                sub_inplace_noalias(dst, x);
                return;
            }
            if (relation == detail::AliasRelation::may_overlap) {
                auto temporary = detail::make_array_like<T>(dst);
                sub_to_noalias(temporary.view(), dst, x);
                copy_to_noalias(dst, temporary.view());
                return;
            }
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x)) {
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                d[i] -= xp[i];
            }
            return;
        }

        detail::for_each_offset2(dst, x, [&](index_t doff, index_t xoff) noexcept { d[doff] -= xp[xoff]; });
    }

    template <typename T, typename X, index_t Rank>
    inline void mul_inplace_noalias(View<T, Rank> dst, View<X, Rank> x) noexcept {
        static_assert(!std::is_const_v<T>, "mul_inplace_noalias: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>>,
                      "mul_inplace_noalias: value types must match");

        detail::require_same_extents(dst, x);
        detail::check_noalias_contract(dst, x);

        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x)) {
            detail::mul_inplace_contiguous_noalias(dst.size(), dst.data(), x.data());
            return;
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        detail::for_each_offset2(dst, x, [&](index_t doff, index_t xoff) noexcept { d[doff] *= xp[xoff]; });
    }

    template <typename T, typename X, index_t Rank>
    inline void mul_inplace(View<T, Rank> dst, View<X, Rank> x) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "mul_inplace: dst must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>>,
                      "mul_inplace: value types must match");

        detail::require_same_extents(dst, x);

        if constexpr (runtime_alias_dispatch) {
            const auto relation = detail::classify_alias(dst, x);
            if (relation == detail::AliasRelation::disjoint) {
                mul_inplace_noalias(dst, x);
                return;
            }
            if (relation == detail::AliasRelation::may_overlap) {
                auto temporary = detail::make_array_like<T>(dst);
                mul_to_noalias(temporary.view(), dst, x);
                copy_to_noalias(dst, temporary.view());
                return;
            }
        }

        T*       d  = dst.data();
        const T* xp = x.data();
        if (is_row_major_contiguous(dst) && is_row_major_contiguous(x)) {
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                d[i] *= xp[i];
            }
            return;
        }

        detail::for_each_offset2(dst, x, [&](index_t doff, index_t xoff) noexcept { d[doff] *= xp[xoff]; });
    }

    template <typename T, typename X, index_t Rank>
    inline void axpy_inplace_noalias(View<T, Rank> y, T alpha, View<X, Rank> x) noexcept {
        static_assert(!std::is_const_v<T>, "axpy_inplace_noalias: y must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>>,
                      "axpy_inplace_noalias: value types must match");

        detail::require_same_extents(y, x);
        detail::check_noalias_contract(y, x);

        if (is_row_major_contiguous(y) && is_row_major_contiguous(x)) {
            detail::axpy_contiguous_noalias(y.size(), y.data(), alpha, x.data());
            return;
        }

        T*       yp = y.data();
        const T* xp = x.data();
        detail::for_each_offset2(
            y, x, [&](index_t yoff, index_t xoff) noexcept { yp[yoff] += alpha * xp[xoff]; });
    }

    template <typename T, typename X, index_t Rank>
    inline void axpy_inplace(View<T, Rank> y, T alpha,
                             View<X, Rank> x) noexcept(!runtime_alias_dispatch) {
        static_assert(!std::is_const_v<T>, "axpy_inplace: y must be writable");
        static_assert(std::is_same_v<T, std::remove_const_t<X>>,
                      "axpy_inplace: value types must match");

        detail::require_same_extents(y, x);

        if constexpr (runtime_alias_dispatch) {
            const auto relation = detail::classify_alias(y, x);
            if (relation == detail::AliasRelation::disjoint) {
                axpy_inplace_noalias(y, alpha, x);
                return;
            }
            if (relation == detail::AliasRelation::may_overlap) {
                auto temporary = detail::make_array_like<T>(y);
                axpby_to_noalias(temporary.view(), T{1}, y, alpha, x);
                copy_to_noalias(y, temporary.view());
                return;
            }
        }

        T*       yp = y.data();
        const T* xp = x.data();
        if (is_row_major_contiguous(y) && is_row_major_contiguous(x)) {
            const index_t n = y.size();
            for (index_t i = 0; i < n; ++i) {
                yp[i] += alpha * xp[i];
            }
            return;
        }

        detail::for_each_offset2(
            y, x, [&](index_t yoff, index_t xoff) noexcept { yp[yoff] += alpha * xp[xoff]; });
    }

    // Reductions and comparisons
    // -----------------------------------------------------------------------------

    template <typename T, index_t Rank>
    inline std::remove_const_t<T> sum(View<T, Rank> x) noexcept {
        using R = std::remove_const_t<T>;
        R result{};

        if (is_row_major_contiguous(x)) {
            const T*      p = x.data();
            const index_t n = x.size();
            for (index_t i = 0; i < n; ++i) {
                result += p[i];
            }
            return result;
        }

        const T* p = x.data();
        detail::for_each_offset(x, [&](index_t o) noexcept { result += p[o]; });
        return result;
    }

    template <typename T, typename U, index_t Rank>
    inline std::remove_const_t<T> dot(View<T, Rank> x, View<U, Rank> y) noexcept {
        static_assert(std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>,
                      "dot: value types must match");
        using R = std::remove_const_t<T>;
        detail::require_same_extents(x, y);

        R result{};
        if (is_row_major_contiguous(x) && is_row_major_contiguous(y)) {
            const T*      xp = x.data();
            const U*      yp = y.data();
            const index_t n  = x.size();
            for (index_t i = 0; i < n; ++i) {
                result += xp[i] * yp[i];
            }
            return result;
        }

        const T* xp = x.data();
        const U* yp = y.data();
        detail::for_each_offset2(
            x, y, [&](index_t xo, index_t yo) noexcept { result += xp[xo] * yp[yo]; });
        return result;
    }

    template <typename T, index_t Rank>
    inline std::remove_const_t<T> squared_norm(View<T, Rank> x) noexcept {
        using R = std::remove_const_t<T>;
        R result{};

        if (is_row_major_contiguous(x)) {
            const T*      p = x.data();
            const index_t n = x.size();
            for (index_t i = 0; i < n; ++i) {
                result += p[i] * p[i];
            }
            return result;
        }

        const T* p = x.data();
        detail::for_each_offset(x, [&](index_t o) noexcept { result += p[o] * p[o]; });
        return result;
    }

    template <typename T, index_t Rank>
    inline detail::abs_result_t<T> max_abs(View<T, Rank> x) noexcept {
        using R = detail::abs_result_t<T>;
        R result{};

        if (is_row_major_contiguous(x)) {
            const T*      p = x.data();
            const index_t n = x.size();
            for (index_t i = 0; i < n; ++i) {
                const R a = detail::abs_value(p[i]);
                result    = result < a ? a : result;
            }
            return result;
        }

        const T* p = x.data();
        detail::for_each_offset(x, [&](index_t o) noexcept {
            const R a = detail::abs_value(p[o]);
            result    = result < a ? a : result;
        });
        return result;
    }

    template <typename T, index_t Rank>
    inline std::remove_const_t<T> min_value(View<T, Rank> x) noexcept {
        using R = std::remove_const_t<T>;
        if constexpr (runtime_shape_check) {
            if (x.empty()) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][shape] min_value: input view is empty, rank=%zu, "
                             "data=%p; %s:%d\n",
                             Rank, const_cast<void*>(static_cast<const void*>(x.data())), __FILE__,
                             __LINE__);
                contract_violation();
            }
        }

        const T* p      = x.data();
        R        result = p[0];
        detail::for_each_offset(
            x, [&](index_t o) noexcept { result = p[o] < result ? p[o] : result; });
        return result;
    }

    template <typename T, index_t Rank>
    inline std::remove_const_t<T> max_value(View<T, Rank> x) noexcept {
        using R = std::remove_const_t<T>;
        if constexpr (runtime_shape_check) {
            if (x.empty()) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][shape] max_value: input view is empty, rank=%zu, "
                             "data=%p; %s:%d\n",
                             Rank, const_cast<void*>(static_cast<const void*>(x.data())), __FILE__,
                             __LINE__);
                contract_violation();
            }
        }

        const T* p      = x.data();
        R        result = p[0];
        detail::for_each_offset(
            x, [&](index_t o) noexcept { result = result < p[o] ? p[o] : result; });
        return result;
    }

    template <typename T, typename U, index_t Rank>
    inline detail::abs_result_t<T> max_abs_diff(View<T, Rank> x, View<U, Rank> y) noexcept {
        static_assert(std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>,
                      "max_abs_diff: value types must match");
        using R = detail::abs_result_t<T>;
        detail::require_same_extents(x, y);

        R        result{};
        const T* xp = x.data();
        const U* yp = y.data();
        detail::for_each_offset2(x, y, [&](index_t xo, index_t yo) noexcept {
            const R d = detail::abs_diff_value(xp[xo], yp[yo]);
            result    = result < d ? d : result;
        });
        return result;
    }

    template <typename T, typename U, index_t Rank>
    inline std::remove_const_t<T> squared_diff_norm(View<T, Rank> x, View<U, Rank> y) noexcept {
        static_assert(std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>,
                      "squared_diff_norm: value types must match");
        using R = std::remove_const_t<T>;
        detail::require_same_extents(x, y);

        R        result{};
        const T* xp = x.data();
        const U* yp = y.data();
        detail::for_each_offset2(x, y, [&](index_t xo, index_t yo) noexcept {
            const R d = xp[xo] - yp[yo];
            result += d * d;
        });
        return result;
    }

    template <typename T, typename U, index_t Rank>
    inline bool all_close(View<T, Rank> x, View<U, Rank> y, std::remove_const_t<T> atol,
                          std::remove_const_t<T> rtol) noexcept {
        static_assert(std::is_same_v<std::remove_const_t<T>, std::remove_const_t<U>>,
                      "all_close: value types must match");
        using R = std::remove_const_t<T>;
        static_assert(std::is_floating_point_v<R>,
                      "all_close is defined only for floating-point values");
        detail::require_same_extents(x, y);

        const T* xp = x.data();
        const U* yp = y.data();
        bool     ok = true;
        detail::for_each_offset2(x, y, [&](index_t xo, index_t yo) noexcept {
            if (!ok) {
                return;
            }
            const R diff      = detail::abs_value(xp[xo] - yp[yo]);
            const R tolerance = atol + rtol * detail::abs_value(yp[yo]);
            ok                = diff <= tolerance;
        });
        return ok;
    }

    // Initialization helpers
    // -----------------------------------------------------------------------------

    template <typename T, index_t Rank>
    inline void iota_inplace(View<T, Rank> dst, T start = T{}, T step = T{1}) noexcept {
        static_assert(!std::is_const_v<T>, "iota_inplace: dst must be writable");
        T value = start;

        if (is_row_major_contiguous(dst)) {
            T*            p = dst.data();
            const index_t n = dst.size();
            for (index_t i = 0; i < n; ++i) {
                p[i] = value;
                value += step;
            }
            return;
        }

        T* p = dst.data();
        detail::for_each_offset(dst, [&](index_t o) noexcept {
            p[o] = value;
            value += step;
        });
    }

    template <typename T, index_t Rank>
    inline void linspace_inplace(View<T, Rank> dst, T first, T last) noexcept {
        static_assert(!std::is_const_v<T>, "linspace_inplace: dst must be writable");
        const index_t n = dst.size();
        if (n == 0) {
            return;
        }
        if (n == 1) {
            fill_inplace(dst, first);
            return;
        }

        const T step = (last - first) / static_cast<T>(n - 1);
        iota_inplace(dst, first, step);

        index_t last_offset = 0;
        for (index_t d = 0; d < Rank; ++d) {
            last_offset += (dst.extent(d) - 1) * dst.stride(d);
        }
        dst.data()[last_offset] = last;
    }

} // namespace vehkko::ndarray
