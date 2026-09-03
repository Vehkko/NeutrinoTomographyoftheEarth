#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

// Configuration
// -----------------------------------------------------------------------------
// Every switch is an integer macro and may be defined to 0 or 1 before this
// header is included. Checks guarded by these constants are removed completely
// when disabled.

#ifndef VEHKKO_NDARRAY_BOUNDS_CHECK
#ifdef NDEBUG
#define VEHKKO_NDARRAY_BOUNDS_CHECK 0
#else
#define VEHKKO_NDARRAY_BOUNDS_CHECK 1
#endif
#endif

#ifndef VEHKKO_NDARRAY_SHAPE_CHECK
#define VEHKKO_NDARRAY_SHAPE_CHECK 1
#endif

#ifndef VEHKKO_NDARRAY_OVERFLOW_CHECK
#define VEHKKO_NDARRAY_OVERFLOW_CHECK 1
#endif

#ifndef VEHKKO_NDARRAY_ALIGNMENT_CHECK
#ifdef NDEBUG
#define VEHKKO_NDARRAY_ALIGNMENT_CHECK 0
#else
#define VEHKKO_NDARRAY_ALIGNMENT_CHECK 1
#endif
#endif

#ifndef VEHKKO_NDARRAY_ALLOCATOR_CHECK
#ifdef NDEBUG
#define VEHKKO_NDARRAY_ALLOCATOR_CHECK 0
#else
#define VEHKKO_NDARRAY_ALLOCATOR_CHECK 1
#endif
#endif

#ifndef VEHKKO_NDARRAY_ALIAS_DISPATCH
#define VEHKKO_NDARRAY_ALIAS_DISPATCH 1
#endif

#ifndef VEHKKO_NDARRAY_NOALIAS_CONTRACT_CHECK
#ifdef NDEBUG
#define VEHKKO_NDARRAY_NOALIAS_CONTRACT_CHECK 0
#else
#define VEHKKO_NDARRAY_NOALIAS_CONTRACT_CHECK 1
#endif
#endif

#ifndef VEHKKO_NDARRAY_LAYOUT_CHECK
#ifdef NDEBUG
#define VEHKKO_NDARRAY_LAYOUT_CHECK 0
#else
#define VEHKKO_NDARRAY_LAYOUT_CHECK 1
#endif
#endif

namespace vehkko::ndarray {

    using index_t                              = std::size_t;
    inline constexpr index_t default_alignment = 64;

    inline constexpr bool runtime_bounds_check    = VEHKKO_NDARRAY_BOUNDS_CHECK != 0;
    inline constexpr bool runtime_shape_check     = VEHKKO_NDARRAY_SHAPE_CHECK != 0;
    inline constexpr bool runtime_overflow_check  = VEHKKO_NDARRAY_OVERFLOW_CHECK != 0;
    inline constexpr bool runtime_alignment_check = VEHKKO_NDARRAY_ALIGNMENT_CHECK != 0;
    inline constexpr bool runtime_allocator_check = VEHKKO_NDARRAY_ALLOCATOR_CHECK != 0;
    inline constexpr bool runtime_alias_dispatch  = VEHKKO_NDARRAY_ALIAS_DISPATCH != 0;
    inline constexpr bool runtime_noalias_contract_check =
        VEHKKO_NDARRAY_NOALIAS_CONTRACT_CHECK != 0;
    inline constexpr bool runtime_layout_check = VEHKKO_NDARRAY_LAYOUT_CHECK != 0;

    // Contract failures do not create or propagate library-defined exceptions.
    // Allocation and element-construction failures may still propagate.
    [[noreturn]] inline void contract_violation() noexcept {
        std::fflush(stderr);
        std::terminate();
    }

    inline void require_contract(bool condition) noexcept {
        if (!condition) {
            contract_violation();
        }
    }

    // Compile-time type checking
    // -----------------------------------------------------------------------------
    // Storage remains intentionally restricted to simple, byte-copyable objects.
    // Non-trivial default construction is permitted; this supports
    // more POD-like numeric types without requiring a lifetime-policy abstraction.

    template <typename T>
    inline constexpr bool is_valid_storage_element_v =
        !std::is_void_v<T> && !std::is_reference_v<T> && !std::is_const_v<T> &&
        !std::is_volatile_v<T> && std::is_trivially_copyable_v<T> &&
        std::is_trivially_destructible_v<T> && std::is_default_constructible_v<T> &&
        std::is_standard_layout_v<T>;

    template <typename T>
    inline constexpr bool is_valid_view_element_v =
        !std::is_reference_v<T> && !std::is_volatile_v<T> && !std::is_void_v<std::remove_cv_t<T>> &&
        std::is_trivially_copyable_v<std::remove_cv_t<T>> &&
        std::is_trivially_destructible_v<std::remove_cv_t<T>> &&
        std::is_standard_layout_v<std::remove_cv_t<T>>;

    // Checked structural arithmetic
    // -----------------------------------------------------------------------------
    // These helpers are used only while storage or layout metadata is established.
    // Element access and algorithm traversal use plain arithmetic after validation.

    constexpr inline bool is_valid_alignment(index_t n) noexcept {
        return n > 0 && (n & (n - 1)) == 0;
    }

    constexpr inline index_t max_index(index_t a, index_t b) noexcept {
        return a > b ? a : b;
    }

    inline index_t checked_add(index_t a, index_t b) noexcept {
        if constexpr (runtime_overflow_check) {
            if (b > std::numeric_limits<index_t>::max() - a) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][overflow] checked_add: a=%zu, b=%zu, "
                             "max=%zu; %s:%d\n",
                             a, b, std::numeric_limits<index_t>::max(), __FILE__, __LINE__);
                contract_violation();
            }
        }
        return a + b;
    }

    inline index_t checked_mul(index_t a, index_t b) noexcept {
        if constexpr (runtime_overflow_check) {
            if (a != 0 && b > std::numeric_limits<index_t>::max() / a) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][overflow] checked_mul: a=%zu, b=%zu, "
                             "max=%zu; %s:%d\n",
                             a, b, std::numeric_limits<index_t>::max(), __FILE__, __LINE__);
                contract_violation();
            }
        }
        return a * b;
    }

    template <index_t Rank>
    inline bool has_zero_extent(const index_t (&extent)[Rank]) noexcept {
        for (index_t d = 0; d < Rank; ++d) {
            if (extent[d] == 0) {
                return true;
            }
        }
        return false;
    }

    template <index_t Rank>
    inline index_t product(const index_t (&extent)[Rank]) noexcept {
        if (has_zero_extent(extent)) {
            return 0;
        }

        index_t n = 1;
        for (index_t d = 0; d < Rank; ++d) {
            n = checked_mul(n, extent[d]);
        }
        return n;
    }

    template <index_t Rank>
    inline void make_row_major_stride(const index_t (&extent)[Rank],
                                      index_t (&stride)[Rank]) noexcept {
        if (has_zero_extent(extent)) {
            for (index_t d = 0; d < Rank; ++d) {
                stride[d] = 0;
            }
            return;
        }

        index_t s = 1;
        for (index_t r = Rank; r > 0; --r) {
            const index_t d = r - 1;
            stride[d]       = s;
            s               = checked_mul(s, extent[d]);
        }
    }

    // Allocator
    // -----------------------------------------------------------------------------

    struct Allocator {
        void* ctx = nullptr;

        void* (*allocate)(void* ctx, index_t bytes, index_t alignment) = nullptr;

        void (*deallocate)(void* ctx, void* ptr, index_t bytes,
                           index_t alignment) noexcept = nullptr;
    };

    inline void* default_allocate(void*, index_t bytes, index_t alignment) {
        return ::operator new[](bytes, std::align_val_t{alignment});
    }

    inline void default_deallocate(void*, void* ptr, index_t, index_t alignment) noexcept {
        ::operator delete[](ptr, std::align_val_t{alignment});
    }

    inline Allocator default_allocator() noexcept {
        return Allocator{nullptr, default_allocate, default_deallocate};
    }

    // Buffer: owning aligned dynamic memory
    // -----------------------------------------------------------------------------

    template <typename T>
    class Buffer {
        static_assert(
            is_valid_storage_element_v<T>,
            "vehkko::ndarray::Buffer<T>: T must be non-cv, non-reference, trivially copyable, "
            "trivially destructible, default constructible, and standard-layout");

      public:
        using element_type = T;
        using value_type   = T;

        Buffer() noexcept = default;

        explicit Buffer(index_t size, index_t alignment = default_alignment,
                        Allocator allocator = default_allocator())
            : size_(size), alignment_(max_index(alignment, alignof(T))), allocator_(allocator) {
            if constexpr (runtime_alignment_check) {
                if (!is_valid_alignment(alignment)) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][alignment] Buffer::Buffer: requested_alignment=%zu "
                                 "is not a positive power of two; %s:%d\n",
                                 alignment, __FILE__, __LINE__);
                    contract_violation();
                }
                if (!is_valid_alignment(alignment_)) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][alignment] Buffer::Buffer: effective_alignment=%zu "
                                 "is not a positive power of two, requested_alignment=%zu, "
                                 "alignof(T)=%zu; %s:%d\n",
                                 alignment_, alignment, alignof(T), __FILE__, __LINE__);
                    contract_violation();
                }
            }
            if constexpr (runtime_allocator_check) {
                if (allocator_.allocate == nullptr) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][allocator] Buffer::Buffer: allocate callback is null, "
                                 "size=%zu, alignment=%zu; %s:%d\n",
                                 size_, alignment_, __FILE__, __LINE__);
                    contract_violation();
                }
                if (allocator_.deallocate == nullptr) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][allocator] Buffer::Buffer: deallocate callback is null, "
                                 "size=%zu, alignment=%zu; %s:%d\n",
                                 size_, alignment_, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            allocate_();
        }

        ~Buffer() noexcept { reset(); }

        Buffer(const Buffer& other)
            : size_(other.size_), alignment_(other.alignment_), allocator_(other.allocator_) {
            allocate_();
            if (size_ != 0)
                std::memcpy(data_, other.data_, bytes());
        }

        Buffer& operator=(const Buffer& other) {
            if (this != &other) {
                Buffer temporary(other);
                *this = std::move(temporary);
            }

            return *this;
        }

        Buffer(Buffer&& other) noexcept
            : size_(other.size_), alignment_(other.alignment_), data_(other.data_), allocator_(other.allocator_) {
            other.clear_state_();
        }

        Buffer& operator=(Buffer&& other) noexcept {
            if (this != &other) {
                reset();
                size_      = other.size_;
                alignment_ = other.alignment_;
                data_      = other.data_;
                allocator_ = other.allocator_;
                other.clear_state_();
            }
            return *this;
        }

        void reset() noexcept {
            if (data_ == nullptr) {
                clear_state_();
                return;
            }

            std::destroy_n(data_, size_);
            allocator_.deallocate(allocator_.ctx, static_cast<void*>(data_), bytes(), alignment_);
            clear_state_();
        }

        T*       data() noexcept { return data_; }
        const T* data() const noexcept { return data_; }

        T*       begin() noexcept { return data_; }
        const T* begin() const noexcept { return data_; }

        T*       end() noexcept { return data_ == nullptr ? nullptr : data_ + size_; }
        const T* end() const noexcept { return data_ == nullptr ? nullptr : data_ + size_; }

        T& front() noexcept {
            if constexpr (runtime_bounds_check) {
                if (size_ == 0) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] Buffer::front: buffer is empty; %s:%d\n",
                                 __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return data_[0];
        }

        const T& front() const noexcept {
            if constexpr (runtime_bounds_check) {
                if (size_ == 0) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] Buffer::front const: buffer is empty; %s:%d\n",
                                 __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return data_[0];
        }

        T& back() noexcept {
            if constexpr (runtime_bounds_check) {
                if (size_ == 0) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] Buffer::back: buffer is empty; %s:%d\n",
                                 __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return data_[size_ - 1];
        }

        const T& back() const noexcept {
            if constexpr (runtime_bounds_check) {
                if (size_ == 0) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] Buffer::back const: buffer is empty; %s:%d\n",
                                 __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return data_[size_ - 1];
        }

        index_t  size() const noexcept { return size_; }
        index_t  bytes() const noexcept { return size_ * sizeof(T); }
        index_t  alignment() const noexcept { return alignment_; }
        bool     empty() const noexcept { return size_ == 0; }
        explicit operator bool() const noexcept { return data_ != nullptr; }

        T& operator[](index_t i) noexcept {
            if constexpr (runtime_bounds_check) {
                if (i >= size_) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] Buffer::operator[]: index=%zu, size=%zu; "
                                 "%s:%d\n",
                                 i, size_, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return data_[i];
        }

        const T& operator[](index_t i) const noexcept {
            if constexpr (runtime_bounds_check) {
                if (i >= size_) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] Buffer::operator[] const: index=%zu, size=%zu; "
                                 "%s:%d\n",
                                 i, size_, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return data_[i];
        }

      private:
        void allocate_() {
            if (size_ == 0) {
                return;
            }

            const index_t byte_count = checked_mul(size_, sizeof(T));
            void*         raw        = allocator_.allocate(allocator_.ctx, byte_count, alignment_);
            if constexpr (runtime_allocator_check) {
                if (raw == nullptr) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][allocator] Buffer::allocate_: allocator returned null, "
                                 "size=%zu, bytes=%zu, alignment=%zu; %s:%d\n",
                                 size_, byte_count, alignment_, __FILE__, __LINE__);
                    contract_violation();
                }
                if (reinterpret_cast<std::uintptr_t>(raw) % alignment_ != 0) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][allocator] Buffer::allocate_: allocator returned "
                                 "misaligned pointer=%p, requested_alignment=%zu, size=%zu, bytes=%zu; "
                                 "%s:%d\n",
                                 raw, alignment_, size_, byte_count, __FILE__, __LINE__);
                    contract_violation();
                }
            }

            T* data = static_cast<T*>(raw);
            try {
                std::uninitialized_default_construct_n(data, size_);
            } catch (...) {
                allocator_.deallocate(allocator_.ctx, raw, byte_count, alignment_);
                throw;
            }
            data_ = data;
        }

        void clear_state_() noexcept {
            size_      = 0;
            alignment_ = max_index(default_alignment, alignof(T));
            data_      = nullptr;
            allocator_ = default_allocator();
        }

      private:
        index_t   size_      = 0;
        index_t   alignment_ = max_index(default_alignment, alignof(T));
        T*        data_      = nullptr;
        Allocator allocator_ = default_allocator();
    };

    // StaticBuffer: owning aligned fixed-capacity stack memory
    // -----------------------------------------------------------------------------

    template <typename T, index_t Capacity, index_t Alignment = default_alignment>
    class StaticBuffer {
        static_assert(is_valid_storage_element_v<T>,
                      "vehkko::ndarray::StaticBuffer<T, Capacity>: T must be non-cv, "
                      "non-reference, trivially copyable, "
                      "trivially destructible, default constructible, and standard-layout");
        static_assert(Capacity > 0, "vehkko::ndarray::StaticBuffer: Capacity must be positive");
        static_assert(is_valid_alignment(Alignment),
                      "vehkko::ndarray::StaticBuffer: Alignment must be a positive "
                      "power of two");

      public:
        using element_type                      = T;
        using value_type                        = T;
        static constexpr index_t capacity_value = Capacity;

        StaticBuffer() = default;

        StaticBuffer(const StaticBuffer& other) noexcept(
            std::is_nothrow_default_constructible_v<T>) {
            std::memcpy(data_, other.data_, bytes());
        }

        StaticBuffer& operator=(const StaticBuffer& other) noexcept {
            if (this != &other) {
                std::memcpy(data_, other.data_, bytes());
            }
            return *this;
        }

        StaticBuffer(StaticBuffer&& other) noexcept(std::is_nothrow_default_constructible_v<T>) {
            std::memcpy(data_, other.data_, bytes());
        }

        StaticBuffer& operator=(StaticBuffer&& other) noexcept {
            if (this != &other) {
                std::memcpy(data_, other.data_, bytes());
            }
            return *this;
        }

        ~StaticBuffer() noexcept = default;

        T*       data() noexcept { return data_; }
        const T* data() const noexcept { return data_; }

        T*       begin() noexcept { return data_; }
        const T* begin() const noexcept { return data_; }
        T*       end() noexcept { return data_ + Capacity; }
        const T* end() const noexcept { return data_ + Capacity; }

        T&       front() noexcept { return data_[0]; }
        const T& front() const noexcept { return data_[0]; }
        T&       back() noexcept { return data_[Capacity - 1]; }
        const T& back() const noexcept { return data_[Capacity - 1]; }

        static constexpr index_t size() noexcept { return Capacity; }
        static constexpr index_t capacity() noexcept { return Capacity; }
        static constexpr index_t bytes() noexcept { return Capacity * sizeof(T); }
        static constexpr index_t alignment() noexcept { return max_index(Alignment, alignof(T)); }
        static constexpr bool    empty() noexcept { return false; }
        explicit                 operator bool() const noexcept { return true; }

        T& operator[](index_t i) noexcept {
            if constexpr (runtime_bounds_check) {
                if (i >= Capacity) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticBuffer::operator[]: index=%zu, "
                                 "capacity=%zu; %s:%d\n",
                                 i, Capacity, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return data_[i];
        }

        const T& operator[](index_t i) const noexcept {
            if constexpr (runtime_bounds_check) {
                if (i >= Capacity) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticBuffer::operator[] const: index=%zu, "
                                 "capacity=%zu; %s:%d\n",
                                 i, Capacity, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return data_[i];
        }

      private:
        alignas(max_index(Alignment, alignof(T))) T data_[Capacity];
    };

    // View: non-owning fixed-rank strided view
    // -----------------------------------------------------------------------------
    // View constness follows span semantics. A const View<T, Rank> still refers to
    // mutable T. Use View<const T, Rank> for read-only elements.

    template <typename T, index_t Rank>
    class View {
        static_assert(is_valid_view_element_v<T>,
                      "vehkko::ndarray::View<T, Rank>: T must be non-reference, "
                      "non-volatile, trivially copyable, trivially destructible, "
                      "and standard-layout; const T is allowed");
        static_assert(Rank > 0, "vehkko::ndarray::View<T, Rank>: Rank must be positive");

      public:
        using element_type            = T;
        using value_type              = std::remove_const_t<T>;
        static constexpr index_t rank = Rank;

        View() noexcept = default;

        View(const View&) noexcept            = default;
        View& operator=(const View&) noexcept = default;

        View(View&&) noexcept            = default;
        View& operator=(View&&) noexcept = default;

        View(T* data, const index_t (&extent)[Rank], const index_t (&stride)[Rank]) noexcept
            : data_(data) {
            for (index_t d = 0; d < Rank; ++d) {
                extent_[d] = extent[d];
                stride_[d] = stride[d];
            }
            validate_layout_();
        }

        template <typename U,
                  std::enable_if_t<std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>,
                                   int> = 0>
        View(const View<U, Rank>& other) noexcept
            : data_(other.data()) {
            for (index_t d = 0; d < Rank; ++d) {
                extent_[d] = other.extent(d);
                stride_[d] = other.stride(d);
            }
        }

        T* data() noexcept { return data_; }
        T* data() const noexcept { return data_; }

        index_t extent(index_t d) const noexcept {
            if constexpr (runtime_bounds_check) {
                if (d >= Rank) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] View::extent: dimension=%zu, rank=%zu; "
                                 "%s:%d\n",
                                 d, Rank, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return extent_[d];
        }

        index_t stride(index_t d) const noexcept {
            if constexpr (runtime_bounds_check) {
                if (d >= Rank) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] View::stride: dimension=%zu, rank=%zu; "
                                 "%s:%d\n",
                                 d, Rank, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return stride_[d];
        }

        const index_t* extents() const noexcept { return extent_; }
        const index_t* strides() const noexcept { return stride_; }

        index_t size() const noexcept {
            index_t n = 1;
            for (index_t d = 0; d < Rank; ++d) {
                n *= extent_[d];
            }
            return n;
        }

        bool empty() const noexcept {
            for (index_t d = 0; d < Rank; ++d) {
                if (extent_[d] == 0) {
                    return true;
                }
            }
            return false;
        }

        explicit operator bool() const noexcept { return data_ != nullptr; }

        template <typename... I>
        T& operator()(I... indices) noexcept {
            return access_(indices...);
        }

        template <typename... I>
        T& operator()(I... indices) const noexcept {
            return access_(indices...);
        }

      private:
        template <typename... I>
        T& access_(I... indices) const noexcept {
            static_assert(sizeof...(I) == Rank,
                          "vehkko::ndarray::View::operator(): wrong number of "
                          "indices");

            const index_t idx[Rank] = {static_cast<index_t>(indices)...};

            if constexpr (runtime_bounds_check) {
                for (index_t d = 0; d < Rank; ++d) {
                    if (idx[d] >= extent_[d]) {
                        std::fprintf(stderr,
                                     "[vehkko::ndarray][bounds] View::operator(): rank=%zu, "
                                     "dimension=%zu, index=%zu, extent=%zu; %s:%d\n",
                                     Rank, d, idx[d], extent_[d], __FILE__, __LINE__);
                        contract_violation();
                    }
                }
            }

            index_t linear = 0;
            for (index_t d = 0; d < Rank; ++d) {
                linear += idx[d] * stride_[d];
            }
            return data_[linear];
        }

        void validate_layout_() const noexcept {
            if constexpr (runtime_layout_check || runtime_overflow_check) {
                bool nonempty = true;
                for (index_t d = 0; d < Rank; ++d) {
                    if (extent_[d] == 0) {
                        nonempty = false;
                        break;
                    }
                }

                if (!nonempty) {
                    return;
                }

                if constexpr (runtime_layout_check) {
                    if (data_ == nullptr) {
                        std::fprintf(stderr,
                                     "[vehkko::ndarray][layout] View::validate_layout_: non-empty view "
                                     "has null data, rank=%zu; %s:%d\n",
                                     Rank, __FILE__, __LINE__);
                        contract_violation();
                    }
                    for (index_t d = 0; d < Rank; ++d) {
                        if (extent_[d] > 1 && stride_[d] == 0) {
                            std::fprintf(stderr,
                                         "[vehkko::ndarray][layout] View::validate_layout_: rank=%zu, "
                                         "dimension=%zu, extent=%zu, stride=0; %s:%d\n",
                                         Rank, d, extent_[d], __FILE__, __LINE__);
                            contract_violation();
                        }
                    }
                }

                if constexpr (runtime_overflow_check) {
                    (void)product(extent_);

                    index_t max_offset = 0;
                    for (index_t d = 0; d < Rank; ++d) {
                        max_offset =
                            checked_add(max_offset, checked_mul(extent_[d] - 1, stride_[d]));
                    }

                    const index_t span_elements = checked_add(max_offset, index_t{1});
                    const index_t span_bytes    = checked_mul(span_elements, sizeof(value_type));

                    if (data_ != nullptr) {
                        const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data_);
                        if (span_bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
                            std::fprintf(stderr,
                                         "[vehkko::ndarray][overflow] View::validate_layout_: address "
                                         "range overflows, data=%p, rank=%zu, span_bytes=%zu; %s:%d\n",
                                         const_cast<void*>(static_cast<const void*>(data_)), Rank,
                                         span_bytes, __FILE__, __LINE__);
                            contract_violation();
                        }
                    }
                }
            }
        }

      private:
        T*      data_ = nullptr;
        index_t extent_[Rank]{};
        index_t stride_[Rank]{};
    };

    // Array: owning row-major dynamic ndarray
    // -----------------------------------------------------------------------------

    template <typename T, index_t Rank>
    class Array {
        static_assert(
            is_valid_storage_element_v<T>,
            "vehkko::ndarray::Array<T, Rank>: T must be non-cv, non-reference, trivially copyable, "
            "trivially destructible, default constructible, and standard-layout");
        static_assert(Rank > 0, "vehkko::ndarray::Array<T, Rank>: Rank must be positive");

      public:
        using element_type            = T;
        using value_type              = T;
        static constexpr index_t rank = Rank;

        Array() noexcept = default;

        explicit Array(const index_t (&extent)[Rank], index_t alignment = default_alignment,
                       Allocator allocator = default_allocator())
            : buffer_(product(extent), alignment, allocator) {
            set_shape_(extent);
        }

        Array(const Array& other)
            : buffer_(other.buffer_) {
            copy_shape_from_(other);
        }

        Array& operator=(const Array& other) {
            if (this != &other) {
                buffer_ = other.buffer_;
                copy_shape_from_(other);
            }

            return *this;
        }

        Array(Array&& other) noexcept
            : buffer_(std::move(other.buffer_)) {
            copy_shape_from_(other);
            other.clear_shape_();
        }

        Array& operator=(Array&& other) noexcept {
            if (this != &other) {
                buffer_ = std::move(other.buffer_);
                copy_shape_from_(other);
                other.clear_shape_();
            }
            return *this;
        }

        void reshape(const index_t (&extent)[Rank]) noexcept {
            const index_t new_size = product(extent);
            if constexpr (runtime_shape_check) {
                if (new_size != size()) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][shape] Array::reshape: rank=%zu, current_size=%zu, "
                                 "requested_size=%zu; %s:%d\n",
                                 Rank, size(), new_size, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            set_shape_(extent);
        }

        T*       data() noexcept { return buffer_.data(); }
        const T* data() const noexcept { return buffer_.data(); }

        T*       begin() noexcept { return buffer_.begin(); }
        const T* begin() const noexcept { return buffer_.begin(); }
        T*       end() noexcept { return buffer_.end(); }
        const T* end() const noexcept { return buffer_.end(); }

        T&       front() noexcept { return buffer_.front(); }
        const T& front() const noexcept { return buffer_.front(); }
        T&       back() noexcept { return buffer_.back(); }
        const T& back() const noexcept { return buffer_.back(); }

        index_t  size() const noexcept { return buffer_.size(); }
        index_t  bytes() const noexcept { return buffer_.bytes(); }
        index_t  alignment() const noexcept { return buffer_.alignment(); }
        bool     empty() const noexcept { return buffer_.empty(); }
        explicit operator bool() const noexcept { return static_cast<bool>(buffer_); }

        index_t extent(index_t d) const noexcept {
            if constexpr (runtime_bounds_check) {
                if (d >= Rank) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] Array::extent: dimension=%zu, rank=%zu; "
                                 "%s:%d\n",
                                 d, Rank, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return extent_[d];
        }

        index_t stride(index_t d) const noexcept {
            if constexpr (runtime_bounds_check) {
                if (d >= Rank) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] Array::stride: dimension=%zu, rank=%zu; "
                                 "%s:%d\n",
                                 d, Rank, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return stride_[d];
        }

        const index_t* extents() const noexcept { return extent_; }
        const index_t* strides() const noexcept { return stride_; }

        View<T, Rank> view() & noexcept { return View<T, Rank>(buffer_.data(), extent_, stride_); }

        View<const T, Rank> view() const& noexcept {
            return View<const T, Rank>(buffer_.data(), extent_, stride_);
        }

        View<T, Rank> view() && noexcept = delete;

        View<const T, Rank> view() const&& noexcept = delete;

        template <typename... I>
        T& operator()(I... indices) noexcept {
            static_assert(sizeof...(I) == Rank,
                          "vehkko::ndarray::Array::operator(): wrong number of indices");

            const index_t idx[Rank] = {static_cast<index_t>(indices)...};
            if constexpr (runtime_bounds_check) {
                for (index_t d = 0; d < Rank; ++d) {
                    if (idx[d] >= extent_[d]) {
                        std::fprintf(stderr,
                                     "[vehkko::ndarray][bounds] Array::operator(): rank=%zu, "
                                     "dimension=%zu, index=%zu, extent=%zu; %s:%d\n",
                                     Rank, d, idx[d], extent_[d], __FILE__, __LINE__);
                        contract_violation();
                    }
                }
            }

            index_t linear = 0;
            for (index_t d = 0; d < Rank; ++d) {
                linear += idx[d] * stride_[d];
            }
            return buffer_.data()[linear];
        }

        template <typename... I>
        const T& operator()(I... indices) const noexcept {
            static_assert(sizeof...(I) == Rank,
                          "vehkko::ndarray::Array::operator(): wrong number of indices");

            const index_t idx[Rank] = {static_cast<index_t>(indices)...};
            if constexpr (runtime_bounds_check) {
                for (index_t d = 0; d < Rank; ++d) {
                    if (idx[d] >= extent_[d]) {
                        std::fprintf(stderr,
                                     "[vehkko::ndarray][bounds] Array::operator() const: rank=%zu, "
                                     "dimension=%zu, index=%zu, extent=%zu; %s:%d\n",
                                     Rank, d, idx[d], extent_[d], __FILE__, __LINE__);
                        contract_violation();
                    }
                }
            }

            index_t linear = 0;
            for (index_t d = 0; d < Rank; ++d) {
                linear += idx[d] * stride_[d];
            }
            return buffer_.data()[linear];
        }

        T&       operator[](index_t i) noexcept { return buffer_[i]; }
        const T& operator[](index_t i) const noexcept { return buffer_[i]; }

      private:
        void set_shape_(const index_t (&extent)[Rank]) noexcept {
            for (index_t d = 0; d < Rank; ++d) {
                extent_[d] = extent[d];
            }
            make_row_major_stride(extent_, stride_);
        }

        void copy_shape_from_(const Array& other) noexcept {
            for (index_t d = 0; d < Rank; ++d) {
                extent_[d] = other.extent_[d];
                stride_[d] = other.stride_[d];
            }
        }

        void clear_shape_() noexcept {
            for (index_t d = 0; d < Rank; ++d) {
                extent_[d] = 0;
                stride_[d] = 0;
            }
        }

      private:
        Buffer<T> buffer_;
        index_t   extent_[Rank]{};
        index_t   stride_[Rank]{};
    };

    // StaticArray: owning row-major fixed-capacity ndarray
    // -----------------------------------------------------------------------------

    template <typename T, index_t Rank, index_t Capacity, index_t Alignment = default_alignment>
    class StaticArray {
        static_assert(is_valid_storage_element_v<T>,
                      "vehkko::ndarray::StaticArray: invalid storage element type");
        static_assert(Rank > 0, "vehkko::ndarray::StaticArray: Rank must be positive");
        static_assert(Capacity > 0, "vehkko::ndarray::StaticArray: Capacity must be positive");

      public:
        using element_type                      = T;
        using value_type                        = T;
        static constexpr index_t rank           = Rank;
        static constexpr index_t capacity_value = Capacity;

        StaticArray() = default;

        explicit StaticArray(const index_t (&extent)[Rank]) noexcept(
            std::is_nothrow_default_constructible_v<T>) {
            reshape(extent);
        }

        StaticArray(const StaticArray&)                     = default;
        StaticArray& operator=(const StaticArray&) noexcept = default;
        StaticArray(StaticArray&&)                          = default;
        StaticArray& operator=(StaticArray&&) noexcept      = default;
        ~StaticArray() noexcept                             = default;

        void reshape(const index_t (&extent)[Rank]) noexcept {
            const index_t n = product(extent);
            if constexpr (runtime_shape_check) {
                if (n > Capacity) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][capacity] StaticArray::reshape: rank=%zu, "
                                 "requested_size=%zu, capacity=%zu; %s:%d\n",
                                 Rank, n, Capacity, __FILE__, __LINE__);
                    contract_violation();
                }
            }

            size_ = n;
            for (index_t d = 0; d < Rank; ++d) {
                extent_[d] = extent[d];
            }
            make_row_major_stride(extent_, stride_);
        }

        T*       data() noexcept { return storage_.data(); }
        const T* data() const noexcept { return storage_.data(); }

        T*       begin() noexcept { return storage_.data(); }
        const T* begin() const noexcept { return storage_.data(); }
        T*       end() noexcept { return storage_.data() + size_; }
        const T* end() const noexcept { return storage_.data() + size_; }

        T& front() noexcept {
            if constexpr (runtime_bounds_check) {
                if (size_ == 0) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticArray::front: array is empty, "
                                 "rank=%zu, capacity=%zu; %s:%d\n",
                                 Rank, Capacity, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return storage_[0];
        }

        const T& front() const noexcept {
            if constexpr (runtime_bounds_check) {
                if (size_ == 0) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticArray::front const: array is empty, "
                                 "rank=%zu, capacity=%zu; %s:%d\n",
                                 Rank, Capacity, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return storage_[0];
        }

        T& back() noexcept {
            if constexpr (runtime_bounds_check) {
                if (size_ == 0) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticArray::back: array is empty, "
                                 "rank=%zu, capacity=%zu; %s:%d\n",
                                 Rank, Capacity, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return storage_[size_ - 1];
        }

        const T& back() const noexcept {
            if constexpr (runtime_bounds_check) {
                if (size_ == 0) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticArray::back const: array is empty, "
                                 "rank=%zu, capacity=%zu; %s:%d\n",
                                 Rank, Capacity, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return storage_[size_ - 1];
        }

        index_t                  size() const noexcept { return size_; }
        static constexpr index_t capacity() noexcept { return Capacity; }
        index_t                  bytes() const noexcept { return size_ * sizeof(T); }
        static constexpr index_t capacity_bytes() noexcept { return Capacity * sizeof(T); }
        static constexpr index_t alignment() noexcept {
            return StaticBuffer<T, Capacity, Alignment>::alignment();
        }
        bool     empty() const noexcept { return size_ == 0; }
        explicit operator bool() const noexcept { return size_ != 0; }

        index_t extent(index_t d) const noexcept {
            if constexpr (runtime_bounds_check) {
                if (d >= Rank) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticArray::extent: dimension=%zu, rank=%zu; "
                                 "%s:%d\n",
                                 d, Rank, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return extent_[d];
        }

        index_t stride(index_t d) const noexcept {
            if constexpr (runtime_bounds_check) {
                if (d >= Rank) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticArray::stride: dimension=%zu, rank=%zu; "
                                 "%s:%d\n",
                                 d, Rank, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return stride_[d];
        }

        const index_t* extents() const noexcept { return extent_; }
        const index_t* strides() const noexcept { return stride_; }

        View<T, Rank> view() & noexcept { return View<T, Rank>(storage_.data(), extent_, stride_); }

        View<const T, Rank> view() const& noexcept {
            return View<const T, Rank>(storage_.data(), extent_, stride_);
        }

        View<T, Rank> view() && noexcept = delete;

        View<const T, Rank> view() const&& noexcept = delete;

        template <typename... I>
        T& operator()(I... indices) noexcept {
            static_assert(sizeof...(I) == Rank,
                          "vehkko::ndarray::StaticArray::operator(): wrong number of indices");

            const index_t idx[Rank] = {static_cast<index_t>(indices)...};
            if constexpr (runtime_bounds_check) {
                for (index_t d = 0; d < Rank; ++d) {
                    if (idx[d] >= extent_[d]) {
                        std::fprintf(stderr,
                                     "[vehkko::ndarray][bounds] StaticArray::operator(): rank=%zu, "
                                     "dimension=%zu, index=%zu, extent=%zu; %s:%d\n",
                                     Rank, d, idx[d], extent_[d], __FILE__, __LINE__);
                        contract_violation();
                    }
                }
            }

            index_t linear = 0;
            for (index_t d = 0; d < Rank; ++d) {
                linear += idx[d] * stride_[d];
            }
            return storage_.data()[linear];
        }

        template <typename... I>
        const T& operator()(I... indices) const noexcept {
            static_assert(sizeof...(I) == Rank,
                          "vehkko::ndarray::StaticArray::operator(): wrong number of indices");

            const index_t idx[Rank] = {static_cast<index_t>(indices)...};
            if constexpr (runtime_bounds_check) {
                for (index_t d = 0; d < Rank; ++d) {
                    if (idx[d] >= extent_[d]) {
                        std::fprintf(stderr,
                                     "[vehkko::ndarray][bounds] StaticArray::operator() const: rank=%zu, "
                                     "dimension=%zu, index=%zu, extent=%zu; %s:%d\n",
                                     Rank, d, idx[d], extent_[d], __FILE__, __LINE__);
                        contract_violation();
                    }
                }
            }

            index_t linear = 0;
            for (index_t d = 0; d < Rank; ++d) {
                linear += idx[d] * stride_[d];
            }
            return storage_.data()[linear];
        }

        T& operator[](index_t i) noexcept {
            if constexpr (runtime_bounds_check) {
                if (i >= size_) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticArray::operator[]: index=%zu, "
                                 "size=%zu, capacity=%zu; %s:%d\n",
                                 i, size_, Capacity, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return storage_[i];
        }

        const T& operator[](index_t i) const noexcept {
            if constexpr (runtime_bounds_check) {
                if (i >= size_) {
                    std::fprintf(stderr,
                                 "[vehkko::ndarray][bounds] StaticArray::operator[] const: index=%zu, "
                                 "size=%zu, capacity=%zu; %s:%d\n",
                                 i, size_, Capacity, __FILE__, __LINE__);
                    contract_violation();
                }
            }
            return storage_[i];
        }

      private:
        StaticBuffer<T, Capacity, Alignment> storage_;
        index_t                              size_ = 0;
        index_t                              extent_[Rank]{};
        index_t                              stride_[Rank]{};
    };

    // Construction helpers
    // -----------------------------------------------------------------------------

    template <typename T>
    inline Array<T, 1> make_array1d(index_t n0, index_t alignment = default_alignment,
                                    Allocator allocator = default_allocator()) {
        const index_t extent[1] = {n0};
        return Array<T, 1>(extent, alignment, allocator);
    }

    template <typename T>
    inline Array<T, 2> make_array2d(index_t n0, index_t n1, index_t alignment = default_alignment,
                                    Allocator allocator = default_allocator()) {
        const index_t extent[2] = {n0, n1};
        return Array<T, 2>(extent, alignment, allocator);
    }

    template <typename T>
    inline Array<T, 3> make_array3d(index_t n0, index_t n1, index_t n2,
                                    index_t   alignment = default_alignment,
                                    Allocator allocator = default_allocator()) {
        const index_t extent[3] = {n0, n1, n2};
        return Array<T, 3>(extent, alignment, allocator);
    }

    template <typename T, index_t Rank>
    inline View<T, Rank> make_view(T* data, const index_t (&extent)[Rank]) noexcept {
        index_t stride[Rank];
        make_row_major_stride(extent, stride);
        return View<T, Rank>(data, extent, stride);
    }

    template <typename T>
    inline View<T, 1> make_view1d(T* data, index_t n0) noexcept {
        const index_t extent[1] = {n0};
        const index_t stride[1] = {1};
        return View<T, 1>(data, extent, stride);
    }

    template <typename T>
    inline View<T, 2> make_view2d(T* data, index_t n0, index_t n1) noexcept {
        const index_t extent[2] = {n0, n1};
        const index_t stride[2] = {n1, 1};
        return View<T, 2>(data, extent, stride);
    }

    template <typename T>
    inline View<T, 3> make_view3d(T* data, index_t n0, index_t n1, index_t n2) noexcept {
        const index_t extent[3] = {n0, n1, n2};
        index_t       stride[3];
        make_row_major_stride(extent, stride);
        return View<T, 3>(data, extent, stride);
    }

    template <typename T, index_t Rank>
    inline View<T, Rank> make_strided_view(T* data, const index_t (&extent)[Rank],
                                           const index_t (&stride)[Rank]) noexcept {
        return View<T, Rank>(data, extent, stride);
    }

    template <typename T>
    inline View<T, 1> make_strided_view1d(T* data, index_t n0, index_t s0) noexcept {
        const index_t extent[1] = {n0};
        const index_t stride[1] = {s0};
        return View<T, 1>(data, extent, stride);
    }

    template <typename T>
    inline View<T, 2> make_strided_view2d(T* data, index_t n0, index_t n1, index_t s0,
                                          index_t s1) noexcept {
        const index_t extent[2] = {n0, n1};
        const index_t stride[2] = {s0, s1};
        return View<T, 2>(data, extent, stride);
    }

    template <typename T>
    inline View<T, 3> make_strided_view3d(T* data, index_t n0, index_t n1, index_t n2, index_t s0,
                                          index_t s1, index_t s2) noexcept {
        const index_t extent[3] = {n0, n1, n2};
        const index_t stride[3] = {s0, s1, s2};
        return View<T, 3>(data, extent, stride);
    }

    template <typename T, index_t Rank>
    inline View<const T, Rank> as_const_view(View<T, Rank> v) noexcept {
        return View<const T, Rank>(v);
    }

    // Layout and shape queries
    // -----------------------------------------------------------------------------

    template <typename T, index_t Rank>
    inline bool is_inner_contiguous(View<T, Rank> v) noexcept {
        return v.stride(Rank - 1) == 1;
    }

    template <typename T, index_t Rank>
    inline bool is_row_major_contiguous(View<T, Rank> v) noexcept {
        if (v.empty()) {
            return true;
        }

        index_t expected = 1;
        for (index_t r = Rank; r > 0; --r) {
            const index_t d = r - 1;
            if (v.stride(d) != expected) {
                return false;
            }
            expected *= v.extent(d);
        }
        return true;
    }

    template <typename T, typename U, index_t Rank>
    inline bool same_extents(View<T, Rank> a, View<U, Rank> b) noexcept {
        for (index_t d = 0; d < Rank; ++d) {
            if (a.extent(d) != b.extent(d)) {
                return false;
            }
        }
        return true;
    }

    template <typename T, typename U, index_t Rank>
    inline bool same_layout(View<T, Rank> a, View<U, Rank> b) noexcept {
        for (index_t d = 0; d < Rank; ++d) {
            if (a.extent(d) != b.extent(d) || a.stride(d) != b.stride(d)) {
                return false;
            }
        }
        return true;
    }

    template <typename T, typename U, index_t Rank>
    inline bool is_same_data(View<T, Rank> a, View<U, Rank> b) noexcept {
        return static_cast<const void*>(a.data()) == static_cast<const void*>(b.data());
    }

    // View derivation
    // -----------------------------------------------------------------------------

    template <typename T>
    inline View<T, 1> row_view(View<T, 2> v, index_t i) noexcept {
        if constexpr (runtime_bounds_check) {
            if (i >= v.extent(0)) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][bounds] row_view: row=%zu, rows=%zu, columns=%zu; "
                             "%s:%d\n",
                             i, v.extent(0), v.extent(1), __FILE__, __LINE__);
                contract_violation();
            }
        }

        const index_t extent[1] = {v.extent(1)};
        const index_t stride[1] = {v.stride(1)};
        const index_t offset    = i * v.stride(0);
        T*            data      = v.data();
        if (data != nullptr && extent[0] != 0) {
            data += offset;
        }
        return View<T, 1>(data, extent, stride);
    }

    template <typename T>
    inline View<T, 1> col_view(View<T, 2> v, index_t j) noexcept {
        if constexpr (runtime_bounds_check) {
            if (j >= v.extent(1)) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][bounds] col_view: column=%zu, rows=%zu, columns=%zu; "
                             "%s:%d\n",
                             j, v.extent(0), v.extent(1), __FILE__, __LINE__);
                contract_violation();
            }
        }

        const index_t extent[1] = {v.extent(0)};
        const index_t stride[1] = {v.stride(0)};
        const index_t offset    = j * v.stride(1);
        T*            data      = v.data();
        if (data != nullptr && extent[0] != 0) {
            data += offset;
        }
        return View<T, 1>(data, extent, stride);
    }

    template <typename T>
    inline View<T, 2> block_view(View<T, 2> v, index_t i0, index_t j0, index_t n0,
                                 index_t n1) noexcept {
        if constexpr (runtime_bounds_check) {
            if (i0 > v.extent(0)) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][bounds] block_view: start_row=%zu, rows=%zu; %s:%d\n",
                             i0, v.extent(0), __FILE__, __LINE__);
                contract_violation();
            }
            if (j0 > v.extent(1)) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][bounds] block_view: start_column=%zu, columns=%zu; "
                             "%s:%d\n",
                             j0, v.extent(1), __FILE__, __LINE__);
                contract_violation();
            }
            if (n0 > v.extent(0) - i0) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][bounds] block_view: start_row=%zu, block_rows=%zu, "
                             "rows=%zu; %s:%d\n",
                             i0, n0, v.extent(0), __FILE__, __LINE__);
                contract_violation();
            }
            if (n1 > v.extent(1) - j0) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][bounds] block_view: start_column=%zu, "
                             "block_columns=%zu, columns=%zu; %s:%d\n",
                             j0, n1, v.extent(1), __FILE__, __LINE__);
                contract_violation();
            }
        }

        const index_t extent[2] = {n0, n1};
        const index_t stride[2] = {v.stride(0), v.stride(1)};
        const index_t offset    = i0 * v.stride(0) + j0 * v.stride(1);
        T*            data      = v.data();
        if (data != nullptr && n0 != 0 && n1 != 0) {
            data += offset;
        }
        return View<T, 2>(data, extent, stride);
    }

    template <typename T, index_t Rank>
    inline View<T, 1> flatten_view(View<T, Rank> v) noexcept {
        if constexpr (runtime_layout_check) {
            if (!is_row_major_contiguous(v)) {
                std::fprintf(stderr,
                             "[vehkko::ndarray][layout] flatten_view: view is not row-major "
                             "contiguous, rank=%zu, size=%zu, data=%p; %s:%d\n",
                             Rank, v.size(),
                             const_cast<void*>(static_cast<const void*>(v.data())), __FILE__,
                             __LINE__);
                contract_violation();
            }
        }

        const index_t extent[1] = {v.size()};
        const index_t stride[1] = {1};
        return View<T, 1>(v.data(), extent, stride);
    }

} // namespace vehkko::ndarray
