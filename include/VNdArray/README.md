# vehkko::ndarray

A lightweight, header-only multidimensional array and strided View library for C++17 numerical code.

Its goal is to provide a simple and explicit multidimensional array data structure layer:

- aligned dynamic and fixed-capacity contiguous storage;
- fixed-Rank, non-owning, strided multidimensional Views;
- basic element-wise algorithms that do not create implicit temporaries;
- safe alias handling by default, together with explicit high-performance `_noalias` APIs;
- all runtime checks can be completely disabled through compile-time macros;
- no third-party dependencies, with an optional algorithm layer.

```text
ndarray.hpp      owning storage + fixed-rank strided View
ndarray_ops.hpp  alias-aware basic algorithms
```

Include only `ndarray.hpp` when only containers and Views are needed; include `ndarray_ops.hpp` when basic algorithms are also needed.

---

## Features

- C++17, header-only, with no third-party dependencies;
- 64-byte alignment by default;
- `Buffer<T>`: dynamic, aligned, copyable and movable one-dimensional storage;
- `StaticBuffer<T, Capacity>`: fixed-capacity stack storage;
- `View<T, Rank>`: fixed-Rank, non-owning, strided multidimensional View;
- `Array<T, Rank>`: dynamic owning row-major array;
- `StaticArray<T, Rank, Capacity>`: fixed-capacity row-major array with a mutable logical shape;
- owning containers have value semantics and perform deep copies; copying a View copies only its handle metadata;
- custom function-pointer Allocator, leaving an extension point for arenas, pinned memory, and similar use cases;
- regular algorithms safely handle dangerous partial overlap by default;
- `_noalias` algorithms perform no allocations and provide a `restrict` fast path for contiguous data;
- direct loops are used for Rank 1–3, while higher Ranks use a generic fixed-Rank iterator;
- structural overflow checks for shapes, strides, byte counts, and address ranges;
- contract failures consistently call `std::terminate()`;
- allocation or element-construction failures from owning operations may propagate to the caller.

---

## Requirements

- C++17 or later;
- GCC, Clang, or another conforming C++17 compiler with aligned-new support;
- elements of owning types must be:
  - neither `const` nor `volatile`, and not references;
  - trivially copyable;
  - trivially destructible;
  - standard-layout;
  - default-constructible.

`View<const T, Rank>` can be used for read-only elements.

---

## Quick Start

```cpp
#include "ndarray_ops.hpp"

#include <iostream>

int main() {
  using namespace vehkko::ndarray;

  // make_array2d<T>(n0, n1, alignment, allocator)
  //
  // T:
  //   Element type, which is double here.
  //
  // n0, n1:
  //   Extents of the two dimensions. Both are 1024 here,
  //   so the array contains 1024 * 1024 doubles in total.
  //
  // alignment:
  //   Optional alignment in bytes for the base address of the underlying
  //   storage. Defaults to default_alignment, which is 64.
  //
  // allocator:
  //   Optional custom allocator. Defaults to default_allocator().
  //
  // The return type is Array<double, 2>, with row-major layout.
  auto x = make_array2d<double>(1024, 1024);
  auto y = make_array2d<double>(1024, 1024);
  auto result = make_array2d<double>(1024, 1024);

  // fill_inplace(dst, value)
  //
  // dst:
  //   Writable View. The algorithm layer accepts only Views, so an owning
  //   Array must explicitly call .view().
  //
  // value:
  //   The value written to every logical element.
  //
  // Here, all elements of x and y are initialized to 1.0 and 2.0,
  // respectively.
  fill_inplace(x.view(), 1.0);
  fill_inplace(y.view(), 2.0);

  // add_to(dst, lhs, rhs)
  //
  // dst:
  //   Output View, which must have the same extents as both inputs.
  //
  // lhs, rhs:
  //   Two read-only input Views. The following element-wise operation is
  //   performed:
  //
  //     dst[i...] = lhs[i...] + rhs[i...]
  //
  // The regular API enables alias dispatch by default:
  // - when non-overlap has been proven, it automatically enters the _noalias
  //   fast path;
  // - exactly identical mappings are handled with safe in-place semantics;
  // - dangerous partial overlap may cause a temporary Array to be created.
  //
  // In this example, the three Arrays are allocated independently, so the
  // contiguous noalias fast path is used.
  add_to(result.view(), x.view(), y.view());

  // operator()(i, j) accesses elements using row-major logical indices.
  // Debug builds check i < extent(0) and j < extent(1) by default;
  // Release builds disable per-element bounds checks by default.
  std::cout << result(0, 0) << '\n'; // Prints 3
}
```

Compile with:

```bash
g++ -std=c++17 -O3 -DNDEBUG -march=native main.cpp -o main
```

Where:

- `-std=c++17`: enables C++17;
- `-O3`: enables high-level optimization;
- `-DNDEBUG`: disables assertions and some checks that are enabled only in Debug builds by default;
- `-march=native`: optionally allows the compiler to generate code for the local instruction set.

---

## Using Only the Container Layer

```cpp
#include "ndarray.hpp"

int main() {
  using namespace vehkko::ndarray;

  // Create an Array<float, 3>.
  //
  // extent = {32, 64, 128}
  // stride = {64 * 128, 128, 1}
  // size   = 32 * 64 * 128
  auto field = make_array3d<float>(32, 64, 128);

  // The three indices correspond to dimensions 0, 1, and 2, respectively.
  field(1, 2, 3) = 4.0f;

  // .view() on a non-const Array returns View<float, 3>, allowing the
  // underlying elements to be read and written without copying data.
  View<float, 3> mutable_view = field.view();

  // A mutable View can be implicitly converted to View<const float, 3>.
  // read_only_view still refers to the same underlying data, but cannot modify
  // the elements.
  View<const float, 3> read_only_view = field.view();

  // reshape(new_extent)
  //
  // new_extent:
  //   The new set of Rank extents.
  //
  // reshape does not reallocate or move data. It only changes the shape and
  // row-major strides. The old and new logical element counts must be equal:
  //
  //   32 * 64 * 128 == 16 * 128 * 128
  //
  // When shape checks are enabled, inequality calls std::terminate().
  const index_t new_shape[3] = {16, 128, 128};
  field.reshape(new_shape);
}
```

`ndarray.hpp` does not depend on the algorithm layer and can be used independently as the unified underlying storage and View type for a project.

---

# Core Types

## `Buffer<T>`

Dynamic, aligned, one-dimensional owning storage.

```cpp
using namespace vehkko::ndarray;

// Buffer<T>(size, alignment, allocator)
//
// size:
//   Number of elements, not bytes.
//
// alignment:
//   Optional alignment in bytes. Defaults to default_alignment == 64.
//   The actual alignment is at least alignof(T).
//
// allocator:
//   Optional. Defaults to default_allocator().
Buffer<double> buffer(1'000'000);

// Explicitly request 128-byte alignment.
Buffer<double> aligned(1024, 128);

buffer[0] = 1.0;
double* ptr = buffer.data();
```

Main properties:

- value semantics with deep-copying copy construction and copy assignment;
- movable;
- default construction creates an empty Buffer;
- default initialization follows the element type's default initialization and does not explicitly zero-fill the storage;
- provides `data()`, `begin()`, `end()`, `front()`, `back()`, `size()`, `bytes()`, and `alignment()`.

Dynamic copies preserve the source size, alignment, and Allocator handle; copy assignment replaces the destination storage accordingly.

Note:

```cpp
Buffer<double> buffer(1024);
```

This does not guarantee that the 1024 `double` elements are initialized to zero. When zero initialization is required, use the following explicitly:

```cpp
zero_inplace(make_view1d(buffer.data(), buffer.size()));
```

## `StaticBuffer<T, Capacity, Alignment>`

Fixed-capacity stack storage.

```cpp
// T:
//   Element type.
//
// Capacity:
//   Compile-time fixed number of elements, not bytes.
//
// Alignment:
//   Optional compile-time alignment in bytes. Defaults to 64.
StaticBuffer<double, 256> scratch;
StaticBuffer<float, 1024, 128> aligned_scratch;
```

Construction does not perform an additional full-capacity zero-fill for convenience. When initialization is required, explicitly call `zero_inplace` or `fill_inplace` from the algorithm layer.

## `Array<T, Rank>`

Dynamic owning, row-major, fixed-Rank array.

```cpp
using namespace vehkko::ndarray;

// The length of the shape array must match the Rank template argument.
const index_t shape[2] = {128, 256};

// Array<T, Rank>(extent, alignment, allocator)
//
// extent:
//   Length of each dimension: 128 rows and 256 columns here.
//
// alignment:
//   Optional. Defaults to 64.
//
// allocator:
//   Optional. Defaults to default_allocator().
Array<double, 2> matrix(shape);

matrix(4, 7) = 3.0;
```

For routine creation, the helper functions are recommended:

```cpp
// Length 1024; returns Array<double, 1>.
auto a = make_array1d<double>(1024);

// extent = {128, 256}; returns Array<double, 2>.
auto b = make_array2d<double>(128, 256);

// extent = {32, 64, 128}; returns Array<float, 3>.
auto c = make_array3d<float>(32, 64, 128);
```

`reshape` does not reallocate and only permits changing to a new row-major shape with the same total number of elements:

```cpp
const index_t shape1[2] = {64, 128};
Array<double, 2> a(shape1);

// 64 * 128 == 32 * 256, so this reshape is valid.
const index_t shape2[2] = {32, 256};
a.reshape(shape2);
```

## `StaticArray<T, Rank, Capacity, Alignment>`

A stack array with fixed physical capacity and a mutable logical shape.

```cpp
// T = double
// Rank = 2
// Capacity = 256 elements
// Alignment is omitted and defaults to 64 bytes
const index_t shape[2] = {8, 16};
StaticArray<double, 2, 256> local(shape);

// 16 * 16 == 256, which still does not exceed Capacity.
const index_t larger_shape[2] = {16, 16};
local.reshape(larger_shape);
```

Where:

- `size()`: current number of logical elements;
- `capacity()`: compile-time fixed maximum number of elements;
- `bytes()`: current logical data size in bytes;
- `capacity_bytes()`: number of bytes corresponding to the full physical capacity.

Suitable for small temporary workspaces and local arrays with a fixed upper bound.

## `View<T, Rank>`

A non-owning, fixed-Rank multidimensional View whose strides are measured in elements.

```cpp
double data[24];

// extent = {4, 6}, so this is logically a 4-row by 6-column array.
const index_t shape[2] = {4, 6};

// make_view(data, extent) automatically constructs row-major strides:
// stride = {6, 1}
auto view = make_view(data, shape);

view(2, 3) = 7.0;
```

Strided View:

```cpp
double data[24];

// make_strided_view1d(data, n0, s0)
//
// data:
//   Address of the first logical element.
//
// n0:
//   Number of logical elements, which is 12 here.
//
// s0:
//   Physical spacing between adjacent logical elements, measured in elements.
//   s0 == 2 means accessing data[0], data[2], data[4], ...
auto every_other = make_strided_view1d(data, 12, 2);
```

A View does not manage the lifetime of the underlying data. The caller must ensure that the underlying storage remains valid for the entire period in which the View is used.

---

# `const` Semantics of View

`View` follows `std::span`-style shallow-const semantics:

```cpp
View<double, 2> view = /* ... */;

// The handle itself is const, which only means that its data/extent/stride
// metadata cannot be changed. Whether the elements are writable is still
// determined by the double template argument.
const View<double, 2> handle = view;

handle(0, 0) = 1.0; // Valid
```

Truly read-only elements require:

```cpp
View<const double, 2> read_only = view;

// read_only(0, 0) = 1.0; // Compilation error
```

Therefore, element writability is determined only by whether `T` is `const`, not by whether the View object itself is `const`.

---

# Derived Views and Layout Queries

```cpp
auto matrix = make_array2d<double>(100, 200);
auto v = matrix.view();

// row_view(v, i)
//
// v:
//   Rank-2 input View.
//
// i:
//   Row index, requiring i < v.extent(0).
//
// Returns a View<T, 1> of length v.extent(1).
// For a row-major matrix, this View is usually contiguous.
auto row = row_view(v, 7);

// col_view(v, j)
//
// j:
//   Column index, requiring j < v.extent(1).
//
// Returns a View<T, 1> of length v.extent(0).
// For a row-major matrix, the stride of this View is usually v.stride(0).
auto column = col_view(v, 9);

// block_view(v, i0, j0, n0, n1)
//
// i0, j0:
//   Starting indices of the block's upper-left corner in the original View.
//
// n0, n1:
//   Extents of the block along the two dimensions.
//
// Requirements:
//   i0 + n0 <= v.extent(0)
//   j0 + n1 <= v.extent(1)
auto block = block_view(v, 10, 20, 30, 40);

// flatten_view(v)
//
// Accepts only row-major contiguous Views.
// Returns a View<T, 1> of length v.size() and stride 1.
// Does not copy the underlying data.
auto flat = flatten_view(v);
```

Layout queries:

```cpp
// Whether the innermost stride is 1.
is_inner_contiguous(v);

// Whether the layout is fully row-major contiguous.
is_row_major_contiguous(v);

// Compares only extents, not data or strides.
same_extents(a, b);

// Compares extents and strides, but not data.
same_layout(a, b);

// Compares only the starting data() addresses.
is_same_data(a, b);
```

`View` supports non-negative strides. Negative strides and reversed Views are not currently supported.

A View used as a write destination should have an injective mapping: distinct logical indices should not refer to the same element. The library checks obviously invalid zero-stride layouts, but does not solve the complete mapping-injectivity problem for arbitrary Rank.

---

# Algorithm Layer

All algorithms accept only `View`:

```cpp
fill_inplace(array.view(), 0.0);
add_to(dst.view(), x.view(), y.view());
```

This avoids overload proliferation from mixed owning/View parameters and keeps the actual data structures processed by the algorithms explicit.

## Initialization and Unary Operations

```cpp
// fill_inplace(dst, value)
// Writes value to every logical element in dst.
fill_inplace(dst, value);

// zero_inplace(dst)
// Equivalent to fill_inplace(dst, T{}).
zero_inplace(dst);

// scale_inplace(dst, alpha)
// dst[i...] *= alpha
scale_inplace(dst, alpha);

// shift_inplace(dst, beta)
// dst[i...] += beta
shift_inplace(dst, beta);

// negate_inplace(dst)
// dst[i...] = -dst[i...]
negate_inplace(dst);

// square_inplace(dst)
// dst[i...] *= dst[i...]
square_inplace(dst);

// iota_inplace(dst, start = T{}, step = T{1})
//
// start:
//   Value of the first logical element. Defaults to T{}.
//
// step:
//   Increment between adjacent logical elements. Defaults to T{1}.
iota_inplace(dst, start, step);

// linspace_inplace(dst, first, last)
//
// first:
//   First logical element.
//
// last:
//   Last logical element.
//
// Both endpoints are included when size() > 1.
linspace_inplace(dst, first, last);
```

Example:

```cpp
auto a = make_array1d<double>(5);

// Result: {2.0, 2.5, 3.0, 3.5, 4.0}
iota_inplace(a.view(), 2.0, 0.5);

// Result: {0.0, 0.25, 0.5, 0.75, 1.0}
linspace_inplace(a.view(), 0.0, 1.0);
```

## Copying and Element-wise Output

```cpp
// copy_to(dst, src)
// dst and src must have the same extents.
// The regular version safely handles complex overlap and may allocate a
// temporary Array when necessary.
copy_to(dst, src);

// copy_to_noalias(dst, src)
// dst and src must not overlap; performs no allocation.
copy_to_noalias(dst, src);

// add/sub/mul/div all perform element-wise operations:
//
// dst[i...] = x[i...] OP y[i...]
add_to(dst, x, y);
sub_to(dst, x, y);
mul_to(dst, x, y);
div_to(dst, x, y);

// axpby_to(dst, alpha, x, beta, y)
//
// dst:
//   Output View.
//
// alpha:
//   Scalar coefficient of x.
//
// x:
//   First input View.
//
// beta:
//   Scalar coefficient of y.
//
// y:
//   Second input View.
//
// Computes:
//   dst[i...] = alpha * x[i...] + beta * y[i...]
axpby_to(dst, alpha, x, beta, y);
```

The corresponding highest-performance APIs are:

```cpp
add_to_noalias(dst, x, y);
sub_to_noalias(dst, x, y);
mul_to_noalias(dst, x, y);
div_to_noalias(dst, x, y);
axpby_to_noalias(dst, alpha, x, beta, y);
```

## In-place Binary Operations

```cpp
// add_inplace(dst, x)
// dst[i...] += x[i...]
add_inplace(dst, x);

// sub_inplace(dst, x)
// dst[i...] -= x[i...]
sub_inplace(dst, x);

// mul_inplace(dst, x)
// dst[i...] *= x[i...]
mul_inplace(dst, x);

// axpy_inplace(y, alpha, x)
//
// y:
//   Writable output that also serves as the original-value input.
//
// alpha:
//   Scalar coefficient of x.
//
// x:
//   Read-only input.
//
// Computes:
//   y[i...] += alpha * x[i...]
axpy_inplace(y, alpha, x);
```

All functions above have corresponding `_noalias` versions.

## Reductions and Comparisons

```cpp
// sum(x)
// Returns the sum of all logical elements.
auto s = sum(x);

// dot(x, y)
// The two Views must have the same extents.
// Returns sum(x[i...] * y[i...]).
auto d = dot(x, y);

// squared_norm(x)
// Returns sum(x[i...] * x[i...]) without computing sqrt.
auto n2 = squared_norm(x);

// max_abs(x)
// Returns the maximum absolute value.
auto m = max_abs(x);

// min_value / max_value require a non-empty View.
auto lo = min_value(x);
auto hi = max_value(x);

// max_abs_diff(x, y)
// Returns max(abs(x[i...] - y[i...])).
auto emax = max_abs_diff(x, y);

// squared_diff_norm(x, y)
// Returns sum((x[i...] - y[i...])^2).
auto e2 = squared_diff_norm(x, y);

// all_close(x, y, atol, rtol)
// Defined only for floating-point element types.
//
// atol:
//   Absolute-error tolerance. There is no default; the caller must provide it
//   explicitly.
//
// rtol:
//   Relative-error tolerance. There is no default; the caller must provide it
//   explicitly.
//
// Each element uses:
//   abs(x - y) <= atol + rtol * abs(y)
bool close = all_close(x, y, atol, rtol);
```

Example:

```cpp
bool close = all_close(
    x.view(),
    y.view(),
    1e-12, // atol: absolute-error floor
    1e-10  // rtol: proportional error relative to y
);
```

---

# Regular APIs and `_noalias`

This is the most important semantic distinction in the algorithm layer.

## Regular APIs

For example:

```cpp
add_to(dst, x, y);
copy_to(dst, src);
```

When alias dispatch is enabled by default, regular APIs:

1. check the shape;
2. determine the mapping relationship between outputs and inputs at low cost;
3. enter the `_noalias` fast path when non-overlap has been proven;
4. execute with in-place semantics when the mappings are exactly identical;
5. use a temporary `Array` for dangerous, complex partial overlap;
6. use `std::memmove` for contiguous overlapping copies;
7. choose forward or backward traversal whenever possible for Rank-1 copies with equal strides, avoiding temporaries.

Regular APIs are therefore safe by default, but complex-overlap paths may allocate temporary storage. Allocation or element-construction failures may propagate to the caller.

## `_noalias` APIs

For example:

```cpp
add_to_noalias(dst, x, y);
```

The contract is:

> No writable output region may overlap any read-only input region; read-only inputs may overlap one another.

The following is valid:

```cpp
// x is used as both read-only inputs.
// The _noalias contract is satisfied as long as dst does not overlap x.
add_to_noalias(dst, x, x);
```

The following violates the contract:

```cpp
// dst and the first read-only input x have the same mapping, violating the
// _noalias contract.
add_to_noalias(x, x, y);
```

The `_noalias` versions:

- perform no allocations;
- provide no safe fallback;
- use a `restrict` pointer for the output on contiguous paths;
- can maintain stable `noexcept` guarantees;
- rely ultimately on the caller, because Debug contract checks can detect only some violations that can be proven at low cost.

---

# Overlap Detection Design

Regular algorithms use a conservative three-state classifier:

```cpp
enum class AliasRelation {
  disjoint,
  same_mapping,
  may_overlap
};
```

The checks are applied in the following order:

1. empty Views;
2. exactly identical data, extents, and strides;
3. bounding byte-address intervals;
4. greatest-common-divisor congruence of stride address lattices;
5. conservatively returning `may_overlap` when non-overlap cannot be proven at low cost.

The classifier never misclassifies real overlap as `disjoint`. It may conservatively classify a small number of complex but actually disjoint layouts as `may_overlap`, avoiding overengineering involving multidimensional Diophantine equations, the Chinese remainder theorem, or element enumeration.

---

# Custom Allocator

Allocator uses simple function-pointer type erasure:

```cpp
struct Allocator {
  // User context passed to allocate/deallocate;
  // nullptr in the default allocator.
  void* ctx;

  // ctx:
  //   The user context above.
  //
  // bytes:
  //   Requested number of bytes, not number of elements.
  //
  // alignment:
  //   Requested alignment in bytes.
  //
  // Returns:
  //   A non-null memory address satisfying alignment.
  void* (*allocate)(void* ctx,
                    index_t bytes,
                    index_t alignment);

  // ctx:
  //   User context.
  //
  // ptr:
  //   Address previously returned by allocate.
  //
  // bytes:
  //   Original allocation size in bytes.
  //
  // alignment:
  //   Original requested alignment.
  void (*deallocate)(void* ctx,
                     void* ptr,
                     index_t bytes,
                     index_t alignment) noexcept;
};
```

Example:

```cpp
#include "ndarray.hpp"

#include <new>

struct Counter {
  std::size_t allocated = 0;
};

void* counted_allocate(void* raw_ctx,
                       vehkko::ndarray::index_t bytes,
                       vehkko::ndarray::index_t alignment) {
  auto& ctx = *static_cast<Counter*>(raw_ctx);

  // bytes is the actual number of bytes requested by this allocation.
  ctx.allocated += bytes;

  // The returned address must satisfy alignment.
  return ::operator new[](bytes, std::align_val_t{alignment});
}

void counted_deallocate(void*,
                        void* ptr,
                        vehkko::ndarray::index_t,
                        vehkko::ndarray::index_t alignment) noexcept {
  // Deallocation must use the same alignment as allocation.
  ::operator delete[](ptr, std::align_val_t{alignment});
}

int main() {
  using namespace vehkko::ndarray;

  Counter counter;

  // Allocator{ctx, allocate, deallocate}
  Allocator allocator{
      &counter,
      counted_allocate,
      counted_deallocate,
  };

  // make_array2d<T>(n0, n1, alignment, allocator)
  //
  // n0 = 128
  // n1 = 256
  // alignment = 64
  // allocator = the counted allocator defined above
  //
  // 64 must be written explicitly here in order to pass the fourth allocator
  // argument.
  auto array = make_array2d<double>(128, 256, 64, allocator);
}
```

This interface leaves an extension point for arenas, NUMA, page-locked host memory, and similar use cases, but the library itself does not currently abstract CUDA address spaces or device-side access.

---

# Configuration Macros

All macros must be defined before the first inclusion of a header and must use the integer value `0` or `1`.

| Macro | Debug Default | Release Default | Meaning |
|---|---:|---:|---|
| `VEHKKO_NDARRAY_BOUNDS_CHECK` | 1 | 0 | element-index and dimension-index checks |
| `VEHKKO_NDARRAY_SHAPE_CHECK` | 1 | 1 | shape consistency, reshape, and capacity checks |
| `VEHKKO_NDARRAY_OVERFLOW_CHECK` | 1 | 1 | overflow checks for shapes, strides, byte sizes, and address ranges |
| `VEHKKO_NDARRAY_ALIGNMENT_CHECK` | 1 | 0 | alignment-argument checks |
| `VEHKKO_NDARRAY_ALLOCATOR_CHECK` | 1 | 0 | allocator function-pointer and return-value checks |
| `VEHKKO_NDARRAY_ALIAS_DISPATCH` | 1 | 1 | safe overlap dispatch for regular algorithms |
| `VEHKKO_NDARRAY_NOALIAS_CONTRACT_CHECK` | 1 | 0 | partial Debug contract validation for `_noalias` |
| `VEHKKO_NDARRAY_LAYOUT_CHECK` | 1 | 0 | checks for obviously invalid View layouts |

Example:

```cpp
// Disable per-element bounds checks.
#define VEHKKO_NDARRAY_BOUNDS_CHECK 0

// Keep structural overflow checks.
#define VEHKKO_NDARRAY_OVERFLOW_CHECK 1

// Keep safe alias dispatch for regular algorithms.
#define VEHKKO_NDARRAY_ALIAS_DISPATCH 1

// Macros must be defined before the first include.
#include "ndarray_ops.hpp"
```

For extreme performance, when the caller assumes full responsibility for all contracts, every check can be disabled:

```cpp
#define VEHKKO_NDARRAY_BOUNDS_CHECK 0
#define VEHKKO_NDARRAY_SHAPE_CHECK 0
#define VEHKKO_NDARRAY_OVERFLOW_CHECK 0
#define VEHKKO_NDARRAY_ALIGNMENT_CHECK 0
#define VEHKKO_NDARRAY_ALLOCATOR_CHECK 0
#define VEHKKO_NDARRAY_ALIAS_DISPATCH 0
#define VEHKKO_NDARRAY_NOALIAS_CONTRACT_CHECK 0
#define VEHKKO_NDARRAY_LAYOUT_CHECK 0

#include "ndarray_ops.hpp"
```

After `VEHKKO_NDARRAY_ALIAS_DISPATCH` is disabled, regular algorithms no longer guarantee correctness under dangerous partial overlap. The caller must ensure the aliasing conditions independently.

---

# Error Handling

The library neither defines nor propagates artificial runtime exception types.

When an enabled contract check fails, for example due to:

- mismatched shapes;
- shape or byte-size overflow;
- a `StaticArray` exceeding its capacity;
- invalid alignment;
- an obviously invalid View layout;

the library calls:

```cpp
std::terminate();
```

Allocation and element-construction failures from owning construction, copying, assignment, or temporary `Array` creation are not converted by the library and may propagate to the caller. Deallocation callbacks remain `noexcept`.
