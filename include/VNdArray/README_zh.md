# vehkko::ndarray

一个面向 C++17 数值代码的轻量 Header-Only 多维数组与跨步 View 库。

它的目标是提供一层简单、显式的多维数组数据结构：

- 对齐的动态与固定容量连续存储；
- 固定 Rank、非拥有、支持 stride 的多维 View；
- 不产生隐式临时量的基础逐元素算法；
- 默认安全的别名处理，以及显式的 `_noalias` 高性能接口；
- 所有运行时检查均可通过编译期宏完全关闭；
- 无第三方依赖，算法层可选。

```text
ndarray.hpp      owning storage + fixed-rank strided View
ndarray_ops.hpp  alias-aware basic algorithms
```

只需要容器与 View 时，仅包含 `ndarray.hpp`；需要基础算法时包含 `ndarray_ops.hpp`。

---

## 特性

- C++17，Header-Only，无第三方依赖；
- 默认 64 字节对齐；
- `Buffer<T>`：动态、对齐、可拷贝、可移动的一维存储；
- `StaticBuffer<T, Capacity>`：栈上固定容量存储；
- `View<T, Rank>`：固定 Rank、非拥有、可跨步多维视图；
- `Array<T, Rank>`：动态 owning row-major 数组；
- `StaticArray<T, Rank, Capacity>`：固定容量、可改变逻辑形状的 row-major 数组；
- owning 容器采用值语义并深拷贝数据；复制 View 只复制其句柄元数据；
- 自定义函数指针 Allocator，为 arena、pinned memory 等扩展保留入口；
- 普通算法默认处理危险的部分重叠；
- `_noalias` 算法无分配，并为连续数据提供 `restrict` 快路径；
- Rank 1–3 采用直接循环，较高 Rank 使用通用固定 Rank 遍历器；
- shape、stride、字节数和地址范围的结构性溢出检查；
- 契约失败统一调用 `std::terminate()`；
- owning 操作中的分配失败或元素构造失败可以传播给调用者。

---

## 要求

- C++17 或更新版本；
- GCC、Clang 或其他支持 aligned new 的标准 C++17 编译器；
- owning 类型的元素必须满足：
  - 非 `const`、非 `volatile`、非引用；
  - trivially copyable；
  - trivially destructible；
  - standard-layout；
  - default-constructible。

`View<const T, Rank>` 可以用于只读元素。

---

## 快速开始

```cpp
#include "ndarray_ops.hpp"

#include <iostream>

int main() {
  using namespace vehkko::ndarray;

  // make_array2d<T>(n0, n1, alignment, allocator)
  //
  // T:
  //   元素类型，这里为 double。
  //
  // n0, n1:
  //   两个维度的 extent。这里分别为 1024 和 1024，
  //   因此数组共有 1024 * 1024 个 double。
  //
  // alignment:
  //   可选，底层首地址对齐字节数，默认 default_alignment，即 64。
  //
  // allocator:
  //   可选，自定义分配器，默认 default_allocator()。
  //
  // 返回类型为 Array<double, 2>，布局为 row-major。
  auto x = make_array2d<double>(1024, 1024);
  auto y = make_array2d<double>(1024, 1024);
  auto result = make_array2d<double>(1024, 1024);

  // fill_inplace(dst, value)
  //
  // dst:
  //   可写 View。算法层只接受 View，因此 owning Array 需要显式调用 .view()。
  //
  // value:
  //   写入每个逻辑元素的值。
  //
  // 此处将 x、y 的所有元素分别初始化为 1.0 和 2.0。
  fill_inplace(x.view(), 1.0);
  fill_inplace(y.view(), 2.0);

  // add_to(dst, lhs, rhs)
  //
  // dst:
  //   输出 View，必须与两个输入具有相同 extent。
  //
  // lhs, rhs:
  //   两个只读输入 View。执行逐元素计算：
  //
  //     dst[i...] = lhs[i...] + rhs[i...]
  //
  // 普通接口默认启用 alias dispatch：
  // - 已证明不重叠时自动进入 _noalias 快路径；
  // - 完全相同映射按安全的原地语义处理；
  // - 危险部分重叠时可能创建临时 Array。
  //
  // 本例中三个 Array 独立分配，因此会进入连续 noalias 快路径。
  add_to(result.view(), x.view(), y.view());

  // operator()(i, j) 按 row-major 逻辑索引访问。
  // Debug 默认检查 i < extent(0)、j < extent(1)；
  // Release 默认关闭逐元素 bounds check。
  std::cout << result(0, 0) << '\n'; // 输出 3
}
```

编译：

```bash
g++ -std=c++17 -O3 -DNDEBUG -march=native main.cpp -o main
```

其中：

- `-std=c++17`：启用 C++17；
- `-O3`：开启高等级优化；
- `-DNDEBUG`：关闭默认仅 Debug 启用的断言与部分检查；
- `-march=native`：允许编译器针对本机指令集生成代码，可选。

---

## 只使用容器层

```cpp
#include "ndarray.hpp"

int main() {
  using namespace vehkko::ndarray;

  // 创建 Array<float, 3>。
  //
  // extent = {32, 64, 128}
  // stride = {64 * 128, 128, 1}
  // size   = 32 * 64 * 128
  auto field = make_array3d<float>(32, 64, 128);

  // 三个索引分别对应第 0、1、2 维。
  field(1, 2, 3) = 4.0f;

  // .view() 在非 const Array 上返回 View<float, 3>，
  // 可读写底层元素，不复制数据。
  View<float, 3> mutable_view = field.view();

  // 可变 View 可以隐式转换为 View<const float, 3>。
  // read_only_view 仍然指向同一份底层数据，但不能修改元素。
  View<const float, 3> read_only_view = field.view();

  // reshape(new_extent)
  //
  // new_extent:
  //   新的 Rank 个 extent。
  //
  // reshape 不重新分配、不移动数据，只修改 shape 和 row-major stride。
  // 新旧逻辑元素总数必须相同：
  //
  //   32 * 64 * 128 == 16 * 128 * 128
  //
  // shape check 开启时，不相等会调用 std::terminate()。
  const index_t new_shape[3] = {16, 128, 128};
  field.reshape(new_shape);
}
```

`ndarray.hpp` 不依赖算法层，可以作为项目统一的底层存储与 View 类型单独使用。

---

# 核心类型

## `Buffer<T>`

动态、对齐的一维 owning storage。

```cpp
using namespace vehkko::ndarray;

// Buffer<T>(size, alignment, allocator)
//
// size:
//   元素个数，不是字节数。
//
// alignment:
//   可选，对齐字节数，默认 default_alignment == 64。
//   实际对齐至少为 alignof(T)。
//
// allocator:
//   可选，默认 default_allocator()。
Buffer<double> buffer(1'000'000);

// 显式要求 128 字节对齐。
Buffer<double> aligned(1024, 128);

buffer[0] = 1.0;
double* ptr = buffer.data();
```

主要性质：

- 值语义，拷贝构造和拷贝赋值均深拷贝；
- 可移动；
- 默认构造为空；
- 默认初始化遵循元素的 default initialization，不主动填零；
- 提供 `data()`、`begin()`、`end()`、`front()`、`back()`、`size()`、`bytes()` 和 `alignment()`。

动态容器的副本保留源对象的 size、alignment 和 Allocator 句柄；拷贝赋值会相应替换目标对象的存储。

注意：

```cpp
Buffer<double> buffer(1024);
```

不会保证 1024 个 `double` 被初始化为零。需要零初始化时应显式使用：

```cpp
zero_inplace(make_view1d(buffer.data(), buffer.size()));
```

## `StaticBuffer<T, Capacity, Alignment>`

栈上固定容量存储。

```cpp
// T:
//   元素类型。
//
// Capacity:
//   编译期固定元素数量，不是字节数。
//
// Alignment:
//   可选，编译期对齐字节数，默认 64。
StaticBuffer<double, 256> scratch;
StaticBuffer<float, 1024, 128> aligned_scratch;
```

构造时不会为了方便而额外执行全容量清零。需要初始化时应显式调用算法层的 `zero_inplace` 或 `fill_inplace`。

## `Array<T, Rank>`

动态 owning、row-major、固定 Rank 数组。

```cpp
using namespace vehkko::ndarray;

// shape 的数组长度必须与模板参数 Rank 一致。
const index_t shape[2] = {128, 256};

// Array<T, Rank>(extent, alignment, allocator)
//
// extent:
//   每一维的长度，这里为 128 行、256 列。
//
// alignment:
//   可选，默认 64。
//
// allocator:
//   可选，默认 default_allocator()。
Array<double, 2> matrix(shape);

matrix(4, 7) = 3.0;
```

日常创建建议使用辅助函数：

```cpp
// 长度为 1024，返回 Array<double, 1>。
auto a = make_array1d<double>(1024);

// extent = {128, 256}，返回 Array<double, 2>。
auto b = make_array2d<double>(128, 256);

// extent = {32, 64, 128}，返回 Array<float, 3>。
auto c = make_array3d<float>(32, 64, 128);
```

`reshape` 不重新分配，仅允许改变为相同元素总数的新 row-major 形状：

```cpp
const index_t shape1[2] = {64, 128};
Array<double, 2> a(shape1);

// 64 * 128 == 32 * 256，因此 reshape 合法。
const index_t shape2[2] = {32, 256};
a.reshape(shape2);
```

## `StaticArray<T, Rank, Capacity, Alignment>`

固定物理容量、可改变逻辑形状的栈上数组。

```cpp
// T = double
// Rank = 2
// Capacity = 256 个元素
// Alignment 省略，默认 64 字节
const index_t shape[2] = {8, 16};
StaticArray<double, 2, 256> local(shape);

// 16 * 16 == 256，仍未超过 Capacity。
const index_t larger_shape[2] = {16, 16};
local.reshape(larger_shape);
```

其中：

- `size()`：当前逻辑元素数量；
- `capacity()`：编译期固定最大元素数量；
- `bytes()`：当前逻辑数据字节数；
- `capacity_bytes()`：全部物理容量对应的字节数。

适合小型临时工作区和固定上限的局部数组。

## `View<T, Rank>`

非 owning、固定 Rank、stride 单位为元素的多维 View。

```cpp
double data[24];

// extent = {4, 6}，因此逻辑上是 4 行 6 列。
const index_t shape[2] = {4, 6};

// make_view(data, extent) 自动构造 row-major stride：
// stride = {6, 1}
auto view = make_view(data, shape);

view(2, 3) = 7.0;
```

跨步 View：

```cpp
double data[24];

// make_strided_view1d(data, n0, s0)
//
// data:
//   第一个逻辑元素的地址。
//
// n0:
//   逻辑元素数量，这里为 12。
//
// s0:
//   相邻逻辑元素之间的物理间隔，单位是元素。
//   s0 == 2 表示访问 data[0], data[2], data[4], ...
auto every_other = make_strided_view1d(data, 12, 2);
```

View 不管理底层数据生命周期。调用者必须保证底层存储在 View 使用期间持续有效。

---

# View 的 const 语义

`View` 遵循 `std::span` 风格的浅 const 语义：

```cpp
View<double, 2> view = /* ... */;

// handle 本身为 const，只表示不能改变其 data/extent/stride 元数据。
// 元素是否可写仍由模板参数 double 决定。
const View<double, 2> handle = view;

handle(0, 0) = 1.0; // 合法
```

真正的只读元素必须使用：

```cpp
View<const double, 2> read_only = view;

// read_only(0, 0) = 1.0; // 编译失败
```

因此，元素是否可写只由 `T` 是否为 `const` 决定，而不是由 View 对象本身是否为 `const` 决定。

---

# View 派生与布局查询

```cpp
auto matrix = make_array2d<double>(100, 200);
auto v = matrix.view();

// row_view(v, i)
//
// v:
//   Rank-2 输入 View。
//
// i:
//   行索引，要求 i < v.extent(0)。
//
// 返回长度为 v.extent(1) 的 View<T, 1>。
// 对 row-major 矩阵，该 View 通常连续。
auto row = row_view(v, 7);

// col_view(v, j)
//
// j:
//   列索引，要求 j < v.extent(1)。
//
// 返回长度为 v.extent(0) 的 View<T, 1>。
// 对 row-major 矩阵，该 View 的 stride 通常为 v.stride(0)。
auto column = col_view(v, 9);

// block_view(v, i0, j0, n0, n1)
//
// i0, j0:
//   子块左上角在原 View 中的起始索引。
//
// n0, n1:
//   子块在两个维度上的 extent。
//
// 要求：
//   i0 + n0 <= v.extent(0)
//   j0 + n1 <= v.extent(1)
auto block = block_view(v, 10, 20, 30, 40);

// flatten_view(v)
//
// 只接受 row-major contiguous View。
// 返回长度为 v.size()、stride 为 1 的 View<T, 1>。
// 不复制底层数据。
auto flat = flatten_view(v);
```

布局查询：

```cpp
// 最内层 stride 是否为 1。
is_inner_contiguous(v);

// 是否为完整 row-major contiguous 布局。
is_row_major_contiguous(v);

// 只比较 extent，不比较 data 或 stride。
same_extents(a, b);

// 比较 extent 和 stride，不比较 data。
same_layout(a, b);

// 只比较 data() 起始地址。
is_same_data(a, b);
```

`View` 支持非负 stride。当前没有负 stride 或反向 View。

作为写入目标的 View 应具有单射映射：不同逻辑索引不应指向同一元素。库会检查明显非法的零 stride 布局，但不会为任意 Rank 求解完整的映射单射性问题。

---

# 算法层

所有算法只接受 `View`：

```cpp
fill_inplace(array.view(), 0.0);
add_to(dst.view(), x.view(), y.view());
```

这样可以避免 owning/View 混合参数的重载膨胀，并使算法真实处理的数据结构保持明确。

## 初始化与一元操作

```cpp
// fill_inplace(dst, value)
// 将 dst 中每个逻辑元素写成 value。
fill_inplace(dst, value);

// zero_inplace(dst)
// 等价于 fill_inplace(dst, T{})。
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
//   第一个逻辑元素的值，默认 T{}。
//
// step:
//   相邻逻辑元素之间的增量，默认 T{1}。
iota_inplace(dst, start, step);

// linspace_inplace(dst, first, last)
//
// first:
//   第一个逻辑元素。
//
// last:
//   最后一个逻辑元素。
//
// 当 size() > 1 时，两端均包含。
linspace_inplace(dst, first, last);
```

示例：

```cpp
auto a = make_array1d<double>(5);

// 结果为 {2.0, 2.5, 3.0, 3.5, 4.0}
iota_inplace(a.view(), 2.0, 0.5);

// 结果为 {0.0, 0.25, 0.5, 0.75, 1.0}
linspace_inplace(a.view(), 0.0, 1.0);
```

## 复制与逐元素输出

```cpp
// copy_to(dst, src)
// dst 和 src 必须具有相同 extent。
// 普通版本允许安全处理复杂重叠，必要时可能分配临时 Array。
copy_to(dst, src);

// copy_to_noalias(dst, src)
// dst 与 src 不得重叠；不分配。
copy_to_noalias(dst, src);

// add/sub/mul/div 均执行逐元素运算：
//
// dst[i...] = x[i...] OP y[i...]
add_to(dst, x, y);
sub_to(dst, x, y);
mul_to(dst, x, y);
div_to(dst, x, y);

// axpby_to(dst, alpha, x, beta, y)
//
// dst:
//   输出 View。
//
// alpha:
//   x 的标量系数。
//
// x:
//   第一个输入 View。
//
// beta:
//   y 的标量系数。
//
// y:
//   第二个输入 View。
//
// 计算：
//   dst[i...] = alpha * x[i...] + beta * y[i...]
axpby_to(dst, alpha, x, beta, y);
```

对应的最高性能接口：

```cpp
add_to_noalias(dst, x, y);
sub_to_noalias(dst, x, y);
mul_to_noalias(dst, x, y);
div_to_noalias(dst, x, y);
axpby_to_noalias(dst, alpha, x, beta, y);
```

## 原地二元操作

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
//   可写输出，同时也是原值输入。
//
// alpha:
//   x 的标量系数。
//
// x:
//   只读输入。
//
// 计算：
//   y[i...] += alpha * x[i...]
axpy_inplace(y, alpha, x);
```

以上函数均有对应的 `_noalias` 版本。

## Reduction 与比较

```cpp
// sum(x)
// 返回所有逻辑元素之和。
auto s = sum(x);

// dot(x, y)
// 两个 View 必须具有相同 extent。
// 返回 sum(x[i...] * y[i...])。
auto d = dot(x, y);

// squared_norm(x)
// 返回 sum(x[i...] * x[i...])，不执行 sqrt。
auto n2 = squared_norm(x);

// max_abs(x)
// 返回最大绝对值。
auto m = max_abs(x);

// min_value / max_value 要求 View 非空。
auto lo = min_value(x);
auto hi = max_value(x);

// max_abs_diff(x, y)
// 返回 max(abs(x[i...] - y[i...]))。
auto emax = max_abs_diff(x, y);

// squared_diff_norm(x, y)
// 返回 sum((x[i...] - y[i...])^2)。
auto e2 = squared_diff_norm(x, y);

// all_close(x, y, atol, rtol)
// 仅适用于浮点元素类型。
//
// atol:
//   绝对误差容限，无默认值，调用者必须显式提供。
//
// rtol:
//   相对误差容限，无默认值，调用者必须显式提供。
//
// 每个元素使用：
//   abs(x - y) <= atol + rtol * abs(y)
bool close = all_close(x, y, atol, rtol);
```

示例：

```cpp
bool close = all_close(
    x.view(),
    y.view(),
    1e-12, // atol：绝对误差下限
    1e-10  // rtol：相对于 y 的比例误差
);
```

---

# 普通接口与 `_noalias`

这是算法层最重要的语义区别。

## 普通接口

例如：

```cpp
add_to(dst, x, y);
copy_to(dst, src);
```

默认开启 alias dispatch 时，普通接口会：

1. 检查 shape；
2. 低成本判断输出与输入的映射关系；
3. 已证明不重叠时进入 `_noalias` 快路径；
4. 完全相同映射时按原地语义执行；
5. 危险的复杂部分重叠时使用临时 `Array`；
6. 连续重叠复制使用 `std::memmove`；
7. Rank-1 相同步长复制尽可能选择正向或反向遍历，避免临时量。

因此普通接口默认安全，但在复杂重叠路径中可能分配临时存储。分配失败或元素构造失败可以传播给调用者。

## `_noalias` 接口

例如：

```cpp
add_to_noalias(dst, x, y);
```

契约是：

> 每个可写输出区域不得与任何只读输入区域重叠；只读输入之间允许相互重叠。

下面是合法的：

```cpp
// x 同时作为两个只读输入。
// 只要 dst 与 x 不重叠，就满足 _noalias 契约。
add_to_noalias(dst, x, x);
```

下面违反契约：

```cpp
// dst 与第一个只读输入 x 是同一映射，违反 _noalias 契约。
add_to_noalias(x, x, y);
```

`_noalias` 版本：

- 不分配；
- 不做安全回退；
- 连续路径使用输出 `restrict` 指针；
- 可保持稳定的 `noexcept`；
- Debug 契约检查只能捕获一部分可以低成本证明的违约，最终责任仍由调用者承担。

---

# 重叠判定设计

普通算法使用一个保守的三态分类器：

```cpp
enum class AliasRelation {
  disjoint,
  same_mapping,
  may_overlap
};
```

判定依次使用：

1. 空 View；
2. 完全相同的 data、extent 和 stride；
3. 字节地址包围区间；
4. stride 地址格点的最大公约数同余；
5. 无法低成本证明时保守返回 `may_overlap`。

分类器不会把真实重叠误判为 `disjoint`。它允许把少量复杂但实际不重叠的布局保守判断为 `may_overlap`，从而避免引入多维丢番图方程、CRT 或元素枚举等过度设计。

---

# 自定义 Allocator

Allocator 使用简单的函数指针类型擦除：

```cpp
struct Allocator {
  // 传递给 allocate/deallocate 的用户上下文；
  // 默认 allocator 中为 nullptr。
  void* ctx;

  // ctx:
  //   上面的用户上下文。
  //
  // bytes:
  //   请求的字节数，不是元素个数。
  //
  // alignment:
  //   请求的字节对齐。
  //
  // 返回：
  //   非空且满足 alignment 的内存地址。
  void* (*allocate)(void* ctx,
                    index_t bytes,
                    index_t alignment);

  // ctx:
  //   用户上下文。
  //
  // ptr:
  //   先前由 allocate 返回的地址。
  //
  // bytes:
  //   原始分配字节数。
  //
  // alignment:
  //   原始请求对齐。
  void (*deallocate)(void* ctx,
                     void* ptr,
                     index_t bytes,
                     index_t alignment) noexcept;
};
```

示例：

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

  // bytes 是本次请求的实际字节数。
  ctx.allocated += bytes;

  // 必须返回满足 alignment 的地址。
  return ::operator new[](bytes, std::align_val_t{alignment});
}

void counted_deallocate(void*,
                        void* ptr,
                        vehkko::ndarray::index_t,
                        vehkko::ndarray::index_t alignment) noexcept {
  // 必须使用与分配时一致的 alignment 释放。
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
  // allocator = 上面定义的 counted allocator
  //
  // 这里必须显式写 64，才能传入第四个 allocator 参数。
  auto array = make_array2d<double>(128, 256, 64, allocator);
}
```

这一接口为 arena、NUMA、page-locked host memory 等扩展保留入口，但当前库本身不抽象 CUDA 地址空间或 device-side 访问。

---

# 配置宏

所有宏必须在第一次包含头文件之前定义，并使用整数 `0` 或 `1`。

| 宏 | Debug 默认 | Release 默认 | 含义 |
|---|---:|---:|---|
| `VEHKKO_NDARRAY_BOUNDS_CHECK` | 1 | 0 | 元素索引和维度编号检查 |
| `VEHKKO_NDARRAY_SHAPE_CHECK` | 1 | 1 | shape 一致性、reshape 与容量检查 |
| `VEHKKO_NDARRAY_OVERFLOW_CHECK` | 1 | 1 | shape、stride、byte size 与地址范围溢出检查 |
| `VEHKKO_NDARRAY_ALIGNMENT_CHECK` | 1 | 0 | alignment 参数检查 |
| `VEHKKO_NDARRAY_ALLOCATOR_CHECK` | 1 | 0 | allocator 函数指针和返回值检查 |
| `VEHKKO_NDARRAY_ALIAS_DISPATCH` | 1 | 1 | 普通算法的安全重叠分派 |
| `VEHKKO_NDARRAY_NOALIAS_CONTRACT_CHECK` | 1 | 0 | `_noalias` 的部分 Debug 契约验证 |
| `VEHKKO_NDARRAY_LAYOUT_CHECK` | 1 | 0 | View 的明显非法布局检查 |

示例：

```cpp
// 关闭逐元素 bounds check。
#define VEHKKO_NDARRAY_BOUNDS_CHECK 0

// 保留结构性溢出检查。
#define VEHKKO_NDARRAY_OVERFLOW_CHECK 1

// 保留普通算法的安全别名分派。
#define VEHKKO_NDARRAY_ALIAS_DISPATCH 1

// 宏必须在第一次 include 之前定义。
#include "ndarray_ops.hpp"
```

极端性能且调用者完全承担契约时，可以关闭全部检查：

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

关闭 `VEHKKO_NDARRAY_ALIAS_DISPATCH` 后，普通算法不再保证危险部分重叠的正确性，调用者必须自行保证别名条件。

---

# 错误处理

库不定义或传播人为的运行时异常类型。

当已启用的契约检查失败时，例如：

- shape 不匹配；
- shape 或 byte size 溢出；
- `StaticArray` 超容量；
- 非法 alignment；
- 明显非法的 View layout；

库调用：

```cpp
std::terminate();
```

owning 容器构造、拷贝、赋值或创建临时 `Array` 时发生的分配失败和元素构造失败，库不会进行转换，可以直接传播给调用者。释放回调仍必须为 `noexcept`。
