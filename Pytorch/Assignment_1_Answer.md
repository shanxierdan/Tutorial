# Assignment 1 参考答案（含 PyTorch 方法原理）

> 对应作业：`shanxierdan/Tutorial/Pytorch/Assignment_1.md`
>
> 说明：以下代码基于 PyTorch。涉及随机初始化时，示例使用 `torch.manual_seed(42)`，因此你的运行结果可能略有不同，但结论应一致。

```python
import torch
import torch.nn as nn
```

## 先看整体关系

本作业中的主要 PyTorch 对象可以分成四层：

```text
Tensor
  保存数据、shape、dtype 和 device
       ↓
Autograd
  记录 Tensor 运算，并自动计算梯度
       ↓
nn.Module / Loss
  定义模型计算和训练目标
       ↓
Optimizer
  使用参数梯度更新模型参数
```

完整训练流程为：

```text
X、Y 是 Tensor
        ↓
model(X) 得到预测
        ↓
criterion(预测, Y) 得到 loss
        ↓
loss.backward() 计算参数梯度
        ↓
optimizer.step() 更新参数
```

需要特别区分：

```text
backward()       只负责计算梯度
optimizer.step() 才真正修改模型参数
```

---

# 第一部分：Tensor 基础

## 作业 1：创建不同类型的 Tensor

### 参考代码

```python
import torch

x1 = torch.tensor([1, 2, 3, 4])
x2 = torch.zeros(2, 3)
x3 = torch.ones(3, 2)
x4 = torch.arange(10)
x5 = torch.randn(4, 5)

tensors = {
    "x1": x1,
    "x2": x2,
    "x3": x3,
    "x4": x4,
    "x5": x5,
}

for name, x in tensors.items():
    print(f"\n{name}:")
    print(x)
    print("shape:", x.shape)
    print("ndim:", x.ndim)
    print("dtype:", x.dtype)
    print("device:", x.device)
```

### 关键结果

```text
x1.shape = torch.Size([4])
x2.shape = torch.Size([2, 3])
x3.shape = torch.Size([3, 2])
x4.shape = torch.Size([10])
x5.shape = torch.Size([4, 5])
```

`x5` 的具体随机数每次可能不同。

### 方法原理

#### 1. Tensor 是什么

Tensor 可以理解为“带有统一数据类型和形状信息的多维数组”。一个 Tensor 除了保存数值，还保存几项重要元数据：

```text
shape   每个维度分别有多大
ndim    一共有多少个维度
dtype   每个元素使用什么数据类型
device  数据实际存放在 CPU 还是 GPU
```

例如，一个 shape 为 `[2, 3]` 的 Tensor 可以看成 2 行 3 列的数据。PyTorch 的大量运算会根据 shape 判断两个 Tensor 能否进行矩阵乘法、逐元素运算或广播。

#### 2. `torch.tensor(data)`

```python
x = torch.tensor([1, 2, 3, 4])
```

该函数根据已有的 Python 数据创建 Tensor。PyTorch 会检查输入值，并推断合适的 dtype：

- 全是整数时，通常推断为 `torch.int64`；
- 含有小数时，通常推断为默认浮点类型 `torch.float32`。

也可以显式指定：

```python
x = torch.tensor([1, 2, 3], dtype=torch.float32)
```

#### 3. `torch.zeros` 和 `torch.ones`

```python
torch.zeros(2, 3)
torch.ones(3, 2)
```

这两个函数先根据传入的尺寸分配 Tensor，再将所有位置分别填充为 0 或 1。传入的参数就是各个维度的大小。

它们经常用于：

- 初始化中间变量；
- 创建掩码；
- 创建与其他 Tensor 相同 shape 的占位数据；
- 初始化累计结果。

#### 4. `torch.arange`

```python
torch.arange(10)
```

生成一个等间隔序列。默认从 0 开始，到 10 之前结束，因此得到 0 到 9。

完整形式为：

```python
torch.arange(start, end, step)
```

其中 `end` 不包含在结果中。例如：

```python
torch.arange(2, 10, 2)
```

得到：

```text
[2, 4, 6, 8]
```

#### 5. `torch.randn`

```python
torch.randn(4, 5)
```

创建 shape 为 `[4, 5]` 的随机 Tensor，每个元素从标准正态分布中采样：

\[
x\sim\mathcal{N}(0,1)
\]

也就是均值为 0、标准差为 1。随机初始化模型参数和生成模拟数据时经常使用它。

如果希望多次运行得到相同随机结果，可以先设置：

```python
torch.manual_seed(42)
```

### 问题回答

1. `shape` 表示 Tensor 每个维度的大小；`ndim` 表示 Tensor 一共有多少个维度。
2. `torch.arange(10)` 默认得到整数 Tensor，通常是 `torch.int64`。
3. `torch.randn(4, 5)` 中，`4` 表示第 0 维大小，`5` 表示第 1 维大小，最终 shape 为 `[4, 5]`。
4. 未指定 `device` 时，Tensor 默认创建在 CPU 上。

---

## 作业 2：数据类型转换

### 参考代码

```python
import torch

x = torch.tensor([1, 2, 3, 4])

x_float32_a = x.float()
x_float32_b = x.to(torch.float32)
x_float64 = x.to(torch.float64)
x_int64 = x_float32_a.to(torch.int64)

print("原始 x:", x, x.dtype)
print("float32，写法一:", x_float32_a, x_float32_a.dtype)
print("float32，写法二:", x_float32_b, x_float32_b.dtype)
print("float64:", x_float64, x_float64.dtype)
print("重新转为 int64:", x_int64, x_int64.dtype)

print("\n转换后原始 x:")
print(x)
print(x.dtype)

print("\n是否为同一个对象:")
print("x is x_float32_a:", x is x_float32_a)
```

### 问题回答

1. `float32` 通常占 4 字节，精度和内存占用较低；`float64` 通常占 8 字节，精度更高，但需要更多内存和计算量。
2. 模型参数、乘法、梯度和损失计算通常需要连续数值，因此输入一般使用浮点类型。整数不适合表示小数梯度。
3. `.to(...)` 返回转换后的 Tensor，通常不会修改原 Tensor。因此一般写成：

```python
x = x.to(torch.float32)
```

当 dtype 或 device 确实发生变化时，会产生转换后的 Tensor；如果目标 dtype 和 device 已经相同，PyTorch 可能直接返回原 Tensor。

### 方法原理

#### 1. 为什么 Tensor 需要统一 dtype

同一个 Tensor 中的所有元素使用同一种数据类型。这样 PyTorch 才能用连续、规则的内存布局和底层数值指令高效计算。

例如：

```text
float32  每个元素通常占 4 字节
float64  每个元素通常占 8 字节
int64    每个元素通常占 8 字节
```

浮点数可以表示小数，因此适合模型预测、loss 和梯度。整数通常用于类别编号、索引和计数。

#### 2. `.float()` 的原理

```python
x.float()
```

是一个便捷写法，基本等价于：

```python
x.to(torch.float32)
```

它请求 PyTorch 将元素转换为 `float32`。

#### 3. `.to(...)` 的原理

`.to(...)` 是一个统一的转换接口，可以改变：

- dtype；
- device；
- 有时还可以同时改变二者。

例如：

```python
x = x.to(dtype=torch.float32, device="cuda")
```

它不会按普通直觉直接修改原变量绑定。它返回转换结果，因此通常需要重新赋值：

```python
x = x.to(device)
```

如果 Tensor 已经具有目标 dtype 和 device，PyTorch 可能直接复用原对象，以避免没有必要的复制。

---

# 第二部分：Shape 与 Tensor 变形

## 作业 3：理解 `reshape`

### 参考代码

```python
import torch

x = torch.arange(12)

shapes = [
    (3, 4),
    (4, 3),
    (2, 2, 3),
    (12, 1),
]

for shape in shapes:
    y = x.reshape(shape)
    print(f"\nreshape 为 {shape}:")
    print(y)
    print("shape:", y.shape)

z = x.reshape(-1, 2)
print("\nreshape(-1, 2):")
print(z)
print("shape:", z.shape)

try:
    wrong = x.reshape(5, 3)
except RuntimeError as error:
    print("\n错误信息:")
    print(error)
```

### 结果

```text
reshape(-1, 2) 的 shape 为 torch.Size([6, 2])
```

`x.reshape(5, 3)` 会报错，因为目标需要 15 个元素，而原 Tensor 只有 12 个元素。

### 问题回答

1. `[3, 4]` 一共需要 `3 × 4 = 12` 个元素，与原 Tensor 元素数相同。
2. `[5, 3]` 需要 15 个元素，与原来的 12 个不一致，因此不能 reshape。
3. `-1` 表示让 PyTorch 根据总元素数和其他维度自动推断。这里第二维为 2，所以第一维自动推断为 `12 ÷ 2 = 6`。

### 方法原理

#### 1. `reshape` 改变的是观察方式

`reshape` 不改变元素的数值和总数量，而是改变这些元素被组织成多少个维度、每个维度多大。

原数据：

```text
[0, 1, 2, ..., 11]
```

既可以按 `[3, 4]` 观察，也可以按 `[2, 2, 3]` 观察，但总元素数始终必须为 12。

#### 2. 为什么总元素数必须保持不变

shape 为：

```text
[d1, d2, ..., dn]
```

的 Tensor 一共有：

\[
d_1d_2\cdots d_n
\]

个元素。`reshape` 只是重新解释这些元素，不会凭空增加或删除数据。

#### 3. `reshape` 与内存

Tensor 的数值通常存放在一段底层内存中。`reshape` 在可能时只创建一个新的 shape 视图，共享原有数据；如果当前内存布局不允许，则可能创建连续副本。

因此教学阶段可以记住：

> `reshape` 一定保持数值和元素总数不变，但不必假设它永远不复制内存。

#### 4. `-1` 自动推断

一个 reshape 中最多只能出现一个 `-1`，因为 PyTorch 需要根据其他维度唯一推断它。例如：

```python
x.reshape(-1, 2)
```

原 Tensor 有 12 个元素，所以缺失维度只能是 6。

---

## 作业 4：理解 `squeeze` 和 `unsqueeze`

### 参考代码

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])

a = x.unsqueeze(0)
b = x.unsqueeze(1)
c = a.squeeze()

print("x:", x.shape)
print("a:", a.shape)
print("b:", b.shape)
print("c:", c.shape)
```

### 结果

```text
x: torch.Size([3])
a: torch.Size([1, 3])
b: torch.Size([3, 1])
c: torch.Size([3])
```

### 问题回答

1. `unsqueeze(0)` 在最前面增加一个维度，`[3]` 变成 `[1, 3]`。
2. `unsqueeze(1)` 在索引 1 的位置增加维度，因此 `[3]` 变成 `[3, 1]`。
3. 不指定参数的 `squeeze()` 会删除所有大小为 1 的维度。
4. 不会。`squeeze()` 只能删除大小为 1 的维度，不能删除大小大于 1 的维度。

### 方法原理

#### 1. `unsqueeze` 只增加维度，不增加数据

```python
x.unsqueeze(dim)
```

在指定位置插入一个大小为 1 的维度。例如：

```text
[3] --unsqueeze(0)--> [1, 3]
[3] --unsqueeze(1)--> [3, 1]
```

数值仍然是原来的三个元素，只是 shape 的解释发生改变。

这在添加 batch 维度时很常见。例如单个样本 shape 为 `[3]`，模型要求输入 `[batch, features]`，就可以变成 `[1, 3]`。

#### 2. `squeeze` 删除大小为 1 的维度

大小为 1 的维度不会区分多个元素，因此可以安全删除。例如：

```text
[1, 3, 1] --squeeze()--> [3]
```

如果指定：

```python
x.squeeze(0)
```

则只尝试删除第 0 维；如果第 0 维大小不为 1，shape 不会改变。

#### 3. 为什么这些方法常用于模型输入

模型经常要求固定的维度约定，例如：

```text
全连接网络：[batch, features]
图像数据：  [batch, channel, height, width]
```

`squeeze` 和 `unsqueeze` 用于补齐或去掉这些结构维度，而不改变实际数据内容。

---

# 第三部分：运算、广播与聚合函数

## 作业 5：理解 `sum` 的 `dim` 参数

### 参考代码

```python
import torch

x = torch.tensor(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]]
)

total = x.sum()
sum_dim0 = x.sum(dim=0)
sum_dim1 = x.sum(dim=1)
sum_keepdim = x.sum(dim=1, keepdim=True)

items = {
    "total": total,
    "sum_dim0": sum_dim0,
    "sum_dim1": sum_dim1,
    "sum_keepdim": sum_keepdim,
}

for name, value in items.items():
    print(name)
    print(value)
    print("shape:", value.shape)
```

### 结果

```text
total = tensor(21.)
shape = torch.Size([])

sum_dim0 = tensor([5., 7., 9.])
shape = torch.Size([3])

sum_dim1 = tensor([ 6., 15.])
shape = torch.Size([2])

sum_keepdim =
tensor([[ 6.],
        [15.]])
shape = torch.Size([2, 1])
```

### 问题回答

1. 不传 `dim` 时，`sum()` 将全部元素相加，结果是标量。
2. `dim=0` 消去第 0 维，相当于逐列求和；`dim=1` 消去第 1 维，相当于逐行求和。
3. `keepdim=True` 会保留被聚合的维度，但该维度大小变为 1。
4. 训练时通常需要一个标量目标，才能直接调用无参数的 `backward()`，并明确优化的是一个总损失。批量样本的损失通常通过 `mean` 或 `sum` 汇总。

### 方法原理

#### 1. `sum` 是聚合运算

聚合运算会把多个元素汇总为更少的元素。常见聚合包括：

```python
x.sum()
x.mean()
x.max()
x.min()
```

`x.sum()` 不指定维度时，将所有元素汇总为一个标量。

#### 2. `dim` 表示被消去的维度

假设：

```text
x.shape = [2, 3]
```

执行：

```python
x.sum(dim=0)
```

会沿第 0 维聚合，因此第 0 维被消去，结果 shape 为 `[3]`。

执行：

```python
x.sum(dim=1)
```

会沿第 1 维聚合，因此第 1 维被消去，结果 shape 为 `[2]`。

与“横着还是竖着”相比，更准确的记法是：

> `dim=k` 表示沿第 k 个轴合并，并默认从结果 shape 中删除该轴。

#### 3. `keepdim=True`

如果后续还需要按原维度进行广播，可以保留该轴：

```python
x.sum(dim=1, keepdim=True)
```

shape 从 `[2, 3]` 变为 `[2, 1]`，而不是 `[2]`。

#### 4. 为什么 loss 常被聚合成标量

一个 batch 可能包含多个样本，每个样本都有一个误差。训练需要一个统一目标，因此通常采用：

\[
L=\frac{1}{N}\sum_{i=1}^{N}L_i
\]

将所有样本误差平均为一个标量。之后 `backward()` 才能计算这个总目标对模型参数的梯度。

---

## 作业 6：广播机制

### 参考代码

```python
import torch

x = torch.tensor(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]]
)

a = torch.tensor([10.0, 20.0, 30.0])
b = torch.tensor([[10.0], [20.0]])

result_a = x + a
result_b = x + b

print("x shape:", x.shape)

print("\nx + a:")
print(result_a)
print("shape:", result_a.shape)

print("\nx + b:")
print(result_b)
print("shape:", result_b.shape)

c = torch.ones(2, 2)

try:
    result_c = x + c
except RuntimeError as error:
    print("\n无法广播时的错误:")
    print(error)
```

### 结果

```text
x + a =
tensor([[11., 22., 33.],
        [14., 25., 36.]])

x + b =
tensor([[11., 12., 13.],
        [24., 25., 26.]])
```

### 问题回答

1. `a.shape` 为 `[3]`。计算时可以理解为它被扩展为两行，每一行都是 `[10, 20, 30]`。
2. `b.shape` 为 `[2, 1]`。大小为 1 的第二维被扩展为 3，因此第一行加 10，第二行加 20。
3. 广播是 PyTorch 在不真正复制全部数据的情况下，将兼容的小 Tensor 视为扩展到更大 shape，再进行逐元素运算的机制。
4. 例如 `[2, 3]` 与 `[2, 2]` 无法广播，因为从最后一维比较时，3 和 2 既不相等，也没有一个为 1。

### 方法原理

#### 1. 广播解决什么问题

逐元素运算通常要求两个 Tensor shape 相同。但很多情况下，小 Tensor 在某些维度上可以自然重复使用，例如给每一行都加同一个向量。

广播允许 PyTorch 在逻辑上扩展较小的 Tensor，而不用手动写：

```python
a.repeat(...)
```

#### 2. 广播匹配规则

PyTorch 从最后一个维度开始向前比较。每对维度满足以下任一条件即可：

1. 两个维度大小相等；
2. 其中一个维度大小为 1；
3. 某个 Tensor 缺少该维度，可以视为大小为 1。

例如：

```text
x: [2, 3]
a:    [3]
```

从末尾比较：

```text
3 和 3 相同
a 缺少前一维，可视为 1
1 可以扩展为 2
```

所以可以广播为 `[2, 3]`。

#### 3. 广播通常不会真正复制全部数据

广播主要是一种逻辑上的 shape 扩展。底层通常通过步长等信息重复读取同一份数据，而不是先创建一个完整的大副本，因此比手动复制更加高效。

#### 4. 广播也可能隐藏错误

如果 shape 意外满足广播规则，代码可能不报错但计算含义错误。因此应养成在关键位置打印 shape 的习惯。

---

# 第四部分：Autograd 自动求导

## 作业 7：标量自动求导

### 手动计算

\[
y=x^2+3x+1
\]

\[
\frac{dy}{dx}=2x+3
\]

当 \(x=4\) 时：

\[
\frac{dy}{dx}=2\times4+3=11
\]

### 参考代码

```python
import torch

x = torch.tensor(4.0, requires_grad=True)
y = x**2 + 3*x + 1

print("x =", x)
print("y =", y)
print("x.grad before backward =", x.grad)
print("y.grad_fn =", y.grad_fn)

y.backward()

print("x.grad after backward =", x.grad)
```

### 结果

```text
y = tensor(29., grad_fn=<AddBackward0>)
x.grad before backward = None
x.grad after backward = tensor(11.)
```

### 问题回答

1. `requires_grad=True` 让 PyTorch 记录与该 Tensor 有关的可求导运算，为反向传播建立计算图。
2. `grad_fn` 记录该非叶子 Tensor 是由哪类运算得到的，以及反向传播时需要使用的函数。
3. 前向传播只建立计算图；在调用 `backward()` 前尚未真正计算梯度，因此 `x.grad` 是 `None`。
4. `x.grad` 保存的是结果对 `x` 的梯度，即 \(\frac{dy}{dx}\)。


### 方法原理

#### 1. 动态计算图

当一个 `requires_grad=True` 的 Tensor 参与运算时，PyTorch 会在前向计算的同时记录运算关系，形成计算图。

例如：

```python
y = x**2 + 3*x + 1
```

可以抽象成：

```text
x
├── 平方 ──┐
├── 乘以3 ─┼── 加法 ── y
└── 常数1 ─┘
```

这个图不是提前固定生成的，而是在 Python 代码实际运行时动态建立的。

#### 2. 叶子节点与非叶子节点

直接创建并要求梯度的 `x` 是叶子节点：

```python
x = torch.tensor(4.0, requires_grad=True)
```

由运算得到的 `y` 是非叶子节点。模型参数通常也是叶子节点，训练所得梯度最终保存在这些参数的 `.grad` 中。

#### 3. `requires_grad=True`

该参数表示：

> 后续需要计算最终结果对这个 Tensor 的梯度。

它并不意味着创建 Tensor 后立刻拥有梯度。只有调用 `backward()` 后，PyTorch 才真正执行反向计算。

#### 4. `grad_fn`

非叶子 Tensor 的 `grad_fn` 指向产生它的反向运算。例如结果最后由加法得到时，通常会显示 `AddBackward0`。

`grad_fn` 不是梯度数值，而是计算图中的反向传播函数节点。

#### 5. `backward()` 与链式法则

`y.backward()` 从 `y` 出发，沿计算图反向应用链式法则。例如：

\[
y=f(g(x))
\]

则：

\[
\frac{dy}{dx}
=
\frac{dy}{dg}
\frac{dg}{dx}
\]

神经网络虽然可能包含很多层，但反向传播的数学基础仍然是反复使用链式法则。

#### 6. 为什么标量可以直接 backward

标量 `y` 对自身的梯度默认是：

\[
\frac{dy}{dy}=1
\]

所以 `y.backward()` 可以把 1 作为起始上游梯度。向量输出则需要额外说明各个输出分量如何组合。

---

## 作业 8：向量输出为什么需要 `sum`

### 直接对向量调用 `backward`

```python
import torch

weights = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True
)

output = weights * 2

try:
    output.backward()
except RuntimeError as error:
    print(error)
```

典型报错：

```text
grad can be implicitly created only for scalar outputs
```

### 方法一：使用 `sum`

```python
weights = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True
)

output = (weights * 2).sum()
output.backward()

print(weights.grad)
```

结果：

```text
tensor([2., 2., 2.])
```

### 方法二：显式传入上游梯度

```python
weights = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True
)

output = weights * 2
output.backward(torch.tensor([1.0, 1.0, 1.0]))

print(weights.grad)
```

结果仍为：

```text
tensor([2., 2., 2.])
```

### 修改上游梯度

```python
weights = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True
)

output = weights * 2
output.backward(torch.tensor([1.0, 2.0, 3.0]))

print(weights.grad)
```

结果：

```text
tensor([2., 4., 6.])
```

### 问题回答

1. 向量输出包含多个分量，其导数通常是 Jacobian。PyTorch 不知道应如何组合多个输出，因此不能自动假定上游梯度。
2. `sum()` 将向量合成为一个标量，标量可以默认从上游梯度 1 开始反向传播。
3. 对 `output.sum()` 求梯度，等价于给 `output` 的每个分量传入上游梯度 1。
4. 每个 `output_i = 2 weight_i`，局部导数均为 2。上游梯度为 `[1, 2, 3]`，根据链式法则得到 `[2×1, 2×2, 2×3] = [2, 4, 6]`。

### 方法原理

#### 1. 向量函数的导数不是单个数

若：

\[
\mathbf{y}=f(\mathbf{x})
\]

输出和输入都是向量，则完整导数是 Jacobian 矩阵：

\[
J_{ij}=\frac{\partial y_i}{\partial x_j}
\]

直接构造完整 Jacobian 可能非常大，神经网络训练通常并不需要它。

#### 2. PyTorch 实际计算向量-Jacobian 乘积

当调用：

```python
output.backward(v)
```

PyTorch 实际计算的是：

\[
\mathbf{v}^{T}J
\]

其中传入的 `v` 就是上游梯度。

这也是为什么 `v` 的 shape 必须与 `output` 相同。

#### 3. `sum()` 等价于传入全 1

令：

\[
z=\sum_i y_i
\]

则：

\[
\frac{\partial z}{\partial y_i}=1
\]

因此：

```python
output.sum().backward()
```

等价于：

```python
output.backward(torch.ones_like(output))
```

#### 4. 上游梯度的直观含义

上游梯度描述“最终目标对当前每个输出分量有多敏感”。反向传播再把这种敏感度乘以当前操作的局部导数，继续传给前面的变量。

---

## 作业 9：观察梯度累积

### 参考代码

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2
y.backward()
print("第一次梯度：", x.grad)

y = x**2
y.backward()
print("第二次梯度：", x.grad)

x.grad.zero_()

y = x**2
y.backward()
print("清零后的梯度：", x.grad)
```

### 结果

```text
第一次梯度： tensor(4.)
第二次梯度： tensor(8.)
清零后的梯度： tensor(4.)
```

### 问题回答

1. \(y=x^2\) 在 \(x=2\) 处的梯度为 4。PyTorch 默认将新梯度加到已有 `.grad` 上，因此第二次是 `4+4=8`。
2. 梯度累积可以支持一个参数由多个计算分支贡献梯度，也可用于梯度累积训练。
3. `x.grad.zero_()` 将已有梯度原地清零。
4. 标准训练流程中使用 `optimizer.zero_grad()` 清空优化器管理参数的梯度。

### 方法原理

#### 1. 为什么梯度默认累积

PyTorch 对 `.grad` 使用加法更新：

```text
新的 grad = 原有 grad + 本次 backward 得到的 grad
```

这样设计是因为同一个参数可能通过多条计算路径影响最终 loss，各条路径的梯度本来就应该相加。

它也支持“梯度累积”：连续处理多个小 batch，累积梯度后再统一更新一次参数。

#### 2. `zero_()` 中下划线的含义

PyTorch 中很多以下划线结尾的方法表示原地操作：

```python
x.grad.zero_()
```

它直接把现有梯度 Tensor 的元素改成 0，而不是返回一个新的全零 Tensor。

常见原地方法还有：

```python
add_()
mul_()
fill_()
```

使用原地操作时要注意，它会修改原对象；在复杂计算图中不恰当的原地修改可能影响 Autograd。

#### 3. `optimizer.zero_grad()`

optimizer 持有需要更新的模型参数。调用：

```python
optimizer.zero_grad()
```

会重置这些参数的梯度，避免当前 batch 的梯度与上一轮意外相加。

在当前 PyTorch 版本中，默认参数通常是：

```python
optimizer.zero_grad(set_to_none=True)
```

这意味着梯度一般被设为 `None`，而不一定被写成一个全零 Tensor。这样通常更节省内存，也可能略微提高性能。下一次 `backward()` 时，新的梯度会重新创建。

如果显式写：

```python
optimizer.zero_grad(set_to_none=False)
```

则会把已有梯度 Tensor 的值清零。

---

# 第五部分：完整训练流程

## 作业 10：修改线性关系

目标关系：

\[
Y=3X+1
\]

因此：

```text
X = 1, 2, 3, 4
Y = 4, 7, 10, 13
```

### 参考代码

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

X = torch.tensor(
    [[1.0],
     [2.0],
     [3.0],
     [4.0]]
)

Y = torch.tensor(
    [[4.0],
     [7.0],
     [10.0],
     [13.0]]
)

X_test = torch.tensor([[5.0]])

model = nn.Linear(in_features=1, out_features=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

for epoch in range(epochs):
    y_pred = model(X)
    loss_value = criterion(y_pred, Y)

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

print("weight:", model.weight.item())
print("bias:", model.bias.item())
print("f(5):", model(X_test).item())
print("final loss:", loss_value.item())
```

### 方法原理

#### 1. `nn.Linear`

```python
model = nn.Linear(in_features=1, out_features=1)
```

表示一个线性层。它实现：

\[
Y=XW^T+b
\]

在当前例子中，每个样本只有一个输入特征和一个输出，因此公式退化为：

\[
\hat y=wx+b
\]

`nn.Linear` 内部自动创建两个可训练参数：

```text
weight  shape 为 [out_features, in_features]
bias    shape 为 [out_features]
```

所以这里：

```text
weight.shape = [1, 1]
bias.shape   = [1]
```

调用：

```python
model(X)
```

实际上会执行该线性变换。

#### 2. 参数为什么会自动训练

`nn.Linear` 创建的 weight 和 bias 都是 `nn.Parameter`，默认设置 `requires_grad=True`。因此前向传播后调用 `loss.backward()`，梯度会被写入：

```python
model.weight.grad
model.bias.grad
```

#### 3. `model.parameters()`

该方法返回模型中所有注册的可训练参数。把它传给 optimizer：

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

相当于告诉优化器：

> 这些参数需要根据梯度进行更新。

#### 4. `nn.MSELoss`

均方误差的基本形式是：

\[
L=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
\]

其中：

- \(\hat y_i\) 是模型预测；
- \(y_i\) 是真实值；
- \(N\) 是参与平均的元素数量。

平方使正负误差都变为非负，并且较大的误差会受到更重惩罚。

默认情况下，`nn.MSELoss()` 使用 `reduction="mean"`，返回所有元素平方误差的平均值，因此结果是标量。

#### 5. `torch.optim.SGD`

SGD 根据当前梯度更新参数。最基本的更新公式是：

\[
\theta_{t+1}
=
\theta_t-\eta\nabla_\theta L
\]

其中：

- \(\theta\) 是模型参数；
- \(\eta\) 是学习率；
- \(\nabla_\theta L\) 是 loss 对参数的梯度。

在代码中：

```python
loss.backward()
```

负责计算梯度，而：

```python
optimizer.step()
```

负责应用更新公式。这两个步骤职责不同。

#### 6. `torch.manual_seed`

线性层的初始参数是随机生成的。设置：

```python
torch.manual_seed(42)
```

会固定 PyTorch 随机数生成器的起始状态，使多次实验更容易复现和公平比较。

### 一组参考结果

使用 `torch.manual_seed(42)` 时，结果约为：

```text
weight: 2.9910
bias: 1.0264
f(5): 15.9815
final loss: 0.000117
```

理论值为：

```text
weight = 3
bias = 1
f(5) = 16
```

### 问题回答

1. weight 应接近 3。
2. bias 应接近 1。
3. 输入 5 时，理论输出为 `3×5+1=16`。
4. `in_features=1` 表示每个样本有 1 个输入特征。
5. `out_features=1` 表示每个样本输出 1 个数。
6. `model.parameters()` 包含该线性层需要学习的 `weight` 和 `bias`。

---

## 作业 11：比较不同学习率

### 参考代码

```python
import torch
import torch.nn as nn

X = torch.tensor(
    [[1.0],
     [2.0],
     [3.0],
     [4.0]]
)

Y = 3 * X + 1

learning_rates = [0.0001, 0.01, 0.1, 1.0]
epochs = 1000

for lr in learning_rates:
    # 保证每种学习率都从相同初始化开始
    torch.manual_seed(42)

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    converged = True

    for epoch in range(epochs):
        y_pred = model(X)
        loss = criterion(y_pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not torch.isfinite(loss):
            converged = False
            break

    print(
        f"lr={lr:<7} "
        f"loss={loss.item():.6g} "
        f"weight={model.weight.item():.6g} "
        f"bias={model.bias.item():.6g} "
        f"finite={converged}"
    )
```

### 参考结果

固定随机种子为 42、训练 1000 轮时：

| 学习率 | 最终 loss | 最终 weight | 最终 bias | 是否正常收敛 |
|---:|---:|---:|---:|---|
| 0.0001 | 1.43951 | 2.43896 | 1.38210 | 下降但过慢 |
| 0.01 | 0.000116985 | 2.99103 | 1.02639 | 是 |
| 0.1 | 约 \(7.96\times10^{-13}\) | 3.00000 | 1.00000 | 是 |
| 1.0 | `inf` | 数值爆炸 | 数值爆炸 | 否 |

数值依赖初始化和训练轮数，表格主要用于观察趋势。

### 方法原理

#### 学习率控制更新步长

SGD 的更新为：

\[
\theta_{t+1}=\theta_t-\eta g_t
\]

其中 \(\eta\) 就是学习率。

- 学习率太小：每一步移动很短，训练稳定但速度慢；
- 学习率适中：能够较快接近最低点；
- 学习率太大：一步可能越过最低点，在两侧震荡；
- 学习率极大：参数和 loss 可能迅速爆炸为 `inf` 或 `nan`。

不同学习率的比较必须使用相同数据、相同训练轮数和相同初始参数，否则不能确定差异究竟来自学习率还是初始化。

### 问题回答

1. 学习率过小时，每次参数变化很小，loss 下降缓慢，在固定训练轮数内可能尚未接近最优值。
2. 学习率过大时，参数可能越过最优点来回震荡，甚至迅速发散，出现极大值、`inf` 或 `nan`。
3. 如果不重新创建模型，后一个学习率会接着前一个模型的训练结果继续训练，比较不公平。每次实验应使用相同的初始参数。
4. 不能。还应观察整个 loss 曲线，因为最终一次 loss 可能偶然较低，但中间存在明显震荡；也可能最终仍在下降，只是学习率过小。

---

## 作业 12：补全训练循环

### 答案

```python
for epoch in range(epochs):
    y_pred = model(X)

    loss_value = criterion(y_pred, Y)

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
```

### 方法原理

这五行构成最基本的训练闭环：

```text
前向传播
→ 计算损失
→ 清空旧梯度
→ 反向传播
→ 更新参数
```

前向传播使用“当前参数”计算预测；反向传播只计算梯度，并不会自动修改参数；真正的参数修改发生在 `optimizer.step()`。

顺序不能随意调整：

- 必须先得到预测，才能计算 loss；
- 必须有 loss，才能反向传播；
- 必须先得到梯度，optimizer 才知道如何更新；
- 普通训练中应在新一轮反向传播前清空旧梯度。

### 逐行解释

```python
y_pred = model(X)
```

使用当前模型参数进行前向传播，得到预测值。

```python
loss_value = criterion(y_pred, Y)
```

比较预测值和真实值，得到损失。

```python
optimizer.zero_grad()
```

清空上一次迭代残留的梯度。

```python
loss_value.backward()
```

从 loss 开始反向传播，计算 loss 对每个可训练参数的梯度。

```python
optimizer.step()
```

根据优化算法、学习率和参数梯度更新模型参数。

---

# 第六部分：CPU 与 GPU

## 作业 13：把训练代码移动到设备

### 参考代码

```python
import torch
import torch.nn as nn

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

X = torch.tensor(
    [[1.0],
     [2.0],
     [3.0],
     [4.0]]
).to(device)

Y = torch.tensor(
    [[4.0],
     [7.0],
     [10.0],
     [13.0]]
).to(device)

X_test = torch.tensor([[5.0]]).to(device)

model = nn.Linear(1, 1).to(device)
criterion = nn.MSELoss()

# 建议在模型移动到设备之后创建 optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1000

print("device =", device)
print("X device =", X.device)
print("Y device =", Y.device)
print("model device =", model.weight.device)

for epoch in range(epochs):
    y_pred = model(X)
    loss_value = criterion(y_pred, Y)

    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

print("weight =", model.weight.item())
print("bias =", model.bias.item())
print("f(5) =", model(X_test).item())
```

### 方法原理

#### 1. CPU 与 GPU 使用不同内存

CPU Tensor 通常存放在系统内存中，CUDA Tensor 存放在 GPU 显存中。矩阵运算需要直接访问参与运算的数据，因此输入和参数必须位于同一个设备。

#### 2. `Tensor.to(device)`

```python
X = X.to(device)
```

会将 Tensor 转换或复制到目标设备，并返回结果。原变量需要重新接收返回值。

#### 3. `model.to(device)`

```python
model = model.to(device)
```

会递归移动模型中注册的：

- parameters；
- buffers。

例如 `nn.Linear` 的 weight 和 bias 都会被移动。

#### 4. criterion 为什么通常不必移动

当前的 `nn.MSELoss()` 只执行算子，本身没有参数或持久 buffer。因此输入在哪个设备，它的计算就在哪个设备执行。

但不能把这一点泛化为所有 loss 模块都一定不需要移动；若某个模块内部保存了 Tensor 参数或 buffer，也应该移动。

#### 5. optimizer 为什么不用 `.to(device)`

optimizer 保存的是对模型参数的引用。模型参数位于 GPU 时，`step()` 就会更新 GPU 参数。

更稳妥的顺序是：

```python
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=...)
```

某些含状态的优化器（如 Adam）会在训练过程中创建动量等状态 Tensor，这些状态通常会跟随对应参数所在设备。

#### 6. 为什么小任务 GPU 可能更慢

GPU 每次计算需要启动 CUDA kernel，并可能涉及 CPU 与 GPU 之间的数据传输。对于只有几个数的任务，启动和调度开销可能远大于实际乘法计算。

GPU 的优势主要出现在大规模、规则、可并行的矩阵与张量运算中。

### 问题回答

1. 同一个运算中的 Tensor 和模型参数需要位于兼容设备上。CPU 内存与 GPU 显存是不同的地址空间，PyTorch 不会自动跨设备完成普通算子。
2. `.to(device)` 返回目标设备上的 Tensor，通常需要重新赋值：

```python
X = X.to(device)
```

3. 当前的 `nn.MSELoss()` 没有可训练参数或需要移动的 buffer，因此不必须调用 `.to(device)`。对于带参数或 buffer 的损失模块，则应移动。
4. optimizer 本身通常不调用 `.to(device)`。它保存对模型参数的引用，参数位于哪个设备，更新就在相应设备进行。建议先 `model.to(device)`，再创建 optimizer。
5. 该任务只有 4 个样本、1 个 weight 和 1 个 bias，计算量极小。GPU 的内核启动和设备调度开销可能大于实际计算时间，因此不一定更快。

---

## 作业 14：故意制造设备错误

### 错误代码

```python
import torch
import torch.nn as nn

model = nn.Linear(1, 1).to("cuda")
X = torch.tensor([[1.0], [2.0]])

model(X)
```

典型错误会说明参与运算的 Tensor 位于不同设备，例如模型参数在 `cuda:0`，输入在 `cpu`。

### 修复方法一：把输入移动到 GPU

```python
X = X.to("cuda")
output = model(X)
print(output)
```

### 修复方法二：把模型放回 CPU

```python
model = model.to("cpu")
output = model(X)
print(output)
```

### 原因

线性层需要计算：

\[
Y=XW^T+b
\]

输入 `X` 与模型参数 `W`、`b` 必须位于同一设备，才能执行矩阵运算和加法。

### 方法原理

PyTorch 的每个底层算子都有明确的设备实现。例如矩阵乘法会选择 CPU kernel 或 CUDA kernel。一次算子不能让一部分输入从 CPU 内存读取、另一部分输入从 GPU 显存读取。

PyTorch 也不会默默自动搬运数据，因为自动传输可能：

- 带来很大的性能开销；
- 造成难以察觉的内存复制；
- 使程序行为不清晰。

因此设备管理由程序员显式完成。遇到 device mismatch 报错时，应逐一检查：

```python
print(X.device)
print(Y.device)
print(next(model.parameters()).device)
```

---

# 综合思考题

## 1. Tensor 与普通 Python 列表的主要区别是什么？

Tensor 是结构化的多维数值数组，具有固定 dtype、shape 和 device，支持高效向量化运算、GPU 计算和自动求导。Python 列表可混合存放不同类型对象，但不直接支持这些数值计算能力。

## 2. Tensor 的 `shape` 为什么非常重要？

模型中的每一层都规定了输入和输出维度。shape 决定矩阵乘法能否执行、batch 有多少样本、每个样本有多少特征。shape 不匹配是训练代码中最常见的错误之一。

## 3. 前向传播和反向传播分别完成什么工作？

前向传播使用当前参数从输入计算预测值和 loss；反向传播沿计算图从 loss 向前计算 loss 对各参数的梯度。

## 4. `loss.backward()` 计算的是谁对谁的梯度？

计算 loss 对所有参与计算且 `requires_grad=True` 的叶子参数的梯度，例如：

\[
\frac{\partial loss}{\partial weight},
\qquad
\frac{\partial loss}{\partial bias}
\]

结果保存在各参数的 `.grad` 中。

## 5. `optimizer.step()` 更新的是什么？

它更新传给 optimizer 的模型参数，例如 weight 和 bias。更新方式由优化器类型决定，SGD 的基本形式是：

\[
\theta \leftarrow \theta-\eta\nabla_\theta L
\]

## 6. 为什么通常要先执行 `optimizer.zero_grad()`？

PyTorch 默认累积梯度。如果不清空，当前 batch 的梯度会与之前 batch 的梯度相加，导致参数更新不符合普通训练流程。

## 7. `MSELoss` 的两个输入分别是什么？

第一个输入通常是模型预测值 `y_pred`，第二个输入是真实目标 `Y`。二者应具有相同或可兼容的 shape。

## 8. 为什么测试数据不能用于更新模型参数？

测试集用于估计模型对未见数据的泛化能力。如果利用测试集更新参数，就相当于提前泄露答案，测试结果不再客观。

## 9. GPU 适合什么类型的计算？

GPU 适合大规模、规则且可并行的数值计算，尤其是大矩阵乘法、卷积和批量 Tensor 运算。对于极小任务，GPU 的额外调度开销可能使其更慢。

## 10. 模型、输入和标签应满足什么设备要求？

参与同一计算流程的模型参数、输入和标签应位于同一设备，例如全部在 CPU，或者全部在同一块 GPU（如 `cuda:0`）。

---

# 最核心的训练流程总结

```python
y_pred = model(X)
loss = criterion(y_pred, Y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

它对应：

```text
用当前参数得到预测
→ 计算预测误差
→ 清除旧梯度
→ 计算新梯度
→ 更新模型参数
```
