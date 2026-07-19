# PyTorch 基础作业：Tensor、Autograd 与训练流程

## 作业要求

每道题需要提交：

1. 完整可运行的 Python 代码；
2. 关键运行结果截图或终端输出；
3. 遇到报错时，记录报错内容及解决方法。

---

# 第一部分：Tensor 基础

## 作业 1：创建不同类型的 Tensor

分别使用以下函数创建 Tensor：

```python
torch.tensor()
torch.zeros()
torch.ones()
torch.arange()
torch.randn()
```

完成以下要求：

1. 创建一个值为 `[1, 2, 3, 4]` 的一维 Tensor；
2. 创建一个形状为 `[2, 3]` 的全零 Tensor；
3. 创建一个形状为 `[3, 2]` 的全一 Tensor；
4. 创建从 0 到 9 的整数 Tensor；
5. 创建一个形状为 `[4, 5]` 的随机 Tensor。

对每个 Tensor 输出：

```python
print(x)
print(x.shape)
print(x.ndim)
print(x.dtype)
print(x.device)
```

回答：

1. `shape` 和 `ndim` 分别表示什么？
2. `torch.arange(10)` 默认是什么数据类型？
3. `torch.randn(4, 5)` 中的两个参数分别表示什么？
4. Tensor 默认创建在哪个设备上？

---

## 作业 2：数据类型转换

创建：

```python
x = torch.tensor([1, 2, 3, 4])
```

依次将其转换为：

```python
torch.float32
torch.float64
torch.int64
```

至少使用两种写法：

```python
x.float()
x.to(torch.float32)
```

回答：

1. `float32` 和 `float64` 有什么区别？
2. 输入数据通常为什么要使用浮点类型？
3. `x.to(...)` 是直接修改原来的 `x`，还是返回一个新的 Tensor？请用代码验证。

---

# 第二部分：Shape 与 Tensor 变形

## 作业 3：理解 `reshape`

创建：

```python
x = torch.arange(12)
```

将它分别变成以下形状：

```text
[3, 4]
[4, 3]
[2, 2, 3]
[12, 1]
```

输出每次变形后的 Tensor 和 shape。

然后尝试：

```python
x.reshape(5, 3)
```

记录报错并解释为什么报错。

回答：

1. 为什么包含 12 个元素的 Tensor 可以变成 `[3, 4]`？
2. 为什么不能变成 `[5, 3]`？
3. `reshape(-1, 2)` 中的 `-1` 表示什么？

---

## 作业 4：理解 `squeeze` 和 `unsqueeze`

创建：

```python
x = torch.tensor([1.0, 2.0, 3.0])
```

依次执行：

```python
a = x.unsqueeze(0)
b = x.unsqueeze(1)
c = a.squeeze()
```

分别输出：

```python
print(x.shape)
print(a.shape)
print(b.shape)
print(c.shape)
```

回答：

1. `unsqueeze(0)` 在哪个位置增加了一个维度？
2. `unsqueeze(1)` 的结果为什么是 `[3, 1]`？
3. `squeeze()` 会删除什么样的维度？
4. 如果某个维度大小不是 1，`squeeze()` 会删除它吗？

---

# 第三部分：运算、广播与聚合函数

## 作业 5：理解 `sum` 的 `dim` 参数

创建：

```python
x = torch.tensor(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]]
)
```

分别执行：

```python
total = x.sum()
sum_dim0 = x.sum(dim=0)
sum_dim1 = x.sum(dim=1)
sum_keepdim = x.sum(dim=1, keepdim=True)
```

输出所有结果及其 shape。

在运行代码前，先手动预测每个结果。

回答：

1. 不传 `dim` 时，`sum()` 做了什么？
2. `dim=0` 和 `dim=1` 分别沿哪个方向求和？
3. `keepdim=True` 有什么作用？
4. 为什么训练中的 loss 通常需要是一个标量？

---

## 作业 6：广播机制

创建：

```python
x = torch.tensor(
    [[1.0, 2.0, 3.0],
     [4.0, 5.0, 6.0]]
)

a = torch.tensor([10.0, 20.0, 30.0])
b = torch.tensor([[10.0], [20.0]])
```

分别计算：

```python
x + a
x + b
```

输出结果和 shape。

回答：

1. `a` 的 shape 是多少？它是怎样与 `x` 相加的？
2. `b` 的 shape 是多少？它是怎样与 `x` 相加的？
3. 什么是广播机制？
4. 尝试创建一个无法与 `x` 广播相加的 Tensor，并记录报错。

---

# 第四部分：Autograd 自动求导

## 作业 7：标量自动求导

创建：

```python
x = torch.tensor(4.0, requires_grad=True)
```

定义：

```python
y = x**2 + 3*x + 1
```

完成：

```python
print("x =", x)
print("y =", y)
print("x.grad before backward =", x.grad)
print("y.grad_fn =", y.grad_fn)

y.backward()

print("x.grad after backward =", x.grad)
```

在运行前，手动计算：

[
\frac{dy}{dx}
]

并计算当 (x=4) 时梯度是多少。

回答：

1. `requires_grad=True` 的作用是什么？
2. `grad_fn` 表示什么？
3. 为什么执行 `backward()` 前，`x.grad` 是 `None`？
4. `x.grad` 保存的是 (x) 对 (y) 的导数，还是 (y) 对 (x) 的导数？

---

## 作业 8：向量输出为什么需要 `sum`

创建：

```python
weights = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True
)

output = weights * 2
```

先尝试：

```python
output.backward()
```

记录报错。

然后使用方法一：

```python
output = (weights * 2).sum()
output.backward()
```

输出：

```python
print(weights.grad)
```

再重新创建 `weights`，使用方法二：

```python
output = weights * 2
output.backward(torch.tensor([1.0, 1.0, 1.0]))
```

回答：

1. 为什么向量输出不能直接调用无参数的 `backward()`？
2. `sum()` 为什么能解决这个问题？
3. 上面两种写法为什么会得到相同的梯度？
4. 把传入的梯度改成：

```python
torch.tensor([1.0, 2.0, 3.0])
```

预测并解释最终的 `weights.grad`。

---

## 作业 9：观察梯度累积

运行：

```python
x = torch.tensor(2.0, requires_grad=True)

y = x**2
y.backward()

print("第一次梯度：", x.grad)
```

然后重新计算：

```python
y = x**2
y.backward()

print("第二次梯度：", x.grad)
```

最后执行：

```python
x.grad.zero_()

y = x**2
y.backward()

print("清零后的梯度：", x.grad)
```

回答：

1. 为什么第二次梯度不是第一次的结果？
2. PyTorch 为什么默认累积梯度？
3. `x.grad.zero_()` 做了什么？
4. 在训练中，哪个函数负责清空梯度？

---

# 第五部分：完整训练流程

## 作业 10：修改线性关系

已知原始训练数据表示：

[
Y=2X
]

现在修改为：

[
Y=3X+1
]

自己构造新的训练数据，例如：

```python
X = torch.tensor(
    [[1.0],
     [2.0],
     [3.0],
     [4.0]]
)
```

根据 (Y=3X+1) 写出对应的 `Y`。

使用：

```python
model = nn.Linear(in_features=1, out_features=1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

训练模型，并输出最终：

```python
model.weight
model.bias
model(torch.tensor([[5.0]]))
```

回答：

1. 理论上最终的 weight 应接近多少？
2. 理论上最终的 bias 应接近多少？
3. 当输入为 5 时，理论输出是多少？
4. `in_features=1` 表示什么？
5. `out_features=1` 表示什么？
6. `model.parameters()` 中包含哪些参数？

---

## 作业 11：比较不同学习率

使用相同的数据和相同模型，分别设置：

```python
lr = 0.0001
lr = 0.01
lr = 0.1
lr = 1.0
```

每种学习率都重新创建模型并训练，记录：

|    学习率 | 最终 loss | 最终 weight | 最终 bias | 是否正常收敛 |
| -----: | ------: | --------: | ------: | ------ |
| 0.0001 |         |           |         |        |
|   0.01 |         |           |         |        |
|    0.1 |         |           |         |        |
|    1.0 |         |           |         |        |

回答：

1. 学习率过小时会出现什么现象？
2. 学习率过大时会出现什么现象？
3. 为什么每次实验都应该重新创建模型？
4. 是否只根据最后一次 loss 就能判断训练过程是否稳定？

---

## 作业 12：补全训练循环

补全以下代码：

```python
for epoch in range(epochs):
    y_pred = __________

    loss_value = __________

    __________
    __________
    __________
```

要求填入：

* 前向传播；
* 计算损失；
* 清空梯度；
* 反向传播；
* 更新参数。

然后逐行解释每一步在做什么。

---

# 第六部分：CPU 与 GPU

## 作业 13：把训练代码移动到设备

在原来的线性回归代码中加入：

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

将以下对象移动到 `device`：

```text
X
Y
X_test
model
```

输出：

```python
print("device =", device)
print("X device =", X.device)
print("Y device =", Y.device)
print("model device =", model.weight.device)
```

回答：

1. 为什么数据和模型必须位于同一个设备？
2. `.to(device)` 是否会自动修改原来的 Tensor？
3. `criterion` 是否必须调用 `.to(device)`？
4. `optimizer` 是否必须调用 `.to(device)`？
5. 为什么这个极小的线性回归任务使用 GPU 不一定比 CPU 更快？

---

## 作业 14：故意制造设备错误

仅在有 CUDA GPU 的环境中完成。

让模型位于 GPU：

```python
model = nn.Linear(1, 1).to("cuda")
```

让数据仍然位于 CPU：

```python
X = torch.tensor([[1.0], [2.0]])
```

然后运行：

```python
model(X)
```

记录报错。

修复代码，并解释错误产生的原因。

---

# 综合思考题

1. Tensor 与普通 Python 列表的主要区别是什么？
2. Tensor 的 `shape` 为什么在深度学习中非常重要？
3. 前向传播和反向传播分别完成什么工作？
4. `loss.backward()` 计算的是谁对谁的梯度？
5. `optimizer.step()` 更新的是什么？
6. 为什么训练循环里通常要先执行 `optimizer.zero_grad()`？
7. `MSELoss` 的两个输入分别是什么？
8. 为什么测试数据不能用于更新模型参数？
9. GPU 适合什么类型的计算？
10. 在一个训练程序中，模型、输入和标签应该满足什么设备要求？

---

