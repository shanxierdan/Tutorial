import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(10086)


# 检查torch和CUDA的版本以及CUDA是否可用
# CUDA是用GPU做深度学习所必须的
# CUDA是NVIDIA推出的并行计算平台和编程模型，它允许开发者使用GPU进行通用计算。PyTorch可以利用CUDA加速深度学习模型的训练和推理过程。
# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())


#  什么是张量？张量是PyTorch中最基本的数据结构，
#它是一个多维数组，可以在CPU或GPU上进行高效的计算。张量可以表示标量、向量、矩阵以及更高维的数据。

# scalar = torch.tensor(3.0)
# vector = torch.tensor([1.0, 2.0, 3.0])
# matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# images = torch.zeros(4, 1, 28, 28)

# print("scalar:", scalar, "shape =", scalar.shape)
# print("vector:", vector, "shape =", vector.shape)
# print("matrix:\n", matrix, "shape =", matrix.shape)
# print("images shape:", images.shape)
# print(images)
# print(vector)
# print(matrix)

# 张量的计算，索引
# x = torch.tensor(
#     [[1, 2, 3],
#      [4, 5, 6]],
#     dtype=torch.float32,
# )

# print("x:\n", x)
# print("shape:", x.shape)
# print("dtype:", x.dtype)

# print("第一行:", x[0])
# print("第二列:", x[:, 1])
# print("单个元素:", x[1, 2].item())

# y = x + 10
# z = x * 2
# flat = x.reshape(-1)

# print("x + 10:\n", y)
# print("x * 2:\n", z)
# print("flatten:", flat, "shape =", flat.shape)


# 张量和NumPy数组之间的转换
# tensor_a = torch.ones(3)
# numpy_b = tensor_a.numpy()

# print("before change")
# print(tensor_a)
# print(numpy_b)

# tensor_a.add_(1)

# print("after tensor_a.add_(1)")
# print(tensor_a)
# print(numpy_b)

# numpy_c = np.array([10, 20, 30], dtype=np.float32)
# tensor_d = torch.from_numpy(numpy_c)

# print("numpy_c:", numpy_c)
# print("tensor_d:", tensor_d)


#张量的核心功能：自动求导
#计算过程的储存格式：计算图
        #      x
        #    /   \
        # 平方    乘以2
        #  │       │
        # x²      2x
        #    \   /
        #     加法
        #      │
        #      y
# x = torch.tensor(3.0, requires_grad=True)

# y = x**2 + 2*x

# print("x:", x)
# print("y:", y)
# print("x.grad before backward:", x.grad)
# print("y.grad_fn:", y.grad_fn)

# y.backward()

# print("x.grad after backward:", x.grad)


#更复杂一点
# a = torch.tensor(2.0, requires_grad=True)
# b = 3 * a
# c = b**2

# print("a:", a)
# print("b:", b)
# print("c:", c)
# print("b.grad_fn:", b.grad_fn)
# print("c.grad_fn:", c.grad_fn)

# c.backward()

# print("dc/da:", a.grad)


#梯度的累加以及归零
# weights = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)

# for step in range(3):
#     output = (weights * 2).sum()
#     output.backward()

#     print(f"step {step + 1}, gradient =", weights.grad)

# weights.grad.zero_()
# print("after zero_():", weights.grad)


# 来个模型试试水
# X_np = np.array([1, 2, 3, 4], dtype=np.float32)
# Y_np = np.array([2, 4, 6, 8], dtype=np.float32)

# w_np = 0.0

# def forward_numpy(x):
#     return w_np * x

# def mse_numpy(y_true, y_pred):
#     return ((y_pred - y_true) ** 2).mean()

# def gradient_numpy(x, y_true, y_pred):
#     return np.mean(2 * x * (y_pred - y_true))

# learning_rate = 0.01
# epochs = 20

# print(f"before training: f(5) = {forward_numpy(5):.3f}")

# for epoch in range(epochs):
#     y_pred = forward_numpy(X_np)
#     loss_value = mse_numpy(Y_np, y_pred)
#     dw = gradient_numpy(X_np, Y_np, y_pred)

#     w_np -= learning_rate * dw

#     if epoch % 2 == 0:
#         print(
#             f"epoch {epoch + 1:02d}: "
#             f"w = {w_np:.3f}, loss = {loss_value:.6f}"
#         )

# print(f"after training: f(5) = {forward_numpy(5):.3f}")


#使用PyTorch实现线性回归，用autograd自动求导代替之前的手写梯度
# X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# def forward_autograd(x):
#     return w * x

# def mse(y_true, y_pred):
#     return ((y_pred - y_true) ** 2).mean()

# learning_rate = 0.01
# epochs = 100

# print(f"before training: f(5) = {forward_autograd(5).item():.3f}")

# for epoch in range(epochs):
#     y_pred = forward_autograd(X)
#     loss_value = mse(Y, y_pred)

#     loss_value.backward()

#     with torch.no_grad():
#         w -= learning_rate * w.grad

#     w.grad.zero_()

#     if epoch % 10 == 0:
#         print(
#             f"epoch {epoch + 1:03d}: "
#             f"w = {w.item():.3f}, loss = {loss_value.item():.6f}"
#         )

# print(f"after training: f(5) = {forward_autograd(5).item():.3f}")


# 还能替换更多！
# - `nn.MSELoss()` 替代手写 MSE
# - `torch.optim.SGD` 替代手写参数更新
# - `optimizer.zero_grad()` 替代 `w.grad.zero_()`
# X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# def forward_with_optimizer(x):
#     return w * x

# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD([w], lr=0.01)

# epochs = 100

# print(f"before training: f(5) = {forward_with_optimizer(5).item():.3f}")

# for epoch in range(epochs):
#     y_pred = forward_with_optimizer(X)
#     loss_value = criterion(y_pred, Y)

#     optimizer.zero_grad()
#     loss_value.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(
#             f"epoch {epoch + 1:03d}: "
#             f"w = {w.item():.3f}, loss = {loss_value.item():.6f}"
#         )

# print(f"after training: f(5) = {forward_with_optimizer(5).item():.3f}")


# 完整训练管线——Model、Loss、Optimizer
# 最后把手写参数 w 和 forward 函数也交给 PyTorch：
# - `nn.Linear` 保存可训练参数并实现 forward
# - `nn.MSELoss` 计算损失
# - `torch.optim.SGD` 更新模型参数
# X = torch.tensor(
#     [[1.0],
#      [2.0],
#      [3.0],
#      [4.0]]
# )

# Y = torch.tensor(
#     [[2.0],
#      [4.0],
#      [6.0],
#      [8.0]]
# )

# X_test = torch.tensor([[5.0]])

# model = nn.Linear(in_features=1, out_features=1)
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# epochs = 1000

# print("initial parameters:")
# print("weight =", model.weight.item())
# print("bias   =", model.bias.item())
# print(f"before training: f(5) = {model(X_test).item():.3f}")

# for epoch in range(epochs):
#     y_pred = model(X)
#     loss_value = criterion(y_pred, Y)

#     optimizer.zero_grad()
#     loss_value.backward()
#     optimizer.step()

#     if epoch % 200 == 0:
#         print(
#             f"epoch {epoch + 1:03d}: "
#             f"weight = {model.weight.item():.3f}, "
#             f"bias = {model.bias.item():.3f}, "
#             f"loss = {loss_value.item():.6f}"
#         )

# print("final parameters:")
# print("weight =", model.weight.item())
# print("bias   =", model.bias.item())
# print(f"after training: f(5) = {model(X_test).item():.3f}")


# 放在GPU上训练
# 学会张量的.to(device) 方法后，我们就可以把数据和模型移动到 GPU 上进行训练了
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )

# print("using device:", device)

# X = torch.tensor(
#     [[1.0],
#      [2.0],
#      [3.0],
#      [4.0]]
# ).to(device)

# Y = torch.tensor(
#     [[2.0],
#      [4.0],
#      [6.0],
#      [8.0]]
# ).to(device)

# X_test = torch.tensor([[5.0]]).to(device)

# # 模型也要移动到同一个设备
# model = nn.Linear(
#     in_features=1,
#     out_features=1
# ).to(device)

# criterion = nn.MSELoss()

# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=0.01
# )

# epochs = 1000

# print("X device:", X.device)
# print("model device:", model.weight.device)

# print("\ninitial parameters:")
# print("weight =", model.weight.item())
# print("bias   =", model.bias.item())
# print(f"before training: f(5) = {model(X_test).item():.3f}")

# for epoch in range(epochs):
#     # 前向传播在 device 上进行
#     y_pred = model(X)

#     # loss 也位于同一个 device
#     loss_value = criterion(y_pred, Y)

#     optimizer.zero_grad()
#     loss_value.backward()
#     optimizer.step()

#     if epoch % 200 == 0:
#         print(
#             f"epoch {epoch + 1:04d}: "
#             f"weight = {model.weight.item():.3f}, "
#             f"bias = {model.bias.item():.3f}, "
#             f"loss = {loss_value.item():.6f}"
#         )

# print("\nfinal parameters:")
# print("weight =", model.weight.item())
# print("bias   =", model.bias.item())
# print(f"after training: f(5) = {model(X_test).item():.3f}")