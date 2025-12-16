import extension
import torch

import sys

print(f"python executable: {sys.executable}")
print(f"python path: {sys.path}")

# direct tensor operations
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
result = extension.tensor_add(a, b)
print(f"Tensor add: {a} + {b} = {result}")
print(f"Result type: {type(result)}")  # torch.Tensor

# gradients
x = torch.randn(3, 3, requires_grad=True)
gelu_result = extension.custom_gelu(x)
print(f"GELU preserves grad: {gelu_result.requires_grad}")

# matmul
m1 = torch.randn(2, 3)
m2 = torch.randn(3, 4)
result = extension.matmul(m1, m2)
print(f"matmul shape: {result.shape}")
