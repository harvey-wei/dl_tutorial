import torch
import torch.nn

a = torch.randn((2, 3, 4, 4))
b = torch.randn((2, 1))

print(f"a.shape: {a.shape}")
print(f"b.shape: {b.shape}")

# Broadcasting
c = a * b
print(c.shape)  # Expected shape: (2, 3, 4, 4)
