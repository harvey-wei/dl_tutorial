import torch

x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = x ** 2  # Equivalent to torch.pow(x, 2)

print(y)  # tensor([1.0, 4.0, 9.0])

z = torch.pow(2.0, x) # 2.0 is auto-promoted to a tensor
print(z)
print(f'z device: {z.device}')
