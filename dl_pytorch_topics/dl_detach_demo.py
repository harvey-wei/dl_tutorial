import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2           # y requires grad
z = y.detach()      # z does NOT require grad z shares the same memory as y but cut off the compute graph

print(y.requires_grad)  # True
print(z.requires_grad)  # False

z[0] = 100
print(y)  # y reflects the change! (z shares the same storage)
y[0] = 101
print(z)

# if you want to be safe
# cut off the computate graph (set requires_grad to False) and then copy
z = y.detach().clone()
print(z.requires_grad)

