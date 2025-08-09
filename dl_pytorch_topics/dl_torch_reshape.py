import torch

x = torch.ones(2, 3, requires_grad=True) # tensor defaults to Fales graa except the parameters nn.module

x = torch.ones (2, 3, requires_grad = True)

y = x[None, :, : ] * 3

l = torch.sum(y)
l.backward()


print(x)
