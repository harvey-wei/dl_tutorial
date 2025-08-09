import torch
import torch.nn.functional as F


# pad argument is a tuple of padding on fron and back of each dimension in reverse order of
# dimensions
# https://discuss.pytorch.org/t/visual-explanation-of-torch
t4d = torch.empty(3, 3, 4, 2)
p1d = (1, 1) # pad last dim by 1 on each side
out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
print(out.size())

p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
out = F.pad(t4d, p2d, "constant", 0)
print(out.size())

t4d = torch.empty(3, 3, 4, 2)
p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
out = F.pad(t4d, p3d, "constant", 0)
print(out.size())


# Create a 2D tensor
tensor = torch.tensor([[1, 2], [3, 4]])

# Pad with 1 unit of padding on all sides
padded_tensor = F.pad(tensor, pad=(1, 1, 1, 1), mode='constant', value=0)
print(padded_tensor)




import torch
import torch.nn.functional as F

x = torch.tensor([[1, 2], [3, 4]])
# shape: (2, 2)

# Pad right by 1, bottom by 2
padded = F.pad(x, pad=(0, 1, 0, 2))  # (left, right, top, bottom)
print(padded)
