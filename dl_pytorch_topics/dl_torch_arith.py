import torch

# Example tensor
tensor = torch.tensor([1, 2, 3, 4])
squared = tensor ** 2
print(squared)  # Output: tensor([ 1,  4,  9, 16])

# For 2D tensors
tensor_2d = torch.tensor([[1, 2], [3, 4]])
squared_2d = tensor_2d ** 2
print(squared_2d)  # Output: tensor([[ 1,  4], [ 9, 16]])
