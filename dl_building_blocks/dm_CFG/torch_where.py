import torch

x = torch.tensor([[1, 2], [3, 4]])

# Explain the usage of torch.where
# torch.where(condition, x, y) returns a tensor where the condition is true and the values from x
# where the condition is fals  and the values from y.
# In this case, we are checking if the elements of x are greater than 2.
result = torch.where(x > 2, x, 0) # condition, true ele, false ele

print(f'\n x {x} \n')
print('After doing torch.where(x > 2, x, 0):')
print(f'\n result {result} \n')
