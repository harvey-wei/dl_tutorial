import torch
import torch.nn.functional as F

# Create a 2D tensor
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print("Original Tensor:")
print(x)

# torch.sum
# (row, col) = (dim_0, dim_1)
sum_dim0 = torch.sum(x, dim=0) # reduce dim_0, aka, row or along axis 0
sum_dim1 = torch.sum(x, dim=1)
print("\nSum along dim=0 (column-wise):", sum_dim0)
print("Sum along dim=1 (row-wise):", sum_dim1)

# torch.min
min_vals_dim1, min_idxs_dim1 = torch.min(x, dim=1)
print("\nMin values along dim=1 (row-wise):", min_vals_dim1)
print("Indices of min values along dim=1:", min_idxs_dim1)

# torch.max
max_vals_dim0, max_idxs_dim0 = torch.max(x, dim=0)
print("\nMax values along dim=0 (column-wise):", max_vals_dim0)
print("Indices of max values along dim=0:", max_idxs_dim0)

# F.softmax
softmax_dim0 = F.softmax(x, dim=0)
softmax_dim1 = F.softmax(x, dim=-1)
print("\nSoftmax along dim=0 (column-wise):")
print(softmax_dim0)
print("\nSoftmax along dim=1 (row-wise):")
print(softmax_dim1)

logits = torch.exp(x)
logits_sum = torch.sum(logits, dim=-1) # (2,)
logits_sum = logits_sum[:, None] # (2, 1)

probs = logits / logits_sum
print("\nProbs:")
print(probs)
