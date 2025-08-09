import torch
from einops import rearrange, einsum, reduce, repeat

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
v = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)

# Row and column vectors
row_vec = rearrange(v, 'n -> 1 n')  # (1, 3)
col_vec = rearrange(v, 'n -> n 1')  # (3, 1)

# Dot product (inner product)
dot_product = einsum(v, v, 'i, i ->')
print("Dot product:", dot_product.item())

# Outer product out_{i, j} = v_{i} v_{j}
outer_product = einsum(v, v, 'i, j -> i j')
print("Outer product:\n", outer_product)

# Matrix product: column @ row. col_{i, j} = col
mat_col_row = einsum(col_vec, row_vec, 'i k, k j -> i j')
print("col_vec @ row_vec:\n", mat_col_row)

# Matrix product: row @ column
mat_row_col = einsum(row_vec, col_vec, 'k i, i k ->')
print("row_vec @ col_vec:\n", mat_row_col)
