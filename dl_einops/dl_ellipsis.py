import torch
from einops import repeat, rearrange, einsum


batch_size = 16
rows_A = 32
cols_A = 16
rows_B = 64
cols_B = cols_A

A = torch.randn(batch_size, rows_A, cols_A)
B = torch.randn(batch_size, rows_B, cols_A )

C = einsum(A, B, '... rows_A cols_A, ... rows_B cols_A -> ... rows_A rows_B')
print(f'C shape {C.shape}')
