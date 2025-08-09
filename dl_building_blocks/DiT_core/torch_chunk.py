import torch

# Example tensor with shape (B, N, C)
B, N, C = 2, 3, 6
x = torch.randn((B, N, C))
# x = torch.arange(B * N * C).reshape(B, N, C)

# Number of chunks to split into
num_chunks = 3

# Split along the last dimension (dim=2) and return tuple of chunk tensor
chunks = torch.chunk(x, num_chunks, dim=-1)

print(f' chunks {chunks}')

# Display the shape of each chunk
for i, chunk in enumerate(chunks):
    print(f"Chunk {i} shape: {chunk.shape}")
