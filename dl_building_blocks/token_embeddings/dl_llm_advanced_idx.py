import torch
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(0)

# Define vocabulary size and embedding dimension
vocab_size = 10
embedding_dim = 4

# Create the embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim)

# Print the full embedding weight for reference
print("Embedding weight matrix:\n", embedding.weight)

# Define a batch of token sequences (2 sequences, each length 2)
tokens = torch.tensor([[1, 3], [4, 7]])  # shape: (2, 2)

# Method 1: Using nn.Embedding forward
# input (*) output (*, H) with H as the embedding_dim
embedded_via_call = embedding(tokens)

# Method 2: Using advanced indexing manually
embedded_via_indexing = embedding.weight[tokens]  # same shape

# Compare outputs
print("\nTokens:\n", tokens)
print("\nEmbedded via nn.Embedding call:\n", embedded_via_call)
print("\nEmbedded via advanced indexing:\n", embedded_via_indexing)

# Check if both are exactly equal
equal = torch.allclose(embedded_via_call, embedded_via_indexing)
print(f"\nAre both methods equal? {'✅ Yes' if equal else '❌ No'}")



