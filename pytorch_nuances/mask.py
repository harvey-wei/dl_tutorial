import torch
import torch.nn as nn


batch_size, text_seq_len, hidden_size = 2, 50, 128
batch_size, vision_seq_len, hidden_size = 2, 30, 128

text_embeddings = torch.randn(batch_size, text_seq_len, hidden_size)
vision_embeddings = torch.randn(batch_size, vision_seq_len, hidden_size)

# vision mask[b, :] store the  seq_index in the text_embeddings
vision_mask = torch.randint(0, text_seq_len, (batch_size, vision_seq_len))


# Replace the some of the text_embeddings with the vision_embeddings by the vision_mask
# text_embeddings[vision_mask]

text_embeddings_copy = text_embeddings.clone()

for b in range(batch_size):
    text_embeddings_copy[b, vision_mask[b, :]] = vision_embeddings[b, :]

print(text_embeddings_copy)

# Use advanced indexing to replace the text_embeddings with the vision_embeddings
# What is advanced indexing?
# Advanced indexing is a way to index into a tensor using another tensor.
# It requires the index tensor for each dimension to be the same shape or broadcastable to the same shape.
# Here, we use the batch_index to index into the text_embeddings, and the vision_mask to index into the vision_embeddings.

print(f"text_embeddings memory address before advanced indexing: {text_embeddings.data_ptr()}")
batch_index = torch.arange(batch_size).unsqueeze(1).expand(-1, vision_seq_len)
text_embeddings[batch_index, vision_mask] = vision_embeddings

print(text_embeddings)

# check if text_embedding changes its memory address
print(f"text_embeddings memory address after advanced indexing: {text_embeddings.data_ptr()}")
print("Advaned indexing with assignment does not change the memory address of the tensor -- inplace operation -- caution for auto-grad")


# check whether these two methods are equivalent
for b in range(batch_size):
    for i in range(vision_seq_len):
        for j in range(hidden_size):
            if not torch.isclose(text_embeddings_copy[b, vision_mask[b, i], j], text_embeddings[b, vision_mask[b, i], j], atol=1e-3):
                print(f"text_embeddings_copy[{b}, {vision_mask[b, i]}, {j}] != text_embeddings[{b}, {vision_mask[b, i]}, {j}]")
                print(text_embeddings_copy[b, vision_mask[b, i], j])
                print(text_embeddings[b, vision_mask[b, i], j])
                break




# print(torch.allclose(text_embeddings_copy, text_embeddings, atol=1e-3))

# compare the text_embeddings_copy and text_embeddings is close enough
# print(torch.allclose(text_embeddings_copy, text_embeddings, atol=1e-6))


print("\n" + "="*50)
print("ADVANCED INDEXING SHAPE REQUIREMENTS")
print("="*50)

# Example 1: Different shapes that work
print("\n1. Different shapes that work:")
x = torch.randn(3, 4, 5)
print(f"x shape: {x.shape}")

# Index tensors with different shapes
batch_idx = torch.tensor([0, 1])  # shape: (2,)
row_idx = torch.tensor([[0, 1], [2, 3]])  # shape: (2, 2)
col_idx = torch.tensor([1, 2])  # shape: (2,)

print(f"batch_idx shape: {batch_idx.shape}")
print(f"row_idx shape: {row_idx.shape}")
print(f"col_idx shape: {col_idx.shape}")

# This works because PyTorch broadcasts the indices
result = x[batch_idx, row_idx, col_idx]
print(f"Result shape: {result.shape}")
print("✓ Advanced indexing works with different index tensor shapes!")

# Example 2: Broadcasting in action
print("\n2. Broadcasting example:")
y = torch.randn(2, 3, 4)
print(f"y shape: {y.shape}")

# Single index for first dimension, multiple for others
idx1 = torch.tensor([0])  # shape: (1,)
idx2 = torch.tensor([[0, 1], [1, 2]])  # shape: (2, 2)
idx3 = torch.tensor([1, 2])  # shape: (2,)

print(f"idx1 shape: {idx1.shape}")
print(f"idx2 shape: {idx2.shape}")
print(f"idx3 shape: {idx3.shape}")

# PyTorch broadcasts these to compatible shapes
result2 = y[idx1, idx2, idx3]
print(f"Result2 shape: {result2.shape}")
print("✓ Broadcasting works in advanced indexing!")

# Example 3: What doesn't work
print("\n3. What doesn't work:")
try:
    # Incompatible shapes that can't be broadcast
    bad_idx1 = torch.tensor([0, 1, 2])  # shape: (3,)
    bad_idx2 = torch.tensor([0, 1])     # shape: (2,)
    # This would fail because 3 and 2 can't be broadcast together
    # result3 = x[bad_idx1, bad_idx2]  # This would raise an error
    print("❌ Incompatible shapes that can't be broadcast will fail")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50)
print("KEY POINTS:")
print("1. Index tensors don't need the same shape")
print("2. They must be broadcastable to compatible shapes")
print("3. The result shape follows broadcasting rules")
print("4. For assignment: right-hand side must be broadcastable to the indexed result")
print("="*50)
