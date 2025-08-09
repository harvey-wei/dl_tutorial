import torch

# Create a list of 1D tensors
tensors = [torch.tensor([1.0]), torch.tensor([2.0])]

# Example 1: Not in-place (does NOT change original tensors)
print("Before += 10:")
print(tensors)


# Normally, t += 10 creates a new tensor, unless:
# The tensor has requires_grad=False and
# PyTorch optimizes it to do an in-place operation (for performance

for t in tensors:
    t += 10  # chagne the tensors as well

print("\nAfter += 10 (not in-place):")
print(tensors)  # Original tensors are unchanged

# Example 2: In-place (modifies original tensors)
print("\nNow applying in-place add_:")

for t in tensors:
    t.add_(10)  # This modifies the tensor in-place

print("\nAfter add_(10) (in-place):")
print(tensors)  # Original tensors are now modified
