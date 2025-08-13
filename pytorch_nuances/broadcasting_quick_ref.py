import torch

print("PYTORCH BROADCASTING QUICK REFERENCE")
print("="*50)

print("\n📋 BROADCASTING RULES:")
print("1. Align shapes from RIGHT to LEFT")
print("2. Dimensions are compatible if:")
print("   • Equal: (3, 4) + (3, 4) ✓")
print("   • One is 1: (3, 4) + (1, 4) ✓")
print("   • One missing: (3, 4) + (4,) ✓")
print("3. Result shape = max size along each dimension")

print("\n🔧 COMMON OPERATIONS:")
print("-" * 30)

# 1. Scalar operations
print("1. Scalar + Tensor:")
a = torch.tensor([1, 2, 3])
print(f"   {a} + 10 = {a + 10}")

# 2. Vector + Matrix
print("\n2. Vector + Matrix:")
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
vector = torch.tensor([10, 20, 30])
print(f"   Matrix: {matrix}")
print(f"   Vector: {vector}")
print(f"   Result: {matrix + vector}")

# 3. Adding dimensions
print("\n3. Adding dimensions:")
tensor = torch.tensor([1, 2, 3])
print(f"   Original: {tensor.shape}")
print(f"   unsqueeze(0): {tensor.unsqueeze(0).shape}")
print(f"   unsqueeze(1): {tensor.unsqueeze(1).shape}")

# 4. Removing dimensions
print("\n4. Removing dimensions:")
tensor_2d = torch.tensor([[1], [2], [3]])
print(f"   Original: {tensor_2d.shape}")
print(f"   squeeze(): {tensor_2d.squeeze().shape}")

print("\n💡 PRACTICAL PATTERNS:")
print("-" * 30)

# Pattern 1: Adding bias
print("1. Adding bias to neural network:")
activations = torch.randn(32, 128)  # batch_size, features
bias = torch.randn(128)            # features
output = activations + bias        # bias broadcasted to (32, 128)
print(f"   Activations: {activations.shape}")
print(f"   Bias: {bias.shape}")
print(f"   Output: {output.shape}")

# Pattern 2: Batch normalization
print("\n2. Batch normalization:")
batch = torch.randn(16, 64, 32)    # batch, channels, features
mean = torch.mean(batch, dim=0)    # (64, 32)
std = torch.std(batch, dim=0)      # (64, 32)
normalized = (batch - mean) / std  # mean/std broadcasted to (16, 64, 32)
print(f"   Batch: {batch.shape}")
print(f"   Mean: {mean.shape}")
print(f"   Normalized: {normalized.shape}")

# Pattern 3: Attention weights
print("\n3. Attention weights:")
scores = torch.randn(8, 12, 50, 50)  # batch, heads, seq_len, seq_len
mask = torch.ones(50, 50)            # seq_len, seq_len
masked_scores = scores + mask        # mask broadcasted to (8, 12, 50, 50)
print(f"   Scores: {scores.shape}")
print(f"   Mask: {mask.shape}")
print(f"   Masked: {masked_scores.shape}")

print("Broadcasting aligned from right, padding with 1s to left, compatible if equal/1")
print("This is why we put batch dimension first in the shape, head_dim second")

print("\n❌ COMMON ERRORS:")
print("-" * 30)
print("1. Incompatible shapes:")
print("   (3,) + (2,) → RuntimeError")
print("   Solution: Reshape or use different dimensions")

print("\n2. Wrong reduction dimension:")
print("   data(5,3,4) - mean(dim=1) → (5,4) vs (5,3,4)")
print("   Solution: mean(dim=0).unsqueeze(0)")

print("\n3. Forgetting to reshape:")
print("   data(10,3) - mean(3,) → OK!")
print("   Solution: mean.unsqueeze(0) → (1,3)")

data = torch.randn(10, 3)
mean = torch.randn(3)
data_mean = data - mean
print(f"   Data: {data.shape}")
print(f"   Mean: {mean.shape}")
print(f"   Data - Mean: {data_mean.shape}")


print("\n🛠️ USEFUL METHODS:")
print("-" * 30)
print("• tensor.unsqueeze(dim)     - Add dimension")
print("• tensor.squeeze()          - Remove size-1 dimensions")
print("• tensor.expand(*sizes)     - Expand without copying")
print("• tensor.repeat(*sizes)     - Repeat with copying")
print("• tensor.view(*sizes)       - Reshape (contiguous)")
print("• tensor.reshape(*sizes)    - Reshape (any memory layout)")
print(". both unsqueeze and squeeze are NOT copying data operations")
print("\n⚡ PERFORMANCE TIPS:")
print("-" * 30)
print("✓ Broadcasting is memory efficient")
print("✓ Use .expand() instead of .repeat() when possible")
print("✓ Avoid unnecessary .clone() operations")
print("✓ Check shapes before operations with .shape")

print("\n" + "="*50)

