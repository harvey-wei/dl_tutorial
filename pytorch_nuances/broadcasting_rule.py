import torch
import numpy as np



'''
Broadcasting Does NOT Copy Input Tensors
Here's the key insight about broadcasting and copying:
✅ What Broadcasting Does NOT Copy:
Input tensors remain unchanged
Original data is not duplicated
Memory addresses stay the same
✅ What Broadcasting DOES Create:
New result tensor with computed values
New memory allocation for the result
On-the-fly computation without intermediate copies

'''

print("="*60)
print("PYTORCH BROADCASTING RULES TUTORIAL")
print("="*60)

print("\n1. WHAT IS BROADCASTING?")
print("-" * 30)
print("Broadcasting is a way to perform operations on tensors with different shapes")
print("PyTorch automatically expands smaller tensors to match larger ones")
print("This allows element-wise operations without explicitly reshaping tensors")

print("\n2. BROADCASTING RULES")
print("-" * 30)
print("Rule 1: Start from the RIGHTMOST dimension")
print("Rule 2: Dimensions are compatible if:")
print("   - They are equal, OR")
print("   - One of them is 1, OR") 
print("   - One of them doesn't exist")
print("   - Padding shape with 1s to make shape length equal")
print("Rule 3: The result shape is the maximum size along each dimension")

print("\n3. BASIC EXAMPLES")
print("-" * 30)

# Example 1: Scalar + Tensor
print("\nExample 1: Scalar + Tensor")
a = torch.tensor([1, 2, 3, 4])
b = 10
print(f"a: {a}, shape: {a.shape}")
print(f"b: {b}, shape: scalar")
result = a + b
print(f"a + b: {result}")
print(f"Result shape: {result.shape}")
print("✓ Scalar is broadcasted to match tensor shape")

# Example 2: Different shapes
print("\nExample 2: Different shapes")
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
y = torch.tensor([10, 20, 30])             # shape: (3,)
print(f"x: {x}")
print(f"x shape: {x.shape}")
print(f"y: {y}")
print(f"y shape: {y.shape}")
result = x + y
print(f"x + y: {result}")
print(f"Result shape: {result.shape}")
print("✓ y is broadcasted from (3,) to (2, 3)")

# Example 3: 1D + 2D
print("\nExample 3: 1D + 2D")
a = torch.tensor([1, 2, 3])           # shape: (3,)
b = torch.tensor([[10], [20], [30]])  # shape: (3, 1)
print(f"a: {a}, shape: {a.shape}")
print(f"b: {b}, shape: {b.shape}")
result = a + b
print(f"a + b: {result}")
print(f"Result shape: {result.shape}")
print("✓ a is broadcasted to (3, 3) and b is broadcasted to (3, 3)")

print("\n4. STEP-BY-STEP BROADCASTING PROCESS")
print("-" * 30)

def explain_broadcasting(tensor1, tensor2, operation="+"):
    print(f"\nBroadcasting: {tensor1.shape} {operation} {tensor2.shape}")
    print(f"Tensor 1: {tensor1}")
    print(f"Tensor 2: {tensor2}")
    
    # Align shapes from right to left
    shape1 = list(tensor1.shape)
    shape2 = list(tensor2.shape)
    
    # Pad with 1s to make lengths equal
    max_len = max(len(shape1), len(shape2))
    shape1 = [1] * (max_len - len(shape1)) + shape1
    shape2 = [1] * (max_len - len(shape2)) + shape2
    
    print(f"Aligned shapes:")
    print(f"  Shape 1: {shape1}")
    print(f"  Shape 2: {shape2}")
    
    # Check compatibility and compute result shape
    result_shape = []
    for i in range(max_len):
        dim1, dim2 = shape1[i], shape2[i]
        if dim1 == dim2:
            result_shape.append(dim1)
        elif dim1 == 1:
            result_shape.append(dim2)
        elif dim2 == 1:
            result_shape.append(dim1)
        else:
            raise ValueError(f"Incompatible dimensions: {dim1} and {dim2}")
    
    print(f"Result shape: {result_shape}")
    
    # Perform operation
    if operation == "+":
        result = tensor1 + tensor2
    elif operation == "*":
        result = tensor1 * tensor2
    
    print(f"Result: {result}")
    return result

# Test the explanation function
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
tensor2 = torch.tensor([10, 20, 30])             # (3,)
explain_broadcasting(tensor1, tensor2)

print("\n5. COMMON BROADCASTING PATTERNS")
print("-" * 30)

# Pattern 1: Adding bias to a matrix
print("\nPattern 1: Adding bias to a matrix")
matrix = torch.randn(3, 4)
bias = torch.randn(4)
print(f"Matrix shape: {matrix.shape}")
print(f"Bias shape: {bias.shape}")
result = matrix + bias
print(f"Result shape: {result.shape}")
print("✓ Bias is broadcasted across all rows")

# Pattern 2: Batch operations
print("\nPattern 2: Batch operations")
batch = torch.randn(5, 3, 4)  # 5 samples, 3 features, 4 values
weights = torch.randn(3, 4)   # 3 features, 4 values
print(f"Batch shape: {batch.shape}")
print(f"Weights shape: {weights.shape}")
result = batch * weights
print(f"Result shape: {result.shape}")
print("✓ Weights are broadcasted across all 5 samples")

# Pattern 3: Reshaping for broadcasting
print("\nPattern 3: Reshaping for broadcasting")
data = torch.randn(10, 3)
mean = torch.mean(data, dim=0)  # shape: (3,)
print(f"Data shape: {data.shape}")
print(f"Mean shape: {mean.shape}")
# To subtract mean from each row, we need to reshape
mean_reshaped = mean.unsqueeze(0)  # shape: (1, 3)
# mean_reshaped = mean
print(f"Mean reshaped: {mean_reshaped.shape}")
result = data - mean_reshaped
print(f"Result shape: {result.shape}")
print("✓ Mean is broadcasted across all 10 rows")

print("\n6. BROADCASTING WITH ADVANCED INDEXING")
print("-" * 30)

# Example with different index shapes
print("\nAdvanced indexing with broadcasting:")
x = torch.randn(3, 4, 5)
print(f"x shape: {x.shape}")

# Different index shapes that broadcast
batch_idx = torch.tensor([0, 1])           # shape: (2,)
row_idx = torch.tensor([[0, 1], [2, 3]])   # shape: (2, 2)
col_idx = torch.tensor([1, 2])             # shape: (2,)

print(f"batch_idx shape: {batch_idx.shape}")
print(f"row_idx shape: {row_idx.shape}")
print(f"col_idx shape: {col_idx.shape}")

# These broadcast to compatible shapes
result = x[batch_idx, row_idx, col_idx]
print(f"Result shape: {result.shape}")
print("✓ Advanced indexing works with broadcastable index shapes")

print("\n7. COMMON MISTAKES AND SOLUTIONS")
print("-" * 30)

# Mistake 1: Incompatible shapes
print("\nMistake 1: Incompatible shapes")
try:
    a = torch.tensor([1, 2, 3])     # shape: (3,)
    b = torch.tensor([1, 2])        # shape: (2,)
    result = a + b
    print(f"If number of dimensions are same possibly padding with 1s to left, each dimension should be either equal or 1")
except RuntimeError as e:
    print(f"Error: {e}")
    print("Solution: Reshape one of the tensors")
    b_reshaped = b.unsqueeze(0)  # shape: (1, 2)
    print(f"b reshaped: {b_reshaped.shape}")
    # Now we need to transpose a to make them compatible
    a_reshaped = a.unsqueeze(1)  # shape: (3, 1)
    result = a_reshaped + b_reshaped
    print(f"Result: {result}")

# Mistake 2: Wrong dimension for reduction
print("\nMistake 2: Wrong dimension for reduction")
data = torch.randn(5, 3, 4)
print(f"Data shape: {data.shape}")
# Wrong way - trying to subtract mean from wrong dimension
try:
    mean_wrong = torch.mean(data, dim=1)  # shape: (5, 4)
    result_wrong = data - mean_wrong
    explain_broadcasting(data, mean_wrong, "-")
    # [5, 3, 4] with left padded shape [1, 5, 4] is not compatible as 3 is not equal to 5
except RuntimeError as e:
    print(f"Error: {e}")
    print("Solution: Use correct dimension and reshape")
    mean_correct = torch.mean(data, dim=0)  # shape: (3, 4)
    mean_correct = mean_correct.unsqueeze(0)  # shape: (1, 3, 4)
    result_correct = data - mean_correct
    print(f"Correct result shape: {result_correct.shape}")

print("\n8. PERFORMANCE CONSIDERATIONS")
print("-" * 30)
print("✓ Broadcasting is memory efficient - no actual copying occurs")
print("✓ Operations are vectorized and fast")
print("✓ Avoid unnecessary reshaping when broadcasting can handle it")
print("✗ Be careful with very large broadcasts as they can use significant memory")

print("\n9. PRACTICAL TIPS")
print("-" * 30)
print("1. Use .unsqueeze() to add dimensions: tensor.unsqueeze(0)")
print("2. Use .squeeze() to remove size-1 dimensions: tensor.squeeze()")
print("3. Use .expand() to repeat without copying: tensor.expand(new_shape)")
print("4. Use .repeat() to repeat with copying: tensor.repeat(times)")
print("5. Check shapes with .shape before operations")
print("6. Use .view() or .reshape() when you need to change the layout")

print("\n10. SUMMARY")
print("-" * 30)
print("Broadcasting allows operations between tensors of different shapes")
# print("Rules: align from right, compatible if equal/1/missing")
print("Rules: align from right, padding with 1s to left, compatible if equal/1")
print("Result shape is maximum along each dimension")
print("Memory efficient and fast when used correctly")
print("Essential for deep learning operations like adding bias, batch normalization, etc.")

print("\n" + "="*60)
print("END OF TUTORIAL")
print("="*60)

