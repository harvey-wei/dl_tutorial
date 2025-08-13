import torch
import time

print("="*60)
print("BROADCASTING: COPYING vs NO COPYING")
print("="*60)

print("\n1. BROADCASTING IS MEMORY EFFICIENT - NO COPYING")
print("-" * 50)

# Create tensors
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([10])

print(f"a: {a}, shape: {a.shape}, memory: {a.data_ptr()}")
print(f"b: {b}, shape: {b.shape}, memory: {b.data_ptr()}")

# Broadcasting operation
result = a + b
print(f"result: {result}, shape: {result.shape}, memory: {result.data_ptr()}")

print("\n✓ Broadcasting creates a NEW tensor for the result")
print("✓ But the original tensors (a, b) are NOT copied")
print("✓ PyTorch computes the result on-the-fly")

print("\n2. COMPARISON: BROADCASTING vs EXPLICIT COPYING")
print("-" * 50)


# Warmup before testing time to rule out the effect of preparation time
'''
1. CPU & GPU runtime optimization effects
When you run a model (or any computation) for the first time:

CPU:

Caches (L1/L2/L3) are cold — the first pass loads data into caches.

Branch predictors and speculative execution aren’t primed.

Dynamic frequency scaling (Turbo Boost) might not yet be engaged.

GPU:

CUDA kernels are being JIT-compiled (if using frameworks like PyTorch/TensorRT).

GPU caches and shared memory are empty.

Power states (P-states) may be in a lower clock mode until load ramps up.

Warmup ensures you measure after these are stabilized.

2. Framework & library initialization
PyTorch, cuDNN, cuBLAS, MKL, etc. often:

Perform kernel autotuning on first call (pick the fastest kernel for your input shape).

Allocate persistent workspaces/buffers.

Initialize handles (e.g., cuBLAS handle).

These one-time costs can heavily skew your first-timing result.

import torch
import time

x = torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize()

start = time.time()
y = torch.matmul(x, x)  # First call: includes cuBLAS init + autotune
torch.cuda.synchronize()
print("First run:", time.time() - start)

start = time.time()
y = torch.matmul(x, x)  # Steady state
torch.cuda.synchronize()
print("Second run:", time.time() - start)

'''
for _ in range(10):
    matrix = torch.randn(1000, 1000)
    vector = torch.randn(1000)
    result_broadcast = matrix + vector
    result_copy = matrix + vector.unsqueeze(0).expand(matrix.shape[0], -1)

# Method 1: Broadcasting (efficient)
print("\nMethod 1: Broadcasting (efficient)")
matrix = torch.randn(1000, 1000)
vector = torch.randn(1000)

start_time = time.time()
result_broadcast = matrix + vector
broadcast_time = time.time() - start_time
print(f"Broadcasting time: {broadcast_time:.6f} seconds")

# Method 2: Explicit copying (less efficient)
print("\nMethod 2: Explicit copying (less efficient)")
start_time = time.time()
# Expand vector to match matrix shape
vector_expanded = vector.unsqueeze(0).expand(matrix.shape[0], -1)
result_copy = matrix + vector_expanded
copy_time = time.time() - start_time
print(f"Explicit copying time: {copy_time:.6f} seconds")

print(f"\nSpeed difference: {copy_time/broadcast_time:.2f}x slower with explicit copying")
print("✓ Broadcasting is faster because it avoids unnecessary memory allocation")

print("\n3. MEMORY USAGE COMPARISON")
print("-" * 50)

# Check memory usage
print("\nMemory usage comparison:")
print(f"matrix: {matrix.element_size() * matrix.numel()} bytes")
print(f"vector: {vector.element_size() * vector.numel()} bytes")
print(f"result_broadcast: {result_broadcast.element_size() * result_broadcast.numel()} bytes")
print(f"vector_expanded: {vector_expanded.element_size() * vector_expanded.numel()} bytes")

print("\n✓ Broadcasting result uses same memory as explicit copying")
print("✓ But broadcasting avoids intermediate tensor creation")
print("✓ vector_expanded does not need additional memory. Returned result is logical storage NOT physical storage! ")

print("\n4. DIFFERENT TYPES OF OPERATIONS")
print("-" * 50)

# In-place operations
print("\nIn-place operations:")
a_inplace = torch.tensor([1, 2, 3, 4])
b_inplace = torch.tensor([10])
print(f"Before: a_inplace = {a_inplace}, memory: {a_inplace.data_ptr()}")
a_inplace += b_inplace  # In-place broadcasting
print(f"After: a_inplace = {a_inplace}, memory: {a_inplace.data_ptr()}")
print("✓ In-place operations modify the original tensor")

# View operations
print("\nView operations:")
original = torch.randn(3, 4)
print(f"Original: {original.shape}, memory: {original.data_ptr()}")
reshaped = original.view(12)
print(f"Reshaped: {reshaped.shape}, memory: {reshaped.data_ptr()}")
print(f"Same memory? {original.data_ptr() == reshaped.data_ptr()}")
print("✓ View operations share the same memory")

print("\n5. WHEN COPYING DOES HAPPEN")
print("-" * 50)

# When you explicitly copy
print("\nExplicit copying:")
original_tensor = torch.tensor([1, 2, 3, 4])
copied_tensor = original_tensor.clone()
print(f"Original: {original_tensor.data_ptr()}")
print(f"Copied: {copied_tensor.data_ptr()}")
print(f"Same memory? {original_tensor.data_ptr() == copied_tensor.data_ptr()}")
print("✗ Explicit .clone() creates a copy with new memory")

# When you use .repeat()
print("\n.repeat() operation:")
small_tensor = torch.tensor([1, 2, 3]) # [3,]
repeated = small_tensor.repeat(3) # [3, 3]
print(f"Small: {small_tensor.data_ptr()}")
print(f"Repeated: {repeated.data_ptr()}")
print(f"Same memory? {small_tensor.data_ptr() == repeated.data_ptr()}")
print("✗ .repeat() creates a copy with new memory")

# When you use .expand() (no copying)
print("\n.expand() operation:")
expandable = torch.tensor([1, 2, 3])
expanded = expandable.unsqueeze(0).expand(3, -1)
print(f"Original: {expandable.data_ptr()}")
print(f"Expanded: {expanded.data_ptr()}")
print(f"Same memory? {expandable.data_ptr() == expanded.data_ptr()}")
print("✓ .expand() shares memory (no copying)")

print("\n6. PERFORMANCE IMPLICATIONS")
print("-" * 50)

# Large tensor demonstration
print("\nLarge tensor performance test:")
large_matrix = torch.randn(50000, 50000)
large_vector = torch.randn(50000)

# Test broadcasting
start_time = time.time()
result1 = large_matrix + large_vector
broadcast_time = time.time() - start_time

# Test explicit expansion
start_time = time.time()
expanded_vector = large_vector.unsqueeze(0).expand(large_matrix.shape[0], -1)
result2 = large_matrix + expanded_vector
expand_time = time.time() - start_time

print(f"Broadcasting time: {broadcast_time:.4f} seconds")
print(f"Explicit expansion time: {expand_time:.4f} seconds")
print(f"Speedup: {expand_time/broadcast_time:.2f}x")

print("\n7. SUMMARY")
print("-" * 50)
print("✓ Broadcasting: NO copying of input tensors")
print("✓ Broadcasting: Creates NEW result tensor")
print("✓ Broadcasting: Memory efficient")
print("✓ Broadcasting: Computationally fast")
print("✗ Explicit copying: Creates copies with new memory")
print("✗ Explicit copying: Uses more memory")
print("✗ Explicit copying: Slower execution")

print("\n8. BEST PRACTICES")
print("-" * 50)
print("✓ Use broadcasting when possible")
print("✓ Use .expand() instead of .repeat() when you can")
print("✓ Use in-place operations (+=, -=) to save memory")
print("✓ Avoid unnecessary .clone() operations")
print("✓ Check memory usage with .element_size() * .numel()")

print("\n" + "="*60)
