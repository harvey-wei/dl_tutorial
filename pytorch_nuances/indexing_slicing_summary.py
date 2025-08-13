import torch
from tabulate import tabulate

print("="*80)
print("PYTORCH INDEXING: RESULTING SHAPES & COPYING BEHAVIOR TABLES")
print("="*80)

print("\n1. RESULTING SHAPES TABLE")
print("-" * 50)

# Create sample tensor for examples
sample_tensor = torch.randn(3, 4, 5)  # shape: (3, 4, 5)

# Define the table data
shape_data = [
    ["Basic Indexing", "Removes indexed dimensions", 
     f"tensor[0] → {sample_tensor[0].shape}", 
     f"tensor[0, 1] → {sample_tensor[0, 1].shape}",
     f"tensor[0, 1, 2] → {sample_tensor[0, 1, 2].shape}"],
    
    ["Slicing", "Keeps sliced dimensions", 
     f"tensor[0:1] → {sample_tensor[0:1].shape}", 
     f"tensor[0:1, 1:2] → {sample_tensor[0:1, 1:2].shape}",
     f"tensor[:, 1:2, 2:3] → {sample_tensor[:, 1:2, 2:3].shape}"],
    
    ["Advanced Indexing", "Shape depends on index tensor shapes", 
     f"tensor[indices] → indices.shape + remaining_dims", 
     f"tensor[indices, :, :] → indices.shape + remaining_dims",
     f"tensor[indices1, indices2, :] → broadcasted_shape + remaining_dims"],
    
    ["Boolean Indexing", "Returns flattened array", 
     f"tensor[mask] → (N,) where N = sum(mask)", 
     f"tensor[mask_2d] → (M,) where M = sum(mask_2d)",
     f"tensor[mask_3d] → (P,) where P = sum(mask_3d)"]
]

# Create the table
shape_headers = ["Method", "Shape Rule", "Example 1", "Example 2", "Example 3"]
shape_table = tabulate(shape_data, headers=shape_headers, tablefmt="grid")

print(shape_table)

print("\n2. COPYING BEHAVIOR TABLE")
print("-" * 50)

# Create sample tensor for memory demonstration
original = torch.randn(3, 4)

# Test different indexing methods
basic_result = original[0, :]
slice_result = original[0:2, 1:3]
indices = torch.tensor([0, 2])
advanced_result = original[indices, :]
mask = original > 0
boolean_result = original[mask]

# Define the copying behavior table data
copying_data = [
    ["Basic Indexing", "❌ No (Views)", "✅ High", 
     f"{original.data_ptr() == basic_result.data_ptr()}", 
     "Modifying result affects original"],
    
    ["Slicing", "❌ No (Views)", "✅ High", 
     f"{original.data_ptr() == slice_result.data_ptr()}", 
     "Modifying result affects original"],
    
    ["Advanced Indexing", "✅ Yes (Copies)", "❌ Lower", 
     f"{original.data_ptr() == advanced_result.data_ptr()}", 
     "Modifying result does NOT affect original"],
    
    ["Boolean Indexing", "✅ Yes (Copies)", "❌ Lower", 
     f"{original.data_ptr() == boolean_result.data_ptr()}", 
     "Modifying result does NOT affect original"]
]

# Create the table
copying_headers = ["Method", "Copying", "Memory Efficiency", "Same Memory Address", "Behavior"]
copying_table = tabulate(copying_data, headers=copying_headers, tablefmt="grid")

print(copying_table)

print("\n3. DETAILED EXAMPLES")
print("-" * 50)

print("\n3.1 Basic Indexing Examples:")
tensor_3d = torch.randn(2, 3, 4)
print(f"Original tensor: {tensor_3d.shape}")

basic_examples = [
    ["tensor[0]", f"{tensor_3d[0].shape}", "Remove first dimension"],
    ["tensor[0, 1]", f"{tensor_3d[0, 1].shape}", "Remove first two dimensions"],
    ["tensor[0, 1, 2]", f"{tensor_3d[0, 1, 2].shape}", "Remove all dimensions (scalar)"],
    ["tensor[:, 1]", f"{tensor_3d[:, 1].shape}", "Remove second dimension"],
    ["tensor[:, :, 2]", f"{tensor_3d[:, :, 2].shape}", "Remove third dimension"]
]

basic_headers = ["Operation", "Result Shape", "Description"]
basic_table = tabulate(basic_examples, headers=basic_headers, tablefmt="grid")
print(basic_table)

print("\n3.2 Slicing Examples:")
slicing_examples = [
    ["tensor[0:1]", f"{tensor_3d[0:1].shape}", "Keep first dimension"],
    ["tensor[0:1, 1:2]", f"{tensor_3d[0:1, 1:2].shape}", "Keep first two dimensions"],
    ["tensor[:, 1:2, 2:3]", f"{tensor_3d[:, 1:2, 2:3].shape}", "Keep all dimensions"],
    ["tensor[::2]", f"{tensor_3d[::2].shape}", "Every other element in first dim"],
    ["tensor[:, ::2]", f"{tensor_3d[:, ::2].shape}", "Every other element in second dim"]
]

slicing_headers = ["Operation", "Result Shape", "Description"]
slicing_table = tabulate(slicing_examples, headers=slicing_headers, tablefmt="grid")
print(slicing_table)

print("\n3.3 Advanced Indexing Examples:")
# Create indices for examples
row_indices = torch.tensor([0, 1])
col_indices = torch.tensor([1, 2])
depth_indices = torch.tensor([0, 3])

advanced_examples = [
    ["tensor[row_indices]", f"{tensor_3d[row_indices].shape}", f"Select rows {row_indices.tolist()} (shape: {row_indices.shape})"],
    ["tensor[row_indices, :, :]", f"{tensor_3d[row_indices, :, :].shape}", f"Select rows {row_indices.tolist()} (shape: {row_indices.shape})"],
    ["tensor[:, col_indices, :]", f"{tensor_3d[:, col_indices, :].shape}", f"Select cols {col_indices.tolist()} (shape: {col_indices.shape})"],
    ["tensor[row_indices, col_indices, :]", f"{tensor_3d[row_indices, col_indices, :].shape}", f"Broadcasted shape: {torch.broadcast_shapes(row_indices.shape, col_indices.shape)}"],
    ["tensor[row_indices, col_indices, depth_indices]", f"{tensor_3d[row_indices, col_indices, depth_indices].shape}", f"Broadcasted shape: {torch.broadcast_shapes(row_indices.shape, col_indices.shape, depth_indices.shape)}"]
]

advanced_headers = ["Operation", "Result Shape", "Description"]
advanced_table = tabulate(advanced_examples, headers=advanced_headers, tablefmt="grid")
print(advanced_table)

print("\n4. MEMORY ADDRESS COMPARISON")
print("-" * 50)

# Create a simple tensor for memory comparison
test_tensor = torch.randn(3, 4)
print(f"Original tensor: {test_tensor.shape}, Memory: {test_tensor.data_ptr()}")

memory_comparison = [
    ["Basic Indexing", f"{test_tensor[0, :].data_ptr()}", f"{test_tensor.data_ptr() == test_tensor[0, :].data_ptr()}"],
    ["Slicing", f"{test_tensor[0:2, 1:3].data_ptr()}", f"{test_tensor.data_ptr() == test_tensor[0:2, 1:3].data_ptr()}"],
    ["Advanced Indexing", f"{test_tensor[torch.tensor([0, 2]), :].data_ptr()}", f"{test_tensor.data_ptr() == test_tensor[torch.tensor([0, 2]), :].data_ptr()}"],
    ["Boolean Indexing", f"{test_tensor[test_tensor > 0].data_ptr()}", f"{test_tensor.data_ptr() == test_tensor[test_tensor > 0].data_ptr()}"]
]

memory_headers = ["Method", "Result Memory Address", "Same as Original?"]
memory_table = tabulate(memory_comparison, headers=memory_headers, tablefmt="grid")
print(memory_table)

print("\n5. PERFORMANCE SUMMARY")
print("-" * 50)

performance_summary = [
    ["Basic Indexing", "✅ Fast", "✅ Memory Efficient", "✅ Creates Views", "✅ Modifies Original"],
    ["Slicing", "✅ Fast", "✅ Memory Efficient", "✅ Creates Views", "✅ Modifies Original"],
    ["Advanced Indexing", "❌ Slower", "❌ Uses More Memory", "❌ Creates Copies", "❌ Doesn't Modify Original"],
    ["Boolean Indexing", "❌ Slower", "❌ Uses More Memory", "❌ Creates Copies", "❌ Doesn't Modify Original"]
]

performance_headers = ["Method", "Speed", "Memory Efficiency", "Copying Behavior", "Modification Behavior"]
performance_table = tabulate(performance_summary, headers=performance_headers, tablefmt="grid")
print(performance_table)

print("\n6. BEST PRACTICES")
print("-" * 50)

best_practices = [
    ["Use Basic Indexing", "When you need single elements or partial dimensions", "Fast and memory efficient"],
    ["Use Slicing", "When you need ranges of elements", "Fast and memory efficient"],
    ["Use Advanced Indexing", "When you need non-contiguous elements", "Slower but flexible"],
    ["Use Boolean Indexing", "When you need conditional selection", "Slower but expressive"],
    ["Avoid in Loops", "Advanced/Boolean indexing in loops", "Can be very slow"],
    ["Use .clone()", "When you explicitly need a copy", "Clear intent"],
    ["Check Shapes", "Always verify resulting shapes", "Prevents bugs"],
    ["Check Memory", "Use .data_ptr() to verify copying", "Understand behavior"]
]

practices_headers = ["Practice", "When to Use", "Why"]
practices_table = tabulate(best_practices, headers=practices_headers, tablefmt="grid")
print(practices_table)

print("\n7. ADVANCED INDEXING SHAPE RULES")
print("-" * 50)

# Create examples to demonstrate the shape rules
tensor_3d = torch.randn(3, 4, 5)

# Single index tensor (no broadcasting needed)
single_indices = torch.tensor([0, 2])  # shape: (2,)
result_single = tensor_3d[single_indices, :, :]  # shape: (2, 4, 5)

# Multiple index tensors (broadcasting needed)
indices1 = torch.tensor([0, 1])  # shape: (2,)
indices2 = torch.tensor([[0, 1], [2, 0]])  # shape: (2, 2)
result_multi = tensor_3d[indices1, indices2, :]  # shape: (2, 2, 5)

advanced_shape_examples = [
    ["Single index tensor", f"tensor[indices] → {result_single.shape}", f"indices.shape: {single_indices.shape}", "No broadcasting needed"],
    ["Multiple index tensors", f"tensor[indices1, indices2] → {result_multi.shape}", f"broadcasted: {torch.broadcast_shapes(indices1.shape, indices2.shape)}", "Broadcasting applied"],
    ["Index + slice", f"tensor[indices, :, :] → {result_single.shape}", f"indices.shape: {single_indices.shape}", "Slice dimensions preserved"],
    ["All indices", f"tensor[indices1, indices2, indices3]", "broadcasted_shape", "All dimensions from indices"]
]

advanced_shape_headers = ["Case", "Result Shape", "Index Shape", "Notes"]
advanced_shape_table = tabulate(advanced_shape_examples, headers=advanced_shape_headers, tablefmt="grid")
print(advanced_shape_table)

print("\nKey Rule: When no broadcasting is needed, the result shape is:")
print("index_tensor.shape + remaining_dimensions")
print("Example: tensor(3,4,5)[indices(2,), :, :] → (2, 4, 5)")

print("\n8. QUICK REFERENCE")
print("-" * 50)

quick_ref = [
    ["Basic Indexing", "tensor[i, j, k]", "Views", "Removes dimensions"],
    ["Slicing", "tensor[i:j:k]", "Views", "Keeps dimensions"],
    ["Advanced Indexing", "tensor[indices]", "Copies", "Index-dependent shape"],
    ["Boolean Indexing", "tensor[mask]", "Copies", "Flattened result"]
]

ref_headers = ["Method", "Syntax", "Copying", "Shape Rule"]
ref_table = tabulate(quick_ref, headers=ref_headers, tablefmt="simple")
print(ref_table)

print("\n" + "="*80)
print("END OF TABLES")
print("="*80)
