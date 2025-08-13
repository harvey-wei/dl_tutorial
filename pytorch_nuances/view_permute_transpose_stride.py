import torch
import numpy as np



'''
In PyTorch, a tensor is contiguous when its logical index order matches its physical memory layout in row-major (C-style) order.
Put another way:
Contiguous → the elements are laid out in memory exactly as you’d expect if you iterated over the tensor in row-major fashion (last dimension changes fastest,
first dimension changes slowest). Iterate over the tensor in memory order in linear order and see whether logical order matches! We can tell whether the tensor is
contiguous or not by viewing the strides and shape. For contiguous tensors, the strides match the row-major order rules. For examples, 
for a 2D tensor, the strides are (width, 1) and the shape is (height, width).
Non-contiguous → the tensor is still backed by the same storage, but its strides tell PyTorch to jump around in memory when moving along dimensions.

Permute and transpose are both operations that change the order of dimensions in a tensor. Hence, they are both non-contiguous operations.
'''

def show_flatten_behavior(t: torch.Tensor):
    print(f"Shape: {t.shape}, Strides: {t.stride()}, Contiguous: {t.is_contiguous()}")
    print("Tensor:\n", t)
    
    # Logical row-major order (iterate over indices in row-major fashion)
    print("\nLogical order (row-major iteration):")
    logical_vals = []
    for idx in torch.cartesian_prod(*(torch.arange(s) for s in t.shape)):
        idx_tuple = tuple(idx.tolist())
        logical_vals.append(t[idx_tuple].item())
    print(logical_vals)
    
    # Physical memory order (flatten view)
    print("\nFlatten (physical memory order):")
    print(t.flatten().tolist())

    # Logical order after making contiguous
    print("\nContiguous flatten (matches logical order):")
    print(t.contiguous().flatten().tolist())
    print("-" * 60)


def show_logical_vs_physical(t: torch.Tensor):
    """
    Shows how the elements are ordered logically (as you see them when iterating rows/cols)
    vs physically (based on the underlying storage and strides).
    """
    print(f"Shape: {t.shape}, Strides: {t.stride()}, Contiguous: {t.is_contiguous()}")
    print("\nLogical order (row-major iteration over tensor dims):")

    # torch.cartesian_prod is a function that returns a Cartesian product (aka all possible combinations) of the given iterables
    # in row-major order!
    for idx in torch.cartesian_prod(*(torch.arange(s) for s in t.shape)):
        idx_tuple = tuple(idx.tolist())
        print(f"Index {idx_tuple} -> value {t[idx_tuple]}")


    print("\nPhysical storage order (based on .storage()):")
    start_pos = t.storage().data_ptr()
    storage_offset = t.storage_offset()
    for i in range(t.numel()):
        print(f"Index {i} -> value {t.storage()[i]}")
        print(f"Position {i} -> {start_pos + (i + storage_offset) * t.element_size()}")
    print(f"Start position: {start_pos}")
    print(f"End position: {start_pos + (t.numel() + storage_offset) * t.element_size()}")

    # Flatten is acutally reshape(-1). Hence, it gives logical order in memory and ven make a copy if input tensor is non-contiguous!
    # flat_vals = list(t.reshape(-1).storage())  # actual memory layout
    # reshpe amounts to .contiguous().view()
    flat_vals = list(t.flatten())
    print(f'Flattened values: {flat_vals}')


def print_tensor_info(tensor, name=""):
    """Print detailed information about a tensor including shape, strides, and memory layout."""
    print(f"\n{name}:" if name else "\nTensor:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Strides: {tensor.stride()}")
    print(f"  Contiguous: {tensor.is_contiguous()}")
    print(f"  Data pointer: {tensor.data_ptr()}")
    print(f"  Content:\n{tensor}")

def demo_physical_vs_logical():
    # Example: swap axes
    x = torch.arange(12).reshape(3, 4)
    print("Original tensor:")
    show_logical_vs_physical(x)

    y = x.permute(1, 0)
    print("\nAfter permute(1, 0):")
    show_logical_vs_physical(y)

def demo_flatten_behavior():
    x = torch.arange(12).reshape(3, 4)
    print("Original tensor:")
    show_flatten_behavior(x)

    y = x.permute(1, 0)
    print("\nAfter permute(1, 0):")
    show_flatten_behavior(y)

    # contiguous return a copy which makes the  logical order match the physical order
    z = x.contiguous()
    print("\nAfter contiguous():")
    show_flatten_behavior(z)


def demonstrate_view_operations():
    """Demonstrate how view operations affect strides."""
    print("=" * 60)
    print("VIEW OPERATIONS AND STRIDES")
    print("=" * 60)
    
    # Create a 2x3x4 tensor
    original = torch.arange(24).reshape(2, 3, 4)
    print_tensor_info(original, "Original tensor (2, 3, 4)")
    
    # View operation - changes shape but keeps same memory layout
    viewed = original.view(2, 12)
    print_tensor_info(viewed, "After view(2, 12)")
    
    # Another view operation
    viewed2 = original.view(6, 4)
    print_tensor_info(viewed2, "After view(6, 4)")
    
    # View with -1 (infer dimension)
    viewed3 = original.view(-1, 6)
    print_tensor_info(viewed3, "After view(-1, 6)")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: View operations maintain the same memory layout!")
    print("The data pointer remains the same, only shape and strides change.")
    print("View operation keeps contiguousness!")
    print("View operation does not copy the data!")
    print("=" * 60)

def demonstrate_permute_operations():
    """Demonstrate how permute operations affect strides."""
    print("\n" + "=" * 60)
    print("PERMUTE OPERATIONS AND STRIDES")
    print("=" * 60)
    
    # Create a 2x3x4 tensor
    original = torch.arange(24).reshape(2, 3, 4)
    print_tensor_info(original, "Original tensor (2, 3, 4)")
    
    # Permute dimensions
    permuted = original.permute(2, 0, 1)  # (4, 2, 3)
    print_tensor_info(permuted, "After permute(2, 0, 1) -> (4, 2, 3)")
    
    # Another permutation
    permuted2 = original.permute(1, 2, 0)  # (3, 4, 2)
    print_tensor_info(permuted2, "After permute(1, 2, 0) -> (3, 4, 2)")
    
    # Permute back to original order
    permuted_back = permuted.permute(1, 2, 0)
    print_tensor_info(permuted_back, "After permute back (1, 2, 0)")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Permute changes the order of dimensions!")
    print("This affects strides but keeps the same data pointer.")
    print("The tensor becomes non-contiguous after permutation.")
    print("=" * 60)

def demonstrate_transpose_operations():
    """Demonstrate how transpose operations affect strides."""
    print("\n" + "=" * 60)
    print("TRANSPOSE OPERATIONS AND STRIDES")
    print("=" * 60)
    
    # Create a 2x3 tensor
    original = torch.arange(6).reshape(2, 3)
    print_tensor_info(original, "Original tensor (2, 3)")
    
    # Transpose
    transposed = original.transpose(0, 1)  # (3, 2)
    print_tensor_info(transposed, "After transpose(0, 1) -> (3, 2)")
    
    # Transpose back
    transposed_back = transposed.transpose(0, 1)
    print_tensor_info(transposed_back, "After transpose back (0, 1)")
    
    # For 3D tensor
    original_3d = torch.arange(24).reshape(2, 3, 4)
    print_tensor_info(original_3d, "Original 3D tensor (2, 3, 4)")
    
    # Transpose first two dimensions
    transposed_3d = original_3d.transpose(0, 1)  # (3, 2, 4)
    print_tensor_info(transposed_3d, "After transpose(0, 1) -> (3, 2, 4)")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Transpose swaps two dimensions!")
    print("Similar to permute but specifically for 2 dimensions.")
    print("=" * 60)

def demonstrate_reshape_operations():
    """Demonstrate how reshape operations affect strides."""
    print("\n" + "=" * 60)
    print("RESHAPE OPERATIONS AND STRIDES")
    print("=" * 60)
    
    # Create a 2x3x4 tensor
    original = torch.arange(24).reshape(2, 3, 4)
    print_tensor_info(original, "Original tensor (2, 3, 4)")
    
    # Reshape to compatible shape (same as view)
    reshaped = original.reshape(2, 12)
    print_tensor_info(reshaped, "After reshape(2, 12)")
    
    # Reshape to incompatible shape (requires copy)
    # This will fail because 24 elements can't be reshaped to 5x5=25
    try:
        reshaped_incompatible = original.reshape(5, 5)
        print_tensor_info(reshaped_incompatible, "After reshape(5, 5)")
    except RuntimeError as e:
        print(f"Error: {e}")
    
    # Reshape to compatible shape that requires copy
    # Create a non-contiguous tensor first
    permuted = original.permute(2, 0, 1)  # (4, 2, 3)
    print_tensor_info(permuted, "Non-contiguous tensor after permute")
    
    # Reshape non-contiguous tensor
    reshaped_non_contiguous = permuted.reshape(2, 12)
    print_tensor_info(reshaped_non_contiguous, "After reshape(2, 12) on non-contiguous tensor")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: Reshape can work like view OR create a copy!")
    print("If tensor is contiguous and shape is compatible -> view-like behavior")
    print("If tensor is non-contiguous -> creates a copy with new memory layout")
    print("=" * 60)

def demonstrate_memory_efficiency():
    """Demonstrate memory efficiency implications."""
    print("\n" + "=" * 60)
    print("MEMORY EFFICIENCY IMPLICATIONS")
    print("=" * 60)
    
    # Create a large tensor
    large_tensor = torch.randn(1000, 1000)
    print(f"Original tensor size: {large_tensor.numel()} elements")
    print(f"Memory usage: {large_tensor.element_size() * large_tensor.numel()} bytes")
    
    # View operation (memory efficient)
    viewed = large_tensor.view(1000, 1000)
    print(f"After view - Same memory pointer: {viewed.data_ptr() == large_tensor.data_ptr()}")
    
    # Permute operation (memory efficient, but non-contiguous)
    permuted = large_tensor.permute(1, 0)
    print(f"After permute - Same memory pointer: {permuted.data_ptr() == large_tensor.data_ptr()}")
    print(f"Permuted is contiguous: {permuted.is_contiguous()}")
    
    # Reshape after permute (creates copy)
    reshaped = permuted.reshape(1000, 1000)
    print(f"After reshape on non-contiguous - Same memory pointer: {reshaped.data_ptr() == large_tensor.data_ptr()}")
    
    print("\n" + "=" * 60)
    print("MEMORY EFFICIENCY SUMMARY:")
    print("- view(): Always memory efficient (no copy)")
    print("- permute(): Memory efficient but makes tensor non-contiguous")
    print("- transpose(): Same as permute for 2 dimensions")
    print("- reshape(): Memory efficient if contiguous, creates copy if not")
    print("=" * 60)

def demonstrate_contiguous_operations():
    """Demonstrate how to make tensors contiguous."""
    print("\n" + "=" * 60)
    print("MAKING TENSORS CONTIGUOUS")
    print("=" * 60)
    
    # Create a non-contiguous tensor
    original = torch.arange(24).reshape(2, 3, 4)
    permuted = original.permute(2, 0, 1)
    print_tensor_info(permuted, "Non-contiguous tensor")
    
    # Make it contiguous
    contiguous = permuted.contiguous()
    print_tensor_info(contiguous, "After .contiguous()")
    
    # Compare memory pointers
    print(f"Same memory pointer: {contiguous.data_ptr() == permuted.data_ptr()}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: .contiguous() creates a copy if needed!")
    print("Only creates a copy if the tensor is not already contiguous.")
    print("=" * 60)

def demonstrate_practical_examples():
    """Show practical examples of when these operations are useful."""
    print("\n" + "=" * 60)
    print("PRACTICAL EXAMPLES")
    print("=" * 60)
    
    # Example 1: Image processing
    print("Example 1: Image processing")
    # Simulate an image: (batch, channels, height, width)
    image = torch.randn(1, 3, 32, 32)
    print(f"Image shape: {image.shape}")
    
    # Convert to (batch, height, width, channels) for some operations
    image_nhwc = image.permute(0, 2, 3, 1)
    print(f"After permute to NHWC: {image_nhwc.shape}")
    
    # Flatten for linear layer
    flattened = image.view(1, -1)
    print(f"Flattened for linear layer: {flattened.shape}")
    
    # Example 2: Matrix operations
    print("\nExample 2: Matrix operations")
    matrix = torch.randn(4, 5)
    print(f"Matrix shape: {matrix.shape}")
    
    # Transpose for matrix multiplication
    matrix_t = matrix.transpose(0, 1)
    print(f"Transposed matrix: {matrix_t.shape}")
    
    # Example 3: Reshaping for different operations
    print("\nExample 3: Reshaping for different operations")
    data = torch.randn(10, 20, 30)
    print(f"Original data: {data.shape}")
    
    # Reshape for batch processing
    batch_reshaped = data.view(-1, 30)
    print(f"Reshaped for batch processing: {batch_reshaped.shape}")
    
    # Reshape back
    back_to_original = batch_reshaped.view(10, 20, 30)
    print(f"Reshaped back: {back_to_original.shape}")
    
    print("\n" + "=" * 60)
    print("PRACTICAL INSIGHTS:")
    print("- Use view() when you want to change shape without copying")
    print("- Use permute()/transpose() when you need to reorder dimensions")
    print("- Use reshape() when you're unsure about tensor contiguity")
    print("- Always check .is_contiguous() for performance-critical code")
    print("=" * 60)

if __name__ == "__main__":
    print("PYTORCH TENSOR OPERATIONS: VIEW, PERMUTE, TRANSPOSE, RESHAPE")
    print("Understanding how these operations affect tensor strides and memory layout")
    
    demonstrate_view_operations()
    demonstrate_permute_operations()
    demonstrate_transpose_operations()
    demonstrate_reshape_operations()
    demonstrate_memory_efficiency()
    demonstrate_contiguous_operations()
    demonstrate_practical_examples()
    demo_physical_vs_logical()
    demo_flatten_behavior()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("1. view(): Changes shape, keeps same memory layout, always efficient")
    print("2. permute(): Reorders dimensions, may make tensor non-contiguous")
    print("3. transpose(): Swaps two dimensions (subset of permute)")
    print("4. reshape(): Changes shape, may create copy if non-contiguous. reshape() is basically contiguous().view() under the hood,")
    print("5. .contiguous(): Ensures contiguous memory layout, may create copy")
    print("=" * 60)
