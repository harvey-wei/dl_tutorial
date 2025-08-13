import torch
import torch.nn.functional as F

def print_tensor_info(tensor, name=""):
    """Print detailed information about a tensor."""
    print(f"\n{name}:" if name else "\nTensor:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Content:\n{tensor}")

def print_tensor_info_with_memory(tensor, name="", original_tensor=None):
    """Print detailed information about a tensor including memory layout."""
    print(f"\n{name}:" if name else "\nTensor:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Strides: {tensor.stride()}")
    print(f"  Contiguous: {tensor.is_contiguous()}")
    print(f"  Data pointer: {tensor.data_ptr()}")
    if original_tensor is not None:
        is_copy = tensor.data_ptr() != original_tensor.data_ptr()
        print(f"  Is copy: {is_copy}")
        if is_copy:
            print(f"  Memory efficiency: Creates new memory allocation")
        else:
            print(f"  Memory efficiency: Shares memory with original")
    print(f"  Content:\n{tensor}")

def demonstrate_torch_where():
    """Demonstrate torch.where operation."""
    print("=" * 60)
    print("TORCH.WHERE OPERATION")
    print("=" * 60)
    
    # Example 1: Basic conditional selection
    print("Example 1: Basic conditional selection")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[10, 20, 30], [40, 50, 60]])
    condition = torch.tensor([[True, False, True], [False, True, False]])
    
    print_tensor_info_with_memory(x, "Tensor x")
    print_tensor_info_with_memory(y, "Tensor y")
    print_tensor_info_with_memory(condition, "Condition")
    
    result = torch.where(condition, x, y)
    print_tensor_info_with_memory(result, "torch.where(condition, x, y)", x)
    
    # Example 2: Broadcasting with where
    print("\nExample 2: Broadcasting with where")
    x = torch.tensor([1, 2, 3, 4, 5])
    condition = x > 3
    print_tensor_info_with_memory(x, "Tensor x")
    print_tensor_info_with_memory(condition, "Condition (x > 3)")
    
    result = torch.where(condition, x, torch.tensor(0))
    print_tensor_info_with_memory(result, "torch.where(condition, x, 0)", x)
    
    # Example 3: Complex condition
    print("\nExample 3: Complex condition")
    x = torch.randn(3, 4)
    print_tensor_info_with_memory(x, "Random tensor x")
    
    # Replace negative values with 0
    result = torch.where(x < 0, torch.tensor(0.0), x)
    print_tensor_info_with_memory(result, "Replace negative values with 0", x)
    
    # Example 4: Three-argument where (condition, x, y)
    print("\nExample 4: Three-argument where")
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[5, 6], [7, 8]])
    condition = torch.tensor([[True, False], [False, True]])
    
    print_tensor_info_with_memory(x, "Tensor x")
    print_tensor_info_with_memory(y, "Tensor y")
    print_tensor_info_with_memory(condition, "Condition")
    
    result = torch.where(condition, x, y)
    print_tensor_info_with_memory(result, "torch.where(condition, x, y)", x)
    
    print("\n" + "=" * 60)
    print("TORCH.WHERE SUMMARY:")
    print("- torch.where(condition, x, y): Select from x where condition is True, y where False")
    print("- Supports broadcasting")
    print("- Useful for conditional element-wise operations")
    print("- MEMORY: Always creates a new tensor (copy operation)")
    print("=" * 60)

def demonstrate_masked_scatter():
    """Demonstrate masked_scatter operation."""
    print("\n" + "=" * 60)
    print("MASKED_SCATTER OPERATION")
    print("=" * 60)
    
    # Example 1: Basic masked_scatter
    print("Example 1: Basic masked_scatter")
    x = torch.zeros(3, 4).float()
    mask = torch.tensor([[True, False, True, False],
                        [False, True, False, True],
                        [True, True, False, False]])
    source = torch.tensor([1, 2, 3, 4, 5, 6]).float()
    
    print_tensor_info_with_memory(x, "Target tensor (zeros)")
    print_tensor_info_with_memory(mask, "Mask")
    print_tensor_info_with_memory(source, "Source tensor")
    
    original_ptr = x.data_ptr()
    result = x.masked_scatter_(mask, source)
    print_tensor_info_with_memory(result, "After masked_scatter_(mask, source)", x)
    print(f"  In-place operation: {result.data_ptr() == original_ptr}")
    
    # Example 2: Scatter specific values
    print("\nExample 2: Scatter specific values")
    x = torch.ones(2, 3).float() * 10
    mask = torch.tensor([[True, False, True],
                        [False, True, False]])
    source = torch.tensor([100, 200, 300]).float()
    
    print_tensor_info_with_memory(x, "Target tensor (all 10s)")
    print_tensor_info_with_memory(mask, "Mask")
    print_tensor_info_with_memory(source, "Source tensor")
    
    original_ptr = x.data_ptr()
    result = x.masked_scatter_(mask, source)
    print_tensor_info_with_memory(result, "After masked_scatter_(mask, source)", x)
    print(f"  In-place operation: {result.data_ptr() == original_ptr}")
    
    # Example 3: Scatter with different shapes
    print("\nExample 3: Scatter with different shapes")
    x = torch.zeros(4, 3).float()
    mask = torch.tensor([[True, False, True],
                        [False, True, False],
                        [True, True, False],
                        [False, False, True]])
    source = torch.arange(1, 8).float()  # 7 elements
    
    print_tensor_info_with_memory(x, "Target tensor")
    print_tensor_info_with_memory(mask, "Mask")
    print_tensor_info_with_memory(source, "Source tensor")
    
    original_ptr = x.data_ptr()
    result = x.masked_scatter_(mask, source)
    print_tensor_info_with_memory(result, "After masked_scatter_(mask, source)", x)
    print(f"  In-place operation: {result.data_ptr() == original_ptr}")
    
    print("\n" + "=" * 60)
    print("MASKED_SCATTER SUMMARY:")
    print("- masked_scatter_(mask, source): In-place operation")
    print("- Scatters elements from source into positions where mask is True")
    print("- Source tensor must have at least as many elements as True values in mask")
    print("- Useful for selective updates")
    print("- MEMORY: In-place operation, no copy (modifies original tensor)")
    '''
    for torch.masked_scatter_
    mask must be the same shape as target.

    source.numel() must equal mask.sum().

    _ at the end of .masked_scatter_ means in-place operation.

    Without _, it returns a new tensor:
    '''
    print("=" * 60)

def demonstrate_prod_cumsum_cumprod():
    """Demonstrate prod, cumsum, and cumprod operations."""
    print("\n" + "=" * 60)
    print("PROD, CUMSUM, CUMPROD OPERATIONS")
    print("=" * 60)
    
    # Example 1: Basic prod operation
    print("Example 1: Basic prod operation")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print_tensor_info_with_memory(x, "Original tensor")
    
    # Product of all elements
    prod_all = x.prod()
    print_tensor_info_with_memory(prod_all, "prod() - product of all elements", x)
    
    # Product along dimension 0
    prod_dim0 = x.prod(dim=0)
    print_tensor_info_with_memory(prod_dim0, "prod(dim=0) - product along rows", x)
    
    # Product along dimension 1
    prod_dim1 = x.prod(dim=1)
    print_tensor_info_with_memory(prod_dim1, "prod(dim=1) - product along columns", x)
    
    # Example 2: cumsum operation
    print("\nExample 2: cumsum operation")
    x = torch.tensor([1, 2, 3, 4, 5])
    print_tensor_info_with_memory(x, "Original tensor")
    
    # Cumulative sum
    cumsum_result = x.cumsum(dim=0)
    print_tensor_info_with_memory(cumsum_result, "cumsum(dim=0) - cumulative sum", x)
    
    # 2D cumsum
    x_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print_tensor_info_with_memory(x_2d, "2D tensor")
    
    cumsum_dim0 = x_2d.cumsum(dim=0)
    print_tensor_info_with_memory(cumsum_dim0, "cumsum(dim=0) - cumulative sum along rows", x_2d)
    
    cumsum_dim1 = x_2d.cumsum(dim=1)
    print_tensor_info_with_memory(cumsum_dim1, "cumsum(dim=1) - cumulative sum along columns", x_2d)
    
    # Example 3: cumprod operation
    print("\nExample 3: cumprod operation")
    x = torch.tensor([1, 2, 3, 4, 5])
    print_tensor_info_with_memory(x, "Original tensor")
    
    # Cumulative product
    cumprod_result = x.cumprod(dim=0)
    print_tensor_info_with_memory(cumprod_result, "cumprod(dim=0) - cumulative product", x)
    
    # 2D cumprod
    x_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print_tensor_info_with_memory(x_2d, "2D tensor")
    
    cumprod_dim0 = x_2d.cumprod(dim=0)
    print_tensor_info_with_memory(cumprod_dim0, "cumprod(dim=0) - cumulative product along rows", x_2d)
    
    cumprod_dim1 = x_2d.cumprod(dim=1)
    print_tensor_info_with_memory(cumprod_dim1, "cumprod(dim=1) - cumulative product along columns", x_2d)
    
    # Example 4: Practical applications
    print("\nExample 4: Practical applications")
    
    # Running average using cumsum
    data = torch.tensor([10, 20, 30, 40, 50, 60])
    print_tensor_info_with_memory(data, "Data points")
    
    cumsum_data = data.cumsum(dim=0)
    print_tensor_info_with_memory(cumsum_data, "Cumulative sum")
    
    # Calculate running average
    indices = torch.arange(1, len(data) + 1, dtype=torch.float)
    running_avg = cumsum_data / indices
    print_tensor_info_with_memory(running_avg, "Running average")
    
    # Example 5: Factorial using cumprod
    print("\nExample 5: Factorial using cumprod")
    n = 5
    numbers = torch.arange(1, n + 1)
    print_tensor_info_with_memory(numbers, "Numbers 1 to 5")
    
    factorial = numbers.cumprod(dim=0)
    print_tensor_info_with_memory(factorial, "Factorial sequence (1!, 2!, 3!, 4!, 5!)")
    
    # Example 6: Memory efficiency comparison
    print("\nExample 6: Memory efficiency comparison")
    large_tensor = torch.randn(1000, 1000)
    original_ptr = large_tensor.data_ptr()
    
    # Test prod
    result_prod = large_tensor.prod()
    print(f"prod() - Same memory pointer: {result_prod.data_ptr() == original_ptr}")
    
    # Test cumsum
    result_cumsum = large_tensor.cumsum(dim=0)
    print(f"cumsum() - Same memory pointer: {result_cumsum.data_ptr() == original_ptr}")
    
    # Test cumprod
    result_cumprod = large_tensor.cumprod(dim=1)
    print(f"cumprod() - Same memory pointer: {result_cumprod.data_ptr() == original_ptr}")
    
    print("\n" + "=" * 60)
    print("PROD, CUMSUM, CUMPROD SUMMARY:")
    print("- prod(): Product of elements (scalar or along dimension)")
    print("- cumsum(): Cumulative sum along specified dimension")
    print("- cumprod(): Cumulative product along specified dimension")
    print("- MEMORY: All create new tensors (copy operations)")
    print("- Useful for running statistics, factorial calculations, etc.")
    print("=" * 60)


def demonstrate_inplace_operation():
    # inplace operation like .add_(), div_(), mul_(), sub_(), etc.
    # will return a new tensor
    x = torch.tensor([1, 2, 3, 4, 5]).float()
    x.add_(2)
    print(x)
    x.div_(2)
    print(x)

    print("\n" + "=" * 60)
    print("INPLACE OPERATION SUMMARY:")
    print("- inplace operation like .add_(), div_(), mul_(), sub_(), etc. modify the original tensor in place")
    print("- Useful for in-place operations")
    print("=" * 60)
def compare_inplace_vs_outofplace():
    x = torch.tensor([1, 2, 3, 4, 5]).float()
    print("Original x:", x, "storage_ptr:", x.data_ptr())

    # Out-of-place addition
    y = x.add(2)  # returns NEW tensor
    print("\n[Out-of-place] x.add(2):", y, "storage_ptr:", y.data_ptr())
    print("x after add():", x, "storage_ptr still:", x.data_ptr())  # unchanged

    # In-place addition
    z = x.add_(2)  # modifies x in-place
    print("\n[In-place] x.add_(2):", z, "storage_ptr:", z.data_ptr())
    print("x after add_():", x, "storage_ptr still:", x.data_ptr())  # same as before

def demonstrate_gather():
    """Demonstrate gather operation."""
    print("\n" + "=" * 60)
    print("GATHER OPERATION")
    print("=" * 60)
    
    # Example 1: Basic gather
    print("Example 1: Basic gather")
    x = torch.tensor([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    indices = torch.tensor([[0, 1, 2],
                           [1, 2, 0],
                           [2, 0, 1]])
    
    print_tensor_info_with_memory(x, "Source tensor")
    print_tensor_info_with_memory(indices, "Indices tensor")
    
    result = torch.gather(x, dim=0, index=indices)
    print_tensor_info_with_memory(result, "gather(x, dim=0, index=indices)", x)
    
    # Example 2: Gather along different dimension
    print("\nExample 2: Gather along dimension 1")
    result = torch.gather(x, dim=1, index=indices)
    print_tensor_info_with_memory(result, "gather(x, dim=1, index=indices)", x)
    
    # Example 3: Gather with broadcasting
    print("\nExample 3: Gather with broadcasting")
    x = torch.tensor([[1, 2, 3, 4],
                     [5, 6, 7, 8]])
    indices = torch.tensor([[0, 2, 1, 3],
                           [1, 3, 0, 2]])
    
    print_tensor_info_with_memory(x, "Source tensor")
    print_tensor_info_with_memory(indices, "Indices tensor")
    
    result = torch.gather(x, dim=1, index=indices)
    print_tensor_info_with_memory(result, "gather(x, dim=1, index=indices)", x)
    
    # Example 4: Gather for max values
    print("\nExample 4: Gather for max values")
    x = torch.randn(3, 4)
    print_tensor_info_with_memory(x, "Random tensor")
    
    # Find max values along dimension 1
    max_values, max_indices = torch.max(x, dim=1, keepdim=True)
    print_tensor_info_with_memory(max_indices, "Max indices")
    
    # Gather the max values
    gathered_max = torch.gather(x, dim=1, index=max_indices)
    print_tensor_info_with_memory(gathered_max, "Gathered max values", x)
    print_tensor_info_with_memory(max_values, "Max values (for comparison)")
    
    print("\n" + "=" * 60)
    print("GATHER SUMMARY:")
    print("- gather(input, dim, index): Collects values from input along specified dimension")
    print("- Index tensor must have same shape as output")
    print("- Useful for collecting specific elements based on indices")
    print("- Often used with max/min operations")
    print("- MEMORY: Always creates a new tensor (copy operation)")
    print("=" * 60)

def demonstrate_pad():
    """Demonstrate padding operations."""
    '''
    PyTorch’s pad applies pads from last dimension inward.
    That’s why for [N, C, H, W] you give (W_left, W_right, H_top, H_bottom) — batch and channel dims aren’t padded.
    '''
    print("\n" + "=" * 60)
    print("PADDING OPERATIONS")
    print("=" * 60)
    
    # Example 1: Basic padding
    print("Example 1: Basic padding")
    x = torch.tensor([[1, 2],
                     [3, 4]])
    print_tensor_info_with_memory(x, "Original tensor")
    
    # # Pad with zeros: (left, right, top, bottom)
    # padded = F.pad(x, pad=(1, 1, 1, 1), mode='constant', value=0)
    # print_tensor_info_with_memory(padded, "Padded with zeros (1, 1, 1, 1)", x)
    
    # # Example 2: Different padding modes
    # print("\nExample 2: Different padding modes")
    # x = torch.tensor([[1, 2],
    #                  [3, 4]])
    # print_tensor_info_with_memory(x, "Original tensor")
    
    # # Reflect padding
    # padded_reflect = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
    # print_tensor_info_with_memory(padded_reflect, "Reflect padding", x)
    
    # # Replicate padding
    # padded_replicate = F.pad(x, pad=(1, 1, 1, 1), mode='replicate')
    # print_tensor_info_with_memory(padded_replicate, "Replicate padding", x)
    
    # # Example 3: Asymmetric padding
    # print("\nExample 3: Asymmetric padding")
    # x = torch.tensor([[1, 2, 3],
    #                  [4, 5, 6]])
    # print_tensor_info_with_memory(x, "Original tensor")
    
    # # Different padding on each side
    # padded_asym = F.pad(x, pad=(2, 1, 1, 2), mode='constant', value=0)
    # print_tensor_info_with_memory(padded_asym, "Asymmetric padding (2, 1, 1, 2)", x)
    
    # # Example 4: 3D tensor padding
    # print("\nExample 4: 3D tensor padding")
    # x = torch.tensor([[[1, 2], [3, 4]],
    #                  [[5, 6], [7, 8]]])
    # print_tensor_info_with_memory(x, "3D tensor")
    
    # # Pad 3D tensor: (left, right, top, bottom, front, back)
    # padded_3d = F.pad(x, pad=(1, 1, 1, 1, 1, 1), mode='constant', value=0)
    # print_tensor_info_with_memory(padded_3d, "3D padding (1, 1, 1, 1, 1, 1)", x)
    
    # # Example 5: Padding for convolution
    # print("\nExample 5: Padding for convolution")
    # x = torch.randn(1, 3, 32, 32)  # Batch, channels, height, width
    # print(f"Input shape: {x.shape}")
    
    # # Pad to maintain spatial dimensions after conv2d
    # padded_conv = F.pad(x, pad=(1, 1, 1, 1), mode='reflect')
    # print(f"Padded shape: {padded_conv.shape}")
    
    # # Apply convolution
    # conv = torch.nn.Conv2d(3, 6, kernel_size=3, padding=0)
    # output = conv(padded_conv)
    # print(f"Output shape: {output.shape}")

    # Example 6: Padding for sequence processing
    x = torch.arange(1, 10).float().reshape(1, 1, 3, 3)  # [N=1, C=1, H=3, W=3]
    print(x[0, 0])

    # pad 1 pixel left, 2 right, 3 top, 0 bottom
    y = F.pad(x, pad=(1, 2, 3, 0), mode="constant", value=0)
    print(y.shape)  # [1, 1, 6, 6]
    print(y[0, 0])
    
    print("\n" + "=" * 60)
    print("PADDING SUMMARY:")
    print("- F.pad(input, pad, mode, value): Add padding to tensor")
    print("- pad format: (left, right, top, bottom) for 2D, (left, right, top, bottom, front, back) for 3D")
    print("- Modes: 'constant', 'reflect', 'replicate', 'circular'")
    print("- Useful for maintaining spatial dimensions in convolutions")
    print("- MEMORY: Always creates a new tensor (copy operation)")
    print("=" * 60)

def demonstrate_memory_efficiency_comparison():
    """Demonstrate memory efficiency differences between operations."""
    print("\n" + "=" * 60)
    print("MEMORY EFFICIENCY COMPARISON")
    print("=" * 60)
    
    # Create a large tensor for demonstration
    large_tensor = torch.randn(1000, 1000)
    print(f"Original tensor size: {large_tensor.numel()} elements")
    print(f"Memory usage: {large_tensor.element_size() * large_tensor.numel()} bytes")
    original_ptr = large_tensor.data_ptr()
    
    # Test torch.where (copy operation)
    print("\n1. torch.where (COPY operation):")
    condition = large_tensor > 0
    result_where = torch.where(condition, large_tensor, torch.tensor(0.0))
    print(f"  Same memory pointer: {result_where.data_ptr() == original_ptr}")
    print(f"  Memory efficiency: {'Inefficient' if result_where.data_ptr() != original_ptr else 'Efficient'}")
    
    # Test masked_scatter (in-place operation)
    print("\n2. masked_scatter_ (NO COPY operation):")
    mask = large_tensor > 0
    source = torch.ones_like(large_tensor) * 0.1
    result_scatter = large_tensor.clone()  # Clone to avoid modifying original
    result_scatter.masked_scatter_(mask, source)
    print(f"  Same memory pointer: {result_scatter.data_ptr() == original_ptr}")
    print(f"  Memory efficiency: {'Efficient' if result_scatter.data_ptr() != original_ptr else 'Inefficient'}")
    
    # Test gather (copy operation)
    print("\n3. gather (COPY operation):")
    indices = torch.randint(0, 1000, (1000, 1000))
    result_gather = torch.gather(large_tensor, dim=1, index=indices)
    print(f"  Same memory pointer: {result_gather.data_ptr() == original_ptr}")
    print(f"  Memory efficiency: {'Inefficient' if result_gather.data_ptr() != original_ptr else 'Efficient'}")
    
    # Test pad (copy operation)
    print("\n4. F.pad (COPY operation):")
    result_pad = F.pad(large_tensor, pad=(1, 1, 1, 1), mode='constant', value=0)
    print(f"  Same memory pointer: {result_pad.data_ptr() == original_ptr}")
    print(f"  Memory efficiency: {'Inefficient' if result_pad.data_ptr() != original_ptr else 'Efficient'}")
    
    print("\n" + "=" * 60)
    print("MEMORY EFFICIENCY INSIGHTS:")
    print("- In-place operations (masked_scatter_) are most memory efficient")
    print("- Copy operations (where, gather, pad) use more memory but are safer")
    print("- Choose based on whether you need to preserve the original tensor")
    print("=" * 60)

def demonstrate_practical_examples():
    """Show practical examples combining these operations."""
    print("\n" + "=" * 60)
    print("PRACTICAL EXAMPLES")
    print("=" * 60)
    
    # Example 1: Image processing with conditional operations
    print("Example 1: Image processing with conditional operations")
    # Simulate an image with noise
    image = torch.randn(3, 4)
    print_tensor_info(image, "Original image")
    
    # Remove noise below threshold
    threshold = 0.5
    denoised = torch.where(torch.abs(image) > threshold, image, torch.tensor(0.0))
    print_tensor_info(denoised, "Denoised image (threshold > 0.5)")
    
    # Example 2: Selective updates with masked_scatter
    print("\nExample 2: Selective updates with masked_scatter")
    weights = torch.randn(4, 4)
    mask = weights > 0  # Only positive weights
    updates = torch.ones_like(weights) * 0.1
    
    print_tensor_info(weights, "Original weights")
    print_tensor_info(mask, "Positive weights mask")
    
    weights.masked_scatter_(mask, updates)
    print_tensor_info(weights, "Updated weights (positive values set to 0.1)")
    
    # Example 3: Gathering top-k values
    print("\nExample 3: Gathering top-k values")
    scores = torch.tensor([[0.1, 0.8, 0.3, 0.9],
                          [0.7, 0.2, 0.6, 0.4]])
    k = 2
    
    # Get top-k indices
    top_k_values, top_k_indices = torch.topk(scores, k, dim=1)
    print_tensor_info(scores, "Scores")
    print_tensor_info(top_k_indices, f"Top-{k} indices")
    
    # Gather top-k values
    gathered_top_k = torch.gather(scores, dim=1, index=top_k_indices)
    print_tensor_info(gathered_top_k, f"Gathered top-{k} values")
    
    # Example 4: Padding for sequence processing
    print("\nExample 4: Padding for sequence processing")
    sequences = [torch.tensor([1, 2, 3]),
                torch.tensor([1, 2, 3, 4, 5]),
                torch.tensor([1])]
    
    print("Original sequences:")
    for i, seq in enumerate(sequences):
        print(f"  Sequence {i}: {seq}")
    
    # Find max length
    max_len = max(len(seq) for seq in sequences)
    print(f"\nMax length: {max_len}")
    
    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        pad_size = max_len - len(seq)
        padded = F.pad(seq, pad=(0, pad_size), mode='constant', value=0)
        padded_sequences.append(padded)
    
    print("Padded sequences:")
    for i, seq in enumerate(padded_sequences):
        print(f"  Sequence {i}: {seq}")
    
    # Stack into batch
    batch = torch.stack(padded_sequences)
    print_tensor_info(batch, "Batched sequences")
    
    print("\n" + "=" * 60)
    print("PRACTICAL INSIGHTS:")
    print("- torch.where: Great for conditional element-wise operations")
    print("- masked_scatter: Efficient for selective in-place updates")
    print("- gather: Essential for collecting specific elements based on indices")
    print("- pad: Critical for maintaining dimensions in neural networks")
    print("=" * 60)

def demo_masked_scatter_fill_vsion_placehodlers_with_vision_tokens():
    B, L, D = 2, 5, 16
    M = 2
    text_tokens = torch.randint(0, 100, (B, 5, D))
    vision_tokens = torch.randn(B, 2, D)

    vision_mask = torch.tensor([[True, True, False, False, False],
                                [False, False, True, False, True]])
    # for each sequence in text_tokens, we choose M vision tokens to replace the text tokens
    assert vision_tokens.numel() == vision_mask.sum()

    # text_tokens.masked_scatter_(vision_mask, vision_tokens) # in-place operation

    filled_text_tokens = text_tokens.masked_scatter(vision_mask, vision_tokens) # return a new tensor

    print(filled_text_tokens)

def demo_torch_gather_cross_entropy_loss():
    B, L, D = 2, 5, 16
    logits = torch.randn(B, L, D)
    targets = torch.randint(0, 100, (B, L))

    # in gather 
    '''
        Here’s a crisp guide to torch.gather with the patterns you’ll actually use.

        What gather does
        out = torch.gather(input, dim, index)
        Picks values from input along dim using integer index.

        Shapes must match on all dims except dim.

        index is LongTensor; out.shape == index.shape.

        index.shape must equal input.shape on every dimension except dim (the dimension you’re gathering along).
        On dim, the size of the output is index.size(dim). The output shape is exactly index.shape.
    '''
    # We need unsqueeze the targets to match the shape of logits and squeeze the last dimension
    predicted_logits = logits.gather(dim=-1, index=targets[:, :, None]).squeeze(-1)

    print(predicted_logits)

    
    



if __name__ == "__main__":
    print("PYTORCH OPERATIONS: WHERE, MASKED_SCATTER, GATHER, PAD")
    print("Understanding how these operations work and when to use them")
    
    demonstrate_torch_where()
    demonstrate_masked_scatter()
    demonstrate_gather()
    demonstrate_pad()
    demonstrate_prod_cumsum_cumprod()
    demonstrate_memory_efficiency_comparison()
    demonstrate_practical_examples()
    compare_inplace_vs_outofplace()
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print("1. torch.where(condition, x, y): Conditional element selection (COPY)")
    print("2. masked_scatter_(mask, source): In-place selective updates (NO COPY)")
    print("3. gather(input, dim, index): Collect elements based on indices (COPY)")
    print("4. F.pad(input, pad, mode, value): Add padding to tensors (COPY)")
    print("5. prod(): Product of elements (COPY)")
    print("6. cumsum(dim): Cumulative sum (COPY)")
    print("7. cumprod(dim): Cumulative product (COPY)")
    print("\nMEMORY EFFICIENCY:")
    print("- COPY operations: torch.where, gather, F.pad, prod, cumsum, cumprod (create new tensors)")
    print("- NO COPY operations: masked_scatter_ (in-place modification)")
    print("=" * 60)