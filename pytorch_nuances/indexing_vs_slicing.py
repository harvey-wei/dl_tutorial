import torch


def element_address(t: torch.Tensor, index):
    """
    Return the byte address of t[index], assuming index length <= t.dim().
    Handles non-contiguous views, storage_offset, and stride-0 (expand) views.
    """
    assert isinstance(index, (tuple, list))
    assert len(index) <= t.dim()

    # Pad missing trailing dims with 0 (e.g., t[i,j] in a 3D tensor means t[i,j,0,...])
    idx = list(index) + [0] * (t.dim() - len(index))

    # Base storage pointer (not the first element’s pointer)
    base_ptr = t.untyped_storage().data_ptr()   # storage base in bytes
    esz = t.element_size()                      # bytes per element
    off = t.storage_offset()                    # elements from storage base to t[0,...,0]
    strides = t.stride()                        # stride in elements per dim

    # Linear offset in ELEMENTS from storage base
    linear = off + sum(i * s for i, s in zip(idx, strides))

    # Byte address
    return base_ptr + linear * esz


print("="*70)
print("PYTORCH INDEXING & SLICING COMPLETE TUTORIAL")
print("="*70)

print("\n1. INTRODUCTION")
print("-" * 40)
print("PyTorch provides multiple ways to access and modify tensor elements:")
print("• Basic indexing: Access single elements")
print("• Slicing: Access ranges of elements")
print("• Advanced indexing: Use tensors as indices")
print("• Boolean indexing: Use boolean masks")
print("• Each method has different copying behavior and shape implications")

print("\n2. BASIC INDEXING")
print("-" * 40)

# Create a sample tensor
tensor_3d = torch.randn(3, 4, 5)
print(f"Original tensor: {tensor_3d.shape}")
print(f"Tensor:\n{tensor_3d}")

print("\n2.1 Single element indexing:")
# Access single element
element = tensor_3d[0, 1, 2]
print(f"tensor_3d[0, 1, 2] = {element}")
print(f"Result shape: {element.shape}")
print(f"Result type: {type(element)}")
print("✓ Returns a scalar tensor (0-dimensional)")
element = torch.tensor(999) # breaks the view of the original tensor
print(f"element =tensor_3d[0, 1, 2]  = {element}")
print(f"tensor_3d[0, 1, 2] -> {tensor_3d[0, 1, 2]}") # not changed to 999
print("Basic indexing creates a view of the original tensor where no colon is used")
'''
"Basic indexing creates a view…"

Careful: that statement is only true if the indexing still leaves some dimensions (e.g., tensor_3d[0, 1, :]), because then you’re slicing, which is a view.

When all dimensions are reduced to a single number (like [0, 1, 2]), you get a new 0D tensor, not a view — so modifying it won’t affect the original tensor.


'''

print("\n2.2 Partial indexing:")
# Access partial dimensions
row = tensor_3d[0, 1]  # Get row at index 1 from first batch
print(f"tensor_3d[0, 1]: {row}")
print(f"Result shape: {row.shape}")
print("✓ Returns a 1D tensor")
row[0] = 999
print(f"tensor_3d[0, 1] = {row}")
print(f"tensor_3d -> {tensor_3d}")
print("Basic indexing creates a view of the original tensor where no colon is used")

# Access 2D slice
slice_2d = tensor_3d[0]  # Get first batch
print(f"tensor_3d[0]: {slice_2d}")
print(f"Result shape: {slice_2d.shape}") # [4, 5] NOT [1, 4, 5] as 0 is index not slice
print("✓ Returns a 2D tensor")
slice_2d[0, 0] = 888
print(f"tensor_3d[0] = {slice_2d}")
print(f"tensor_3d[0].shape = {slice_2d.shape}") 
print(f"tensor_3d -> {tensor_3d}")
print("Basic indexing creates a view of the original tensor where no colon is used")

col = tensor_3d[:, 1] # [3, 5] NOT [3, 1, 5] as 1 is index not slice
print(f"tensor_3d[:, 1] = {col}")
print(f"tensor_3d -> {tensor_3d}")
print("Basic indexing creates a view of the original tensor where no colon is used")
col[0] = 7777
print(f"tensor_3d[:, 1] = {col}")
print(f"tensor_3d[:, 1].shape = {col.shape}")
print(f"tensor_3d[:, 1] after modification -> {tensor_3d[:, 1]}")
print(f"tensor_3d -> of basic indexing creates a view of the original tensor where no colon is used")

col_ = tensor_3d[:, :1] # [3, 1, 5]
print(f"tensor_3d[:, :1] = {col_}")
print(f"tensor_3d[:, :1].shape = {col_.shape}")
col_[0, 0] = 8866
print(f"tensor_3d[:, :1] after modification -> {tensor_3d[:, :1]}")
print(f"Pure Slicing creates a view of the original tensor where no colon is used")

col_2 = tensor_3d[:, torch.tensor([1])] # [3, 5]
print(f"tensor_3d[:, torch.tensor([1])] = {col_2}")
print(f"tensor_3d[:, torch.tensor([1])].shape = {col_2.shape}")
col_2[0, 0] = 8855
print(f"tensor_3d[:, torch.tensor([1])] after modification -> {tensor_3d[:, torch.tensor([1])]}")
print(f"Advanced indexing with slcing creates a copy of the original tensor where index tensors are used.")





print("\n2.3 Memory behavior:")
print(f"Original tensor memory at row [0, 1, 2]: {element_address(tensor_3d, (0, 1, 2))}")
print(f"Original tensor memory at row [0, 1]: {element_address(tensor_3d, (0, 1))}")
print(f"Original tensor memory at row [0]: {element_address(tensor_3d, (0,))}")
print(f"Row memory (tensor_3d[0, 1]): {row.data_ptr()}")
print(f"Slice 2D memory (tensor_3d[0]): {slice_2d.data_ptr()}")
print("✓ Basic indexing creates VIEWS (no copying)")
print("✓ Basic indexing removed indexed dimensions while keeping the remaining dimensions!")

print("\n3. SLICING")
print("-" * 40)

print("\n3.1 Basic slicing syntax:")
print("tensor[start:end:step]")
print("• start: inclusive (default 0)")
print("• end: exclusive (default end)")
print("• step: stride (default 1)")

# Create a 2D tensor for slicing examples
tensor_2d = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"\n2D tensor:\n{tensor_2d}")

print("\n3.2 Row slicing:")
# Get first two rows
rows_01 = tensor_2d[0:2]
print(f"tensor_2d[0:2]:\n{rows_01}")
print(f"Result shape: {rows_01.shape}")

# Get every other row
rows_step = tensor_2d[::2]
print(f"tensor_2d[::2]:\n{rows_step}")
print(f"Result shape: {rows_step.shape}")

print("\n3.3 Column slicing:")
# Get first two columns
cols_01 = tensor_2d[:, 0:2]
print(f"tensor_2d[:, 0:2]:\n{cols_01}")
print(f"Result shape: {cols_01.shape}")

# Get every other column
cols_step = tensor_2d[:, ::2]
print(f"tensor_2d[:, ::2]:\n{cols_step}")
print(f"Result shape: {cols_step.shape}")

print("\n3.4 2D slicing:")
# Get 2x2 submatrix
submatrix = tensor_2d[0:2, 1:3]
print(f"tensor_2d[0:2, 1:3]:\n{submatrix}")
print(f"Result shape: {submatrix.shape}")
print(f"Slicing keeps the number of dimensions!")

print("\n3.5 Memory behavior:")
print(f"Original tensor memory: {tensor_2d.data_ptr()}")

print(f"Row 01 memory: {element_address(tensor_2d, (0, 1))}")
print(f"")

print(f"Row slice memory: {rows_01.data_ptr()}")
print(f"Column slice memory: {cols_01.data_ptr()}")
print(f"Submatrix memory: {submatrix.data_ptr()}")
print("✓ Slicing creates VIEWS (no copying)")

# check memoery behavior by modifying the sliced tensor and compare the original tensor
rows_01[0, 0] = 999
print(f"Original tensor after modification: {tensor_2d}")
print(f"Sliced tensor after modification: {rows_01}")
print(f"Memory addresses same? {tensor_2d.data_ptr() == rows_01.data_ptr()}")
print(f"Modifying rows_01 affects tensor_2d: {torch.equal(tensor_2d, rows_01)}")
print('Basic indexing creates a view of the original tensor where no colon is used')
print('Slicing creates a view of the original tensor where a colon is used')
print('Advanced indexing creates a copy of the oliginal tensor where index tensors are used.')


print("\n4. ADVANCED INDEXING")
print("-" * 40)

print("\n4.1 Integer array indexing:")
# Create tensor and indices
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row_indices = torch.tensor([0, 2])
col_indices = torch.tensor([1, 2])

print(f"Original tensor:\n{tensor}")
print(f"Row indices: {row_indices}")
print(f"Column indices: {col_indices}")

# Advanced indexing
selected = tensor[row_indices, col_indices]
print(f"tensor[row_indices, col_indices]: {selected}")
print(f"Result shape: {selected.shape}")
print("✓ Returns elements at (0,1) and (2,2)")
print(f"Row indices shape: {row_indices.shape}")
print(f"Column indices shape: {col_indices.shape}")
print("The resulting shape is the same as the shape of the indices if no broadcasting is needed")
print(f"The resulting shape is the same as the shape of the indices if no broadcasting is needed: {selected.shape == row_indices.shape}")

print("\n4.2 Broadcasting in advanced indexing:")
# Different shaped indices
row_idx = torch.tensor([0, 1])  # shape: (2,)
col_idx = torch.tensor([[0, 1], [2, 0]])  # shape: (2, 2)

print(f"Row index: {row_idx.shape}")
print(f"Column index: {col_idx.shape}")

# These broadcast to compatible shapes
result = tensor[row_idx, col_idx]
print(f"Result: {result}")
print(f"Result shape: {result.shape}")
print("✓ Indices are broadcasted to compatible shapes")

print("\n4.3 Memory behavior:")
print(f"Original tensor memory: {tensor.data_ptr()}")
print(f"Advanced indexed result memory: {selected.data_ptr()}")
print("✗ Advanced indexing creates COPIES (not views)")

# check memory behavior by modifying the advanced indexed tensor and compare the original tensor
result[0, 1] = 999
print(f"Original tensor after modification: {tensor}")
print(f"Advanced indexed result after modification: {result}")
# print(f"Memory addresses same? {tensor.data_ptr() == result.data_ptr()}")
print(f"Modifying result affects tensor: {torch.equal(tensor, result)}")
print(f"Advanced indexing creates a copy of the original tensor!")

# advanced indexing with list of integers
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
indices = [0, 2]
selected = tensor[indices]
print(f"tensor[indices]: {selected}")
print(f"Result shape: {selected.shape}")
selected[0] = 999
print(f"tensor[indices]: {selected}")
print(f"tensor -> {tensor}")
print(f"Advanced indexing creates a copy of the original tensor!")


print("\n5. BOOLEAN INDEXING")
print("-" * 40)

print("\n5.1 Basic boolean indexing:")
# Create tensor and boolean mask
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask = tensor >= 5 # mask shape is the same as the indexed tensor

print(f"Original tensor:\n{tensor}")
print(f"Boolean mask (tensor > 5):\n{mask}")

# Boolean indexing
selected = tensor[mask]
print(f"tensor[mask]: {selected}")
print(f"Result shape: {selected.shape}") # [# of trues in mask,]
print("✓ Returns flattened array of elements where mask is True in row major order of the indexed tensor")

# Check memory behavior by modifying the boolean indexed tensor and compare the original tensor
selected[1] = 999
print(f"Original tensor after modification: {tensor}")
print(f"Boolean indexed result after modification: {selected}")

# What if we transpose the tensor?
tensor_transposed = tensor.permute(1, 0)
mask_transposed = tensor_transposed >= 5
selected_transposed = tensor_transposed[mask_transposed]
print(f"tensor_transposed: {tensor_transposed}")
print(f"tensor_transposed[mask_transposed]: {selected_transposed}")
print(f"Result shape: {selected_transposed.shape}")
print("✓ Returns flattened array of elements where mask is True in row major order of the tranposed tensor")

print("\n5.2 Multi-dimensional boolean indexing:")
# 2D boolean mask
mask_2d = torch.tensor([[True, False, True], [False, True, False], [True, True, False]]) # mask shape is the same as the indexed tensor
print(f"2D mask:\n{mask_2d}")

selected_2d = tensor[mask_2d]
print(f"tensor[mask_2d]: {selected_2d}")
print(f"Result shape: {selected_2d.shape}") # a flattened tensor

print("\n5.3 Memory behavior:")
selected_2d[2] = 999
print(f"Original tensor after modification: {tensor}")
print(f"Boolean indexed result after modification: {selected_2d}")
# print(f"Memory addresses same? {tensor.data_ptr() == selected_2d.data_ptr()}")
# print(f"Modifying selected_2d affects tensor: {torch.equal(tensor, selected_2d)}")
print("Boolean indexing creates a copy of the original tensor!")

print(f"Original tensor memory: {tensor.data_ptr()}")
print(f"Boolean indexed result memory: {selected.data_ptr()}")
print("✗ Boolean indexing creates COPIES (not views)")

print("\n5.4 Boolean indexing with one dimension mask:")
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask_rows = torch.tensor([True, False, True])
# mask_cols = torch.tensor([True, False, True])
selected_2d = tensor_2d[mask_rows] # shape: (2, 3)
print(f"tensor_2d: {tensor_2d}")
print(f"mask_rows: {mask_rows}")
# print(f"mask_cols: {mask_cols}")
print(f"tensor_2d[mask_rows]: {selected_2d}")
print(f"Result shape: {selected_2d.shape}")
selected_2d[0] = 999 # set the first row  of selected_2d all to 999
print(f"Original tensor after modification: {tensor_2d}")
print(f"Boolean indexed result after modification: {selected_2d}")
# print(f"Memory addresses same? {tensor_2d.data_ptr() == selected_2d.data_ptr()}")
# print(f"Modifying selected_2d affects tensor_2d: {torch.equal(tensor_2d, selected_2d)}")
print("Boolean indexing creates a copy of the original tensor!")

print("\n5.5 Boolean indexing with multiple dimensions mask -> advanced indexing:")
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask_rows = torch.tensor([True, False, True]) # select rows [0, 2]
mask_cols = torch.tensor([True, True, False]) # select cols [0, 1]
selected_2d = tensor_2d[mask_rows, mask_cols] # select [0,0] and [2,1] ->  shape [2,]
print(f"tensor_2d: {tensor_2d}")
# print(f"mask_2d: {mask_2d}")
print(f"tensor_2d[mask_rows, mask_cols]: {selected_2d}")
print(f"Result shape: {selected_2d.shape}")
selected_2d[0] = 999 # set the first row  of selected_2d all to 999
print(f"Original tensor after modification: {tensor_2d}")
print(f"Boolean indexed result after modification: {selected_2d}")
# print(f"Memory addresses same? {tensor_2d.data_ptr() == selected_2d.data_ptr()}")
# print(f"Modifying selected_2d affects tensor_2d: {torch.equal(tensor_2d, selected_2d)}")
print("Boolean indexing creates a copy of the original tensor!")

print("\n5.6 Boolean indexing with multiple dimensions mask:")
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mask_rows = torch.tensor([True, False, True]) # select rows [0, 2]
mask_cols = torch.tensor([True, True, False]) # select cols [0, 1]
selected_2d = tensor_2d[mask_rows, :][:, mask_cols] 
print(f"tensor_2d: {tensor_2d}")
print(f"tensor_2d[mask_rows, :][:, mask_cols]: {selected_2d}")
selected_2d[0] = 999 # set the first row  of selected_2d all to 999
print(f"Original tensor after modification: {tensor_2d}")
print(f"Boolean indexed result after modification: {selected_2d}")
# print(f"Memory addresses same? {tensor_2d.data_ptr() == selected_2d.data_ptr()}")
# print(f"Modifying selected_2d affects tensor_2d: {torch.equal(tensor_2d, selected_2d)}")
print("Boolean indexing creates a copy of the original tensor!")

print("\n6. RESULTING SHAPES")
print("-" * 40)

print("\n6.1 Shape rules for different indexing methods:")

# Create a 3D tensor
tensor_3d = torch.randn(5, 3, 4)
print(f"3D tensor shape: {tensor_3d.shape}")

print("\nBasic indexing shapes:")
print(f"tensor_3d[0] -> {tensor_3d[0].shape}")  # Remove first dimension
print(f"tensor_3d[0, 1] -> {tensor_3d[0, 1].shape}")  # Remove first two dimensions
print(f"tensor_3d[0, 1, 2] -> {tensor_3d[0, 1, 2].shape}")  # Remove all dimensions

print("\nSlicing shapes:")
print(f"tensor_3d[0:1] -> {tensor_3d[0:1].shape}")  # Keep first dimension
print(f"tensor_3d[0:1, 1:2] -> {tensor_3d[0:1, 1:2].shape}")  # Keep first two dimensions
print(f"tensor_3d[:, 1:2, 2:3] -> {tensor_3d[:, 1:2, 2:3].shape}")  # Keep all dimensions

print("\nAdvanced indexing shapes:")
indices = torch.tensor([0, 1])
tensor_3d_indices = tensor_3d[indices]
tensor_3d_mixed = tensor_3d[indices, :, :]
print(f"tensor_3d_indices -> {tensor_3d_indices.shape}") # shape: (2, 3, 4)
print(f"tensor_3d_mixed -> {tensor_3d_mixed.shape}") # shape: (2, 3, 4)
tensor_3d_mixed[0, 0, 0] = 999
tensor_3d_indices[0, 0, 0] = 9999
print(f'After modification:')
print(f'tensor_3d_indices -> {tensor_3d_indices}')
print(f'tensor_3d_mixed -> {tensor_3d_mixed}')
print(f'tensor_3d -> {tensor_3d}')
print("Mixed indexing of advanced indexing creates a copy of the original tensor!")

# print(f"tensor_3d[indices] -> {tensor_3d[indices].shape}")  # Index tensor shape
# print(f"tensor_3d[indices, :, :] -> {tensor_3d[indices, :, :].shape}")  # Mixed indexing
# print(f"tensor_3d[indices, :, :] -> {tensor_3d[indices, :, :].shape}")  # Mixed indexing

print("\n6.2 Shape prediction rules:")
print("• Basic indexing: Removes indexed dimensions")
print("• Slicing: Keeps sliced dimensions")
print("• Advanced indexing: Result shape depends on index tensor shapes")
print("• Boolean indexing: Returns flattened array")

print("\n7. COPYING BEHAVIOR")
print("-" * 40)

print("\n7.1 When copying occurs:")
print("✓ Basic indexing: NO copying (creates views)")
print("✓ Slicing: NO copying (creates views)")
print("✗ Advanced indexing: COPIES are created")
print("✗ Boolean indexing: COPIES are created")
print("✗ Mixed indexing: May create copies")

print("\n7.2 Demonstration:")

# Create tensor
original = torch.randn(3, 4)
print(f"Original tensor: {original.shape}")

print("\nBasic indexing  (no copy):")
basic_result = original[0, :] # here : is not a slice, can also be written as original[0]
print(f"Memory addresses same? {original.data_ptr() == basic_result.data_ptr()}")
print(f"Modifying basic_result affects original: {torch.equal(original[0], basic_result)}")

print("\nSlicing (no copy):")
slice_result = original[0:2, 1:3]
print(f"Memory addresses same? {original.data_ptr() == slice_result.data_ptr()}")
print(f"Modifying slice_result affects original: {torch.equal(original[0:2, 1:3], slice_result)}")

print("\nAdvanced indexing (which use index tensors) (copy):")
indices = torch.tensor([0, 2])
advanced_result = original[indices, :]
print(f"Memory addresses same? {original.data_ptr() == advanced_result.data_ptr()}")
print(f"Modifying advanced_result does NOT affect original")

print("\nBoolean indexing (copy):")
mask = original > 0
boolean_result = original[mask]
print(f"Memory addresses same? {original.data_ptr() == boolean_result.data_ptr()}")
print(f"Modifying boolean_result does NOT affect original")

print("\n8. PRACTICAL EXAMPLES")
print("-" * 40)

print("\n8.1 Neural network data processing:")
# Simulate batch of images
batch_images = torch.randn(32, 3, 224, 224)  # batch, channels, height, width
print(f"Batch images: {batch_images.shape}")

# Get first image (basic indexing no copy)
first_image = batch_images[0]  # shape: (3, 224, 224)
print(f"First image: {first_image.shape}")

# Get first 10 images (slicing no copy)
first_10 = batch_images[0:10]  # shape: (10, 3, 224, 224)
print(f"First 10 images: {first_10.shape}")

# Get specific images (advanced indexing COPYING)
specific_indices = torch.tensor([0, 5, 10, 15])
specific_images = batch_images[specific_indices]  # shape: (4, 3, 224, 224)
print(f"Specific images: {specific_images.shape}")

print("\n8.2 Attention mechanism:")
# Attention weights
attention = torch.randn(8, 12, 50, 50)  # batch, heads, seq_len, seq_len
print(f"Attention weights: {attention.shape}")

# Get attention for first batch, first head. Basic indexing no copy
first_attention = attention[0, 0]  # shape: (50, 50)
print(f"First attention: {first_attention.shape}")

# Get attention for specific positions via advanced indexing COPYING
pos_indices = torch.tensor([0, 10, 20])
pos_attention = attention[:, :, pos_indices, pos_indices]  # shape: (8, 12, 3, 3)
print(f"Position attention: {pos_attention.shape}")

print("\n8.3 Data filtering:")
# Create dataset
data = torch.randn(100, 10)  # 100 samples, 10 features
labels = torch.randint(0, 3, (100,))  # 3 classes
print(f"Data: {data.shape}, Labels: {labels.shape}")

# Filter by class
class_0_mask = labels == 0
class_0_data = data[class_0_mask]  # shape: (N, 10) where N is number of class 0 samples
print(f"Class 0 data: {class_0_data.shape}")

# Get samples with high values
high_value_mask = data > 0.5
high_value_data = data[high_value_mask]  # shape: (M,) where M is number of high values
print(f"High value data: {high_value_data.shape}")

print("\n9. PERFORMANCE CONSIDERATIONS")
print("-" * 40)

print("\n9.1 Memory efficiency:")
print("• Views (basic indexing, slicing): Memory efficient")
print("• Copies (advanced indexing, boolean): Uses more memory")
print("• Large tensors: Consider memory usage when choosing indexing method")

print("\n9.2 Speed considerations:")
print("• Views: Fast access, no data copying")
print("• Copies: Slower due to memory allocation and copying")
print("• Repeated operations: Views are more efficient")

print("\n9.3 Best practices:")
print("• Use basic indexing/slicing when possible")
print("• Avoid unnecessary advanced indexing in loops")
print("• Consider in-place operations for modifications")
print("• Use .clone() explicitly when you need a copy")

print("\n10. COMMON MISTAKES")
print("-" * 40)

print("\n10.1 Assuming views are copies:")
tensor = torch.randn(3, 4)
view = tensor[0, :]
view[0] = 999
print(f"Original tensor affected: {tensor[0, 0]}")

print("\n10.2 Assuming copies are views:")
tensor = torch.randn(3, 4)
indices = torch.tensor([0, 1])
copy_result = tensor[indices, :]
copy_result[0, 0] = 999
print(f"Original tensor NOT affected: {tensor[0, 0]}")

print("\n10.3 Wrong shape expectations:")
tensor = torch.randn(3, 4)
# Expecting 2D but getting 1D
result = tensor[0, 0]  # Scalar, not 2D
print(f"Result shape: {result.shape}")

print("\n11. SUMMARY")
print("-" * 40)
print("• Basic indexing: Views, removes dimensions")
print("• Slicing: Views, keeps dimensions")
print("• Advanced indexing: Copies, shape depends on indices")
print("• Boolean indexing: Copies, flattened result")
print("• Memory efficiency: Views > Copies")
print("• Speed: Views > Copies")
print("• Always check shapes and memory behavior")

print("\n" + "="*70)


'''
here you go — a crisp PyTorch Indexing: View vs Copy cheat sheet you can keep by your side 👇

PyTorch Indexing Cheat Sheet (View vs Copy)
Quick rules
Pure slicing (: / start:stop:step) → View

Mix of slices + integers → View (dims indexed by an integer are removed)

All integers → 0-D Tensor (copy-like, not a view)

Advanced indexing (Long/Bool tensors, Python lists) → Copy

Boolean mask → Copy

Ellipsis ... is just shorthand for enough : — behaves like slicing (view)


'''