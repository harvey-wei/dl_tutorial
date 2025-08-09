import numpy as np
import time
import os

# Configuration
shape = (40000, 20000)  # ~1.49 GB for float32
dtype = np.float32
filename_memmap = "large_memmap.dat"
filename_npy = "large_array.npy"

# Step 1: Generate large array and save as binary
# array = np.random.rand(*shape).astype(dtype)

# Save for memmap test
# with open(filename_memmap, "wb") as f:
#     array.tofile(f)
#
# Save as .npy
# np.save(filename_npy, array)

# Step 2: Load with memmap
start_memmap = time.time()
fp_memmap = np.memmap(filename_memmap, dtype=dtype, mode='r', shape=shape)
_ = fp_memmap[1000, :100].copy()  # Trigger read from disk
# _ = np.array(fp_memmap)  # Force full load read other form disk
memmap_time = time.time() - start_memmap

# Step 3: Load with np.load
start_npy = time.time()
loaded_npy = np.load(filename_npy)
# Is blocked by loading from disk to memory
_ = loaded_npy[1000, :100]  # Already in RAM
npy_time = time.time() - start_npy

# Clean up
# os.remove(filename_memmap)
# os.remove(filename_npy)


{
    "Memmap load time (including full load)": memmap_time,
    "NumPy load time (.npy)": npy_time
}

print("Memmap load time (including full load):", memmap_time)
print("NumPy load time (.npy):", npy_time)
