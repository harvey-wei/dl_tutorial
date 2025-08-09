import numpy as np
import time

def load_memmap():
    # ------------------------------
    # Step 1: Create or open a memory-mapped file
    # ------------------------------
    # This creates a 10000 x 10000 float32 array backed by a file on disk.
    # M ode 'w+' means: read/write access and create a new file or overwrite if it exists.
    # The data is not loaded into RAM yet — just a virtual memory mapping is created.
    fp = np.memmap('data.dat', dtype='float32', mode='w+', shape=(10000, 10000))

    # ------------------------------
    # Step 2: Write to a small part of the array
    # ------------------------------
    # Here we fill the first 10 rows of the array with random values.
    # Only this portion (10 x 10000 floats = ~0.38 MB) is modified.
    fp[0:10] = np.random.rand(10, 10000)

    # Ensure the written data is flushed from memory to disk
    # This is important if you're done writing and want the data safely stored
    fp.flush()

    # ------------------------------
    # Step 3: Read a small slice of the array
    # ------------------------------
    # This reads the first 5 elements of the first row.
    # Thanks to memory-mapping, only the corresponding memory page(s)
    # are brought into RAM by the operating system.
    print(fp[0, :5])  # Efficient — reads only a few bytes from disk

def load():
    # ------------------------------
    # Step 1: Create a regular NumPy array in memory
    # ------------------------------
    # This allocates ~3.7 GB of RAM immediately
    array = np.zeros((10000, 10000), dtype='float32')

    # ------------------------------
    # Step 2: Write to part of the array
    # ------------------------------
    # Fill the first 10 rows with random values (stored in RAM)
    array[0:10] = np.random.rand(10, 10000)

    # ------------------------------
    # Step 3: Save the full array to disk as a .npy file
    # ------------------------------
    # This writes the entire array (all 100 million float32s) to disk
    np.save('data.npy', array)

    # ------------------------------
    # Step 4: Load the entire array back from disk
    j# ------------------------------
    # This loads the entire array into memory (RAM)
    array_loaded = np.load('data.npy')

    # Access the first few elements of the first row
    print(array_loaded[0, :5])
