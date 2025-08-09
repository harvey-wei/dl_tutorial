import numpy as np

# Step 1: Create a 1D array which can be viewed as both col and row vector
v = np.array([1, 2, 3]) # [3,]
print("Original 1D vector:", v)
print("Shape:", v.shape)  # (3,)

w = np.random.rand(3, 3)

print("\nRandom 3x3 matrix:\n", w)

wv = w @ v # [4,]
print("\nMatrix-vector product (w @ v):", wv)

# vw = v @ w
# print("\nVector-matrix product (v @ w):", vw)

# Step 2: Convert to row and column vectors
row_vec = v.reshape(1, -1)   # (1, 3)
col_vec = v.reshape(-1, 1)   # (3, 1)

w_v_col = w @ col_vec

print("\nMatrix-vector product (w @ col_vec):", w_v_col)

print("\nRow vector:\n", row_vec)
print("Row vector shape:", row_vec.shape)

print("\nColumn vector:\n", col_vec)
print("Column vector shape:", col_vec.shape)


# Step 3: Dot product of v with itself (inner product)
dot_product = v @ v
print("\nDot product (v @ v):", dot_product)

# Step 4: Outer product (v outer v)
outer_product = np.outer(v, v)
print("\nOuter product (np.outer(v, v)):\n", outer_product)

# Step 5: Matrix product - column vector @ row vector
mat_col_row = col_vec @ row_vec
print("\nMatrix product (col_vec @ row_vec):\n", mat_col_row)

# Step 6: Matrix product - row vector @ column vector
mat_row_col = row_vec @ col_vec
print("\nMatrix product (row_vec @ col_vec):\n", mat_row_col)
