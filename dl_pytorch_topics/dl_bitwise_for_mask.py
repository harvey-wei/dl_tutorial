import numpy as np

a = 60  # 0b00111100
b = 13  # 0b00001101

print(a & b)   # AND → 12 (0b00001100)
print(a | b)   # OR  → 61 (0b00111101)
print(a ^ b)   # XOR → 49 (0b00110001)
print(~a)      # NOT → -61  (two's complement)
print(a << 2)  # Left shift → 240 (0b11110000)
print(a >> 2)  # Right shift → 15  (0b00001111)


mask = 0b1000  # check 4th bit
if a & mask:
    print("4th bit is set")
a &= ~mask      # clear that bit
a |= mask       # set that bit
a ^= mask       # toggle it :contentReference[oaicite:5]{index=5}



import numpy as np

arr = np.array([1, 2, 3, 4, 5])
mask = (arr > 2) & (arr < 5)  # AND operator
print(mask)  # [False, False, True, True, False]

filtered = arr[mask]
print(filtered)  # [3, 4]

# invert mask (logical NOT)
not_mask = ~mask  # [True, True, False, False, True]
print(arr[not_mask])  # [1, 2, 5] :contentReference[oaicite:7]{index=7}



import torch

# Example tensor
x = torch.tensor([[10, 20, 30],
                  [40, 50, 60],
                  [70, 80, 90]])

# Create a mask: True where value >= 50
mask = (x >= 50) & (x <= 80) # the same shape as x
print(mask)
# tensor([[False, False, False],
#         [False,  True,  True],
#         [ True,  True,  False]])

# Use mask to index
filtered = x[mask]
print(filtered)
print(f'filtered shape {filtered.shape}')
# tensor([50, 60, 70, 80])
