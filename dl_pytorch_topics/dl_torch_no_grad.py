import torch
import torch.nn as nn
import time

# Simple model and input
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 3, kernel_size=3, padding=1)
)

input_tensor = torch.randn(64, 3, 128, 128)  # batch of 64 RGB images

# -------------------------------
# 1. WITH gradient tracking
# -------------------------------
start = time.time()

for _ in range(100):
    output = model(input_tensor)

end = time.time()
print(f"With gradient tracking: {end - start:.4f} seconds")

# -------------------------------
# 2. WITHOUT gradient tracking
# -------------------------------
start = time.time()

with torch.no_grad():
    for _ in range(100):
        output = model(input_tensor)

end = time.time()
print(f"With torch.no_grad(): {end - start:.4f} seconds")
