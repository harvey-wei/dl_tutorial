import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
)

print(model[0])  # nn.Conv2d(3, 16, ...)
print(model[1])  # nn.ReLU()
print(model[-1]) # Last layer (nn.Conv2d(16, 32, ...))
