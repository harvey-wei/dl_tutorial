import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

x = np.linspace(-5, 5, 100)
beta = 2.0

# reference: https://arxiv.org/pdf/1710.05941
swish = x / (1 + np.exp(- beta * x)) # Swish = x * sigmoid(beta * x)
silu = F.silu(torch.from_numpy(x).to("cuda")).cpu().numpy()
relu = np.maximum(0, x)

plt.figure(figsize=(8, 5))
plt.plot(x, swish, label='Swish (x * sigmoid(beta x))', linewidth=2)
plt.plot(x, silu, label='Silu (x * sigmoid(x))', linewidth=2)
plt.plot(x, relu, label='ReLU (ReLU(x))', linestyle='--', linewidth=2)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.title('Swish vs ReLU Activation Functions')
plt.xlabel('Input x')
plt.ylabel('Activation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
