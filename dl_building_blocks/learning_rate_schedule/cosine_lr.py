import numpy as np
import matplotlib.pyplot as plt

# Parameters
eta_max = 0.01   # Max learning rate
eta_min = 0.0001 # Min learning rate
T = 1000         # Total steps
T_wp = int(0.05 * T)

# Compute learning rates
t = np.arange(T + 1)

eta_t = [0.0] * T
eta_t[:T_wp] = eta_min + t[:T_wp] * (eta_max - eta_min) / T_wp
eta_t[T_wp:] = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t[T_wp:] / T))

# Plot
plt.figure(figsize=(8, 4))
plt.plot(t, eta_t, label='Cosine LR Schedule')
plt.xlabel('Step (t)')
plt.ylabel('Learning Rate')
plt.title('Cosine Learning Rate Schedule')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
