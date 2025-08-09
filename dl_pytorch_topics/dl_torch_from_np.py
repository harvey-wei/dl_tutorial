import numpy as np
import torch

a = np.array([1, 2, 3], dtype=np.float32)
t = torch.from_numpy(a)

a[0] = 100
print(t)  # tensor([100.,   2.,   3.])

