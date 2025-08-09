import torch


'''
Method	In-Place?	Returns New Tensor?	Changes Original Tensor?
tensor.to()	❌ No	✅ Yes	❌ No

'''
tensor = torch.randn(2, 2)  # Initially float32 on CPU

tensor = tensor.to(torch.float64)  # Now float64
print(tensor.dtype)  # dtype=torch.float64

cuda0 = torch.device('cuda:0')
tensor = tensor.to(cuda0)  # Now on GPU
print(tensor.device)  # device=cuda:0

tensor = tensor.to(cuda0, dtype=torch.float64)  # This is redundant now

other = torch.randn((), dtype=torch.float64, device=cuda0)
tensor = tensor.to(other, non_blocking=True)  # Matches other’s dtype/device
