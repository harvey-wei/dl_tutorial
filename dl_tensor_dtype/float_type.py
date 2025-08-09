import torch

def print_info(dtype):
    info = torch.finfo(dtype)
    print(f"\n--- {dtype} ---")
    print(f"Max:        {info.max:.3e}")
    print(f"Min (norm): {info.min:.3e}")
    print(f"Subnormal:  {info.tiny:.3e}")
    print(f"Epsilon:    {info.eps:.3e}")
    print(f"Resolution: {info.resolution:.3e}")
    print(f"Bits:       {info.bits}")

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    print_info(dtype)
