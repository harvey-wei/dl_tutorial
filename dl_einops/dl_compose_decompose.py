import torch
from einops import rearrange

x = torch.randn(2, 3, 4)  # shape: [2, 3, 4]

# Flatten the last two dims into One new dim
# y = rearrange(x, 'b h w -> b (h w)')  # shape: [2, 12]
y = rearrange(x, 'b h w -> b (h w)')

# Reverse: split one axis into two
# z = rearrange(y, 'b (h w) -> b h w', h=3, w=4)  # shape: [2, 3, 4]
z = rearrange(y, 'b (h w) -> b h w', h=3, w=4)

w = rearrange(x, 'b h w -> b w h') #[2, 4, 3]


print(f'x shape {x.shape}')
print(f'y shape {y.shape}')
print(f'z shape {z.shape}')
print(f'w shape {w.shape}')
