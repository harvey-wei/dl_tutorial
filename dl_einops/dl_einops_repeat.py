import torch
from einops import rearrange, repeat, einsum

image = torch.randn(32, 32)

# repeat along new axis
image = repeat(image, 'h w -> h w c', c=3)
print(f'image shape {image.shape}')

# repeat along new axis
img_rgbr = repeat(image, 'h w c -> h w c a', a=2)

# repeat along w -> three images along width axis
img_repeate_w = repeat(image, 'h w c -> h (repeat w) c', repeat=3)

# order axes matters, we can repeat each element times
# if repeat = 1 one image, now repeat = 3 -> element along c repeats 3 times in width
img_rep = repeat(image, 'h w c -> h (w repeat) c', repeat=3)

# repeat along h -> three images along height axis
img_repeate_h = repeat(image, 'h w c -> (repeat h) w c', repeat=2)

# repeat alog h and w
img_repeate_hw = repeat(image, 'h w c -> (2 h) (2 w) c')




# img_rgbr = rearrange(image, 'h w c -> h w c a', a = 1)
print(f'img_rgbr shape {img_rgbr.shape}')

a = torch.randn(16, 16)
# a = a[..., None]
a = repeat(a, 'h w -> h w c', c=1)
print(f'a shape {a.shape}')

b = torch.randn(16, 16)
b = b[..., None] # [16, 16, 1]



mask = torch.tensor([True, False, True])
print(~mask)     # → array([False,  True, False])


print(int(True))
print(int(False))

