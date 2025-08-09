import torch
from einops import einsum, rearrange

B, H, W, C = 16, 52, 12, 3
d_dim = 72
D = torch.rand([B, H, W, C, ])
A = torch.rand([d_dim, C])

Y = D @ A.T

Y_new = einsum(D, A, "b h w c, d_dim c -> b h w d_dim ")
print(torch.allclose(Y, Y_new, atol=1e-4))



images = torch.randn([B, H, W, C])
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)


dim_by = rearrange(dim_by, "dim_val -> 1 dim_val 1 1 1")
images = rearrange(images, "B H W C -> B 1 H W C")
dimmed_images = images * dim_by




# None is unsqueeze
# dim_by =  dim_by[None, :, None, None, None]
# images = images[:, None, :, :, :]
#
# dimmed_images = images * dim_by
print(f"shape of dimmed_images{dimmed_images.shape}")


a = torch.rand(4)
b = torch.rand(4)
c = einsum(a, b, 'd, d ->')
d = a.dot(b)
print(f'c {c}')
print(f'd {d}')
print(torch.allclose(c, d, atol=1e-6))
