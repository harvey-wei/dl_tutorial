import einops
import torch

channels_last = torch.randn(64, 32, 32, 3) # [batch, height, width, channel]
B = torch.randn(32 * 32, 32 * 32)

# copy
channels_first = einops.rearrange(channels_last, "b h w c -> b c (h w)")
print(f'shape of channels_first {channels_first.shape}')
print(f'shape of channels_last {channels_last.shape}')


# h, w = 32, 32
channels_first_transformed = einops.einsum(channels_first, B, "b c in, out in -> b c out")
channels_last_transformed = einops.rearrange(channels_first_transformed, "b c (h w) -> b h w c",
                                             h=32, w=32)
print(f'shape of channels_last_transformed {channels_last_transformed.shape}')

