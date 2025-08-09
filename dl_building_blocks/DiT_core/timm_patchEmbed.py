from timm.models.vision_transformer import PatchEmbed
import torch

# Parameters
img_size = (256, 512)
patch_size = (8, 16)
in_chans = 3
embed_dim = 768

'''
PatchEmbed divides an input image into non-overlapping patches and projects each patch into a vector
embedding. This is typically implemented using a convolutional layer with a kernel size and stride equal to the patch size.
'''
# Create PatchEmbed instance
patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

# Input image tensor
x = torch.randn(1, 3, img_size[0], img_size[1])  # Batch size 1, 3 channels, 224x224 image

# Apply PatchEmbed
x = patch_embed(x)  # Output shape: (1, num_patches, embed_dim)

# Calculate the target shape
num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
target_shape = (1, num_patches, embed_dim)
# Check the output shape
assert x.shape == target_shape, f"Expected output shape {target_shape}, but got {x.shape}"


print(f"Output shape after PatchEmbed: {x.shape}")
