from numpy import stack
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

# Create two random RGB images (3 channels, 64x64)
img1 = torch.rand(3, 64, 64)  # shape: (C, H, W)
img2 = torch.rand(3, 64, 64)

# -------------------------------
# 1. Use torch.stack to make a batch
# -------------------------------
stacked = torch.stack([img1, img2])  # shape: (2, 3, 64, 64) defaults to dim 0
print("torch.stack shape:", stacked.shape)

stacked_ = torch.stack([img1, img2], dim=1) # (3, 2, 64, 64)
print("torch.stack shape:", stacked_.shape)


# -------------------------------
# 2. Use torch.cat along channels
# -------------------------------
cat_channels = torch.cat([img1, img2], dim=0)  # shape: (6, 64, 64)
print("torch.cat (channels) shape:", cat_channels.shape)

# -------------------------------
# 3. Use torch.cat side-by-side (width)
# -------------------------------
cat_width = torch.cat([img1, img2], dim=2)  # shape: (3, 64, 128)
print("torch.cat (width) shape:", cat_width.shape)

# -------------------------------
# 4. Visualize all 3
# -------------------------------

# Helper to convert (C, H, W) to (H, W, C) for plotting
def show_tensor_img(t, title=""):
    np_img = TF.to_pil_image(t).convert("RGB")
    plt.imshow(np_img)
    plt.title(title)
    plt.axis("off")

plt.figure(figsize=(10, 4))

# Original image 1
plt.subplot(1, 3, 1)
show_tensor_img(img1, "Image 1")

# Stacked batch (show first image in batch)
plt.subplot(1, 3, 2)
show_tensor_img(stacked[1], "stacked[1] (Batch)")

# Concatenated width-wise
plt.subplot(1, 3, 3)
show_tensor_img(cat_width, "cat (side-by-side)")

plt.tight_layout()
plt.show()


'''

Feature	            torch.cat	                                torch.stack
Adds new dimension	❌ No	                                    ✅ Yes (inserts new dimension)
Shape requirement	All tensors must match except along dim	    All tensors must have exactly the same shape
Common use case	    Extend tensors along existing axis	        Combine tensors into a batch
Default dim	        dim=0	                                    dim=0
'''
