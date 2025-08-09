import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Simulate a batch of 3 RGB images of size 64x64
B, H, W, C = 3, 64, 64, 3
images = np.random.randint(0, 256, size=(B, H, W, C), dtype=np.uint8)  # shape: (3, 64, 64, 3)

# Loop through each image in the batch
for i, img in enumerate(images):  # Each img: shape (64, 64, 3)
    print(f"Image {i} shape: {img.shape}")

    # Convert to PIL and show it
    pil_img = Image.fromarray(img)
    pil_img.show()  # This will open the image in your default viewer

    # Or display inline using matplotlib (for Jupyter notebooks)
    plt.imshow(pil_img)
    plt.title(f"Image {i}")
    plt.axis("off")
    plt.show()
