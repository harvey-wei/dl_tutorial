from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

url = "https://cataas.com/cat"  # Random cat image from cataas.com

if (not os.path.exists("cat.jpg")):
    urllib.request.urlretrieve(url, "cat.jpg")

    print("Cat image saved as cat.jpg")


# Load image using PIL
pil_img = Image.open("cat.jpg")
pil_np = np.array(pil_img)  # Convert to numpy array

# Load image using OpenCV
cv_img = cv2.imread("cat.jpg") # This loads as BGR by default and as a numpy array
cv_img_rgb = cv_img[:, :, ::-1]
# # cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for comparison
#
# # Make the image partially transparent
# pil_np = np.concatenate((pil_np, 0.5 * np.ones((pil_np.shape[0], pil_np.shape[1], 1))), axis=-1)

# Print shapes
print("PIL Image shape:", pil_np.shape)        # (H, W, 3)
print("OpenCV Image shape:", cv_img.shape)     # (H, W, 3)
print("OpenCV RGB-converted shape:", cv_img_rgb.shape)

# Plot both side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("PIL Image (RGB)")
plt.imshow(pil_np)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("OpenCV Image (RGB)")
plt.imshow(cv_img_rgb)
plt.axis("off")

plt.tight_layout()
plt.show()
