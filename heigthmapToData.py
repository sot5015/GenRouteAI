import cv2
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

# Load the image
img = cv2.imread("Schiaparelli_crater_scale.jpg")
color_bar = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Crop the color bar manually (you might need to adjust these based on the image)
# color_bar = img_rgb[0:20, 0:img.shape[1]]  # Crop the horizontal strip of the colorbar

# Get the width of the color bar
width = color_bar.shape[1]

# Define the corresponding value range
min_val, max_val = -2000, 2500
values = np.linspace(min_val, max_val, width)

# Sample the middle row to get color for each pixel
colors = color_bar[color_bar.shape[0] // 2].reshape(-1, 3)

# Create mapping of value -> color
value_color_map = dict(zip(values, [tuple(c) for c in colors]))

# # (Optional) Display a few samples
# for i in range(0, width, width // 10):
#     print(f"Value: {values[i]:.0f}, Color: {value_color_map[values[i]]}")

# # (Optional) Plot the extracted colors
# plt.figure(figsize=(10, 1))
# plt.imshow([colors], extent=[min_val, max_val, 0, 1])
# plt.title("Extracted Colormap")
# plt.yticks([])
# plt.xlabel("Value (m)")
# plt.show()

# print(value_color_map)
# Build KDTree for fast color lookup
tree = KDTree(colors.astype(np.float32))

# Load your data image
data_img = cv2.imread("test1.jpg")
data_rgb = cv2.cvtColor(data_img, cv2.COLOR_BGR2RGB)
h, w, _ = data_rgb.shape
pixels = data_rgb.reshape(-1, 3)

# Query nearest color in colormap for each pixel
_, idx = tree.query(pixels.astype(np.float32), k=1)
value_array = values[idx.flatten()].reshape(h, w)
print(value_array[150:160, 150:160])
# Result: value_array is the same shape as your image with real values
# print(value_array)