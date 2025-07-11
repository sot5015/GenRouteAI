import numpy as np
import matplotlib.pyplot as plt
import os

"""
Script for visualizing elevation and costmap data:
- Loads elevation (.npy) and costmap (.npy) files
- Displays maps and overlays cost contours
- Plots histograms and scatter plots for analysis
- Useful for checking data quality before path planning
"""
# --- Paths ---
elevation_path = "data/elevation/hei7.npy"
costmap_path = "data/costmaps/hei7_costmap.npy"

# --- Load elevation ---
if not os.path.exists(elevation_path):
    raise FileNotFoundError(f"File not found: {elevation_path}")

elevation = np.load(elevation_path)

# Plot elevation
plt.figure(figsize=(8, 6))
plt.imshow(elevation, cmap="gray")
plt.colorbar()
plt.title("Elevation Map (hei7.npy)")
plt.show()

# --- Load costmap ---
if not os.path.exists(costmap_path):
    raise FileNotFoundError(f"File not found: {costmap_path}")

costmap = np.load(costmap_path)

# --- Check for NaN, Inf, min/max ---
print("===== Costmap Stats =====")
print("Min:", np.min(costmap))
print("Max:", np.max(costmap))
print("Mean:", np.mean(costmap))
print("Std:", np.std(costmap))
print("Any NaNs?", np.isnan(costmap).any())
print("Any Infs?", np.isinf(costmap).any())

# Plot costmap
plt.figure(figsize=(8, 6))
plt.imshow(costmap, cmap="viridis")
plt.colorbar()
plt.title("Costmap (hei7_costmap.npy)")
plt.show()

# --- Overlay costmap on elevation (contours) ---
plt.figure(figsize=(8, 6))
plt.imshow(elevation, cmap="gray")
plt.contour(costmap, levels=10, colors='red')
plt.title("Elevation Map with Costmap Contours")
plt.colorbar()
plt.show()

# --- Overlay costmap on elevation (transparency) ---
plt.figure(figsize=(8, 6))
plt.imshow(elevation, cmap="gray")
plt.imshow(costmap, cmap="viridis", alpha=0.5)
plt.title("Elevation Map + Costmap Overlay")
plt.colorbar()
plt.show()

# --- Histogram of cost values ---
plt.figure(figsize=(6,4))
plt.hist(costmap.flatten(), bins=100)
plt.title("Histogram of Costmap Values")
plt.xlabel("Cost value")
plt.ylabel("Frequency")
plt.show()

# --- Elevation vs Cost scatter plot ---
plt.figure(figsize=(6,4))
plt.scatter(elevation.flatten(), costmap.flatten(), s=1, alpha=0.3)
plt.xlabel("Elevation")
plt.ylabel("Cost")
plt.title("Elevation vs Cost Scatter Plot")
plt.show()

# --- Fake path test (OPTIONAL) ---
# Αν θέλεις να τεστάρεις αν μπορείς να σχεδιάσεις μονοπάτι:
fake_path = [(10,10), (20,20), (30,30), (40,40)]
plt.imshow(costmap, cmap="viridis")
y, x = zip(*fake_path)
plt.plot(x, y, color="red", linewidth=2)
plt.title("Fake path over costmap")
plt.show()