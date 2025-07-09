import numpy as np
import matplotlib.pyplot as plt
import os

# Folder with elevation npy files
elevation_dir = "data/elevation"

# List all npy files
files = [f for f in os.listdir(elevation_dir) if f.endswith(".npy")]

print(f"Found {len(files)} elevation maps.")

for f in files:
    path = os.path.join(elevation_dir, f)
    arr = np.load(path)

    print(f"Plotting {f} ...")
    
    plt.figure(figsize=(6, 5))
    plt.imshow(arr, cmap="viridis")
    plt.colorbar(label="Elevation (m)")
    plt.title(f"Elevation Map: {f}")
    plt.show()