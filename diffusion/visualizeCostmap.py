import numpy as np
import matplotlib.pyplot as plt
import os

################# code for visualizing costmaps #################

# Folder containing costmaps
costmap_dir = "data/costmaps"

# List all .npy files
costmap_files = [
    f for f in os.listdir(costmap_dir)
    if f.endswith(".npy")
]

print(f"Found {len(costmap_files)} costmaps.")

for file in costmap_files:
    path = os.path.join(costmap_dir, file)
    
    # Load costmap
    costmap = np.load(path)
    
    # Mask unreachable areas
    mask = costmap < 1e6
    masked_costmap = np.where(mask, costmap, np.nan)

    # Print basic stats
    print(f"Costmap: {file}")
    print("  min cost:", np.nanmin(masked_costmap))
    print("  max cost:", np.nanmax(masked_costmap))
    print("  shape:", costmap.shape)
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.imshow(masked_costmap, cmap="viridis")
    plt.colorbar(label="Cost to reach")
    plt.title(f"Costmap: {file}")
    plt.show()