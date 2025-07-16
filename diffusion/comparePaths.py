import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from runPipeline import a_star

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
gt_costmap_path = "data/costmaps/hei22_costmap.npy"
pred_costmap_path = "data/costmaps/pred_costmap22.npy"

start = (0, 0)
goal = (250, 250)

# ----------------------------------------------------
# LOAD GROUND TRUTH COSTMAP
# ----------------------------------------------------
costmap_gt = np.load(gt_costmap_path)

# Normalize GT to [-1, 1] as in training
cmin = costmap_gt.min()
cmax = costmap_gt.max()
if (cmax - cmin) > 0:
    costmap_gt_norm = (costmap_gt - cmin) / (cmax - cmin)
    costmap_gt_norm = costmap_gt_norm * 2 - 1
else:
    costmap_gt_norm = np.zeros_like(costmap_gt)

# Resize to (256,256)
costmap_gt_tensor = torch.tensor(costmap_gt_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
costmap_gt_resized = F.interpolate(
    costmap_gt_tensor,
    size=(256, 256),
    mode="bilinear",
    align_corners=False
).squeeze().numpy()

# Bring GT to [0,1] for A*
gt_costmap_vis = (costmap_gt_resized + 1) / 2
gt_costmap_vis = np.clip(gt_costmap_vis, 1e-5, 1.0)

# ----------------------------------------------------
# LOAD PREDICTED COSTMAP
# ----------------------------------------------------
pred_costmap_vis = np.load(pred_costmap_path)

# ----------------------------------------------------
# RUN A* ON BOTH
# ----------------------------------------------------
path_gt, total_cost_gt = a_star(gt_costmap_vis, start, goal)
path_pred, total_cost_pred = a_star(pred_costmap_vis, start, goal)

# ----------------------------------------------------
# PRINT RESULTS
# ----------------------------------------------------
print("=== GROUND TRUTH ===")
print("Path length:", len(path_gt))
print("Total cost:", total_cost_gt)

print("\n=== PREDICTED COSTMAP ===")
print("Path length:", len(path_pred))
print("Total cost:", total_cost_pred)

# ----------------------------------------------------
# PLOT COMPARISON
# ----------------------------------------------------
# Plot GT
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(gt_costmap_vis, cmap="viridis")
y_gt, x_gt = zip(*path_gt)
plt.plot(x_gt, y_gt, color="red")
plt.title("GT Costmap + Path")
plt.colorbar()

# Plot Predicted
plt.subplot(1,2,2)
plt.imshow(pred_costmap_vis, cmap="viridis")
y_pred, x_pred = zip(*path_pred)
plt.plot(x_pred, y_pred, color="red")
plt.title("Predicted Costmap + Path")
plt.colorbar()

plt.tight_layout()
plt.show()