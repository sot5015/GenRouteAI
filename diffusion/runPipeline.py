import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import heapq
import os
from model import UNet
from betaSchedule import linear_beta_schedule, get_alpha, get_alpha_bar


"""
runPipeline.py

Script for testing the diffusion model pipeline for costmap prediction and path planning.

Pipeline Overview:
------------------
1. Loads a ground truth heightmap (elevation map) and costmap from disk.
2. Normalizes the heightmap and randomly masks regions to simulate missing terrain data.
3. Uses a trained diffusion model (UNet) to predict the complete costmap from the masked heightmap.
4. Applies bias correction to align the predicted costmap's global mean to the ground truth.
5. Computes the Mean Absolute Error (MAE) between the predicted and ground truth costmaps.
6. Plots:
    - Ground truth costmap
    - Predicted costmap
    - Absolute difference map
7. Runs the A* path planning algorithm on the predicted costmap.
8. Visualizes the optimal path:
    - Over the predicted costmap (currently implemented)
    - (could be) Over the masked or reconstructed heightmap

Notes:
------
- The purpose of this pipeline is to test whether the diffusion model can reconstruct realistic costmaps from incomplete heightmaps, and how well the predicted costmaps can support downstream tasks like path planning.
- The red path plotted shows the optimal route computed on the predicted costmap, visualized for analysis.

"""


# ---------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------

def generate_random_mask(shape, block_size=64, num_blocks=10):
    mask = np.ones(shape, dtype=np.float32)
    H, W = shape
    for _ in range(num_blocks):
        top = np.random.randint(0, max(1, H - block_size))
        left = np.random.randint(0, max(1, W - block_size))
        mask[top:top+block_size, left:left+block_size] = 0.0
    return mask

def heuristic(a, b):
    """Euclidean distance as heuristic"""
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(costmap, start, goal):
    rows, cols = costmap.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = reconstruct_path(came_from, current)
            total_cost = g_score[goal]
            return path, total_cost

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                step_cost = costmap[neighbor]
                tentative_g = g_score[current] + step_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current

    return None, float("inf")

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

heightmap_path = "data/elevation/hei22.npy"
costmap_gt_path = "data/costmaps/hei22_costmap.npy"

heightmap = np.load(heightmap_path)
costmap_gt = np.load(costmap_gt_path)

# Normalize heightmap to [0,1]
heightmap_norm = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)

# Generate mask
mask = generate_random_mask(heightmap.shape, block_size=64, num_blocks=10)
masked_heightmap = heightmap_norm * mask + (1 - mask) * 0.5

# Plot masked heightmap
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(heightmap_norm, cmap="gray")
plt.title("Original Heightmap")

plt.subplot(1,3,2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1,3,3)
plt.imshow(masked_heightmap, cmap="gray")
plt.title("Masked Heightmap")

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# DIFFUSION MODEL PREDICTION (direct x0_pred)
# ---------------------------------------------------

device = "mps" if torch.backends.mps.is_available() else "cpu"
T = 100

# Load model
ckpt_path = "results/diffusion-20250713-220255/diffusion_model.pt"
model = UNet(in_channels=2).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# Prepare input
masked_heightmap_tensor = torch.tensor(masked_heightmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
masked_heightmap_tensor = F.interpolate(
    masked_heightmap_tensor,
    size=(256, 256),
    mode="bilinear",
    align_corners=False
)

# Direct prediction
t = torch.full((1,), 0, device=device, dtype=torch.long)
x_input = torch.cat([masked_heightmap_tensor, masked_heightmap_tensor], dim=1)

with torch.no_grad():
    x0_pred = model(x_input, t)

x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
pred_costmap = x0_pred.squeeze().cpu().numpy()

# ---------------------------------------------------
# BIAS CORRECTION + MAE
# ---------------------------------------------------

cmin = costmap_gt.min()
cmax = costmap_gt.max()
if (cmax - cmin) > 0:
    costmap_gt_norm = (costmap_gt - cmin) / (cmax - cmin)
    costmap_gt_norm = costmap_gt_norm * 2 - 1
else:
    costmap_gt_norm = np.zeros_like(costmap_gt)

costmap_gt_tensor = torch.tensor(costmap_gt_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
costmap_gt_resized = F.interpolate(
    costmap_gt_tensor,
    size=(256, 256),
    mode="bilinear",
    align_corners=False
).squeeze().numpy()

mean_gt = costmap_gt_resized.mean()
mean_pred = pred_costmap.mean()
bias = mean_pred - mean_gt

pred_costmap_corrected = pred_costmap - bias
pred_costmap_corrected = np.clip(pred_costmap_corrected, -1.0, 1.0)

mae_corrected = np.mean(np.abs(pred_costmap_corrected - costmap_gt_resized))
print("MAE after bias correction:", mae_corrected)

# Bring to [0,1] for A* and plotting
pred_costmap_vis = (pred_costmap_corrected + 1) / 2
pred_costmap_vis = np.clip(pred_costmap_vis, 1e-5, 1.0)  # avoid zero costs for A*

# Save predicted costmap
np.save("data/costmaps/pred_costmap22.npy", pred_costmap_vis)

# ---------------------------------------------------
# PLOT GT vs Predicted Costmap
# ---------------------------------------------------

costmap_gt_vis = (costmap_gt_resized + 1) / 2
costmap_gt_vis = np.clip(costmap_gt_vis, 0, 1)

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.imshow(costmap_gt_vis, cmap="viridis")
plt.title("Ground Truth Costmap")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(pred_costmap_vis, cmap="viridis")
plt.title("Predicted Costmap")
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(np.abs(pred_costmap_vis - costmap_gt_vis), cmap="hot")
plt.title("Absolute Difference")
plt.colorbar()

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# A* PATH PLANNING
# ---------------------------------------------------

start = (0, 0)
goal = (250, 250)

path, total_cost = a_star(pred_costmap_vis, start, goal)

if path is None:
    print("No path found.")
else:
    print(f"Path found with length: {len(path)}")
    print(f"Total path cost: {total_cost}")

    plt.figure(figsize=(8,6))
    plt.imshow(pred_costmap_vis, cmap="viridis")
    y_coords, x_coords = zip(*path)
    plt.plot(x_coords, y_coords, color="red", linewidth=2)
    plt.title("Optimal Path over Predicted Costmap")
    plt.colorbar(label="Predicted Cost")
    plt.show()
    # y_coords, x_coords = zip(*path)
    