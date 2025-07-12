import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import UNet
from betaSchedule import linear_beta_schedule, get_alpha, get_alpha_bar

# Purpose:
# Test if the diffusion model can reconstruct a costmap 
# given a heightmap with missing data (masked regions).
# Compares the model's prediction to the ground-truth
# and visualizes the difference.

# ----- Load original heightmap & costmap -----
heightmap_path = "data/elevation/hei7.npy"
costmap_path = "data/costmaps/hei7_costmap.npy"

heightmap = np.load(heightmap_path)
costmap_gt = np.load(costmap_path)

# Normalize heightmap to [0, 1]
heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)

# ----- Create mask -----
def generate_random_mask(shape, block_size=32, num_blocks=10):
    mask = np.ones(shape, dtype=np.float32)
    H, W = shape
    for _ in range(num_blocks):
        top = np.random.randint(0, max(1, H - block_size))
        left = np.random.randint(0, max(1, W - block_size))
        mask[top:top+block_size, left:left+block_size] = 0.0
    return mask

mask = generate_random_mask(heightmap.shape, block_size=64, num_blocks=5)
masked_heightmap = heightmap * mask + (1 - mask) * 0.5
# Plot mask and masked heightmap
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(heightmap, cmap="gray")
plt.title("Original Heightmap")

plt.subplot(1,3,2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1,3,3)
plt.imshow(masked_heightmap, cmap="gray")
plt.title("Masked Heightmap")

plt.show()

# ----- Prepare model -----
device = "mps" if torch.backends.mps.is_available() else "cpu"
T = 1000

beta = linear_beta_schedule(T).to(device)
alpha = get_alpha(beta).to(device)
alpha_bar = get_alpha_bar(alpha).to(device)

# Load trained model
ckpt_path = "results/diffusion-20250712-173753/diffusion_model.pt"
model = UNet(in_channels=2).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# ----- Prepare input -----
# Resize masked heightmap
masked_heightmap_tensor = torch.tensor(masked_heightmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
masked_heightmap_tensor = F.interpolate(
    masked_heightmap_tensor,
    size=(256, 256),
    mode="bilinear",
    align_corners=False
)

# Start from pure noise
x_t = torch.randn((1, 1, 256, 256), device=device)

# ----- Reverse Sampling Loop -----
for curr_t in reversed(range(T)):
    t = torch.full((1,), curr_t, device=device, dtype=torch.long)

    # Concatenate noise and masked heightmap
    x_input = torch.cat([x_t, masked_heightmap_tensor], dim=1)

    with torch.no_grad():
        predicted_noise = model(x_input, t)

    beta_t = beta[curr_t]
    alpha_t = alpha[curr_t]
    alpha_bar_t = alpha_bar[curr_t]

    # Estimate x0
    x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

    # Sample x_{t-1}
    if curr_t > 0:
        noise = torch.randn_like(x_t)
        x_t = (
            torch.sqrt(alpha_t) * x0_pred +
            torch.sqrt(1 - alpha_t) * noise
        )
    else:
        x_t = x0_pred

# Convert result
pred_costmap = x_t.squeeze().cpu().numpy()
pred_costmap_vis = (pred_costmap + 1) / 2
pred_costmap_vis = np.clip(pred_costmap_vis, 0, 1)

# ----- Resize ground-truth costmap -----
costmap_gt_tensor = torch.tensor(costmap_gt, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
costmap_gt_resized = F.interpolate(
    costmap_gt_tensor,
    size=(256, 256),
    mode="bilinear",
    align_corners=False
).squeeze().numpy()

# Normalize GT costmap for comparison
costmap_gt_resized_norm = costmap_gt_resized / (costmap_gt_resized.max() + 1e-8)

# ----- Plot -----
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(costmap_gt_resized_norm, cmap="viridis")
plt.title("Ground Truth Costmap (resized)")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(pred_costmap_vis, cmap="viridis")
plt.title("Predicted Costmap (masked input)")
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(np.abs(pred_costmap_vis - costmap_gt_resized_norm), cmap="hot")
plt.title("Absolute Difference")
plt.colorbar()

plt.tight_layout()
plt.show()

# ----- Optional metrics -----
mae = np.mean(np.abs(pred_costmap_vis - costmap_gt_resized_norm))
print("MAE:", mae)