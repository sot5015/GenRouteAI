import torch
from torch.utils.data import DataLoader
from model import UNet
from forward import forward_diffusion_sample
from betaSchedule import linear_beta_schedule, get_alpha, get_alpha_bar
from dataset import HeightmapDataset
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----- Hyperparameters -----
T = 1000
BATCH_SIZE = 24
LR = 5e-5
NUM_EPOCHS = 1000
SAVE_EVERY = 10
MEAN_PENALTY_WEIGHT = 0.1     # <--- μπορείς να το πειράξεις αν θες

# ----- Device check: cross-platform -----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Using device:", device)

# ----- Diffusion schedule -----
beta = linear_beta_schedule(T).to(device)
alpha = get_alpha(beta).to(device)
alpha_bar = get_alpha_bar(alpha).to(device)

# ----- Dataset -----
heightmap_paths = [
    os.path.join("data/elevation", f)
    for f in os.listdir("data/elevation")
    if f.endswith(".npy")
]
costmap_paths = [
    os.path.join("data/costmaps", os.path.splitext(f)[0] + "_costmap.npy")
    for f in os.listdir("data/elevation")
    if f.endswith(".npy")
]
assert len(heightmap_paths) == len(costmap_paths), "Mismatch in dataset!"

dataset = HeightmapDataset(
    heightmap_paths=heightmap_paths,
    costmap_paths=costmap_paths,
    target_size=(256, 256)
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- Run folder -----
now = datetime.now().strftime("%Y%m%d-%H%M%S")
save_dir = f"results/diffusion-{now}"
os.makedirs(save_dir, exist_ok=True)
print(f"Checkpoints & logs will be saved in: {save_dir}")

# ----- Model -----
model = UNet(in_channels=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model.train()

loss_history = []
start_time = time.time()
progress_bar = tqdm(range(NUM_EPOCHS), desc="Training epochs")

for epoch in progress_bar:
    epoch_losses = []
    for heightmap, costmap, mask in dataloader:
        B = costmap.shape[0]
        costmap = costmap.to(device)
        heightmap = heightmap.to(device)

        # Sample timestep
        t = torch.randint(0, T, (B,), device=device)
        x_t, noise = forward_diffusion_sample(costmap, t, alpha_bar)

        # Concatenate noisy costmap + heightmap
        x_input = torch.cat([x_t, heightmap], dim=1)  # (B, 2, H, W)

        # Model prediction
        predicted_noise = model(x_input, t)

        # --- Compute losses ---
        mse_loss = torch.mean((predicted_noise - noise) ** 2)

        mean_pred = predicted_noise.mean()
        mean_target = noise.mean()
        mean_bias_loss = (mean_pred - mean_target) ** 2

        # Total loss
        loss = mse_loss + MEAN_PENALTY_WEIGHT * mean_bias_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    loss_history.append(mean_epoch_loss)
    progress_bar.set_postfix(loss=mean_epoch_loss)

    if (epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
        ckpt_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(
            f"[Epoch {epoch+1}] Loss: {mean_epoch_loss:.6f} "
            f"| Last MSE Loss: {mse_loss.item():.6f} "
            f"| Mean Bias Loss: {mean_bias_loss.item():.6f} "
            f"| Saved: {ckpt_path}"
        )

# ----- End -----
elapsed = time.time() - start_time
print(f"Total training time: {elapsed/60:.2f} minutes")

plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Mean Loss")
plt.title("Training Loss Curve")
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.close()
print(f"Loss curve saved: {save_dir}/loss_curve.png")

torch.save(model.state_dict(), os.path.join(save_dir, "diffusion_model.pt"))
print(f"Final model saved: {save_dir}/diffusion_model.pt")
print("Training complete.")