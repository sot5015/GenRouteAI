import torch
from torch.utils.data import DataLoader
from model import UNet
from forward import forward_diffusion_sample
from betaSchedule import linear_beta_schedule, get_alpha, get_alpha_bar
from dataset import HeightmapDataset
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----- Hyperparameters -----
T = 1000
BATCH_SIZE = 24
LR = 5e-5
NUM_EPOCHS = 500

# ----- Device -----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ----- Diffusion schedule -----
beta = linear_beta_schedule(T).to(device)
alpha = get_alpha(beta).to(device)
alpha_bar = get_alpha_bar(alpha).to(device)

# ----- Prepare file lists -----
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

assert len(heightmap_paths) == len(costmap_paths), "Mismatched heightmap and costmap counts!"

# ----- Dataset -----
dataset = HeightmapDataset(
    heightmap_paths=heightmap_paths,
    costmap_paths=costmap_paths,
    target_size=(256, 256)
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- Create run folder -----
now = datetime.now().strftime("%Y%m%d-%H%M%S")
save_dir = f"results/diffusion-{now}"
os.makedirs(save_dir, exist_ok=True)
print(f"Checkpoints & logs will be saved in: {save_dir}")

# ----- Model -----
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model.train()

loss_history = []

# ----- Training loop -----
for epoch in tqdm(range(NUM_EPOCHS), desc="Training epochs"):
    epoch_losses = []
    for heightmap, costmap, mask in dataloader:
        B = heightmap.shape[0]
        costmap = costmap.to(device)

        t = torch.randint(0, T, (B,), device=device)

        x_t, noise = forward_diffusion_sample(costmap, t, alpha_bar)

        predicted_noise = model(x_t, t)

        loss = torch.mean((predicted_noise - noise) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses.append(loss.item())

    mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
    loss_history.append(mean_epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"[Epoch {epoch+1}] Mean loss: {mean_epoch_loss:.4f}")

# Plot loss curve at the end of training
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Mean Loss")
plt.title("Training Loss Curve")
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
plt.close()
print(f"Saved loss curve plot to {save_dir}/loss_curve.png")

# Save final model
torch.save(model.state_dict(), os.path.join(save_dir, "diffusion_model.pt"))
print(f"Model saved to {save_dir}/diffusion_model.pt")

print("Training complete.")