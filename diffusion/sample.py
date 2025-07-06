import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import UNet
from betaSchedule import linear_beta_schedule, get_alpha, get_alpha_bar

@torch.no_grad()
def sample(model, alpha, alpha_bar, T, shape, device):
    """
    Run reverse diffusion sampling.

    Args:
        model: trained UNet model
        alpha: tensor of α_t values
        alpha_bar: tensor of α̅_t values
        T: number of diffusion steps
        shape: tuple (C, H, W)
        device: computation device

    Returns:
        A tensor of generated sample of shape (C, H, W)
    """

    # Start from pure Gaussian noise
    x_t = torch.randn((1,) + shape, device=device)

    for t in tqdm(reversed(range(T)), total=T, desc="Sampling"):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)

        # Predict noise
        eps_theta = model(x_t, t_batch)

        # Compute coefficients
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]

        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        sqrt_alpha_recip = torch.sqrt(1.0 / alpha_t)

        mean = sqrt_alpha_recip * (
            x_t - (1 - alpha_t) / sqrt_one_minus_alpha_bar * eps_theta
        )

        if t > 0:
            beta_t = 1 - alpha_t
            sigma = torch.sqrt(beta_t)
            z = torch.randn_like(x_t)
            x_t = mean + sigma * z
        else:
            x_t = mean

        # Optionally clamp to avoid exploding values
        x_t = torch.clamp(x_t, -5.0, 5.0)

    return x_t.squeeze(0)


def main(args):
    # ----- Device -----
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Sampling device:", device)

    # ----- Diffusion Schedule -----
    T = 1000
    beta = linear_beta_schedule(T).to(device)
    alpha = get_alpha(beta).to(device)
    alpha_bar = get_alpha_bar(alpha).to(device)

    # ----- Load trained model -----
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    model.eval()

    # ----- Prepare output folder -----
    save_dir = Path(args.save_dir or ".")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = save_dir / f"samples_{timestamp}"
    run_dir.mkdir(exist_ok=True, parents=True)
    print(f"Samples will be saved in: {run_dir}")

    # ----- Generate samples -----
    for i in range(args.num_samples):
        sample_img = sample(
            model,
            alpha,
            alpha_bar,
            T,
            shape=(1, 256, 256),
            device=device
        )

        # Normalize to [0, 1] for visualization
        sample_img_vis = (sample_img + 1) / 2
        sample_img_vis = sample_img_vis.clamp(0, 1)
        
        # Save each image
        out_path = run_dir / f"sample_{i}.png"
        plt.imsave(
            out_path,
            sample_img_vis.squeeze().cpu().numpy(),
            cmap="viridis"
        )
        print(f"Saved sample: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to diffusion_model.pt")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save samples")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    args = parser.parse_args()

    main(args)