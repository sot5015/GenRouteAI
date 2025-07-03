import torch
from tqdm import tqdm
from model import UNet
from betaSchedule import linear_beta_schedule, get_alpha, get_alpha_bar
import matplotlib.pyplot as plt

@torch.no_grad()
def sample(model, alpha, alpha_bar, T, shape, device="cpu"):
    """
    Run reverse diffusion sampling to generate a new sample from noise.

    Parameters:
    - model: trained UNet model
    - alpha: tensor of alpha_t values (length T)
    - alpha_bar: tensor of alpha_bar_t values (length T)
    - T: total number of diffusion steps
    - shape: tuple with desired sample shape, e.g. (1, 64, 64)
    - device: "cpu" or "cuda"

    Returns:
    - Generated sample tensor with shape `shape`
    """

    # Start from pure Gaussian noise
    x_t = torch.randn((1,) + shape).to(device)

    # Loop backwards through time steps
    for t in tqdm(reversed(range(T)), total=T, desc="Sampling"):
        # Create a batch of timesteps
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)

        # Predict noise using the trained model
        eps_theta = model(x_t, t_batch)

        # Extract scalars for this timestep
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t]

        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)
        sqrt_alpha_recip = torch.sqrt(1.0 / alpha_t)

        # Compute the mean of the posterior p(x_{t-1} | x_t)
        mean = sqrt_alpha_recip * (x_t - (1 - alpha_t) / sqrt_one_minus_alpha_bar * eps_theta)

        if t > 0:
            # Add noise for stochastic sampling
            beta_t = 1 - alpha_t
            sigma = torch.sqrt(beta_t)
            z = torch.randn_like(x_t)
            x_t = mean + sigma * z
        else:
            # Final step - no noise
            x_t = mean

    # Remove batch dimension and return sample
    return x_t.squeeze(0)

# ----------------------------
if __name__ == "__main__":
    device = "cpu"

    # Hyperparams
    T = 1000
    size = (1, 256, 256)

    # Load beta schedule
    beta = linear_beta_schedule(T)
    alpha = get_alpha(beta)
    alpha_bar = get_alpha_bar(alpha)

    # Load trained model
    model = UNet().to(device)
    model.load_state_dict(torch.load("diffusion_model.pt", map_location=device))
    model.eval()

    # Sample
    x_sample = sample(model, alpha, alpha_bar, T, size, device)

    # Plot
    import matplotlib.pyplot as plt
    plt.imshow(x_sample.squeeze().cpu(), cmap="viridis")
    plt.title("Generated Sample")
    plt.colorbar()
    plt.show()