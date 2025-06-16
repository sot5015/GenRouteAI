import torch


def forward_diffusion_sample(x_0, t, alpha_bar, noise = None):
    """
    Adds noise to the input x_0 at timestep t using alpha_bar values.
    
    Parameters:
    - x_0: original clean image (e.g., elevation map), shape (B, C, H, W)
    - t: tensor of timesteps, shape (B,)
    - alpha_bar: precomputed alpha_bar schedule (1D tensor of length T)
    - noise: optional noise (default: Gaussian)

    Returns:
    - x_t: noisy version of x_0
    - noise: the noise that was added (used for training)
    """
    if noise is None:
        noise = torch.randn_like(x_0)  # Sample standard Gaussian noise

    # Gather alpha_bar_t for each sample in the batch and reshape
    sqrt_alpha_bar = torch.sqrt(alpha_bar[t]).reshape(-1, 1, 1, 1)           # shape (B, 1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar[t]).reshape(-1, 1, 1, 1)

    # Compute noisy x_t
    x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
    return x_t, noise

if __name__ == "__main__":
    # Dummy test
    B, C, H, W = 4, 1, 64, 64
    T = 1000
    x_0 = torch.randn(B, C, H, W)
    t = torch.randint(0, T, size=(B,))

    from betaSchedule import linear_beta_schedule, get_alpha, get_alpha_bar
    beta = linear_beta_schedule(T)
    alpha = get_alpha(beta)
    alpha_bar = get_alpha_bar(alpha)

    x_t, noise = forward_diffusion_sample(x_0, t, alpha_bar)
    print("x_t shape:", x_t.shape)