import torch

def linear_beta_schedule(timesteps):
    """
    Create a linear schedule for beta from small to larger values.
    Used to gradually increase noise during the forward diffusion.
    """
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_alpha(beta):
    """
    Compute alpha values from beta: αₜ = 1 - βₜ
    """
    return 1. - beta

def get_alpha_bar(alpha):
    """
    Compute cumulative product of alphas: α̅ₜ = ∏ₛ₌₁ᵗ αₛ
    """
    return torch.cumprod(alpha, dim=0)

# Optional test run
if __name__ == "__main__":
    T = 1000
    beta = linear_beta_schedule(T)
    alpha = get_alpha(beta)
    alpha_bar = get_alpha_bar(alpha)
    print("beta[:5]:", beta[:5])
    print("alpha_bar[-1]:", alpha_bar[-1])