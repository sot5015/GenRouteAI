import torch

def linear_beta_schedule(timesteps):
    """
    Create a linear schedule for beta from small to larger values.
    Used to gradually increase noise during the forward diffusion.
    """
    beta_start = 1e-4               # Minimum noise value at t=1
    beta_end = 0.02                 # Maximum noise value at t=T
    return torch.linspace(beta_start, beta_end, timesteps)  # βₜ from t=1 to t=T

def get_alpha(beta):
    """
    Compute alpha values from beta: αₜ = 1 - βₜ
    """
    return 1. - beta                # αₜ = 1 - βₜ

def get_alpha_bar(alpha):
    """
    Compute cumulative product of alphas: α̅ₜ = ∏ₛ₌₁ᵗ αₛ
    """
    return torch.cumprod(alpha, dim=0)  # α̅ₜ = α₁ * α₂ * ... * αₜ

# Optional test run
if __name__ == "__main__":
    T = 1000                                        # Number of diffusion steps
    beta = linear_beta_schedule(T)                 # Linear βₜ schedule
    alpha = get_alpha(beta)                        # Compute αₜ
    alpha_bar = get_alpha_bar(alpha)               # Compute α̅ₜ

    print("beta[:5]:", beta[:5])                   # First few beta values
    print("alpha_bar[-1]:", alpha_bar[-1])         # Final α̅ₜ (very small)