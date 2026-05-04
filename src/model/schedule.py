import torch
import numpy as np


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    """
    Cosine schedule from Improved DDPM (Nichol & Dhariwal, 2021).
    Produces a smoother noise schedule than the original linear schedule,
    which is important for motion since it preserves structure longer.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, min=1e-4, max=0.9999)


class NoiseSchedule:
    """
    Precomputes and stores all quantities needed for DDPM forward/reverse.
    All tensors are registered on the given device.
    """

    def __init__(self, timesteps: int = 1000, device="cpu"):
        self.T = timesteps
        self.device = device

        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), alphas_cumprod[:-1]]
        )

        self.betas               = betas
        self.alphas              = alphas
        self.alphas_cumprod      = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev

        # quantities used in the forward (noising) process
        self.sqrt_alphas_cumprod       = alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        # quantities used in the reverse (denoising) process
        self.posterior_variance = (
            betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        self.sqrt_recip_alphas = (1.0 / alphas).sqrt()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise=None):
        """
        Forward process: add noise to x0 at timestep t.
        x0   : (B, T, D)
        t    : (B,) integer timesteps
        Returns x_t of the same shape as x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_acp = self.sqrt_alphas_cumprod[t]             # (B,)
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t] # (B,)

        # reshape for broadcasting over (T, D)
        sqrt_acp   = sqrt_acp[:, None, None]
        sqrt_omacp = sqrt_omacp[:, None, None]

        return sqrt_acp * x0 + sqrt_omacp * noise, noise

    def predict_x0_from_eps(self, x_t, t, eps):
        """Recover x0 prediction from predicted noise eps."""
        sqrt_acp   = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return (x_t - sqrt_omacp * eps) / sqrt_acp

    def p_sample(self, x_t, t, eps_pred):
        """Reverse DDPM step: compute x_{t-1} from x_t and predicted noise."""
        beta_t       = self.betas[t][:, None, None]
        alpha_t      = self.alphas[t][:, None, None]
        acp_t        = self.alphas_cumprod[t][:, None, None]
        sqrt_omacp_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        x0_pred = (x_t - sqrt_omacp_t * eps_pred) / acp_t.sqrt()
        x0_pred = x0_pred.clamp(-5, 5)

        # alphas_cumprod[0] == 1.0 exactly, so 1-acp_t == 0 at t=0.
        # Guard the denominator; the t==0 branch never uses mean anyway.
        nonzero_t = (t > 0).float()[:, None, None]
        denom = (1 - acp_t).clamp(min=1e-8)
        coef1 = (self.alphas_cumprod_prev[t][:, None, None].sqrt() * beta_t) / denom
        coef2 = (alpha_t.sqrt() * (1 - self.alphas_cumprod_prev[t][:, None, None])) / denom
        mean = coef1 * x0_pred + coef2 * x_t

        noise = torch.randn_like(x_t)
        var = self.posterior_variance[t][:, None, None]
        # At t=0 the posterior is degenerate; return x0_pred directly.
        return torch.where(nonzero_t > 0, mean + var.sqrt() * noise, x0_pred)
