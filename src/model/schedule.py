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

        # SNR(t) = ᾱ_t / (1 − ᾱ_t). Clamped denominator avoids ÷0 at t=0.
        self.snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise=None):
        """
        Forward process: add noise to x0 at timestep t.
        x0   : (B, T, D)
        t    : (B,) integer timesteps
        Returns (x_t, noise), both the same shape as x0. The noise is returned
        so the training loss can regress against the exact epsilon target.

        LEDITS++ Stage 1: edit-friendly inversion runs the forward process in
        reverse (t=0 → t=N), stepping x_t → x_{t+1} at each inversion step.
        The inversion recurrence (Huberman-Spiegelglas et al. 2024, Eq. 5) uses
        alphas_cumprod and sqrt_one_minus_alphas_cumprod from this schedule.
        The resulting (x_1, ..., x_N) sequence provides the noisy starting
        points for hard frame inpainting in Stage 3.
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
        """Recover x0 prediction from predicted noise eps.

        LEDITS++ Stage 2 mask M2: the guidance vector ψ = ε_θ(x_t, c_edit) − ε_θ(x_t, ∅)
        is NOT directly this function, but predict_x0_from_eps applied to each gives the
        corresponding x0 predictions. The element-wise magnitude |ψ| thresholded at the
        λ-th percentile produces the noise-estimate mask M2 over (J × F) space.
        """
        sqrt_acp   = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return (x_t - sqrt_omacp * eps) / sqrt_acp

    def min_snr_weight(self, t, gamma):
        """Per-sample Min-SNR weight: min(SNR(t), γ) / SNR(t) (Hang et al. 2023).

        Used by the diffusion eps-MSE. Only suppresses low-noise/low-t samples
        (SNR(t) ≫ γ there); it stays ≈1 for t roughly ≳500, so it does NOT protect
        against high-noise instability — see `x0_confidence_weight` for that.
        """
        snr_t = self.snr[t]  # (B,)
        return snr_t.clamp(max=gamma) / snr_t

    def x0_confidence_weight(self, t):
        """Per-sample weight reflecting how reliable x0_pred is at timestep t: ᾱ_t.

        `predict_x0_from_eps` divides by sqrt(ᾱ_t), which →0 as t→T, so any eps
        prediction error is amplified without bound at high noise — and FK-based
        losses (smplh_geometric_losses) compose that error multiplicatively down
        the kinematic chain, making it worse still. ᾱ_t ≈1 at low noise (x0_pred
        reliable) and →0 at high noise (x0_pred meaningless), so weighting by it
        fades out any loss computed on x0_pred exactly where that loss stops being
        meaningful. NOT the same as `min_snr_weight` (see there for why).
        """
        return self.alphas_cumprod[t]  # (B,)

    def posterior_mean(self, x0, x_t, t):
        """DDPM posterior mean μ̃_t(x0, x_t) = E[x_{t-1} | x_t, x0].

        Shared by p_sample (reverse sampling) and the edit-friendly inversion
        (LEDITS++ Stage 1), so both define x_{t-1} consistently. x0 is the
        (predicted or true) clean signal; x_t the current noisy sample.
        """
        beta_t   = self.betas[t][:, None, None]
        alpha_t  = self.alphas[t][:, None, None]
        acp_t    = self.alphas_cumprod[t][:, None, None]
        acp_prev = self.alphas_cumprod_prev[t][:, None, None]

        denom = (1 - acp_t).clamp(min=1e-8)
        coef1 = (acp_prev.sqrt() * beta_t) / denom
        coef2 = (alpha_t.sqrt() * (1 - acp_prev)) / denom
        return coef1 * x0 + coef2 * x_t

    def p_sample(self, x_t, t, eps_pred):
        """Reverse DDPM step: compute x_{t-1} from x_t and predicted noise.

        LEDITS++ Stage 3: after computing the masked eps_hat (Eq. 1 in the proposal),
        this function performs the reverse step. After each call, frames whose mask
        column is all-zero must be overwritten with q_sample(x0_source, t-1) — the
        source motion noised to t-1 — to guarantee zero drift on unedited frames.
        """
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred).clamp(-5, 5)
        mean    = self.posterior_mean(x0_pred, x_t, t)

        # alphas_cumprod[0] == 1.0 exactly, so 1-acp_t == 0 at t=0 → posterior
        # is degenerate; return x0_pred directly there.
        nonzero_t = (t > 0).float()[:, None, None]
        noise = torch.randn_like(x_t)
        var = self.posterior_variance[t][:, None, None]
        return torch.where(nonzero_t > 0, mean + var.sqrt() * noise, x0_pred)
