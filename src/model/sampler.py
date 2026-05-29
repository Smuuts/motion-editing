"""
DDPM sampler with classifier-free guidance.

Given a trained MotionDiT and a text prompt, generates a motion by
iteratively denoising from pure noise. Uses classifier-free guidance
to blend conditional and unconditional predictions.

Usage:
    sampler = DDPMSampler(model, schedule, device)
    motion = sampler.sample(text_context, length=120, guidance_scale=4.0)
    # motion: (F, 263) normalised feature tensor

LEDITS++ implementation will extend this class with two additional methods:
  invert(x0, num_steps): Stage 1 — runs the forward process x_0 → x_N without
      text conditioning, storing cross-attention maps at each step for mask M1
      and returning the full noisy sequence (x_1, ..., x_N) for frame inpainting.
  edit(x_N, c_edit, mask_M, x0_source, guidance_scale, num_steps): Stage 3 —
      denoises x_N using Eq. 1 (masked SEGA guidance), replacing unedited frames
      from the precomputed noisy sequence after each step.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


class DDPMSampler:
    def __init__(self, model, schedule, device):
        self.model    = model
        self.schedule = schedule
        self.device   = device

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,       # (1, L, context_dim) text embedding
        length: int = 120,            # number of frames to generate
        guidance_scale: float = 4.0,  # CFG scale; 1.0 = no guidance
        num_steps: int = 1000,        # can reduce for faster sampling
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
                Returns a single motion of shape (length, 263), normalised feature vectors.
                Denormalise with: motion * std + mean, then call recover_from_ric() to get
                joint positions for visualisation.
        """
        self.model.eval()
        B = 1

        # start from pure noise
        x = torch.randn(B, length, self.model.input_dim, device=self.device)

        stride = max(1, self.schedule.T // num_steps)
        timesteps = list(reversed(range(stride, self.schedule.T, stride)))

        it = tqdm(timesteps, desc="Sampling") if show_progress else timesteps

        for i, t in enumerate(it):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)

            # conditional prediction
            eps_cond = self.model(x, t_batch, context)

            # unconditional prediction (null context)
            eps_uncond = self.model(x, t_batch, context=None)

            # Plain CFG — for LEDITS++ Stage 3 this is replaced by Eq. 1:
            #   ε̂_0(x_t, c_e) = ε̂_0(x_t, ∅) + Σ_i s_e · M_i · [ε_θ(x_t, c_e) − ε_θ(x_t, ∅)]
            # where M_i = M1_i ∩ M2_i is the per-instruction spatiotemporal mask.
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # DDPM reverse step. For LEDITS++ Stage 3: after this step, frames
            # whose mask column is all-zero must be overwritten with
            # schedule.q_sample(x0_source, t-1) to guarantee zero drift.
            x = self._ddpm_step(x, t_batch, eps)

        return x[0]  # (length, 263)

    def _ddpm_step(self, x_t, t, eps_pred):
        """One step of the DDPM reverse process."""
        s = self.schedule

        beta_t        = s.betas[t][:, None, None]
        alpha_t       = s.alphas[t][:, None, None]
        acp_t         = s.alphas_cumprod[t][:, None, None]
        sqrt_omacp_t  = s.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        # predicted x0
        x0_pred = (x_t - sqrt_omacp_t * eps_pred) / acp_t.sqrt()
        x0_pred = x0_pred.clamp(-5, 5)  # stability clip

        # posterior mean
        nonzero_t = (t > 0).float()[:, None, None]
        denom = (1 - acp_t).clamp(min=1e-8)
        coef1 = (s.alphas_cumprod_prev[t][:, None, None].sqrt() * beta_t) / denom
        coef2 = (alpha_t.sqrt() * (1 - s.alphas_cumprod_prev[t][:, None, None])) / denom
        mean  = coef1 * x0_pred + coef2 * x_t

        noise = torch.randn_like(x_t)
        var   = s.posterior_variance[t][:, None, None]
        return torch.where(nonzero_t > 0, mean + var.sqrt() * noise, x0_pred)
