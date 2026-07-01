"""
DDPM sampler with classifier-free guidance.

Given a trained MotionDiT and a text prompt, generates a motion by
iteratively denoising from pure noise. Uses classifier-free guidance
to blend conditional and unconditional predictions.

Usage:
    sampler = DDPMSampler(model, schedule, device)
    motion = sampler.sample(text_context, length=120, guidance_scale=4.0)
    # motion: (F, 263) normalised feature tensor

Sampling always walks the full timestep grid (T-1, T-2, ..., 1): schedule.p_sample()
assumes x is exactly one step of noise ahead (x_t -> x_{t-1}), so skipping steps would
condition the model on the wrong timestep for the noise level x actually carries.
Strided/DDIM-style subsampling needs a respaced schedule (recomputed alphas_cumprod
ratios between the kept steps) and is not implemented — pass num_steps == schedule.T.

LEDITS++ implementation will extend this class with two additional methods:
  invert(x0, num_steps): Stage 1 — runs the forward process x_0 → x_N without
      text conditioning, storing cross-attention maps at each step for mask M1
      and returning the full noisy sequence (x_1, ..., x_N) for frame inpainting.
  edit(x_N, c_edit, mask_M, x0_source, guidance_scale, num_steps): Stage 3 —
      denoises x_N using Eq. 1 (masked SEGA guidance), replacing unedited frames
      from the precomputed noisy sequence after each step.
"""

import torch
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
        num_steps: int | None = None, # must equal schedule.T; see module docstring
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
                Returns a single motion of shape (length, 263), normalised feature vectors.
                Denormalise with: motion * std + mean, then call recover_from_ric() to get
                joint positions for visualisation.
        """
        self.model.eval()
        B = 1

        if num_steps is None:
            num_steps = self.schedule.T
        if num_steps != self.schedule.T:
            raise ValueError(
                f"DDPMSampler only supports full-resolution sampling (num_steps == "
                f"schedule.T == {self.schedule.T}); got num_steps={num_steps}. "
                "Strided sampling is not implemented — see the module docstring."
            )

        # start from pure noise
        x = torch.randn(B, length, self.model.input_dim, device=self.device)

        timesteps = list(range(self.schedule.T - 1, 0, -1))

        it = tqdm(timesteps, desc="Sampling") if show_progress else timesteps

        for t in it:
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
            x = self.schedule.p_sample(x, t_batch, eps)

        return x[0]  # (length, 263)
