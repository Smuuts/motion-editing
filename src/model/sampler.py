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

            # conditional prediction. to_eps is the identity for an eps-head and the
            # exact x0->eps conversion for an x0-head (Option 5), so CFG and the
            # reverse step below are unchanged for either parameterisation.
            eps_cond = self.schedule.to_eps(self.model(x, t_batch, context), x, t_batch)

            # unconditional prediction (null context)
            eps_uncond = self.schedule.to_eps(
                self.model(x, t_batch, context=None), x, t_batch)

            # Plain CFG — for LEDITS++ Stage 3 this is replaced by Eq. 1:
            #   ε̂_0(x_t, c_e) = ε̂_0(x_t, ∅) + Σ_i s_e · M_i · [ε_θ(x_t, c_e) − ε_θ(x_t, ∅)]
            # where M_i = M1_i ∩ M2_i is the per-instruction spatiotemporal mask.
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # DDPM reverse step. For LEDITS++ Stage 3: after this step, frames
            # whose mask column is all-zero must be overwritten with
            # schedule.q_sample(x0_source, t-1) to guarantee zero drift.
            x = self.schedule.p_sample(x, t_batch, eps)

        return x[0]  # (length, 263)

    @torch.no_grad()
    def sample_paired(
        self,
        contexts: torch.Tensor,       # (B, L, context_dim) one text embedding per row
        length: int = 120,
        guidance_scale: float = 4.0,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate one motion per context row along ONE shared noise path. (B, F, D).

        Identical to `sample` except that the initial x_T and every posterior draw z_t
        are drawn once and reused across the batch, so two rows can only diverge through
        their text. That is the paired-sample trick Option 6 is built on
        (docs/AttentionGrounding_Options.md): with independent noise, two generations of
        the same prompt already differ everywhere, and the text-driven difference is not
        recoverable from a single pair.

        The pairing is exact, not approximate: rows that carry the *same* context stay
        bitwise identical for all T steps, which is what `probe_gen_diff.py` asserts as
        its plumbing check. Note that the shared z_t makes the rows statistically
        dependent — this returns a matched set for differencing, not B i.i.d. samples.
        """
        self.model.eval()
        B = contexts.shape[0]
        D = self.model.input_dim

        if num_steps is None:
            num_steps = self.schedule.T
        if num_steps != self.schedule.T:
            raise ValueError(
                f"DDPMSampler only supports full-resolution sampling (num_steps == "
                f"schedule.T == {self.schedule.T}); got num_steps={num_steps}.")

        def shared_noise():
            """One (1, F, D) draw, broadcast to the whole batch."""
            return torch.randn(1, length, D, device=self.device,
                               generator=generator).expand(B, -1, -1)

        x = shared_noise().contiguous()

        timesteps = list(range(self.schedule.T - 1, 0, -1))
        it = tqdm(timesteps, desc="Sampling (paired)") if show_progress else timesteps

        for t in it:
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            eps_cond = self.schedule.to_eps(self.model(x, t_batch, contexts), x, t_batch)
            eps_uncond = self.schedule.to_eps(
                self.model(x, t_batch, context=None), x, t_batch)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            x = self.schedule.p_sample(x, t_batch, eps, noise=shared_noise())

        return x  # (B, length, D)
