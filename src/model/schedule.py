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

    # Which quantity the network's output head represents. "eps" is the original
    # (and default) parameterisation; "x0" is Option 5 of
    # docs/AttentionGrounding_Options.md — see `to_eps` and `min_snr_weight`.
    PREDICT_TYPES = ("eps", "x0")

    @classmethod
    def from_config(cls, config: dict, device="cpu"):
        """Build a schedule from a checkpoint/training config.

        Use this anywhere a config dict is available instead of
        `NoiseSchedule(timesteps=...)`: it carries `predict_type` across, so an
        x0-trained checkpoint is interpreted correctly by the sampler, the inversion
        and the mask statistics. Missing key -> "eps", so every pre-Option-5
        checkpoint behaves exactly as before.
        """
        return cls(timesteps=config.get("timesteps", 1000), device=device,
                   predict_type=config.get("predict_type", "eps"))

    def __init__(self, timesteps: int = 1000, device="cpu", predict_type: str = "eps"):
        if predict_type not in self.PREDICT_TYPES:
            raise ValueError(f"predict_type must be one of {self.PREDICT_TYPES}, "
                             f"got {predict_type!r}")
        self.T = timesteps
        self.device = device
        self.predict_type = predict_type

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

    def predict_eps_from_x0(self, x_t, t, x0):
        """Inverse of `predict_x0_from_eps`: ε = (x_t − √ᾱ_t·x0) / √(1−ᾱ_t).

        The conversion shim for an x0-prediction network (Option 5). Exact, and affine
        in the network output with (x_t, t) held fixed — which is why SEGA guidance and
        the scale-0 exact-reconstruction property survive the switch unchanged
        (docs/AttentionGrounding_Options.md §5.3). Ill-conditioned only as t→0, where
        √(1−ᾱ_t)→0; clamped, and inversion/guidance barely matter there.
        """
        sqrt_acp   = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t][:, None, None].clamp(min=1e-4)
        return (x_t - sqrt_acp * x0) / sqrt_omacp

    def to_eps(self, model_out, x_t, t):
        """Interpret a raw network output as ε, whatever the network predicts.

        THE inference-side boundary for Option 5: every consumer that treats the model
        output as a noise estimate (sampler, verify_backbone) routes through here, so an
        x0-trained checkpoint runs the entire existing pipeline unchanged. Identity in
        "eps" mode — the default path is byte-for-byte what it was.

        NOTE this keeps the 1/√ᾱ_t amplification on the guidance term (and hence the
        need for `guidance_alpha_floor`). The editing stack no longer routes through it
        unconditionally: it picks its space via `resolve_space`/`to_space`, so an
        x0-trained checkpoint takes the x0-native path of
        docs/AttentionGrounding_Options.md §5.3 instead.
        """
        if self.predict_type == "eps":
            return model_out
        return self.predict_eps_from_x0(x_t, t, model_out)

    def to_x0(self, model_out, x_t, t):
        """Interpret a raw network output as a clean-signal estimate x̂0.

        Mirror of `to_eps` for the training-side geometric losses: an x0-head's output
        IS x̂0 (no conversion, no 1/√ᾱ_t error amplification), an ε-head's must be
        converted.
        """
        if self.predict_type == "x0":
            return model_out
        return self.predict_x0_from_eps(x_t, t, model_out)

    def resolve_space(self, space: str | None = "auto") -> str:
        """Which quantity the *editing* stack should do its arithmetic in: "eps" or "x0".

        "auto" (or None) resolves to the checkpoint's own `predict_type`, which is the
        intended way to select it: an x0-trained checkpoint then runs the x0-native
        LEDITS++ path (docs/AttentionGrounding_Options.md §5.3) and an ε-trained one the
        historical ε-space path, with nothing to pass. An explicit "eps"/"x0" forces the
        other space — that is the control needed to attribute a measured change to the
        space rather than to the checkpoint, and the escape hatch for running the
        x0-native path on an ε head (see `to_space`).
        """
        if space is None or space == "auto":
            return self.predict_type
        if space not in self.PREDICT_TYPES:
            raise ValueError(f"space must be 'auto' or one of {self.PREDICT_TYPES}, "
                             f"got {space!r}")
        return space

    def to_space(self, model_out, x_t, t, space: str):
        """Interpret a raw network output in `space`: `to_eps` or `to_x0`.

        Both conversions are affine in `model_out` at fixed (x_t, t), which is why the
        SEGA contrast and the whole LEDITS++ stack survive the choice: substituting one
        into the other reproduces it exactly (§5.3). What the choice does change is
        *conditioning*, because only one of the two directions is a no-op for a given
        head — the converted direction carries a 1/√ᾱ_t (to x0) or √ᾱ_t/√(1−ᾱ_t) (to ε)
        factor that amplifies the head's own error at one end of the schedule.
        """
        return (self.to_eps(model_out, x_t, t) if space == "eps"
                else self.to_x0(model_out, x_t, t))

    def diffusion_target(self, x0, noise):
        """The regression target matching `predict_type`: ε or the clean signal x0."""
        return noise if self.predict_type == "eps" else x0

    def min_snr_weight(self, t, gamma):
        """Per-sample Min-SNR weight (Hang et al. 2023), in the form matching
        `predict_type`.

        Min-SNR clamps the *effective* weight on the clean-signal error ‖x̂0 − x0‖² to
        min(SNR_t, γ). What that costs depends on the parameterisation, because the two
        objectives are reweightings of one another (docs/AttentionGrounding_Options.md
        §5.2):

            ‖ε̂ − ε‖² = SNR_t · ‖x̂0 − x0‖²

          eps-head: loss already carries an implicit SNR_t  -> weight min(SNR,γ)/SNR
          x0-head : loss carries no implicit factor         -> weight min(SNR,γ)

        The two differ by exactly SNR_t. **Leaving the eps form in place under an x0
        head double-applies the correction and re-suppresses high noise — i.e. undoes
        the entire point of Option 5**, which is why this is parameterisation-aware
        rather than a caller's responsibility.

        In eps mode this only suppresses low-noise/low-t samples (SNR ≫ γ there) and
        stays ≈1 for t ≳ 500, so it does NOT protect against high-noise instability —
        see `x0_confidence_weight`.
        """
        snr_t = self.snr[t]  # (B,)
        if self.predict_type == "x0":
            return snr_t.clamp(max=gamma)
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

        **Only meaningful for an eps-head.** With predict_type="x0" the network outputs
        x̂0 directly, there is no division by √ᾱ_t and so no error amplification to damp
        — applying this weight would just fade out the geometric losses at high noise
        for no reason (docs/AttentionGrounding_Options.md §5.4). train.py resolves this
        automatically; `--geo_conf_weight` overrides it.
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

    def p_sample(self, x_t, t, eps_pred, noise=None):
        """Reverse DDPM step: compute x_{t-1} from x_t and predicted noise.

        LEDITS++ Stage 3: after computing the masked eps_hat (Eq. 1 in the proposal),
        this function performs the reverse step. After each call, frames whose mask
        column is all-zero must be overwritten with q_sample(x0_source, t-1) — the
        source motion noised to t-1 — to guarantee zero drift on unedited frames.

        `noise` overrides the posterior noise draw z_t (broadcast against x_t). Passing
        ONE z shared by every batch element denoises a batch of prompts along a single
        noise path, so the samples differ only through their text — the paired-sample
        trick Option 6 rests on (docs/AttentionGrounding_Options.md; the caller is
        DDPMSampler.sample_paired).
        """
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred).clamp(-5, 5)
        mean    = self.posterior_mean(x0_pred, x_t, t)

        # alphas_cumprod[0] == 1.0 exactly, so 1-acp_t == 0 at t=0 → posterior
        # is degenerate; return x0_pred directly there.
        nonzero_t = (t > 0).float()[:, None, None]
        noise = torch.randn_like(x_t) if noise is None else noise
        var = self.posterior_variance[t][:, None, None]
        return torch.where(nonzero_t > 0, mean + var.sqrt() * noise, x0_pred)
