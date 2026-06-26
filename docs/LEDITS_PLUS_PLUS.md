# LEDITS++ for 3D Human Motion — How It Works in This Thesis

This document explains, step by step, how the training-free instruction-following
motion editor works: the maths, the intuition, and how each piece maps onto the
code in this repository. It ends with an implementation-status section
(what exists, where, and what still needs to be built).

The method is a motion-domain adaptation of **LEDITS++** (Brack et al., 2024),
which itself combines three ingredients:

1. **Edit-friendly DDPM inversion** (Huberman-Spiegelglas et al., 2024) — invert a
   real sample into a noise space from which it can be reconstructed *exactly*,
   with no source-text prompt and no per-sample optimisation.
2. **SEGA semantic guidance** (Brack et al., 2023) — push the denoiser along the
   direction of an edit concept.
3. **Implicit masking** — localise the edit to only the regions the instruction is
   actually about.

The whole method runs **at inference time only**: no training data, no
fine-tuning, no source annotation, no gradients through the model.

---

## 0. Notation and the data representation

| Symbol | Meaning |
|---|---|
| $x_0 \in \mathbb{R}^{F\times D}$ | source motion, $F$ frames, $D=263$ HumanML3D features |
| $T$ | number of diffusion steps (1000) |
| $\bar\alpha_t$ | cumulative noise schedule coefficient at step $t$ |
| $\varepsilon_\theta(x_t, c)$ | the model's noise prediction given context $c$ |
| $\varnothing$ | the *unconditional* context (the model's learned `null_text_emb`) |
| $c_e$ | embedding of an edit instruction (e.g. "raise the right arm") |
| $G$ | number of body-part groups (7 for GroupDiT, 1 for MotionDiT) |

Each frame's 263-dim vector packs root kinematics, 21 local joint positions, 21
6-D joint rotations, 22 joint velocities, and 4 foot contacts. The partition of
these 263 channels into the 7 body-part groups (root, left/right leg, spine,
left/right arm, head) lives in [`src/model/body_groups.py`](../src/model/body_groups.py).

**Why GroupDiT matters.** A plain frame-level model (`MotionDiT`) gives one
attention row per *frame*, so a mask can only say *which frames* are edited. A
**`GroupDiT`** tokenises each frame into 7 body-part tokens, so its cross-attention
is `(B, heads, F·G, L_text)` and a mask can say *which body part at which frame* —
the genuine spatiotemporal localisation the proposal needs.

---

## 1. Backbone prerequisites

LEDITS++ inversion is not architecture-agnostic in practice; the backbone must
satisfy two properties, both checked by
[`src/verify_backbone.py`](../src/verify_backbone.py):

1. **ε-prediction, not velocity.** The inversion maths below is written in terms of
   noise estimates $\varepsilon_\theta$. A flow-matching/velocity model would forfeit
   the exact-reconstruction guarantee. Our [`MotionDiT`/`GroupDiT`](../src/model/dit.py)
   regress noise.
2. **An accurate unconditional branch.** Stages 1 and 3 both call
   $\varepsilon_\theta(x_t, \varnothing)$. That branch is the *learned* `null_text_emb`,
   trained by classifier-free-guidance dropout in [`src/train.py`](../src/train.py).
   This is why the null is a learned parameter, not a zero vector — the editor must
   use exactly the null the model was trained with.

The model also exposes its cross-attention maps for masking via
`store_attn=True` / `get_attn_maps()` ([`src/model/dit.py`](../src/model/dit.py)).

---

## 2. Stage 1 — Edit-friendly inversion

**Goal:** turn the source motion $x_0$ into a per-step noise representation
$\{z_t\}$ such that re-running the reverse diffusion process with those exact noises
returns $x_0$ *perfectly* — and provides noisy versions of $x_0$ at every step for
later inpainting. No text is used here.

A naive DDIM inversion accumulates reconstruction error and is not exact. The
edit-friendly construction (Huberman-Spiegelglas) avoids this by **decoupling the
$x_t$'s from each other**:

**Step 1a — build an independent noisy sequence.** For every $t$ draw an
*independent* Gaussian $\varepsilon_t$ and set

$$ x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon_t, \qquad \varepsilon_t \sim \mathcal N(0, I)\ \text{i.i.d.} $$

Crucially the $x_t$ are **not** a single consistent forward trajectory; each is its
own noisy view of $x_0$. (Code: `xs[t]` in `MotionEditor.invert`.)

**Step 1b — extract the edit-friendly noise maps.** Walk $t = T-1 \to 1$ and define
the noise that *makes the DDPM reverse step land exactly on the stored* $x_{t-1}$:

$$ \hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t,\varnothing)}{\sqrt{\bar\alpha_t}}, \qquad \mu_t = \mu_\theta(\hat x_0, x_t), \qquad z_t = \frac{x_{t-1} - \mu_t}{\sigma_t} $$

where $\mu_\theta$ is the DDPM posterior mean and $\sigma_t = \sqrt{\tilde\beta_t}$ is
the posterior standard deviation.

**Perfect-reconstruction guarantee.** By construction $x_{t-1} = \mu_t + \sigma_t z_t$.
So if we later run the reverse process $x_{t-1} \leftarrow \mu_\theta(x_t) + \sigma_t z_t$
with the *same* unconditional $\varepsilon_\theta$, we recover the whole sequence and
hence $x_0$ exactly. This is verified in the test below (max abs error $= 0$).

> **Where:** [`src/editing/inversion.py`](../src/editing/inversion.py) →
> `MotionEditor.invert()` returns an `InversionState` holding `xs` (the $x_t$
> sequence, used for inpainting) and `zs` (the $z_t$ maps).
> The posterior mean is shared with the sampler via
> `NoiseSchedule.posterior_mean` in [`src/model/schedule.py`](../src/model/schedule.py).

> **Proposal vs. implementation:** the proposal targets the *sde-dpm-solver++*
> recurrence for speed. What is implemented is the mathematically exact **DDPM**
> form. Swapping in dpm-solver++ replaces the per-step $\mu_t/\sigma_t$ recurrence
> only; the structure (independent $x_t$, stored $z_t$, exact reconstruction) is
> unchanged. This is deliberately left as the remaining derivation work.

---

## 3. Stage 2 — Spatiotemporal implicit masking

**Goal:** for an edit instruction, decide *which (frame, body-part group) cells*
the edit is allowed to touch. The final mask is the intersection of two
complementary masks, both built in $(F\times G)$ space and **averaged over the whole
inversion trajectory** (every stored $x_t$), not a single noise level.

### 3.1 Semantic mask $M_1$ (cross-attention)

For each $x_t$, run a conditional forward pass $\varepsilon_\theta(x_t, c_e)$ with
attention capture, average the cross-attention over transformer layers, heads, and
the instruction's **content-token columns** (BOS/EOS/padding excluded), then reshape
the token axis to $(F, G)$. Accumulate over $t$. High values = "the edit text
attends to this body part at this frame."

Selecting the right token columns is encoder-specific, so it is delegated to the
encoder itself: `text_encoder.token_info(text)` returns the content-token positions
that match *that encoder's* `L_text` dimension (CLIP=77 or T5=max_length). See
[`src/model/text_encoder.py`](../src/model/text_encoder.py).

### 3.2 Noise-estimate mask $M_2$ (guidance magnitude)

For each $x_t$ compute the guidance vector

$$ \psi = \varepsilon_\theta(x_t, c_e) - \varepsilon_\theta(x_t, \varnothing) $$

take $|\psi|$ (263 channels), aggregate per group to $(F,G)$, and accumulate. High
values = "conditioning on the edit actually changes the prediction here." $M_2$
captures fine-grained motion boundaries that attention alone misses.

### 3.3 Intersection

Threshold each accumulated statistic at a percentile ($\lambda$) over valid
(non-padding) frames, then

$$ M = M_1 \cap M_2, \qquad M[\text{padding frames}] = 0. $$

The $(F,G)$ mask is then expanded two ways:
- to a **per-channel $(F,263)$ float mask** for the guidance term (via
  `GROUP_CHANNELS`), and
- to a **per-frame "edited" flag** ($(F,)$ bool) for the Stage-3 inpainting.

> **Where:** [`src/editing/masking.py`](../src/editing/masking.py) —
> `collect_statistics()` accumulates $M_1$/$M_2$ raw stats over the trajectory;
> `build_mask()` thresholds and intersects; `group_mask_to_channels()` does the
> $(F,G)\to(F,263)$ expansion. Orchestrated by `MotionEditor.collect_masks()`.

---

## 4. Stage 3 — SEGA-guided masked denoising with inpainting

**Goal:** denoise from $x_{T-1}$ back to an edited $\hat x_0$, applying the edit only
inside the mask and keeping every other frame *byte-for-byte identical* to the
source.

At each step $t = T-1 \to 1$ compute the **masked multi-edit SEGA estimate**
(proposal Eq. 1):

$$ \hat\varepsilon(x_t) = \varepsilon_\theta(x_t,\varnothing) + \sum_{i=1}^{K} s_{e,i}\; M_i \odot \big[\varepsilon_\theta(x_t, c_{e,i}) - \varepsilon_\theta(x_t, \varnothing)\big] $$

- $\varepsilon_\theta(x_t,\varnothing)$ is the unconditional base direction.
- For each instruction $i$, the bracket is the SEGA edit direction; $s_{e,i}$ is its
  guidance scale; $M_i$ (the per-channel mask) restricts it spatiotemporally.
- Per-instruction masks $M_i$ prevent cross-edit interference when several edits are
  applied at once.

Then take the reverse step **reusing the stored edit-friendly noise** $z_t$:

$$ x_{t-1} = \mu_\theta(\hat x_0(\hat\varepsilon), x_t) + \sigma_t\, z_t. $$

**Hard frame inpainting.** Immediately overwrite every *unedited* frame (those whose
mask is all-zero) with the exact source value from inversion:

$$ x_{t-1}[\text{unedited frames}] \leftarrow x_{t-1}^{\text{inversion}} = \text{xs}[t-1]. $$

Because the stored $z_t$ already reconstruct the source on those frames, and we
re-pin them to the source at every step, unedited frames have **provably zero
drift** — a constraint images don't need but motion does, for physical plausibility.

> **Where:** [`src/editing/inversion.py`](../src/editing/inversion.py) →
> `MotionEditor.edit()`.

---

## 5. Implicit attention mask vs. LLM fallback (secondary research question)

The proposal's secondary question: *are the cross-attention maps body-part grounded
enough to derive $M_1$ implicitly?* This is answered by
[`src/analyse_attention.py`](../src/analyse_attention.py), which scores whether the
expected body-part group wins the attention for prompts like "raise the right arm".

- **If grounded** → use the implicit attention $M_1$ of §3.1.
- **If not** → fall back to an **LLM-derived explicit joint-group mask**: an LLM maps
  the instruction to group names (`GROUP_NAMES`), which become the $(F,G)$ mask
  directly (optionally expanded along `PARENTS`), still intersected with $M_2$.

The Phase-0 finding recorded for this project is that attention was **not**
sufficiently grounded, so the **LLM fallback is the intended path** — and it is the
main piece of Stage 2 still to be written (see below).

---

## 6. End-to-end flow

```
x0 (source motion, normalised)
      │
      ▼  Stage 1  MotionEditor.invert
   InversionState{ xs[t], zs[t] }          ← exact reconstruction guarantee
      │
      ▼  Stage 2  MotionEditor.collect_masks → masking.collect_statistics / build_mask
   per-instruction masks { m_channel (F,263), edited (F,) }
      │
      ▼  Stage 3  MotionEditor.edit  (Eq. 1 + hard inpainting)
   x̂0 (edited motion: targeted cells changed, everything else identical)
```

Library usage:

```python
editor = MotionEditor(ema_model, schedule, device, is_group=True)
state  = editor.invert(x0)                                  # Stage 1
ctxs   = [text_encoder.encode([e]) for e in edits]
toks   = [text_encoder.token_info(e)[0] for e in edits]
masks  = editor.collect_masks(state, ctxs, toks, valid_frames)   # Stage 2
x_edit = editor.edit(state, ctxs, masks, scales=[5.0])           # Stage 3
```

---

## 7. Implementation status

### Done — and where

| Piece | Location | Notes |
|---|---|---|
| ε-prediction DiT backbone (frame + group) | [`model/dit.py`](../src/model/dit.py) | cross-attention exposed for masking |
| Body-part group partition | [`model/body_groups.py`](../src/model/body_groups.py) | 263-channel partition, `GROUP_NAMES` |
| Noise schedule + shared posterior mean | [`model/schedule.py`](../src/model/schedule.py) | `q_sample`, `predict_x0_from_eps`, `posterior_mean` |
| Learned unconditional branch | [`train.py`](../src/train.py) | CFG dropout trains `null_text_emb` |
| Encoder content-token selection | [`model/text_encoder.py`](../src/model/text_encoder.py) | `token_info()`, CLIP + T5 |
| **Stage 1 — edit-friendly inversion** | [`editing/inversion.py`](../src/editing/inversion.py) | `invert()`; exact-reconstruction verified |
| **Stage 2 — implicit M1 ∩ M2 masking** | [`editing/masking.py`](../src/editing/masking.py) | attention + noise-estimate masks |
| **Stage 3 — masked SEGA + inpainting** | [`editing/inversion.py`](../src/editing/inversion.py) | `edit()`; zero-drift verified |
| Backbone prerequisite check | [`verify_backbone.py`](../src/verify_backbone.py) | Phase 0.1 |
| Attention-grounding analysis | [`analyse_attention.py`](../src/analyse_attention.py) | Phase 0.2, secondary question |

Verified on toy models: inversion reconstructs the source with max-abs-error $=0$,
and unedited frames have exactly zero drift after an edit (both MotionDiT and
GroupDiT).

### To do — in priority order

1. **Runnable edit entry-point** (`src/edit_motion.py`): load a trained EMA
   checkpoint, edit one clip + instruction, render a gen-vs-source video. Needed to
   exercise the pipeline on real (trained) weights for the first time.
2. **LLM explicit-mask fallback** in `masking.py`: instruction → `GROUP_NAMES` →
   $(F,G)$ mask, intersected with $M_2$. This is the path Phase 0 selected and is the
   main missing Stage-2 branch.
3. **MotionFix editing evaluation harness**: edit each (source, instruction) pair and
   score R@1/2/3, FID, and MPJPE-on-unedited via the T2M evaluator. The current
   [`evaluate.py`](../src/evaluate.py) scores generation, not editing.
4. **sde-dpm-solver++ acceleration** of inversion/editing: replace the full-grid
   per-step recurrence so editing isn't $T\times(1{+}K)$ forward passes. (Research
   step from the proposal; the DDPM form is the correct fallback meanwhile.)
5. **Hyper-parameter sweeps** (proposal ablations): $\lambda$ percentiles, guidance
   scale $s_e \in \{1,2.5,5,7.5,10\}$, $M_1$-only vs $M_2$-only vs intersection,
   edit-friendly vs DDIM inversion, single- vs multi-edit, with/without inpainting.
