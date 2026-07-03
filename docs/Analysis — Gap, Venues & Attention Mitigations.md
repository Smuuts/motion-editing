---
title: "LEDITS++ Motion — Gap Analysis, Venue Fit & Attention-Map Mitigations"
date: 2026-07-02
tags: [thesis, motion-editing, ledits, attention, related-work]
---

# LEDITS++ → Motion: Research Analysis

Advisory note for Jan Mathy's thesis *Training-Free Instruction-Following Editing of 3D Human Motion*. Covers three asks: (1) does the research gap still exist and where to publish, (2) why the cross-attention maps don't localize (arms dominate), and (3) concrete mitigations. All cited PDFs are in [[Related Work]] (organized into 5 subfolders). Papers referenced as `Name (arXiv:ID)`.

> **TL;DR**
> - **The gap still exists but has narrowed.** Don't claim "first training-free motion editing" (MoMo, MotionCLR, SALAD are there). The defensible, unclaimed core is: *bringing LEDITS++/SEGA semantic guidance + edit-friendly **DDPM** inversion to **unannotated real** motion with **relative** instructions, training-free.* Lead with "SEGA-for-motion" (fully open) and "zero training triplets."
> - **The arm-dominance is the textbook attention-sink / register-token pathology, made worse by (a) DiT cross-attention being inherently non-localized and (b) the entangled 263-dim HumanML3D vector.** It is *expected*, not a bug in the student's code.
> - **Highest-leverage cheap fixes (no retraining):** contrastive attention (subtract the null-condition map), drop the start/sink token, and **lean the pipeline on M2 (guidance-magnitude) while demoting M1 (attention)**. Then look at **self-attention** (esp. the body-part↔body-part / "SkelAttn" axis) rather than cross-attention for *where* on the body.
> - **Before falling back to manual masks, run a probing protocol** — it both tells you which layers/heads/steps localize and converts a negative result into a publishable diagnostic.
> - **Best venues:** HuMoGen @ CVPR 2027 workshop (anchor), 3DV 2027 or SCA 2027 (archival), ICLR 2027 (if fast). CVPR main track is a reach at this scale.

---

## 1. Does the research gap still exist?

**Verdict: Yes, but it must be scoped precisely.** No paper does the exact combination proposed. But three 2024–2025 papers now sit adjacent and *must* be cited and distinguished:

| Paper | What it does | Why the gap survives |
|---|---|---|
| **MoMo / Monkey-See-Monkey-Do** (arXiv:2406.06508) | **Only prior motion-inversion editing.** DDIM inversion + self-attention manipulation, training-free, real motions. | Targets **motion transfer**, not instruction-following. Uses plain **DDIM** (not edit-friendly DDPM), no SEGA, no relative language. → *Cite as the closest mechanism precedent; differentiate on task + machinery.* |
| **MotionCLR** (arXiv:2410.18977, ICLR'25) | Training-free attention-map editing (emphasize/erase/replace). | Edits its **own generations from a known prompt**; needs source text; edits are attention-weight ops, not semantic guidance on an unannotated clip. |
| **SALAD** (arXiv:2503.13836, CVPR'25) | Skeleton-aware latent diffusion; **zero-shot attention-based text editing**. Word↔joint↔frame attention. | Edits **generated** motion via **absolute prompt swaps**, no inversion of a real clip, no relative deltas. **But see §3 — it is also your closest architectural template AND a partial threat.** |
| **MotionReFit** (arXiv:2503.20724) | Instruction-driven **spatiotemporal** editing (spatial + temporal). | **Fully supervised** (trained on composed triplets; body-part discriminator). Uses unannotated data only at *training* time. → *Lead with "zero training triplets."* |
| PartMotionEdit (2512.24200), SimMotionEdit (2503.18211), Cross-Axis (2606.01014), MotionLab (2502.02358) | Supervised part-level / unified editors. | All require training + trained vocabulary; define the gap by contrast. |

**Defensible novelty axes (and who threatens each):**
1. Edit-friendly **DDPM** inversion for motion — only MoMo does motion inversion, with DDIM, for transfer. *Open.*
2. **SEGA / LEDITS++ semantic guidance for motion** — never applied to motion. **Fully open, cleanest single claim.**
3. **Relative free-form instructions** ("more/higher") — supervised editors take absolute-ish edits; attention editors take prompts. *Open.*
4. **No source annotation at inference** — SALAD/MotionCLR need source prompt / generated motion; MoMo (transfer) is the only annotation-free real-motion precedent. *Mostly open for editing.*
5. **Training-free implicit spatiotemporal masks** — PartMotionEdit/MotionReFit localize but via supervised training. *Open in the training-free regime.*

**Weak points to pre-empt in the write-up:** MoMo already established training-free motion inversion+attention editing (so lead with *edit-friendly-DDPM + SEGA*, not "inversion" per se), and MotionReFit already claims instruction-driven spatiotemporal localization (so lead with *zero training triplets*).

---

## 2. The attention problem — diagnosis (why arms dominate)

The symptom (cross-attention concentrated on left/right arm regardless of the instruction) has **three compounding causes**, all documented:

### 2a. It's an attention sink / high-norm register-token artifact
This is the primary and most likely cause. In transformers, softmax forces attention weights to sum to 1, so heads with weak content preference **dump excess mass onto a few fixed tokens** ("attention sinks"), which develop **massive, roughly input-invariant activations** and act as implicit bias slots.
- Sinks: `StreamingLLM (arXiv:2309.17453)`; massive activations: `Sun et al. (arXiv:2402.17762)`.
- ViT analog — the closest match to your symptom: high-norm "register" tokens emerge on **low-information tokens**, hoard global info, and **wreck localization/dense prediction**: `ViTs Need Registers (arXiv:2309.16588)`.
- Confirmed in **diffusion transformers** specifically: `Attention Sinks in DiTs — Causal Analysis (arXiv:2605.09313)`, `Registers Matter for Pixel DiTs (arXiv:2605.16147)`, `Taming Outlier Tokens in DiTs (arXiv:2605.05206)`.
- **Tell-tale test:** if the arm concentration is *input-independent* (same joints for every instruction, even a null prompt), it's a sink, not semantics.

### 2b. DiT cross-attention is inherently less localized than U-Net
The whole Prompt-to-Prompt / LEDITS++ premise rests on **U-Net convolutional locality**. Pure-transformer DiT attention is measurably noisier and needs post-hoc refinement.
- `Revelio (arXiv:2411.16725)`: DiT features show "no clear spatially localized information" vs. clean localization in U-Net.
- `AnchorDiff (2605.26460)`, `AttnRouter (2605.01480)`: raw MM-DiT attention is "noisy / poorly localized."
- Nuance: modern DiTs *can* localize, but only in **specific middle/late layers, specific heads (~25%), early-to-mid timesteps** — not by blind averaging (`Seg4Diff 2509.18096`, `HeadHunter 2506.10978`, `Stable Flow 2411.14430`).

### 2c. The 263-dim HumanML3D representation is entangled
Even with a joint-token axis, each joint's information is smeared across root-relative position + 6D rotation + velocity + foot contacts, so attention over the token space doesn't cleanly map to one anatomical joint.
- `CoMo (arXiv:2403.13900)`: the holistic vector is a **"superimposed representation of body parts"** that must be disentangled — the strongest direct support.
- `Rethinking Diffusion / Redundant Reps (arXiv:2411.16575, CVPR'25)`: ~196/263 dims redundant; heterogeneous (continuous + 6D + categorical) → breaks z-normalization and the DDPM Gaussian assumption.
- `Absolute Coordinates / ACMDM (arXiv:2505.19377)`: local-relative + previous-frame redundancy **degrades** generation; absolute per-joint coords are simpler and better.
- `MotionStreamer (arXiv:2503.15451)` + [HumanML3D issue #26]: the 6D `rot_data` loses twist and is kinematically inconsistent.

### 2d. Dataset prior
Most HumanML3D captions describe arm/hand actions; locomotion is the default. Text conditioning is therefore globally correlated with arm joints — a bias correctable by contrastive attention (§3, fix #1).

---

## 3. Mitigations — ranked

### Tier A — cheap, no retraining (do these first, this week)

**A1. Contrastive / difference attention — subtract the null-condition map.** *(Highest leverage.)*
Use `A_diff = A(c_edit) − A(∅)` (or `A(c_edit) − A(source)`), not raw attention. The input-independent sink/arm bias is present even for a null prompt and **cancels**, exposing the content-driven signal. This is the attention-level analog of what M2 already does at the score level. Support: `P2P (2208.01626)`, `DiffEdit (2210.11427)`, `SEGA (2301.12247)`, DiT sink causal analysis.

**A2. Drop the start/⟨sot⟩ token (and any identified sink token), re-softmax over content tokens.** Attend-and-Excite found the start token soaks disproportionate attention and must be excluded before thresholding. `Attend-and-Excite (arXiv:2301.13826)`. Combine with Gaussian smoothing (k=3, σ=0.5) to kill single-joint spikes.

**A3. Lean the pipeline on M2, demote M1.** *(Rescues the method even if attention never localizes.)*
LEDITS++ itself states M2 = `|ε(c_edit) − ε(∅)|` is **finer-grained** than the attention mask M1, and that the **intersection beats either alone**. If M1 (attention) is unreliable, make M1 a *permissive* semantic gate and let the **guidance-magnitude M2 drive the boundary**. M2 is representation-free — it measures where the edit actually changes the motion. Tricks from `DiffEdit`: clip outliers, average `|ψ|` over several mid/structural timesteps before thresholding. Refs: `LEDITS++ (2311.16711)` Eqs. 12–14, `SEGA`, `DiffEdit`.

**A4. Aggregation hygiene — stop blindly averaging over all layers/heads/steps.**
- **Layers:** select middle/"vital" layers, not all (`Seg4Diff`, `Stable Flow`).
- **Heads:** only ~25% of heads carry clean semantics; select or cluster heads rather than averaging (`HeadHunter 2506.10978`).
- **Timesteps:** aggregate **early-to-mid (high-noise, structural)** steps; skip the very first steps for editing (LEDITS++ found `t∈[0.9T,0.8T]≈t=T`); ignore late texture-only steps.
- **Baseline recipe:** DAAM-style aggregate (upscale + sum over heads/layers/steps), then λ-percentile threshold. `DAAM (2210.04885)`.

**A5. Read body-part structure from SELF-attention, not (only) cross-attention.** *(Key architectural insight — you have the right tensors.)*
MotionCLR and SALAD agree on the division of labor:
- **Cross-attention** = *word → frame* timing ("which action, when"). Reliable for **temporal** localization (MotionCLR reports ~74% IoU of the action word with root velocity).
- **Frame↔frame self-attention** = motion structure/texture/order (style transfer via Q-injection; resequencing via map-shift).
- **Body-part↔body-part self-attention** ("SkelAttn") = **where on the body** — this is the axis you uniquely have and the one SALAD exploits for word↔joint binding and left↔right mirroring.
→ For "raise the *right arm*": get *when* from cross-attention, get *which joints* from the skeletal self-attention (and word→joint from cross-attention if it survives probing). MoMo adds: in self-attention, Q = motion "outline", K = "motif". Refs: `MotionCLR (2410.18977)`, `SALAD (2503.13836)`, `MoMo (2406.06508)`.

**A6. Training-free register redirect / sink suppression.** If §2a is confirmed: (i) detect the few MLP "register neurons" that spike on arm tokens and transplant their activation into appended register tokens at inference, zeroing at real joints (`ViTs Don't Need Trained Registers, arXiv:2506.08010`); or (ii) suppress arm-key pre-softmax logits (add `log η`, η<1) or zero their value vectors (`Attention Sinks in DiTs, 2605.09313`).

**A7. Per-joint variance normalization / inverse-variance reweighting.** *(Directly targets arm dominance; very cheap.)*
Arms/wrists are the highest-**displacement** end-effectors, so raw attention *energy* is structurally dominated by them regardless of the instruction. Normalize attention **across the joint axis** — z-score per joint, or reweight each joint by its inverse temporal variance — so a fixed high-motion joint no longer wins by default. Grounding: `Rethinking Diffusion / Redundant Reps (2411.16575)` shows the 7 feature groups have mismatched SD causing "error amplification" (compute per-joint variance directly with HumanML3D's own `cal_mean_variance.ipynb`); `Motion-X (2307.00818)` uses per-part temporal std-dev as its motion metric (hands/arms ≫ body); `PartMotionEdit (2512.24200)` dynamically predicts **per-body-part weights across timesteps** — the learned version of this. Also consider an **inverse action/part-frequency** reweight: HumanML3D is long-tailed and arm-gesture + walking are "head" classes.

### Tier B — needs retraining (backbone phase / if time permits)

**B1. Add learnable register tokens** to the motion DiT, discarded at output — the clean architectural fix for the sink pathology (`ViTs Need Registers 2309.16588`; DiT versions 2605.16147 / 2605.05206).

**B2. Change the motion representation** (most likely *root-cause* fix if §2c dominates):
- Per-joint **absolute coordinates** (`ACMDM 2505.19377`) — supports spatial/temporal editing natively.
- Per-joint **2D token maps** preserving joint structure (`MoGenTS 2409.17686`).
- Per-part codebooks / **skeleto-temporal latent** with an explicit joint axis (`SALAD 2503.13836`, `ParCo 2403.18512`).
- **SALAD is the closest architectural precedent** — it keeps joints/frames/words as separate axes and *shows word→joint attention DOES localize* ("waving arms" → arm joints). ⚠️ **Compatibility caveat:** SALAD uses **v-prediction latent diffusion**, which is *incompatible* with edit-friendly DDPM inversion (the proposal already flags this for flow-matching). You cannot copy SALAD's backbone wholesale — but you can adopt its **skeleto-temporal latent structure with a DDPM/ε objective**. This reconciliation is itself a nice contribution.

### Tier C — validate before you give up (and turn a negative into a contribution)

**C1. Run a probing protocol** *before* defaulting to manual masks. This is important both scientifically and for publishability (it answers the thesis's *secondary research question* with statistics instead of eyeballing heatmaps).
- Ground truth: per-joint/per-frame kinetic energy from the generated motion → binary active mask.
- **Minimal-pair captions:** "raise the right *arm*" vs "right *leg*"; "*right*" vs "*left*"; ordering swaps for temporal axis.
- **Metrics:** IoU/pointing-game/Spearman ρ of `A_diff` vs. GT active set; **counterfactual shift** (does attention move to the correct part when you swap one token?); vs. **uniform + random-token baselines**; cross-check against gradient/ablation importance; paired significance test with effect sizes.
- Refs: `DAAM (2210.04885)`, DINO emergent segmentation, Hewitt & Liang control tasks, Jain & Wallace / Wiegreffe & Pinter (attention-as-explanation cautions), Pointing Game. Motion precedents that *do* get word↔part attention (after restructuring): `AttT2M (2309.00796)`, `FineMoGen (2312.15004)`, `Guided Attention captioning (2310.07324)`, `KinMo (2411.15472)`.

**C2. Reframe manual masks / the negative result.** Manual masks = the proposal's "explicit LLM-derived joint-group mask" fallback. Legitimate, but it shrinks the "core novel contribution." Reframe as a **diagnostic contribution**: *"We show DiT cross-attention over HumanML3D does not localize anatomically, characterize why (arm sink + entangled representation), and propose a self-attention/SkelAttn + guidance-magnitude alternative."* Far more publishable than a bare fallback.

### Suggested order of attack
1. **A1 + A2 + A3 + A7** (an afternoon each) — contrastive attention, drop sink token, M2-driven masks, per-joint variance normalization. These four are cheap, independent, and each directly attacks the arm-dominance.
2. **C1** probing — decide *with data* whether attention localizes, and in which layers/heads/steps.
3. **A5** — pivot the "where on the body" signal to the **body-part self-attention** axis.
4. If still failing → **A6** (register redirect) as training-free; **B2** (skeleton-native representation with DDPM objective) as the root-cause fix in the backbone phase.

---

## 4. Publication venues

**Reality check (today = 2 July 2026):** SIGGRAPH Asia 2026, NeurIPS 2026, SCA 2026, SIGGRAPH 2026, BMVC 2026 windows are **closed**. Live cycle for you is **late-2026 → mid-2027**. This is master's-scale work (single author, one benchmark, method+system+analysis with a possibly-negative headline), which pushes top main tracks toward "reach."

| Tier | Venue | Deadline (this cycle) | Notes |
|---|---|---|---|
| **Anchor (safe/fast)** | **HuMoGen @ CVPR 2027** (workshop) | ~Mar 2027 | **Best topical fit.** Tolerates analysis/negative results; CVPR visibility; achievable bar. |
| **Solid archival** | **3DV 2027** | ~28 Aug 2026 | Nearest reputable archival target; 3D/human-motion audience. |
| **Solid archival** | **SCA 2027** | ~Apr 2027 | Best *graphics/animation* fit (MotionFix's community). |
| **Solid** | WACV 2028 Applications track | ~summer 2027 | Best "method + system" home. |
| **Reach** | CVPR 2027 main | abstract ~13–15 Nov 2026 | Topic fits (SALAD landed here) but scale/negative headline are hard. Prefer the workshop unless a supervisor co-drives. |
| **Reach (best ML fit)** | ICLR 2027 | ~late Sep 2026 | MotionCLR precedent; rewards attention *analysis*. Timing aggressive. |
| **Negative-result home** | ICBINB (NeurIPS workshop) / ReScience | rolling | For the attention-analysis contribution specifically. |

**Recommendation:** anchor on **HuMoGen @ CVPR 2027**; aim the full archival paper at **3DV 2027** (Aug 2026, if results come together) or **SCA 2027**; consider **ICLR 2027** only if results lock by late summer. Frame the attention finding as a *localization diagnostic*, never as "it didn't work."

---

## 5. Related-work map

See [[Related Work]] — 43 PDFs in 5 subfolders:
- **1 — Core Methods (LEDITS++ lineage):** LEDITS++, LEDITS v1, SEGA, edit-friendly DDPM inversion, Prompt-to-Prompt, DiffEdit, Attend-and-Excite, Self-Guidance, DDPM-inversion audio.
- **2 — Motion Editing:** MotionFix, MotionCLR, SALAD, MoMo, DNO, SimMotionEdit, MotionLab, MotionReFit, PartMotionEdit, Cross-Axis, InterEdit.
- **3 — Motion Representations & Tokenization:** HumanML3D, TMR, Rethinking-Diffusion/Redundant-Reps, Absolute-Coordinates, MotionStreamer, MoGenTS, ParCo, Bailando, MoMask, CoMo, KinMo.
- **4 — Attention Analysis (sinks/registers/DiT localization):** ViTs-Need-Registers, ViTs-Don't-Need-Trained-Registers, Massive-Activations, StreamingLLM-Sinks, Attention-Sinks-in-DiTs, Registers-Matter-Pixel-DiTs, Taming-Outlier-Tokens, Understanding-Cross/Self-Attn-SD.
- **5 — Attention Validation & Probing:** DAAM, AttT2M, FineMoGen, Guided-Attention-captioning.

**Key papers not downloaded (findable by arXiv ID):** Revelio (2411.16725), AnchorDiff (2605.26460), AttnRouter (2605.01480), Seg4Diff (2509.18096), HeadHunter (2506.10978), Stable Flow (2411.14430), HiSTF-Mamba (2503.06897), GUESS (2401.02142). Several 2026-dated IDs (Cross-Axis 2606.01014, InterEdit 2603.13082, PartMotionEdit 2512.24200) are recent preprints — verify before citing.
