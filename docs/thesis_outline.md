# Master Thesis Outline
### Training-Free Instruction-Following Editing of 3D Human Motion

A section-by-section structure with bullet points of what belongs where. Tailored
to this project (LEDITS++ adapted to motion, HumanML3D, GroupDiT backbone,
edit-friendly inversion + spatiotemporal masking + SEGA). Adjust depth to your
department's page expectations; a typical CS MA thesis is 50–80 pages.

---

## Front matter
- Title page, declaration of originality, abstract (EN), optional abstract (DE)
- Acknowledgements
- Table of contents, list of figures, list of tables, list of abbreviations/symbols

## Abstract (~250 words)
- The problem: editing unannotated motion from free-form instructions, no retraining
- The gap: supervised methods overfit a small vocabulary; training-free methods only
  do kinematics or token swaps
- Your contribution in one line: a full motion-domain adaptation of LEDITS++ with a
  spatiotemporal mask and zero-drift inpainting
- Headline results (R@k / FID vs. baselines; OOD generalisation)

---

## 1. Introduction
- **Motivation** — animation/VR/games pipelines, cost of re-rigging or retraining;
  natural language as the authoring interface
- **Problem statement** — formal goal: given source $x_0 \in \mathbb{R}^{F\times D}$
  and instruction $e$, produce $\hat x_0$ realising $e$ on the targeted joints/frames
  while leaving everything else unchanged; **inference-time only** constraint
  (no training data, no source annotation, no fine-tuning, no per-sample optimisation)
- **Research questions**
  - RQ1 (primary): can LEDITS++-style semantic guidance enable training-free
    instruction-following motion editing?
  - RQ2 (secondary): are a motion DiT's cross-attention maps body-part- and
    temporally-grounded enough for an implicit mask, or is an LLM fallback needed?
- **Contributions** (bulleted, explicit) — the masking architecture, the
  inversion adaptation, the zero-drift inpainting, the empirical study
- **Thesis structure** — one sentence per chapter

## 2. Related Work
- **Supervised motion editing** — MotionFix/TMED, SimMotionEdit, PartMotionEdit,
  MotionLab; strength (in-distribution quality) and shared failure (OOD vocabulary)
- **Training-free motion editing** — DNO (kinematic noise optimisation), MotionCLR /
  SALAD (prompt-token attention); limitation: no relative semantic instructions on
  unannotated clips
- **Image-editing predecessors** — LEDITS++, edit-friendly DDPM inversion, SEGA;
  what transfers and what doesn't
- **Why the transfer is non-trivial** — 2D 16×16 attention vs. spatiotemporal
  $J\times F$ (~17× richer), needing simultaneous body-part and temporal locality
- **Positioning table** — methods × {training-free, semantic, relative-edit,
  unannotated-source, inversion} to make the gap visual

## 3. Background / Preliminaries
- **Diffusion models** — DDPM forward/reverse, ε-prediction objective, the noise
  schedule, classifier-free guidance
- **Why ε-prediction (not velocity/flow)** — required by the inversion maths
- **DiT / cross-attention conditioning** — why dedicated cross-attention (not joint
  attention) so maps over joint–frame space are extractable
- **Edit-friendly DDPM inversion** — the noise-space idea and exact reconstruction
- **SEGA semantic guidance** — concept directions
- **The HumanML3D representation** — the 263-dim layout (root, positions, 6D
  rotations, velocities, foot contacts); normalisation; recovery to joint positions
- **Evaluation background** — the T2M evaluator, FID, R-Precision, MPJPE

## 4. Method
> The core chapter. Present the pipeline as three stages; keep maths precise.
- **4.0 Overview** — pipeline figure (x0 → invert → mask → edit → x̂0); the
  inference-time constraints restated
- **4.1 Backbone** — MotionDiT vs. **GroupDiT**: tokenising each frame into 7
  body-part groups; why this gives true spatiotemporal mask resolution; the
  263-channel → group partition; the two prerequisites (ε-prediction, learned
  unconditional branch)
- **4.2 Stage 1 — Edit-friendly inversion** — independent noisy sequence $x_t$;
  extraction of edit-friendly noise $z_t$; the exact-reconstruction guarantee;
  storing $x_t$ for inpainting; note on the DDPM vs. sde-dpm-solver++ recurrence
- **4.3 Stage 2 — Spatiotemporal implicit masking** (the central contribution)
  - semantic mask $M_1$ from cross-attention over content tokens, averaged over
    layers/heads/timesteps; reshaping the token axis to $(F,G)$
  - noise-estimate mask $M_2$ from $|\varepsilon_\theta(x_t,c)-\varepsilon_\theta(x_t,\varnothing)|$
  - intersection $M = M_1 \cap M_2$; percentile thresholding; padding handling
  - channel expansion and per-frame "edited" reduction
- **4.4 Implicit vs. LLM-derived explicit mask** — the RQ2 decision rule; the LLM
  joint-group fallback and optional `PARENTS` expansion
- **4.5 Stage 3 — SEGA-guided masked denoising** — Eq. 1 (masked multi-edit
  guidance); reuse of stored $z_t$; **hard frame inpainting** and the zero-drift
  argument; multi-edit composition and cross-edit interference
- **4.6 Design choices & trade-offs** — guidance scale, thresholds, full-grid
  inversion cost

## 5. Implementation
- **System overview** — repo/module map (backbone, schedule, encoders, editing
  package); training vs. editing code paths
- **Backbone training** — HumanML3D pipeline, T5/CLIP text encoders, EMA, Min-SNR
  weighting, MDM geometric losses, CFG dropout; hyper-parameters table
- **Editing pipeline** — `invert` / `collect_masks` / `edit`; how attention maps are
  captured; mask construction in code
- **Reproducibility** — environment, seeds, checkpoints, run commands
- (Keep this chapter engineering-focused; defer *why* to Method, *how-well* to
  Experiments)

## 6. Experiments
- **6.0 Setup** — datasets (HumanML3D, MotionFix), metrics (R@1/2/3, FID,
  MPJPE-on-unedited), baselines (DNO, MotionCLR, SALAD; supervised upper bounds TMED,
  SimMotionEdit), hardware, protocol (no source prompts, no MotionFix training)
- **6.1 Backbone validation (Phase 0.1)** — one-step reconstruction MPJPE and noise
  MSE vs. noise level; cond vs. uncond; the PASS criterion
- **6.2 Attention-grounding study (Phase 0.2, answers RQ2)** — per-prompt heatmaps,
  alignment score, the implicit-vs-LLM decision
- **6.3 Main results on MotionFix (answers RQ1)** — table vs. all baselines;
  qualitative edits (figures/video stills)
- **6.4 Out-of-distribution generalisation** — the 150 hand-built instruction set
  (cross-cultural gestures, VR actions, compound multi-part), 750 sequences; isolate
  the training-free advantage
- **6.5 Ablations** — no-mask vs. $M_1$-only vs. $M_2$-only vs. intersection;
  implicit vs. LLM mask; edit-friendly vs. DDIM inversion; guidance-scale sweep
  $s_e \in \{1,2.5,5,7.5,10\}$; single- vs. multi-edit; with/without inpainting
- **6.6 Source preservation** — MPJPE on unedited joints; drift analysis

## 7. Discussion
- Interpret results against RQ1/RQ2; where you beat training-free baselines and where
  the gap to supervised remains
- Failure cases (qualitative) and likely causes
- Sensitivity to thresholds / scales; cost of full-grid inversion
- **Limitations** — 22-joint body only, ≤196 frames, single-clip, compute cost,
  reliance on backbone quality / mask grounding

## 8. Conclusion and Future Work
- Restate contributions and headline findings (1 paragraph each)
- **Future work** — sde-dpm-solver++ acceleration; full-body + hand articulation
  ($J=52$); longer sequences / sliding window / transition edits; cross-modal edit
  signals (video / keyframe sketch via the shared embedding space)

## References
- BibTeX; ensure all baselines, datasets, and method predecessors are cited

## Appendices
- A. Full HumanML3D 263-dim channel layout and the body-part group partition
- B. Hyper-parameters and training curves
- C. Extended qualitative results / additional edit examples
- D. The 150 OOD instructions (full list)
- E. Derivations (edit-friendly inversion recurrence; zero-drift proof)
- F. Reproducibility checklist / commands

---

### Writing-order suggestion (not the reading order)
1. Method (4) — you understand it best; write it first
2. Background (3) — only the concepts Method actually uses
3. Experiments + Results (6) — as runs complete
4. Related Work (2) and Introduction (1) — once the story is fixed
5. Discussion, Conclusion, Abstract — last

### Mapping to the proposal's planned content
- Proposal §1 → Introduction (1) + research questions
- Proposal §2 → Related Work (2)
- Proposal §3 (backbone, 3 stages) → Method (4) + Implementation (5)
- Proposal §4 (MotionFix, OOD, ablations) → Experiments (6)
- Proposal §5 (extension points) → Future Work (8)
