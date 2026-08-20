"""
Supervised fine-tune of a pretrained SMPL-H checkpoint on MotionFix triplets.

  args.py     the CLI surface and where each default comes from
  data.py     triplet featurisation cache, dataset, collation, length bucketing
  labels.py   the grounding label set (caption parser + velocity-difference fallback)
  loop.py     the objective — noise the SOURCE, regress the TARGET — and one epoch of it
  setup.py    assembling a run: backbone, normalisation, loaders, optimiser
  runner.py   the outer loop: validation, checkpointing, early stopping

`src/finetune_motionfix.py` is the entry point.
"""
