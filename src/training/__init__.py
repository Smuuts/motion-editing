"""
Training: config resolution, run assembly, the epoch loops and the shared loss terms.

  config.py    CLI args -> the run config saved next to the checkpoint
  assemble.py  building data / model / EMA / grounding / optimiser from that config
  trainer.py   the Trainer wiring and the epoch loop
  epoch.py     one train pass and one validate pass
  losses.py    the loss terms both passes share, so they cannot drift apart
  optim.py     optimiser and LR schedule
  plotting.py  the loss curve
  grounding/   the TokenCompose cross-attention grounding loss
  motionfix/   the supervised fine-tune on MotionFix triplets
"""
