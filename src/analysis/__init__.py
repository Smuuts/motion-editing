"""
Analysis/probe logic shared by the `analyse_*` / `probe_*` / `visualise_*` entry
points in src/.

The scripts own argument parsing, output paths and printing; the measurement itself
lives here so two scripts reporting "the same" number really do compute it the same
way (see docs/ARCHITECTURE.md "Mask instruction-invariance is reported decomposed").
"""
