"""
Minimal logger that writes to a JSONL file and optionally to Weights & Biases.
"""

import os
import json
from datetime import datetime


class Logger:
    def __init__(self, output_dir: str, use_wandb: bool = False,
                 project: str = "motion-dit", run_name: str = None):
        self.log_path = os.path.join(output_dir, "metrics.jsonl")
        self.use_wandb = use_wandb

        if use_wandb:
            try:
                import wandb
                wandb.init(project=project, name=run_name or output_dir)
                self.wandb = wandb
            except ImportError:
                print("wandb not installed — logging to file only.")
                self.use_wandb = False

    def log(self, metrics: dict):
        metrics["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self.use_wandb:
            self.wandb.log(metrics)
