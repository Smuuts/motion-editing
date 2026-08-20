"""
The training run: `Trainer(config)` assembles everything, `Trainer.run()` executes it.

Construction lives in `training/assemble.py`; this module is the wiring and the epoch
loop, and writes the checkpoints, `losses.json` and the loss curve. train.py stays a CLI.
"""

import json
import os

import torch

from training import assemble
from training.epoch import train_one_epoch, validate_one_epoch
from training.plotting import save_loss_graph
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.cli import resolve_device
from utils.logger import get_logger

log = get_logger(__name__)


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config["output_dir"]
        self.device = resolve_device(config.get("device"))
        log.info("Using device: %s", self.device)

        self.train_loader, self.val_loader = assemble.build_data(config)
        dataset = self.train_loader.dataset
        self.geo_fn, self.geo_weights, self.geo_conf_weight = \
            assemble.build_geometric_losses(config, dataset, self.device)
        (self.text_encoder, self.model, self.ema, self.schedule, self.amp_dtype,
         self.scaler) = assemble.build_model_stack(config, dataset, self.device)
        self.grounding = assemble.build_grounding(config, self.model, dataset,
                                                  self.text_encoder)
        self.optimizer, self.scheduler = assemble.build_optim(config, self.model)

        self.run_logger = get_logger("run").attach_run(self.output_dir)
        self.losses_path = os.path.join(self.output_dir, "losses.json")
        self.start_epoch, self.train_losses, self.val_losses = 0, [], []
        if config["resume"]:
            self._load_resume()

    def _load_resume(self):
        ckpt_dir = self.config["resume"]
        if ckpt_dir == "latest":
            ckpt_dir = os.path.join(self.output_dir, "checkpoint_latest")
        self.start_epoch = load_checkpoint(ckpt_dir, self.model, self.ema,
                                           self.optimizer, self.scheduler)
        if os.path.exists(self.losses_path):
            with open(self.losses_path) as f:
                saved = json.load(f)
            self.train_losses = saved.get("train", [])
            self.val_losses = saved.get("val", [])

    # ── the loop ────────────────────────────────────────────────────────────────
    def run(self):
        epochs = self.config["epochs"]
        log.info("")
        log.info("Starting training from epoch %d", self.start_epoch)
        if self.grounding.enabled and self.start_epoch < self.grounding.warmup_epochs:
            log.info("Grounding loss starts at epoch %d (warmup — from-scratch "
                     "attention is random noise before that).",
                     self.grounding.warmup_epochs)

        for epoch in range(self.start_epoch, epochs):
            avg_loss, epoch_stats = self._train(epoch)
            # Read the LR used for the epoch that just ran BEFORE advancing the
            # scheduler — get_last_lr() after step() reports next epoch's LR.
            lr_used = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            self.train_losses.append((epoch, avg_loss))

            val_loss, val_stats = self._validate(epoch)
            self._write_progress(epoch, avg_loss, lr_used, {**epoch_stats, **val_stats})
            log.info(_epoch_line(epoch, avg_loss, lr_used, epoch_stats, val_loss))

            if (epoch + 1) % self.config["save_every"] == 0:
                self._save(epoch)

        self._save(epochs - 1)
        save_loss_graph(self.output_dir, self.train_losses, val_losses=self.val_losses)
        log.info("Training complete.")

    def _train(self, epoch):
        c = self.config
        return train_one_epoch(
            self.model, self.ema, self.text_encoder, self.schedule, self.optimizer,
            self.scaler, self.train_loader, self.device, c["cfg_dropout"],
            self.run_logger, epoch, c["log_every"],
            snr_gamma=c["snr_gamma"], geo_fn=self.geo_fn,
            hml3d_pos_weight=self.geo_weights["pos"],
            hml3d_vel_weight=self.geo_weights["vel"],
            hml3d_foot_weight=self.geo_weights["foot"],
            attn_entropy_weight=c["attn_entropy_weight"],
            geo_conf_weight=self.geo_conf_weight,
            amp_dtype=self.amp_dtype,
            grounding=self.grounding)

    def _validate(self, epoch):
        c = self.config
        if self.val_loader is None or (epoch + 1) % c["val_every"] != 0:
            return None, {}
        # fork_rng so validation's sampling does not perturb the training RNG stream.
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            val_loss, val_stats = validate_one_epoch(
                self.ema.ema_model, self.text_encoder, self.schedule, self.val_loader,
                self.device, epoch,
                snr_gamma=c["snr_gamma"], geo_fn=self.geo_fn,
                hml3d_pos_weight=self.geo_weights["pos"],
                hml3d_vel_weight=self.geo_weights["vel"],
                hml3d_foot_weight=self.geo_weights["foot"],
                geo_conf_weight=self.geo_conf_weight,
                amp_dtype=self.amp_dtype,
                grounding_cfg=self.grounding)
        self.val_losses.append((epoch, val_loss))
        self.run_logger.metrics({"val/epoch_loss": val_loss, "val/epoch": epoch})
        return val_loss, val_stats

    def _write_progress(self, epoch, avg_loss, lr_used, epoch_stats):
        with open(self.losses_path, "w") as f:
            json.dump({"train": self.train_losses, "val": self.val_losses}, f, indent=2)
        save_loss_graph(self.output_dir, self.train_losses, val_losses=self.val_losses)
        self.run_logger.metrics({"train/epoch_loss": avg_loss, "train/epoch": epoch,
                                 "train/lr": lr_used, **epoch_stats})

    def _save(self, epoch):
        save_checkpoint(self.output_dir, epoch, self.model, self.ema, self.optimizer,
                        self.scheduler, self.config)


def _epoch_line(epoch, avg_loss, lr_used, stats, val_loss):
    """The one-line epoch summary, with only the terms this run actually computed."""
    line = f"Epoch {epoch:4d} | loss {avg_loss:.4f} | lr {lr_used:.2e}"
    if "train/geo_pos_raw_epoch" in stats:
        # "_raw" = unweighted magnitude; the loss adds weight * this value.
        line += "".join(f" | {k}_raw {stats[f'train/geo_{k}_raw_epoch']:.4f}"
                        for k in ("pos", "vel", "foot"))
    if "train/ground_m_S_epoch" in stats:
        # m_S is the readable one (the loss is its square). src_corr next to it is the
        # shortcut monitor: both rising together means the model is learning a motion
        # detector, not word routing.
        line += f" | m_S {stats['train/ground_m_S_epoch']:.3f}"
        if "train/ground_src_corr_epoch" in stats:
            line += f" | src_r {stats['train/ground_src_corr_epoch']:+.3f}"
    if val_loss is not None:
        line += f" | val {val_loss:.4f}"
    return line
