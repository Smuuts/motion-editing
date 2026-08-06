"""
The training run: everything train.py needs assembled from one config dict.

`Trainer(config)` builds the data, model, EMA, schedule, losses, optimiser and logger;
`Trainer.run()` executes the epoch loop and writes checkpoints, `losses.json` and the
loss curve. train.py stays a CLI.
"""

import glob
import json
import os

import numpy as np
import torch
from torch.amp import GradScaler

from data.dataset import build_dataloader
from model.dit import build_model
from model.schedule import NoiseSchedule
from model.text_encoder import build_text_encoder, get_encoder_dims
from training.config import resolve_amp_dtype
from training.epoch import train_one_epoch, validate_one_epoch
from training.optim import build_optimizer, build_scheduler
from training.plotting import save_loss_graph
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.cli import resolve_device
from utils.ema import EMA
from utils.logger import Logger
from utils.skeleton import build_geo_fn


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config["output_dir"]
        self.device = resolve_device(config.get("device"))
        print(f"Using device: {self.device}")

        self._build_data()
        self._build_losses()
        self._build_model()
        self._build_optim()

        self.logger = Logger(self.output_dir)
        self.losses_path = os.path.join(self.output_dir, "losses.json")
        self.start_epoch, self.train_losses, self.val_losses = 0, [], []
        if config["resume"]:
            self._load_resume()

    # ── construction ────────────────────────────────────────────────────────────
    def _build_data(self):
        c = self.config
        loader_kwargs = dict(batch_size=c["batch_size"], max_frames=c["max_frames"],
                             num_workers=c["num_workers"], feature_mode=c["feature_mode"])
        self.train_loader = build_dataloader(c["data_root"], split="train", **loader_kwargs)
        print(f"Training on {len(self.train_loader.dataset)} clips")

        self.val_loader = None
        if c["val_every"] > 0:
            self.val_loader = build_dataloader(c["data_root"], split="val", **loader_kwargs)
            print(f"Validation on {len(self.val_loader.dataset)} clips "
                  f"(every {c['val_every']} epochs)")

    def _build_losses(self):
        """Geometric losses + the diffusion-target bookkeeping that depends on
        predict_type. geo_fn is a closure (x0_pred, motion, mask) -> {pos,vel,foot} so
        the epoch loop stays representation-agnostic."""
        c = self.config
        ds = self.train_loader.dataset
        mean_t = torch.from_numpy(ds.mean).float().to(self.device)
        std_t  = torch.from_numpy(ds.std).float().to(self.device)

        self.geo_weights = {"pos": c["hml3d_pos_weight"], "vel": c["hml3d_vel_weight"],
                            "foot": c["hml3d_foot_weight"]}
        self.geo_fn, label = build_geo_fn(
            c["feature_mode"], mean_t, std_t, self.device,
            pos_weight=c["hml3d_pos_weight"], vel_weight=c["hml3d_vel_weight"],
            foot_weight=c["hml3d_foot_weight"], smplh_model_path=c["smplh_model_path"])
        if self.geo_fn is not None:
            active = ", ".join(f"{k}={w}" for k, w in self.geo_weights.items() if w)
            print(f"{label} geometric losses enabled ({active})")

        # AUTO: the alpha_bar_t damping only makes sense when x0 is DERIVED from eps.
        # An x0 head outputs x0 directly, so there is no error amplification to damp.
        self.geo_conf_weight = (c["predict_type"] == "eps"
                                if c["geo_conf_weight"] is None else c["geo_conf_weight"])
        if c["predict_type"] == "x0":
            gamma = c["snr_gamma"]
            weighting = ("plain (unweighted) — 40% of training weight on t>=600 vs 3% for eps"
                         if gamma == 0.0 else
                         f"min(SNR,{gamma}) x0-form — WARNING: this reproduces the eps "
                         f"baseline's weighting (~3% on t>=600) and largely cancels Option 5")
            print(f"Prediction target: x0 (Option 5). Loss weighting: {weighting}. "
                  f"Geometric-loss confidence weight "
                  f"{'ON (forced via --geo_conf_weight)' if self.geo_conf_weight else 'OFF (auto)'}. "
                  "Loss values are NOT comparable to eps runs — compare x0 runs only "
                  "against other x0 runs.")

    def _build_model(self):
        c = self.config
        context_dim, text_seq_len = get_encoder_dims(c)
        self.text_encoder = self._build_text_encoder(context_dim, text_seq_len)

        if c["ctx_pad_mask"] and c["text_encoder"] == "clip":
            print("NOTE: --ctx_pad_mask is a no-op with the CLIP encoder (CLIP does not "
                  "zero its padding embeddings, so no context column is all-zero).")
        # Pass the FULL config so new model-defining keys (e.g. attention-regime flags)
        # never need hand-copying here — same pattern as utils/model_io.load_model.
        self.model = build_model({
            **c,
            "input_dim":    self.train_loader.dataset.feature_dim,
            "context_dim":  context_dim,
            "text_seq_len": text_seq_len,
        }, device=self.device)
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params / 1e6:.1f}M")

        self.ema = EMA(self.model, decay=c["ema_decay"])
        self.schedule = NoiseSchedule.from_config(c, device=self.device)
        # GradScaler exists to keep fp16 GRADIENTS out of the subnormal range. bf16 has
        # fp32's exponent range and fp32 needs no scaling at all, so it is enabled for
        # fp16 only — leaving it on under bf16 costs a pointless inf-check per step.
        self.amp_dtype = resolve_amp_dtype(c.get("amp_dtype", "auto"))
        self.scaler = GradScaler(device=self.device.type,
                                 enabled=(self.device.type == "cuda"
                                          and self.amp_dtype is torch.float16))
        print(f"AMP: autocast dtype {str(self.amp_dtype).replace('torch.', '')}"
              f"{'  (GradScaler on)' if self.scaler.is_enabled() else '  (GradScaler off)'}")

    def _build_text_encoder(self, context_dim, text_seq_len):
        """None when precomputed embeddings exist (the encoder is never loaded), after
        checking they match the configured encoder — a silent mismatch would train
        against embeddings from a different text model."""
        emb_dir = self.train_loader.dataset.text_emb_dir
        if emb_dir is None:
            return build_text_encoder(self.config, device=self.device)

        sample_file = next(glob.iglob(os.path.join(emb_dir, "*.npy")), None)
        if sample_file:
            shape = np.load(sample_file).shape        # (num_ann, L, dim)
            if (int(shape[1]), int(shape[2])) != (text_seq_len, context_dim):
                raise ValueError(
                    f"Precomputed embeddings in '{emb_dir}' have shape "
                    f"(*, {shape[1]}, {shape[2]}), but the configured encoder "
                    f"(--text_encoder {self.config['text_encoder']}) expects "
                    f"(*, {text_seq_len}, {context_dim}). Re-run precompute_text.py with "
                    f"matching encoder settings, or point --data_root to a directory "
                    f"without a stale text_emb/ folder.")
        print(f"Precomputed text embeddings found — skipping "
              f"{self.config['text_encoder'].upper()} model load.")
        return None

    def _build_optim(self):
        c = self.config
        self.optimizer = build_optimizer(self.model, c["lr"], c["weight_decay"])
        self.scheduler = build_scheduler(self.optimizer, c["epochs"], c["warmup_epochs"],
                                         decay=not c["no_lr_decay"])

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
            self.val_losses   = saved.get("val", [])

    # ── the loop ────────────────────────────────────────────────────────────────
    def run(self):
        c = self.config
        print(f"\nStarting training from epoch {self.start_epoch}")
        for epoch in range(self.start_epoch, c["epochs"]):
            avg_loss, geo_epoch = train_one_epoch(
                self.model, self.ema, self.text_encoder, self.schedule, self.optimizer,
                self.scaler, self.train_loader, self.device, c["cfg_dropout"],
                self.logger, epoch, c["log_every"],
                snr_gamma=c["snr_gamma"], geo_fn=self.geo_fn,
                hml3d_pos_weight=self.geo_weights["pos"],
                hml3d_vel_weight=self.geo_weights["vel"],
                hml3d_foot_weight=self.geo_weights["foot"],
                attn_entropy_weight=c["attn_entropy_weight"],
                geo_conf_weight=self.geo_conf_weight,
                amp_dtype=self.amp_dtype,
            )
            # Read the LR used for the epoch that just ran BEFORE advancing the
            # scheduler — get_last_lr() after step() reports next epoch's LR.
            lr_used = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            self.train_losses.append((epoch, avg_loss))

            log_line = f"Epoch {epoch:4d} | loss {avg_loss:.4f} | lr {lr_used:.2e}"
            if geo_epoch:
                # "_raw" = unweighted magnitude; the loss adds weight * this value.
                log_line += "".join(
                    f" | {k}_raw {geo_epoch[f'train/geo_{k}_raw_epoch']:.4f}"
                    for k in ("pos", "vel", "foot"))

            val_loss = self._validate(epoch)
            if val_loss is not None:
                log_line += f" | val {val_loss:.4f}"

            self._write_progress(epoch, avg_loss, lr_used, geo_epoch)
            print(log_line)

            if (epoch + 1) % c["save_every"] == 0:
                self._save(epoch)

        self._save(c["epochs"] - 1)
        save_loss_graph(self.output_dir, self.train_losses, val_losses=self.val_losses)
        print("Training complete.")

    def _validate(self, epoch):
        c = self.config
        if self.val_loader is None or (epoch + 1) % c["val_every"] != 0:
            return None
        # fork_rng so validation's sampling doesn't perturb the training RNG stream.
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            val_loss = validate_one_epoch(
                self.ema.ema_model, self.text_encoder, self.schedule, self.val_loader,
                self.device, epoch,
                snr_gamma=c["snr_gamma"], geo_fn=self.geo_fn,
                hml3d_pos_weight=self.geo_weights["pos"],
                hml3d_vel_weight=self.geo_weights["vel"],
                hml3d_foot_weight=self.geo_weights["foot"],
                geo_conf_weight=self.geo_conf_weight,
                amp_dtype=self.amp_dtype,
            )
        self.val_losses.append((epoch, val_loss))
        self.logger.log({"val/epoch_loss": val_loss, "val/epoch": epoch})
        return val_loss

    def _write_progress(self, epoch, avg_loss, lr_used, geo_epoch):
        with open(self.losses_path, "w") as f:
            json.dump({"train": self.train_losses, "val": self.val_losses}, f, indent=2)
        save_loss_graph(self.output_dir, self.train_losses, val_losses=self.val_losses)
        self.logger.log({"train/epoch_loss": avg_loss, "train/epoch": epoch,
                         "train/lr": lr_used, **geo_epoch})

    def _save(self, epoch):
        save_checkpoint(self.output_dir, epoch, self.model, self.ema, self.optimizer,
                        self.scheduler, self.config)
