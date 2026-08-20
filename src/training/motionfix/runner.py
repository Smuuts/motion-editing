"""
The fine-tune's outer loop: epochs, validation, checkpointing and early stopping.
"""

import json
import os
import shutil

import torch

from training.motionfix.loop import run_epoch
from utils.checkpoint import save_checkpoint
from utils.logger import get_logger

log = get_logger(__name__)

# Above this ratio to the best validation loss, the epoch line says how far it has drifted.
_REGRESSION_NOTICE = 1.02


def _epoch_line(epoch, train_stats, val_loss=None, marker=""):
    """The one-line epoch summary, in the fixed column order the run log is read in."""
    line = (f"epoch {epoch:3d}  loss {train_stats['loss']:.4f}  "
            f"diff {train_stats['diff']:.4f}  ground {train_stats['ground']:.4f}  "
            f"m_S {train_stats['m_S']:.3f}  m_tok {train_stats['m_tok']:.3f}  "
            f"items/step {train_stats['n_items']:.1f}")
    if train_stats["skipped"]:
        line += f"  SKIPPED {train_stats['skipped']}"
    if val_loss is not None:
        line += f"  | val {val_loss:.4f}"
    return line + marker


def _link_best(output_dir, epoch):
    """Point `checkpoint_best` at this epoch's snapshot, replacing whatever was there."""
    best = os.path.join(output_dir, "checkpoint_best")
    if os.path.islink(best) or os.path.isfile(best):
        os.remove(best)
    elif os.path.isdir(best):
        shutil.rmtree(best)
    os.symlink(f"checkpoint_epoch_{epoch:04d}", best)


def _validate(ctx, args, epoch):
    """One validation pass, with the grounding cache swapped to the val labels."""
    grounding = ctx.grounding
    with torch.no_grad():
        if grounding is not None:
            grounding.cache = ctx.label_caches["val"]
        stats = run_epoch(ctx.model, ctx.ema, ctx.schedule, ctx.optimiser,
                          ctx.lr_scheduler, ctx.scaler, ctx.loaders["val"], ctx.device,
                          args, grounding, ctx.text_encoder, epoch, ctx.geo_fn,
                          train=False)
        if grounding is not None:
            grounding.cache = ctx.label_caches["train"]
    return stats


def run_finetune(ctx, args, config_out):
    """Run every epoch, writing `metrics.jsonl`, periodic snapshots and the best model."""
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    history, best_val, best_epoch, stale = [], float("inf"), -1, 0

    for epoch in range(args.epochs):
        if "train" in ctx.samplers:
            ctx.samplers["train"].set_epoch(epoch)
        train_stats = run_epoch(ctx.model, ctx.ema, ctx.schedule, ctx.optimiser,
                                ctx.lr_scheduler, ctx.scaler, ctx.loaders["train"],
                                ctx.device, args, ctx.grounding, ctx.text_encoder,
                                epoch, ctx.geo_fn, train=True)

        is_val_epoch = (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1
        if not is_val_epoch:
            history.append({"epoch": epoch, **train_stats})
            log.info(_epoch_line(epoch, train_stats))
        else:
            val_stats = _validate(ctx, args, epoch)
            marker = ""
            if val_stats["loss"] < best_val:
                # Saved on improvement, not on a fixed cadence: this fine-tune's
                # validation loss bottoms early (~epoch 14 measured) and then rises for
                # the rest of the run, so a --save_every grid can miss the best model
                # entirely — the first run of this script did exactly that.
                best_val, best_epoch, stale = val_stats["loss"], epoch, 0
                save_checkpoint(args.output_dir, epoch, ctx.model, ctx.ema,
                                ctx.optimiser, ctx.lr_scheduler,
                                {**config_out, "best_val": best_val})
                _link_best(args.output_dir, epoch)
                marker = "  <- best"
            else:
                stale += 1
                if val_stats["loss"] > best_val * _REGRESSION_NOTICE:
                    marker = (f"  (val {val_stats['loss'] / best_val - 1:+.1%} vs best "
                              f"@ep{best_epoch})")
            history.append({"epoch": epoch, **train_stats, "val": val_stats})
            log.info(_epoch_line(epoch, train_stats, val_stats["loss"], marker))

        with open(metrics_path, "a") as f:
            f.write(json.dumps(history[-1]) + "\n")
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(args.output_dir, epoch, ctx.model, ctx.ema, ctx.optimiser,
                            ctx.lr_scheduler, config_out)
        if args.early_stop and stale >= args.early_stop:
            log.info("  early stop: %d validations without improvement since epoch %d",
                     stale, best_epoch)
            break

    _report_outcome(args.output_dir, best_val, best_epoch)
    return history


def _report_outcome(output_dir, best_val, best_epoch):
    log.info("done -> %s", output_dir)
    if best_epoch < 0:
        return
    log.info("  BEST validation %.4f at epoch %d -> %s/checkpoint_best",
             best_val, best_epoch, output_dir)
    log.info("  `checkpoint_latest` is the LAST epoch, which is not the best one when "
             "the run overfits — evaluate `checkpoint_best` unless you want the final "
             "weights.")
