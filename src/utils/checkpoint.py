import os
import json
import torch


def save_checkpoint(output_dir, epoch, model, ema, optimizer, scheduler, config):
    ckpt_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch:04d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(),         os.path.join(ckpt_dir, "model.pt"))
    torch.save(ema.ema_model.state_dict(), os.path.join(ckpt_dir, "ema.pt"))
    torch.save(optimizer.state_dict(),     os.path.join(ckpt_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(),     os.path.join(ckpt_dir, "scheduler.pt"))
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump({**config, "epoch": epoch}, f, indent=2)
    latest = os.path.join(output_dir, "checkpoint_latest")
    if os.path.islink(latest):
        os.remove(latest)
    os.symlink(os.path.basename(ckpt_dir), latest)
    print(f"  Saved checkpoint: {ckpt_dir}")


def load_checkpoint(ckpt_dir, model, ema, optimizer, scheduler):
    def _load(name):
        return torch.load(os.path.join(ckpt_dir, name), weights_only=True)

    model.load_state_dict(_load("model.pt"))
    ema.ema_model.load_state_dict(_load("ema.pt"))
    optimizer.load_state_dict(_load("optimizer.pt"))
    scheduler.load_state_dict(_load("scheduler.pt"))
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        saved = json.load(f)
    print(f"  Resumed from epoch {saved['epoch']}")
    return saved["epoch"] + 1
