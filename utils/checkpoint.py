from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    cfg: Dict[str, Any],
    epoch: int,
    best_metric: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a training checkpoint that is safe for paper reproduction.

    Args:
        path: file path (.pt).
        model: nn.Module.
        cfg: resolved config dict.
        epoch: current epoch.
        best_metric: best metric value so far.
        optimizer: optional optimizer.
        scaler: optional AMP GradScaler.
        extra: optional additional payload.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "cfg": cfg,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        try:
            payload["scaler"] = scaler.state_dict()
        except Exception:
            pass
    if extra:
        payload["extra"] = dict(extra)

    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load a checkpoint saved by save_checkpoint().
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be a dict, got {type(ckpt)}")
    return ckpt
