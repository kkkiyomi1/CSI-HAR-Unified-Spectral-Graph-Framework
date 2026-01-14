from __future__ import annotations

import copy
import hashlib
import os
from typing import Any, Dict, Optional, Tuple

import yaml


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively updates dict dst with src."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _ensure_keys(cfg: Dict[str, Any], keys: Tuple[str, ...]) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        obj = {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML config must be a dict at root. Got {type(obj)}")
    return obj


def resolve_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Loads a YAML config and applies optional overrides (dict merge).
    """
    base = load_yaml(config_path)
    cfg = copy.deepcopy(base)
    if overrides:
        _deep_update(cfg, overrides)

    # Minimal schema validation (kept intentionally lightweight)
    _ensure_keys(cfg, ("exp_name", "dataset", "dataloader", "model", "optimizer", "trainer", "output"))
    _ensure_keys(cfg["dataset"], ("name", "params"))
    _ensure_keys(cfg["model"], ("name", "params"))

    return cfg


def config_fingerprint(cfg: Dict[str, Any]) -> str:
    """
    Stable hash of resolved config for reproducibility.
    """
    dumped = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(dumped).hexdigest()[:12]


def make_run_dir(out_dir: str, exp_name: str) -> str:
    run_dir = os.path.join(out_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_resolved_config(cfg: Dict[str, Any], run_dir: str) -> str:
    """
    Saves the resolved config to run_dir/config.resolved.yaml.
    Returns the saved path.
    """
    path = os.path.join(run_dir, "config.resolved.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path
