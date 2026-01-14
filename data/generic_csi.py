from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .registry import register_dataset


def _atomic_save_npy(dst_path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = dst_path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst_path)


def _safe_unwrap_phase(phase: np.ndarray) -> np.ndarray:
    """
    Phase unwrap along time axis (axis=0).
    phase: [T,K]
    """
    return np.unwrap(phase, axis=0)


def _sanitize_phase(phase: np.ndarray) -> np.ndarray:
    """
    A simple but practical phase sanitation:
    - subtract the first frame (remove global offset)
    - clip extreme values (robustness)
    """
    p0 = phase[:1]
    out = phase - p0
    out = np.clip(out, -10.0, 10.0)
    return out


def _resample_time_linear(X: np.ndarray, T_new: int) -> np.ndarray:
    """
    Resample along time axis using linear interpolation.
    X: [T,K]
    """
    T_old = X.shape[0]
    if T_old == T_new:
        return X.astype(np.float32)
    t_old = np.linspace(0.0, 1.0, T_old, endpoint=False)
    t_new = np.linspace(0.0, 1.0, T_new, endpoint=False)
    out = np.empty((T_new, X.shape[1]), dtype=np.float32)
    for k in range(X.shape[1]):
        out[:, k] = np.interp(t_new, t_old, X[:, k])
    return out


def _as_complex(arr: np.ndarray) -> np.ndarray:
    """
    Convert various representations to complex:
    - if already complex: return
    - if last dim == 2: treat as (real, imag)
    """
    if np.iscomplexobj(arr):
        return arr
    if arr.ndim >= 2 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
