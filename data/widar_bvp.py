from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from .registry import register_dataset


# ---------------------------
# Label parsing helpers
# ---------------------------

_PAT_STRICT = re.compile(r"user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-", re.IGNORECASE)


def parse_bvp_label_from_path(path: str) -> int:
    """
    Parse label from Widar BVP filename.
    Typical pattern includes "...-<label>-..." where label is group(5) in strict regex.
    Fallback: try find trailing digits split by '-' or '_'.
    """
    name = os.path.basename(path)
    m = _PAT_STRICT.search(name)
    if m:
        try:
            return int(m.group(5))
        except Exception:
            return -1

    base = os.path.splitext(name)[0]
    toks = [t for t in re.split(r"[-_]", base) if t]
    for t in toks[::-1]:
        if t.isdigit():
            return int(t)
    return -1


# ---------------------------
# Atomic cache write
# ---------------------------

def _atomic_save_npy(dst_path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = dst_path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst_path)


# ---------------------------
# Dataset
# ---------------------------

@dataclass
class WidarBVPParams:
    """
    Preprocessing parameters for Widar3.0 BVP velocity spectrum.
    """
    bvp_T: int = 64
    temporal_feats: List[str] = None
    ma_win: int = 5

    # label behaviors
    label_source: str = "auto"           # ["manifest", "auto", "filename"]
    filename_label_field: int = 1        # used when label_source="filename"

    # optional class filtering (kept here but typically handled by manifest/splits)
    allowed_classes: Optional[List[int]] = None

    # cache
    cache_dir: Optional[str] = None

    def __post_init__(self):
        if self.temporal_feats is None:
            self.temporal_feats = ["level", "diff"]


@register_dataset("widar3_bvp")
class WidarBVP(Dataset):
    """
    Widar3.0 BVP adapter:
    - loads .mat
    - picks a 3D array (prefer velocity_spectrum_ro)
    - resamples time axis to fixed T
    - flattens (N,N) to K
    - builds temporal features into C channels
    - returns x in R^{C x T x K}
    """
    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        bvp_T: int = 64,
        temporal_feats: Optional[List[str]] = None,
        ma_win: int = 5,
        label_source: str = "auto",
        filename_label_field: int = 1,
        allowed_classes: Optional[List[int]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.paths = list(paths)
        self.manifest_labels = list(labels)
        self.p = WidarBVPParams(
            bvp_T=bvp_T,
            temporal_feats=temporal_feats or ["level", "diff"],
            ma_win=ma_win,
            label_source=label_source,
            filename_label_field=filename_label_field,
            allowed_classes=allowed_classes,
            cache_dir=cache_dir,
        )
        if self.p.cache_dir:
            os.makedirs(self.p.cache_dir, exist_ok=True)

        self.labels = [self._resolve_label(i) for i in range(len(self.paths))]

        # optional filtering
        if self.p.allowed_classes:
            allowed = set(int(x) for x in self.p.allowed_classes)
            keep = [i for i, y in enumerate(self.labels) if int(y) in allowed]
            self.paths = [self.paths[i] for i in keep]
            self.manifest_labels = [self.manifest_labels[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            if len(self.paths) == 0:
                raise RuntimeError("allowed_classes filtering resulted in 0 samples.")

    def _resolve_label(self, idx: int) -> int:
        src = (self.p.label_source or "auto").lower().strip()
        if src == "manifest":
            return int(self.manifest_labels[idx])
        if src == "filename":
            return self._label_from_filename(self.paths[idx], self.p.filename_label_field)
        # auto
        return parse_bvp_label_from_path(self.paths[idx])

    @staticmethod
    def _label_from_filename(path: str, field: int) -> int:
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        parts = [t for t in re.split(r"[-_]", stem) if t]
        if parts and parts[0].lower().startswith("user"):
            idx = field
        else:
            idx = field - 1
        if idx < 0 or idx >= len(parts):
            return -1
        try:
            return int(parts[idx])
        except Exception:
            return -1

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_key(self, path: str) -> str:
        tag = f"T{self.p.bvp_T}_tf{','.join(self.p.temporal_feats)}_mw{self.p.ma_win}"
        h = hashlib.md5((path + "|" + tag).encode("utf-8")).hexdigest()
        return os.path.join(self.p.cache_dir, f"{h}.npy")

    def _pick_3d(self, mat: Dict) -> np.ndarray:
        if "velocity_spectrum_ro" in mat:
            v = mat["velocity_spectrum_ro"]
            if isinstance(v, np.ndarray) and v.ndim == 3:
                return v
        for v in mat.values():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                return v
        raise RuntimeError("No 3D array found in .mat file.")

    def _resample_T(self, V: np.ndarray) -> np.ndarray:
        """
        V: [N, N, T]
        """
        N, _, T = V.shape
        if T == self.p.bvp_T:
            return V.astype(np.float32)

        t_old = np.linspace(0.0, 1.0, T, endpoint=False)
        t_new = np.linspace(0.0, 1.0, self.p.bvp_T, endpoint=False)
        out = np.empty((N, N, self.p.bvp_T), dtype=np.float32)

        # interpolation per (i,j)
        for i in range(N):
            for j in range(N):
                out[i, j, :] = np.interp(t_new, t_old, V[i, j, :])
        return out

    @staticmethod
    def _moving_avg_same(A: np.ndarray, w: int) -> np.ndarray:
        """
        A: [T, K] - apply moving average along T for each feature K.
        """
        if w <= 1:
            return A.copy()
        pad = w // 2
        kernel = np.ones(w, dtype=np.float32) / float(w)
        out = np.empty_like(A, dtype=np.float32)
        Ap = np.pad(A, ((pad, pad), (0, 0)), mode="edge")
        for j in range(A.shape[1]):
            out[:, j] = np.convolve(Ap[:, j], kernel, mode="valid")
        return out

    def _build_x(self, path: str) -> np.ndarray:
        """
        Build x in R^{C x T x K}
        """
        mat = sio.loadmat(path, squeeze_me=True)
        V = self._pick_3d(mat).astype(np.float32)

        # some BVP arrays may have negative values; clamp for log transforms
        V = np.maximum(V, 0.0)

        V = self._resample_T(V)  # [N,N,T]
        N, _, T = V.shape

        # flatten N*N into K, transpose to [T,K]
        M = V.reshape(N * N, T).T  # [T,K]

        feats: List[np.ndarray] = []
        eps = 1e-6

        for tag in self.p.temporal_feats:
            t = str(tag).lower().strip()
            win = self.p.ma_win
            m = re.search(r"(\d+)", t)
            if m:
                win = max(1, int(m.group(1)))

            if t.startswith("level"):
                X = np.log1p(M)

            elif t.startswith("diffnorm"):
                d = np.diff(M, axis=0, prepend=M[:1])
                base = self._moving_avg_same(M, max(3, win))
                X = np.log1p(np.abs(d) / (base + eps))

            elif t.startswith("diff2"):
                d1 = np.diff(M, axis=0, prepend=M[:1])
                d2 = np.diff(d1, axis=0,
