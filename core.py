from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# SciPy is used for Widar .mat reading.
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

try:
    import yaml
except Exception as e:
    raise RuntimeError("PyYAML is required. Please install via `pip install pyyaml`.") from e

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# 0) Small utilities: config, seed, env, logging, jsonl
# =========================================================

def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dst with src."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        obj = {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a dict. Got {type(obj)}")
    return obj


def resolve_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load YAML config and apply dict overrides (deep merge).
    Minimal validation is performed here to keep it flexible.
    """
    cfg = load_yaml(config_path)
    cfg = copy.deepcopy(cfg)
    if overrides:
        _deep_update(cfg, overrides)

    # minimal required keys
    for k in ["exp_name", "dataset", "dataloader", "model", "optimizer", "trainer", "output"]:
        if k not in cfg:
            raise ValueError(f"Config missing required key '{k}'")

    if "name" not in cfg["dataset"]:
        raise ValueError("Config.dataset must include 'name'")
    if "params" not in cfg["dataset"]:
        cfg["dataset"]["params"] = {}

    if "name" not in cfg["model"]:
        raise ValueError("Config.model must include 'name'")
    if "params" not in cfg["model"]:
        cfg["model"]["params"] = {}

    return cfg


def config_fingerprint(cfg: Dict[str, Any]) -> str:
    dumped = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(dumped).hexdigest()[:12]


def make_run_dir(out_dir: str, exp_name: str) -> str:
    run_dir = os.path.join(out_dir, exp_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_resolved_config(cfg: Dict[str, Any], run_dir: str) -> str:
    path = os.path.join(run_dir, "config.resolved.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["python"] = platform.python_version()
    info["platform"] = platform.platform()
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
    return info


def setup_logger(run_dir: str, name: str = "csihar") -> logging.Logger:
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class JSONLWriter:
    """Append-only JSONL writer for metrics/records."""
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def write(self, obj: Dict[str, Any]) -> None:
        rec = dict(obj)
        rec.setdefault("time", datetime.utcnow().isoformat(timespec="seconds") + "Z")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


@dataclass
class AverageMeter:
    name: str
    sum: float = 0.0
    cnt: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * int(n)
        self.cnt += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.cnt)


@dataclass
class MetricBook:
    meters: Dict[str, AverageMeter] = dataclasses.field(default_factory=dict)

    def update(self, name: str, val: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter(name=name)
        self.meters[name].update(val, n)

    def as_dict(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self.meters.items()}


def save_checkpoint(
    path: str,
    model: nn.Module,
    cfg: Dict[str, Any],
    epoch: int,
    best_metric: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint must be dict, got {type(ckpt)}")
    return ckpt


# =========================================================
# 1) Manifest v2 (JSONL) + legacy support
# =========================================================

@dataclass
class ManifestItem:
    path: str
    label: int
    meta: Dict[str, Any]


def _parse_json_line(ln: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(ln)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def load_manifest(path: str) -> List[ManifestItem]:
    """
    Supported formats:
      1) JSONL (Manifest v2):
         {"path":"...","label":3,"subject":"u1",...}
      2) Legacy lines:
         /path/to/file
         /path/to/file 3
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    out: List[ManifestItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue

            obj = _parse_json_line(ln)
            if obj is not None:
                p = str(obj["path"])
                y = int(obj["label"])
                meta = {k: v for k, v in obj.items() if k not in ("path", "label")}
                out.append(ManifestItem(path=p, label=y, meta=meta))
                continue

            # legacy
            if " " in ln:
                p, y_str = ln.rsplit(" ", 1)
                try:
                    y = int(y_str)
                except Exception:
                    p, y = ln, -1
            else:
                p, y = ln, -1
            out.append(ManifestItem(path=p, label=y, meta={}))

    return out


def manifest_paths_labels(items: List[ManifestItem]) -> Tuple[List[str], List[int]]:
    return [it.path for it in items], [int(it.label) for it in items]


# =========================================================
# 2) Dataset adapters (Widar3.0 BVP + GenericCSI)
# =========================================================

def _atomic_save_npy(dst_path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = dst_path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst_path)


# ---------- Widar3.0 label parsing ----------
_PAT_STRICT = re.compile(r"user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-", re.IGNORECASE)


def parse_bvp_label_from_path(path: str) -> int:
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


class WidarBVP(Dataset):
    """
    Widar3.0 BVP adapter:
      - reads .mat
      - picks a 3D array (prefer 'velocity_spectrum_ro')
      - resamples time to fixed T
      - flattens (N,N) -> K
      - produces x in R^{C x T x K} via temporal features
    """
    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        bvp_T: int = 64,
        temporal_feats: Optional[List[str]] = None,
        ma_win: int = 5,
        label_source: str = "auto",          # "manifest" | "auto" | "filename"
        filename_label_field: int = 1,
        cache_dir: Optional[str] = None,
        allowed_classes: Optional[List[int]] = None,
    ):
        self.paths = list(paths)
        self.manifest_labels = [int(y) for y in labels]
        self.bvp_T = int(bvp_T)
        self.temporal_feats = temporal_feats or ["level", "diff"]
        self.ma_win = int(ma_win)
        self.label_source = str(label_source).lower().strip()
        self.filename_label_field = int(filename_label_field)
        self.cache_dir = cache_dir
        self.allowed_classes = [int(x) for x in allowed_classes] if allowed_classes else None

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.labels = [self._resolve_label(i) for i in range(len(self.paths))]

        # optional filter
        if self.allowed_classes:
            allowed = set(self.allowed_classes)
            keep = [i for i, y in enumerate(self.labels) if int(y) in allowed]
            self.paths = [self.paths[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            self.manifest_labels = [self.manifest_labels[i] for i in keep]
            if len(self.paths) == 0:
                raise RuntimeError("allowed_classes filtering resulted in 0 samples.")

    def _resolve_label(self, idx: int) -> int:
        if self.label_source == "manifest":
            return int(self.manifest_labels[idx])
        if self.label_source == "filename":
            return self._label_from_filename(self.paths[idx])
        return parse_bvp_label_from_path(self.paths[idx])

    def _label_from_filename(self, path: str) -> int:
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        parts = [t for t in re.split(r"[-_]", stem) if t]
        if parts and parts[0].lower().startswith("user"):
            j = self.filename_label_field
        else:
            j = self.filename_label_field - 1
        if j < 0 or j >= len(parts):
            return -1
        try:
            return int(parts[j])
        except Exception:
            return -1

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_key(self, p: str) -> str:
        tag = f"T{self.bvp_T}_tf{','.join(self.temporal_feats)}_mw{self.ma_win}"
        h = hashlib.md5((p + "|" + tag).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npy")

    def _pick_3d(self, mat: Dict[str, Any]) -> np.ndarray:
        if "velocity_spectrum_ro" in mat:
            v = mat["velocity_spectrum_ro"]
            if isinstance(v, np.ndarray) and v.ndim == 3:
                return v
        for v in mat.values():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                return v
        raise RuntimeError("No 3D array found in .mat file.")

    def _resample_T(self, V: np.ndarray) -> np.ndarray:
        N, _, T = V.shape
        if T == self.bvp_T:
            return V.astype(np.float32)
        t_old = np.linspace(0.0, 1.0, T, endpoint=False)
        t_new = np.linspace(0.0, 1.0, self.bvp_T, endpoint=False)
        out = np.empty((N, N, self.bvp_T), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                out[i, j, :] = np.interp(t_new, t_old, V[i, j, :])
        return out

    @staticmethod
    def _moving_avg_same(A: np.ndarray, w: int) -> np.ndarray:
        if w <= 1:
            return A.copy()
        pad = w // 2
        kernel = np.ones(w, dtype=np.float32) / float(w)
        out = np.empty_like(A, dtype=np.float32)
        Ap = np.pad(A, ((pad, pad), (0, 0)), mode="edge")
        for j in range(A.shape[1]):
            out[:, j] = np.convolve(Ap[:, j], kernel, mode="valid")
        return out

    def _build_x(self, p: str) -> np.ndarray:
        mat = sio.loadmat(p, squeeze_me=True)
        V = self._pick_3d(mat).astype(np.float32)
        V = np.maximum(V, 0.0)
        V = self._resample_T(V)  # [N,N,T]
        N, _, T = V.shape
        M = V.reshape(N * N, T).T  # [T,K]

        feats: List[np.ndarray] = []
        eps = 1e-6

        for tag in self.temporal_feats:
            t = str(tag).lower().strip()
            win = self.ma_win
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
                d2 = np.diff(d1, axis=0, prepend=d1[:1])
                X = np.log1p(np.abs(d2))
            elif t.startswith("diff"):
                d = np.diff(M, axis=0, prepend=M[:1])
                X = np.log1p(np.abs(d))
            elif t.startswith("highpass"):
                mu = self._moving_avg_same(M, win)
                X = np.log1p(np.abs(M - mu))
            elif t.startswith("movstd"):
                mu = self._moving_avg_same(M, win)
                mu2 = self._moving_avg_same(M * M, win)
                X = np.log1p(np.sqrt(np.maximum(mu2 - mu * mu, 0.0) + eps))
            else:
                raise ValueError(f"Unknown temporal feature: {tag}")

            feats.append(X.astype(np.float32))

        return np.stack(feats, axis=0).astype(np.float32)  # [C,T,K]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.paths[idx]
        y = int(self.labels[idx])
        if y < 0:
            raise ValueError(f"Invalid label for sample: {p} (label_source={self.label_source})")

        if self.cache_dir:
            cp = self._cache_key(p)
            if os.path.exists(cp):
                x = np.load(cp)
            else:
                x = self._build_x(p)
                _atomic_save_npy(cp, x)
        else:
            x = self._build_x(p)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ---------- Generic CSI adapter for WiAR / CSI survey ----------

def _safe_unwrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.unwrap(phase, axis=0)


def _sanitize_phase(phase: np.ndarray) -> np.ndarray:
    p0 = phase[:1]
    out = phase - p0
    out = np.clip(out, -10.0, 10.0)
    return out


def _resample_time_linear(X: np.ndarray, T_new: int) -> np.ndarray:
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
    if np.iscomplexobj(arr):
        return arr
    if arr.ndim >= 2 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    return arr.astype(np.complex64)


def _flatten_to_TK(csi: np.ndarray) -> np.ndarray:
    if csi.ndim == 1:
        csi = csi[:, None]
    if csi.ndim == 2:
        T, K = csi.shape
        # heuristic: treat larger dim as time
        if T < K:
            return csi.T
        return csi

    sizes = list(csi.shape)
    time_axis = int(np.argmax(sizes))
    arr_t = np.moveaxis(csi, time_axis, 0)  # [T,...]
    T = arr_t.shape[0]
    rest = int(np.prod(arr_t.shape[1:]))
    return arr_t.reshape(T, rest)


class GenericCSIDataset(Dataset):
    """
    Generic CSI adapter (WiAR / CSI survey) after converting raw samples to .npz/.npy/.pt.

    Input:
      - .npz: key "csi" by default, value is complex or (real, imag) as last dim=2
      - .npy: array
      - .pt : tensor or dict with "csi"
    Output:
      x in R^{C x T x K} (float32), y int64
    """
    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        file_key: str = "csi",
        T: int = 64,
        resample: str = "linear",
        cache_dir: Optional[str] = None,
        view: Optional[Dict[str, Any]] = None,
        normalize: Optional[Dict[str, Any]] = None,
    ):
        self.paths = list(paths)
        self.labels = [int(y) for y in labels]

        self.file_key = str(file_key)
        self.T = int(T)
        self.resample = str(resample).lower().strip()
        self.cache_dir = cache_dir

        view = view or {}
        normalize = normalize or {}

        self.view_mode = str(view.get("mode", "amp_phase")).lower().strip()
        self.phase_unwrap = bool(view.get("phase_unwrap", True))
        self.phase_sanitize = bool(view.get("phase_sanitize", True))

        self.per_sample_norm = bool(normalize.get("per_sample", True))
        self.eps = float(normalize.get("eps", 1e-6))

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        for i, y in enumerate(self.labels):
            if y < 0:
                raise ValueError(f"Invalid label at index {i}: {y}")

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_key(self, p: str) -> str:
        tag = (
            f"T{self.T}_rs{self.resample}_vm{self.view_mode}_"
            f"uw{int(self.phase_unwrap)}_san{int(self.phase_sanitize)}_"
            f"pn{int(self.per_sample_norm)}"
        )
        h = hashlib.md5((p + "|" + tag).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npy")

    def _load_raw(self, p: str) -> np.ndarray:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".npz":
            z = np.load(p, allow_pickle=True)
            if self.file_key not in z:
                raise KeyError(f"Key '{self.file_key}' not found in npz: {p}")
            return np.array(z[self.file_key])
        if ext == ".npy":
            return np.load(p, allow_pickle=True)
        if ext in [".pt", ".pth"]:
            obj = torch.load(p, map_location="cpu")
            if isinstance(obj, dict) and self.file_key in obj:
                obj = obj[self.file_key]
            if torch.is_tensor(obj):
                return obj.detach().cpu().numpy()
            return np.array(obj)
        raise ValueError(f"Unsupported file extension: {ext} ({p})")

    def _build_x(self, p: str) -> np.ndarray:
        raw = self._load_raw(p)
        csi = _as_complex(np.array(raw))
        Xc = _flatten_to_TK(csi)  # complex [T,K]

        if self.resample == "linear":
            Xr = _resample_time_linear(np.real(Xc).astype(np.float32), self.T)
            Xi = _resample_time_linear(np.imag(Xc).astype(np.float32), self.T)
            Xc = (Xr + 1j * Xi).astype(np.complex64)
        elif self.resample == "none":
            pass
        else:
            raise ValueError(f"Unknown resample mode: {self.resample}")

        amp = np.abs(Xc).astype(np.float32)
        phase = np.angle(Xc).astype(np.float32)

        if self.phase_unwrap:
            phase = _safe_unwrap_phase(phase)
        if self.phase_sanitize:
            phase = _sanitize_phase(phase)

        if self.per_sample_norm:
            mu = amp.mean()
            sig = amp.std()
            amp = (amp - mu) / (sig + self.eps)

            mu_p = phase.mean()
            sig_p = phase.std()
            phase = (phase - mu_p) / (sig_p + self.eps)

        feats: List[np.ndarray] = []
        if self.view_mode == "amp":
            feats = [amp]
        elif self.view_mode == "phase":
            feats = [phase]
        elif self.view_mode == "amp_phase":
            feats = [amp, phase]
        else:
            raise ValueError(f"Unknown view mode: {self.view_mode}")

        x = np.stack(feats, axis=0).astype(np.float32)  # [C,T,K]
        return x

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.paths[idx]
        y = int(self.labels[idx])

        if self.cache_dir:
            cp = self._cache_key(p)
            if os.path.exists(cp):
                x = np.load(cp)
            else:
                x = self._build_x(p)
                _atomic_save_npy(cp, x)
        else:
            x = self._build_x(p)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def build_dataset_from_config(
    ds_cfg: Dict[str, Any],
    paths: List[str],
    labels: List[int],
) -> Dataset:
    """
    Factory: map dataset.name -> Dataset adapter.
    """
    name = str(ds_cfg.get("name", "")).lower().strip()
    params = ds_cfg.get("params", {}) or {}

    if name in ["widar3_bvp", "widar_bvp", "widar3"]:
        return WidarBVP(paths=paths, labels=labels, **params)
    if name in ["generic_csi", "generic", "csi"]:
        return GenericCSIDataset(paths=paths, labels=labels, **params)

    raise KeyError(f"Unknown dataset '{name}'. Supported: widar3_bvp, generic_csi")


# =========================================================
# 3) Model: WiPromptTCN (TCN + Temporal Attention)
# =========================================================

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv = nn.utils.weight_norm(conv)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        crop = self.conv.padding[0]
        if crop > 0:
            out = out[:, :, :-crop]
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + res)


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)              # [B,T,C]
        scores = torch.tanh(self.proj(xt))  # [B,T,1]
        w = F.softmax(scores, dim=1)
        out = (xt * w).sum(dim=1)           # [B,C]
        return out


class WiPromptTCN(nn.Module):
    """
    Input: x in [B,C,T,K]
      - flatten (C,K) per frame -> [B,T,C*K]
      - LazyLinear to hidden
      - Dilated TCN
      - Temporal attention pooling
      - Head -> embedding + logits
    """
    def __init__(
        self,
        n_classes: int,
        tcn_channels: int = 128,
        tcn_layers: int = 4,
        tcn_kernel: int = 5,
        tcn_dropout: float = 0.1,
        emb_dim: int = 256,
        logit_scale: float = 1.0,
    ):
        super().__init__()
        self.logit_scale = float(logit_scale)
        self.input_proj = nn.LazyLinear(int(tcn_channels))

        hidden = int(tcn_channels)
        layers: List[nn.Module] = []
        for i in range(int(tcn_layers)):
            dilation = 2 ** i
            layers.append(TCNBlock(hidden, hidden, int(tcn_kernel), dilation, float(tcn_dropout)))
        self.tcn = nn.Sequential(*layers)

        self.attn = TemporalAttentionPool(hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, int(emb_dim)),
            nn.ReLU(inplace=True),
        )
        self.cls = nn.Linear(int(emb_dim), int(n_classes))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, T, K = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B, T, C * K)  # [B,T,C*K]
        feat = self.input_proj(x_flat)                        # [B,T,H]
        feat = feat.transpose(1, 2)                           # [B,H,T]
        feat_t = self.tcn(feat)                               # [B,H,T]
        pooled = self.attn(feat_t)                            # [B,H]
        h = self.head(pooled)                                 # [B,emb]
        logits = self.cls(h) * self.logit_scale
        return h, logits


def build_model_from_config(model_cfg: Dict[str, Any], n_classes: int) -> nn.Module:
    name = str(model_cfg.get("name", "")).lower().strip()
    p = model_cfg.get("params", {}) or {}
    if name not in ["wiprompt_tcn", "wiprompt", "tcn_attn"]:
        raise KeyError(f"Unknown model '{name}'. Supported: wiprompt_tcn")

    return WiPromptTCN(
        n_classes=n_classes,
        tcn_channels=int(p.get("tcn_channels", 128)),
        tcn_layers=int(p.get("tcn_layers", 4)),
        tcn_kernel=int(p.get("tcn_kernel", 5)),
        tcn_dropout=float(p.get("tcn_dropout", 0.1)),
        emb_dim=int(p.get("emb_dim", 256)),
        logit_scale=float(p.get("logit_scale", 1.0)),
    )


# =========================================================
# 4) Losses: Supervised InfoNCE + Cheeger surrogate
# =========================================================

def supervised_info_nce(h: torch.Tensor, y: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    B = h.size(0)
    if B <= 1:
        return h.new_tensor(0.0)

    h = F.normalize(h, dim=1)
    sim = (h @ h.t()) / max(float(temperature), 1e-6)

    diag = torch.eye(B, device=h.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -1e4)

    y = y.view(-1, 1)
    pos = (y == y.t()) & (~diag)
    pos_cnt = pos.sum(dim=1)
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return h.new_tensor(0.0)

    logp = F.log_softmax(sim, dim=1)
    pos_logp = (logp * pos.float()).sum(dim=1) / torch.clamp(pos_cnt.float(), min=1.0)
    return (-pos_logp[valid]).mean()


def build_knn_graph(
    z: torch.Tensor,
    k: int,
    sigma: Union[str, float] = "median",
    sigma_min: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = z.size(0)
    if N <= 1:
        d = z.new_zeros((N,))
        L = z.new_zeros((N, N))
        return d, L

    D = torch.cdist(z, z)  # [N,N]
    if sigma == "median":
        with torch.no_grad():
            mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
            tri = D[mask]
            sig = tri.median()
            sig = torch.clamp(sig, min=float(sigma_min))
    else:
        sig = torch.tensor(float(sigma), device=z.device)

    k_curr = min(int(k) + 1, N)
    knn = torch.topk(-D, k=k_curr, dim=1).indices[:, 1:]  # exclude self

    W = torch.zeros_like(D)
    rows = torch.arange(N, device=z.device).view(-1, 1).expand_as(knn)
    dist_sq = D[rows, knn] ** 2
    weights = torch.exp(-dist_sq / (2 * sig * sig + 1e-8))
    W[rows, knn] = weights

    W = 0.5 * (W + W.t())
    W.fill_diagonal_(0.0)

    d = W.sum(dim=1)
    L = torch.diag(d) - W
    return d, L


def cheeger_surrogate(
    z: torch.Tensor,
    y: torch.Tensor,
    logits: torch.Tensor,
    k: int = 10,
    sigma: Union[str, float] = "median",
    sigma_min: float = 1e-3,
) -> torch.Tensor:
    N = z.size(0)
    if N <= 1:
        return z.new_tensor(0.0)

    d, L = build_knn_graph(z, k=k, sigma=sigma, sigma_min=sigma_min)
    C = logits.size(1)

    P = F.softmax(logits, dim=1)
    Y = P.clone()
    labeled = (y >= 0)
    if labeled.any():
        Y[labeled] = 0.0
        Y[labeled, y[labeled]] = 1.0

    Ddiag = torch.diag(d)
    total = z.new_tensor(0.0)
    for c in range(C):
        yc = Y[:, c:c+1]
        num = (yc.t() @ L @ yc).squeeze()
        den = (yc.t() @ Ddiag @ yc).squeeze()
        total = total + num / (den + 1e-5)
    return total


# =========================================================
# 5) Addons: EMA, EarlyStopping, CosineWarmup
# =========================================================

@dataclass
class EarlyStopping:
    patience: int = 20
    mode: str = "max"  # "max" or "min"
    best: Optional[float] = None
    bad_count: int = 0

    def step(self, value: float) -> bool:
        v = float(value)
        if self.best is None:
            self.best = v
            self.bad_count = 0
            return False
        improved = (v > self.best) if self.mode == "max" else (v < self.best)
        if improved:
            self.best = v
            self.bad_count = 0
            return False
        self.bad_count += 1
        return self.bad_count >= int(self.patience)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    def restore(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}


class CosineWithWarmup:
    """Step once per epoch."""
    def __init__(self, optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int, eta_min: float):
        self.opt = optimizer
        self.total = max(1, int(total_epochs))
        self.warm = max(0, int(warmup_epochs))
        self.eta_min = float(eta_min)
        self.base_lrs = [pg["lr"] for pg in self.opt.param_groups]
        self.epoch = 0

    def step(self) -> None:
        self.epoch += 1
        e = self.epoch
        for i, pg in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if self.warm > 0 and e <= self.warm:
                lr = base * (e / self.warm)
            else:
                t = (e - self.warm) / max(1, (self.total - self.warm))
                t = min(max(t, 0.0), 1.0)
                lr = self.eta_min + 0.5 * (base - self.eta_min) * (1.0 + math.cos(t * math.pi))
            pg["lr"] = float(lr)


# =========================================================
# 6) Evaluation
# =========================================================

@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    return_confusion: bool = False,
    return_preds: bool = False,
) -> Dict[str, Any]:
    model.eval()
    tot = 0
    corr = 0

    conf = None
    if return_confusion:
        conf = torch.zeros((n_classes, n_classes), dtype=torch.long)

    all_pred = []
    all_true = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, logits = model(x)
        pred = logits.argmax(dim=1)

        corr += (pred == y).sum().item()
        tot += int(y.numel())

        if conf is not None:
            for t, p in zip(y.view(-1), pred.view(-1)):
                conf[int(t.item()), int(p.item())] += 1

        if return_preds:
            all_pred.append(pred.detach().cpu())
            all_true.append(y.detach().cpu())

    top1 = corr / tot if tot > 0 else 0.0
    out: Dict[str, Any] = {"top1": float(top1), "total": int(tot)}
    if conf is not None:
        out["confusion"] = conf.numpy()
    if return_preds:
        out["pred"] = torch.cat(all_pred, dim=0).numpy() if all_pred else np.zeros((0,), dtype=np.int64)
        out["true"] = torch.cat(all_true, dim=0).numpy() if all_true else np.zeros((0,), dtype=np.int64)
    return out


# =========================================================
# 7) DataLoader builder (label remap + collate)
# =========================================================

def _build_label_remap(train_labels: List[int]) -> Dict[int, int]:
    uniq = sorted(set(int(y) for y in train_labels if int(y) >= 0))
    if not uniq:
        raise RuntimeError("No valid labels found in training set.")
    return {lab: i for i, lab in enumerate(uniq)}


def _make_collate(remap: Dict[int, int]):
    def collate(batch):
        xs, ys = [], []
        for x, y in batch:
            yi = int(y.item())
            if yi not in remap:
                raise ValueError(f"Label {yi} not in remap. Check split consistency.")
            xs.append(x)
            ys.append(remap[yi])
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
    return collate


def build_loaders_from_manifests(
    cfg: Dict[str, Any],
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, int], Dict[int, int]]:
    tr_items = load_manifest(train_manifest)
    va_items = load_manifest(val_manifest)
    te_items = load_manifest(test_manifest)
    tr_paths, tr_labels = manifest_paths_labels(tr_items)
    va_paths, va_labels = manifest_paths_labels(va_items)
    te_paths, te_labels = manifest_paths_labels(te_items)

    ds_cfg = cfg["dataset"]
    train_ds = build_dataset_from_config(ds_cfg, tr_paths, tr_labels)
    val_ds = build_dataset_from_config(ds_cfg, va_paths, va_labels)
    test_ds = build_dataset_from_config(ds_cfg, te_paths, te_labels)

    # build remap from training dataset labels if available
    raw_train_labels = getattr(train_ds, "labels", tr_labels)
    remap = _build_label_remap(list(raw_train_labels))

    counts: Dict[int, int] = {}
    for y in raw_train_labels:
        y = int(y)
        if y in remap:
            ny = remap[y]
            counts[ny] = counts.get(ny, 0) + 1

    dl_cfg = cfg.get("dataloader", {})
    bs = int(dl_cfg.get("batch_size", 64))
    nw = int(dl_cfg.get("num_workers", 4))
    pf = int(dl_cfg.get("prefetch_factor", 2))
    pw = bool(dl_cfg.get("persistent_workers", True))
    pm = bool(dl_cfg.get("pin_memory", True))
    drop_last = bool(dl_cfg.get("drop_last", True))

    collate_fn = _make_collate(remap)

    def _pf_arg():
        return pf if nw > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        prefetch_factor=_pf_arg(),
        persistent_workers=pw if nw > 0 else False,
        pin_memory=pm,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        prefetch_factor=_pf_arg(),
        persistent_workers=pw if nw > 0 else False,
        pin_memory=pm,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        prefetch_factor=_pf_arg(),
        persistent_workers=pw if nw > 0 else False,
        pin_memory=pm,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader, remap, counts


# =========================================================
# 8) Trainer (AMP + accum + clip + EMA + early stop + cosine warmup)
# =========================================================

class SafeAugmentor(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, C, T, K = x.shape

        if C > 1 and torch.rand(1).item() < self.p:
            c_idx = int(torch.randint(0, C, (1,)).item())
            x[:, c_idx] = 0.0

        if torch.rand(1).item() < self.p:
            x = x + torch.randn_like(x) * 0.02
        return x


class Trainer:
    """
    A single-class trainer to keep the repository compact but "research-grade".
    """
    def __init__(
        self,
        cfg: Dict[str, Any],
        model: nn.Module,
        n_classes: int,
        device: torch.device,
        run_dir: str,
        logger: logging.Logger,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.cfg = cfg
        self.model = model.to(device)
        self.n_classes = int(n_classes)
        self.device = device
        self.run_dir = run_dir
        self.logger = logger

        aug_cfg = cfg.get("augment", {}) or {}
        self.aug = SafeAugmentor(p=float(aug_cfg.get("prob", 0.5))).to(device)

        # optimizer
        opt_cfg = cfg.get("optimizer", {}) or {}
        lr = float(opt_cfg.get("lr", 3e-4))
        wd = float(opt_cfg.get("weight_decay", 1e-4))
        fused = False
        if device.type == "cuda" and hasattr(torch.optim.AdamW, "fused"):
            fused = True
        self.opt = AdamW(self.model.parameters(), lr=lr, weight_decay=wd, fused=fused)

        # AMP scaler
        tr_cfg = cfg.get("trainer", {}) or {}
        amp_on = bool(tr_cfg.get("amp", True)) and (not bool(tr_cfg.get("force_fp32", False)))
        try:
            from torch.amp import GradScaler
            self.scaler = GradScaler("cuda", enabled=amp_on and device.type == "cuda")
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=amp_on and device.type == "cuda")

        # scheduler
        sch_cfg = cfg.get("scheduler", {}) or {}
        self.sched = None
        if bool(sch_cfg.get("enabled", True)):
            self.sched = CosineWithWarmup(
                self.opt,
                total_epochs=int(tr_cfg.get("epochs", 30)),
                warmup_epochs=int(sch_cfg.get("warmup_epochs", 3)),
                eta_min=float(sch_cfg.get("lr_min", 5e-5)),
            )

        # EMA
        ema_cfg = cfg.get("ema", {}) or {}
        self.ema = EMA(self.model, decay=float(ema_cfg.get("decay", 0.999))) if bool(ema_cfg.get("enabled", False)) else None

        # losses config
        self.loss_cfg = cfg.get("loss", {}) or {}
        self.ce_cfg = self.loss_cfg.get("ce", {}) or {}
        self.nce_cfg = self.loss_cfg.get("nce", {}) or {}
        self.chg_cfg = self.loss_cfg.get("cheeger", {}) or {}

        self.class_weights = class_weights.to(device) if class_weights is not None else None

        self.metrics = JSONLWriter(os.path.join(run_dir, "metrics.jsonl"))

        out_cfg = cfg.get("output", {}) or {}
        self.best_path = os.path.join(out_cfg.get("out_dir", "./runs"), cfg.get("exp_name", "exp"), "best.pt")

    def _ce_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ls = float(self.ce_cfg.get("label_smoothing", 0.0))
        if ls > 0:
            C = logits.size(1)
            logp = F.log_softmax(logits, dim=1)
            y_one = F.one_hot(y, num_classes=C).float()
            y_smooth = (1.0 - ls) * y_one + ls / float(C)
            return -(y_smooth * logp).sum(dim=1).mean()

        if bool(self.ce_cfg.get("class_balanced", False)) and self.class_weights is not None:
            return F.cross_entropy(logits, y, weight=self.class_weights)
        return F.cross_entropy(logits, y)

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        self.opt.zero_grad(set_to_none=True)

        tr_cfg = self.cfg.get("trainer", {}) or {}
        accum = max(1, int(tr_cfg.get("accum_steps", 1)))
        clip = float(tr_cfg.get("grad_clip", 1.0))

        use_aug = bool((self.cfg.get("augment", {}) or {}).get("enabled", True))

        try:
            from torch.amp import autocast
            amp_ctx = autocast("cuda", enabled=self.scaler.is_enabled())
        except Exception:
            amp_ctx = torch.cuda.amp.autocast(enabled=self.scaler.is_enabled())

        total = 0.0
        steps = 0

        it = loader
        if tqdm is not None:
            it = tqdm(loader, desc=f"train ep{epoch}", leave=False)

        for step, (x, y) in enumerate(it):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            if use_aug:
                with torch.no_grad():
                    x = self.aug(x)

            with amp_ctx:
                h, logits = self.model(x)
                loss = self._ce_loss(logits, y)

                # InfoNCE
                if bool(self.nce_cfg.get("enabled", False)) and epoch >= int(self.nce_cfg.get("warmup_epoch", 0)):
                    lam = float(self.nce_cfg.get("lambda", 0.0))
                    if lam > 0:
                        loss = loss + lam * supervised_info_nce(h, y, float(self.nce_cfg.get("temperature", 0.1)))

                # Cheeger surrogate
                if bool(self.chg_cfg.get("enabled", False)) and epoch >= int(self.chg_cfg.get("warmup_epoch", 5)):
                    lam = float(self.chg_cfg.get("lambda", 0.0))
                    if lam > 0:
                        h_norm = F.normalize(h, dim=1)
                        loss = loss + lam * cheeger_surrogate(
                            h_norm, y, logits,
                            k=int(self.chg_cfg.get("knn_k", 10)),
                            sigma=self.chg_cfg.get("sigma", "median"),
                            sigma_min=float(self.chg_cfg.get("sigma_min", 1e-3)),
                        )

                loss = loss / float(accum)

            if torch.isnan(loss):
                raise RuntimeError("NaN loss encountered. Check data / lr / amp stability.")

            self.scaler.scale(loss).backward()

            if ((step + 1) % accum == 0) or (step + 1 == len(loader)):
                if clip > 0:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

                if self.ema:
                    self.ema.update(self.model)

            total += float(loss.item()) * float(accum)
            steps += 1

        return total / max(1, steps)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader) -> Dict[str, Any]:
        tr_cfg = self.cfg.get("trainer", {}) or {}
        epochs = int(tr_cfg.get("epochs", 30))

        es_cfg = self.cfg.get("early_stop", {}) or {}
        use_es = bool(es_cfg.get("enabled", True))
        early = EarlyStopping(patience=int(es_cfg.get("patience", 20)), mode=str(es_cfg.get("mode", "max")))

        best_val = -1e9
        best_ep = -1

        self.logger.info(f"[run] fingerprint={config_fingerprint(self.cfg)}")

        for ep in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader, ep)

            # validation
            if self.ema:
                self.ema.apply(self.model)
                val_res = evaluate_split(self.model, val_loader, self.device, self.n_classes)
                self.ema.restore(self.model)
            else:
                val_res = evaluate_split(self.model, val_loader, self.device, self.n_classes)

            val_top1 = float(val_res.get("top1", 0.0))

            if self.sched:
                self.sched.step()

            self.metrics.write({"epoch": ep, "train_loss": float(train_loss), "val_top1": val_top1})
            self.logger.info(f"[ep {ep:03d}] train_loss={train_loss:.4f} | val_top1={val_top1:.4f} | time={time.time()-t0:.1f}s")

            if val_top1 > best_val:
                best_val = val_top1
                best_ep = ep
                save_checkpoint(
                    self.best_path,
                    model=self.model,
                    cfg=self.cfg,
                    epoch=ep,
                    best_metric=best_val,
                    optimizer=self.opt,
                    scaler=self.scaler,
                    extra={"val": val_res},
                )
                self.logger.info(f"[save] best updated @ ep{ep}: val_top1={best_val:.4f}")

            if use_es and early.step(val_top1):
                self.logger.info(f"[early stop] triggered at ep{ep}")
                break

        # Load best and test
        if os.path.exists(self.best_path):
            ckpt = load_checkpoint(self.best_path, map_location=str(self.device))
            self.model.load_state_dict(ckpt["model"], strict=False)

        test_res = evaluate_split(self.model, test_loader, self.device, self.n_classes, return_preds=True)
        test_top1 = float(test_res.get("top1", 0.0))

        # export predictions
        pred_path = os.path.join(self.run_dir, "test_predictions.jsonl")
        w = JSONLWriter(pred_path)
        pred = test_res.get("pred", np.zeros((0,), dtype=np.int64))
        true = test_res.get("true", np.zeros((0,), dtype=np.int64))
        for i in range(len(pred)):
            w.write({"i": int(i), "true": int(true[i]), "pred": int(pred[i])})

        self.metrics.write({"epoch": best_ep, "test_top1": test_top1, "split": "test"})
        self.logger.info(f"[TEST] best_ep={best_ep} | top1={test_top1:.4f} | pred_saved={pred_path}")

        return {"best_val": best_val, "best_ep": best_ep, "test": test_res}


# =========================================================
# 9) Override parsing for CLI (--set a.b.c=value)
# =========================================================

def parse_kv_overrides(kvs: List[str]) -> Dict[str, Any]:
    """
    Parse overrides like:
      --set trainer.epochs=50
      --set loss.cheeger.enabled=true
      --set dataset.params.temporal_feats=["level","diff","highpass"]
    Values are parsed by JSON when possible; otherwise string.
    """
    out: Dict[str, Any] = {}
    for kv in kvs or []:
        if "=" not in kv:
            raise ValueError(f"Invalid --set '{kv}', expected key=value")
        k, v = kv.split("=", 1)
        key = k.strip()
        raw = v.strip()

        # try JSON parse
        try:
            val = json.loads(raw)
        except Exception:
            # fallback: parse common bool
            low = raw.lower()
            if low in ["true", "false"]:
                val = (low == "true")
            else:
                val = raw

        # assign into nested dict
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = val
    return out
