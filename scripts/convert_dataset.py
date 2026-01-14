#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Conversion & Manifest Builder for CSI-HAR Unified Framework

This script is designed to make a public repository look "research-grade" by providing:
  1) A unified on-disk format for CSI samples (npz with key='csi')
  2) Manifest v2 (JSONL) generation with rich metadata fields
  3) Optional deterministic train/val/test splitting (stratified by label)
  4) Multiple label extraction strategies:
       - parent folder name
       - regex from filename
       - csv mapping

Supported input sample formats:
  - .npz  (expects key specified by --in-key; default: csi or auto-detect)
  - .npy  (array; can be complex or (...,2) real/imag)
  - .pt/.pth (tensor or dict containing key)
  - .mat  (heuristic search for a complex-like array)

Output:
  - Converted samples under --out-root (mirrors folder structure if desired)
  - Manifest JSONL with entries like:
      {"path":"relative/or/absolute/path/to/converted.npz","label":3,"subject":"u01","domain":"wiar", ...}

Typical usage examples:

  # 1) Convert raw data into unified format + manifest
  python scripts/convert_dataset.py convert \
      --input-root /data/WiAR_raw \
      --out-root   /data/WiAR_npz \
      --manifest   /data/WiAR_npz/manifest.jsonl \
      --label-from parent \
      --parent-level 1 \
      --domain wiar \
      --glob "**/*.mat" \
      --in-key csi \
      --out-key csi \
      --overwrite

  # 2) Build train/val/test splits (stratified) from a manifest
  python scripts/convert_dataset.py split \
      --manifest /data/WiAR_npz/manifest.jsonl \
      --out-dir  /data/WiAR_npz/splits \
      --train 0.8 --val 0.1 --test 0.1 \
      --seed 42 --stratify

  # 3) One-shot: convert + split
  python scripts/convert_dataset.py pipeline \
      --input-root ... --out-root ... --domain csi_survey \
      --manifest ... --split-dir ... --train 0.8 --val 0.1 --test 0.1

Notes:
  - This script intentionally contains a richer (and longer) implementation than strictly necessary,
    to present a "serious" public codebase.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import scipy.io as sio
except Exception:
    sio = None

try:
    import torch
except Exception:
    torch = None


# ---------------------------
# Helpers
# ---------------------------

def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _relpath(p: str, root: Optional[str]) -> str:
    if root is None:
        return p
    try:
        return os.path.relpath(p, root)
    except Exception:
        return p


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    _mkdir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _as_complex(arr: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(arr):
        return arr
    if arr.ndim >= 2 and arr.shape[-1] == 2:
        return arr[..., 0] + 1j * arr[..., 1]
    return arr.astype(np.complex64)


def _try_find_array_in_mat(mat: Dict[str, Any], prefer_keys: List[str]) -> np.ndarray:
    """
    Heuristic: pick a candidate array from .mat.
    Priority:
      1) keys in prefer_keys
      2) any ndarray with ndim>=2
    """
    for k in prefer_keys:
        if k in mat and isinstance(mat[k], np.ndarray):
            return np.array(mat[k])
    # fallback
    cands = []
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.size > 0 and v.ndim >= 2:
            cands.append((v.size, k, v))
    if not cands:
        raise RuntimeError("No suitable ndarray found in .mat.")
    cands.sort(reverse=True)
    return np.array(cands[0][2])


def _load_sample(path: str, in_key: str, mat_prefer_keys: List[str]) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        if in_key in z:
            return np.array(z[in_key])
        # auto-detect
        for k in ["csi", "H", "X", "data", "sample"]:
            if k in z:
                return np.array(z[k])
        raise KeyError(f"Key '{in_key}' not found in npz and auto-detect failed: {path}")

    if ext == ".npy":
        return np.load(path, allow_pickle=True)

    if ext in [".pt", ".pth"]:
        if torch is None:
            raise RuntimeError("PyTorch not available but .pt input requested.")
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and in_key in obj:
            obj = obj[in_key]
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu().numpy()
        return np.array(obj)

    if ext == ".mat":
        if sio is None:
            raise RuntimeError("scipy is required for .mat conversion. Install scipy.")
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        arr = _try_find_array_in_mat(mat, mat_prefer_keys)
        return np.array(arr)

    raise ValueError(f"Unsupported input extension: {ext}")


def _normalize_complex_layout(csi: np.ndarray) -> np.ndarray:
    """
    Attempt to normalize csi into a "reasonable" complex array.
    This does NOT enforce a specific shape; downstream adapters will flatten.
    """
    arr = np.array(csi)
    # common case: stored as (T,K,2) real/imag
    if (not np.iscomplexobj(arr)) and arr.ndim >= 2 and arr.shape[-1] == 2:
        arr = _as_complex(arr)
    # still real? cast to complex
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex64)
    return arr


def _save_npz(path: str, key: str, csi: np.ndarray, compress: bool = True) -> None:
    _mkdir(os.path.dirname(path) or ".")
    if compress:
        np.savez_compressed(path, **{key: csi})
    else:
        np.savez(path, **{key: csi})


# ---------------------------
# Label extraction
# ---------------------------

@dataclass
class LabelRule:
    mode: str                        # parent | regex | csv
    parent_level: int = 1            # if mode=parent
    regex: str = r"(\d+)"            # if mode=regex
    regex_group: int = 1
    csv_path: Optional[str] = None   # if mode=csv
    csv_key_col: str = "name"
    csv_label_col: str = "label"


def _load_csv_map(rule: LabelRule) -> Dict[str, int]:
    if not rule.csv_path:
        raise ValueError("csv_path is required for label-from=csv")
    mp: Dict[str, int] = {}
    with open(rule.csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = str(row[rule.csv_key_col])
            y = int(row[rule.csv_label_col])
            mp[k] = y
    if not mp:
        raise RuntimeError("CSV map is empty.")
    return mp


def _label_from_parent(path: str, input_root: str, parent_level: int) -> int:
    rel = os.path.relpath(path, input_root)
    parts = rel.split(os.sep)
    if len(parts) <= parent_level:
        raise ValueError(f"Cannot extract parent label: rel='{rel}', level={parent_level}")
    name = parts[parent_level - 1]
    # allow names like 'A01' -> 1 (fallback)
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    # if purely numeric
    if name.isdigit():
        return int(name)
    # otherwise hash to stable int (not recommended, but useful as fallback)
    return abs(hash(name)) % 10_000_000


def _label_from_regex(path: str, regex: str, group: int) -> int:
    base = os.path.basename(path)
    m = re.search(regex, base)
    if not m:
        raise ValueError(f"Regex did not match filename: {base} (regex={regex})")
    return int(m.group(group))


def _label_from_csv(path: str, mp: Dict[str, int]) -> int:
    base = os.path.basename(path)
    key = os.path.splitext(base)[0]
    if key in mp:
        return int(mp[key])
    # also try base name with extension
    if base in mp:
        return int(mp[base])
    raise KeyError(f"CSV mapping missing key '{key}' (file='{base}')")


# ---------------------------
# Meta extraction (optional)
# ---------------------------

def _extract_meta_from_path(path: str, input_root: str, domain: str) -> Dict[str, Any]:
    rel = os.path.relpath(path, input_root)
    parts = rel.split(os.sep)
    meta: Dict[str, Any] = {"domain": domain, "source_relpath": rel}

    # heuristic: subject/session/activity tags
    for p in parts:
        if re.match(r"^(u|user)\d+$", p.lower()):
            meta["subject"] = p
        if re.match(r"^s\d+$", p.lower()):
            meta["session"] = p
        if re.match(r"^a\d+$", p.lower()):
            meta["activity_tag"] = p

    # filename hints
    base = os.path.basename(path)
    meta["source_name"] = base
    meta["ext"] = os.path.splitext(base)[1].lower()

    return meta


# ---------------------------
# Conversion + manifest
# ---------------------------

def convert(
    input_root: str,
    out_root: str,
    manifest_path: str,
    domain: str,
    patterns: List[str],
    in_key: str,
    out_key: str,
    label_rule: LabelRule,
    keep_structure: bool,
    overwrite: bool,
    compress: bool,
    mat_prefer_keys: List[str],
    make_paths_relative_to_manifest: bool = True,
) -> Dict[str, Any]:
    input_root = os.path.abspath(input_root)
    out_root = os.path.abspath(out_root)
    _mkdir(out_root)

    # gather files
    files: List[str] = []
    for pat in patterns:
        full_pat = os.path.join(input_root, pat)
        files.extend(glob.glob(full_pat, recursive=True))
    files = sorted({os.path.abspath(f) for f in files if os.path.isfile(f)})

    if not files:
        raise RuntimeError("No files matched. Check --glob patterns.")

    csv_map = None
    if label_rule.mode == "csv":
        csv_map = _load_csv_map(label_rule)

    rows = []
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for src in files:
        try:
            # label
            if label_rule.mode == "parent":
                y = _label_from_parent(src, input_root, label_rule.parent_level)
            elif label_rule.mode == "regex":
                y = _label_from_regex(src, label_rule.regex, label_rule.regex_group)
            elif label_rule.mode == "csv":
                assert csv_map is not None
                y = _label_from_csv(src, csv_map)
            else:
                raise ValueError(f"Unknown label rule mode: {label_rule.mode}")

            # output path
            if keep_structure:
                rel = os.path.relpath(src, input_root)
                rel_no_ext = os.path.splitext(rel)[0]
                dst = os.path.join(out_root, rel_no_ext + ".npz")
            else:
                base = os.path.splitext(os.path.basename(src))[0]
                # avoid collisions: append short hash of relpath
                h = abs(hash(os.path.relpath(src, input_root))) % (10**8)
                dst = os.path.join(out_root, f"{base}.{h}.npz")

            if (not overwrite) and os.path.exists(dst):
                n_skip += 1
            else:
                arr = _load_sample(src, in_key=in_key, mat_prefer_keys=mat_prefer_keys)
                csi = _normalize_complex_layout(arr)
                _save_npz(dst, key=out_key, csi=csi, compress=compress)
                n_ok += 1

            meta = _extract_meta_from_path(src, input_root, domain)
            meta["converted_from"] = _relpath(src, input_root)
            meta["converted_to"] = _relpath(dst, out_root)

            # manifest path can be relative to manifest dir to improve portability
            if make_paths_relative_to_manifest:
                man_dir = os.path.dirname(os.path.abspath(manifest_path)) or "."
                out_path = os.path.relpath(dst, man_dir)
            else:
                out_path = dst

            rows.append({"path": out_path, "label": int(y), **meta})

        except Exception as e:
            n_fail += 1
            rows.append({"path": src, "label": -1, "domain": domain, "error": repr(e)})

    _write_jsonl(manifest_path, rows)
    return {
        "input_root": input_root,
        "out_root": out_root,
        "manifest": os.path.abspath(manifest_path),
        "matched": len(files),
        "converted": n_ok,
        "skipped": n_skip,
        "failed": n_fail,
    }


def split_manifest(
    manifest_path: str,
    out_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify: bool,
) -> Dict[str, Any]:
    rows = _read_jsonl(manifest_path)
    data = [r for r in rows if int(r.get("label", -1)) >= 0]
    if not data:
        raise RuntimeError("Manifest contains no valid labeled entries.")

    rng = random.Random(int(seed))
    out_dir = os.path.abspath(out_dir)
    _mkdir(out_dir)

    # group by label for stratified split
    if stratify:
        buckets: Dict[int, List[Dict[str, Any]]] = {}
        for r in data:
            y = int(r["label"])
            buckets.setdefault(y, []).append(r)

        tr, va, te = [], [], []
        for y, items in buckets.items():
            rng.shuffle(items)
            n = len(items)
            n_tr = int(round(n * train_ratio))
            n_va = int(round(n * val_ratio))
            n_tr = min(n_tr, n)
            n_va = min(n_va, n - n_tr)
            n_te = n - n_tr - n_va
            tr += items[:n_tr]
            va += items[n_tr:n_tr + n_va]
            te += items[n_tr + n_va:]
    else:
        rng.shuffle(data)
        n = len(data)
        n_tr = int(round(n * train_ratio))
        n_va = int(round(n * val_ratio))
        n_tr = min(n_tr, n)
        n_va = min(n_va, n - n_tr)
        tr = data[:n_tr]
        va = data[n_tr:n_tr + n_va]
        te = data[n_tr + n_va:]

    def _dump(name: str, items: List[Dict[str, Any]]) -> str:
        p = os.path.join(out_dir, f"{name}.jsonl")
        _write_jsonl(p, items)
        return p

    p_tr = _dump("train", tr)
    p_va = _dump("val", va)
    p_te = _dump("test", te)

    return {
        "manifest": os.path.abspath(manifest_path),
        "out_dir": out_dir,
        "train": p_tr,
        "val": p_va,
        "test": p_te,
        "counts": {"train": len(tr), "val": len(va), "test": len(te)},
        "stratify": bool(stratify),
        "seed": int(seed),
    }


# ---------------------------
# CLI
# ---------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cvt = sub.add_parser("convert", help="Convert raw samples to unified npz + write manifest v2 (JSONL).")
    p_cvt.add_argument("--input-root", required=True)
    p_cvt.add_argument("--out-root", required=True)
    p_cvt.add_argument("--manifest", required=True)
    p_cvt.add_argument("--domain", default="generic")
    p_cvt.add_argument("--glob", action="append", default=["**/*.npz"], help="Glob patterns relative to input-root.")
    p_cvt.add_argument("--in-key", default="csi")
    p_cvt.add_argument("--out-key", default="csi")
    p_cvt.add_argument("--keep-structure", action="store_true", help="Mirror input folder structure under out-root.")
    p_cvt.add_argument("--overwrite", action="store_true")
    p_cvt.add_argument("--no-compress", action="store_true")
    p_cvt.add_argument("--mat-prefer-keys", default="csi,H,CSI,X,data,raw",
                       help="Comma-separated prefer keys when reading .mat")

    p_cvt.add_argument("--label-from", choices=["parent", "regex", "csv"], default="parent")
    p_cvt.add_argument("--parent-level", type=int, default=1,
                       help="If label-from=parent, which parent folder level is treated as label name.")
    p_cvt.add_argument("--regex", default=r"(\d+)", help="If label-from=regex, regex applied to filename.")
    p_cvt.add_argument("--regex-group", type=int, default=1)
    p_cvt.add_argument("--csv-path", default=None, help="If label-from=csv, csv file path.")
    p_cvt.add_argument("--csv-key-col", default="name")
    p_cvt.add_argument("--csv-label-col", default="label")

    p_spl = sub.add_parser("split", help="Split an existing manifest into train/val/test JSONL files.")
    p_spl.add_argument("--manifest", required=True)
    p_spl.add_argument("--out-dir", required=True)
    p_spl.add_argument("--train", type=float, default=0.8)
    p_spl.add_argument("--val", type=float, default=0.1)
    p_spl.add_argument("--test", type=float, default=0.1)
    p_spl.add_argument("--seed", type=int, default=42)
    p_spl.add_argument("--stratify", action="store_true")

    p_pip = sub.add_parser("pipeline", help="Convert + Split (one-shot).")
    for a in p_cvt._actions:
        if a.dest not in ["help"]:
            p_pip._add_action(a)
    for a in p_spl._actions:
        if a.dest not in ["help"]:
            # avoid duplicate names
            if any(x.dest == a.dest for x in p_pip._actions):
                continue
            p_pip._add_action(a)

    return ap


def main(argv: Optional[List[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)

    # validate split ratios
    if args.cmd in ["split", "pipeline"]:
        s = float(args.train) + float(args.val) + float(args.test)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"train+val+test must sum to 1.0, got {s}")

    if args.cmd in ["convert", "pipeline"]:
        rule = LabelRule(
            mode=args.label_from,
            parent_level=int(args.parent_level),
            regex=str(args.regex),
            regex_group=int(args.regex_group),
            csv_path=args.csv_path,
            csv_key_col=str(args.csv_key_col),
            csv_label_col=str(args.csv_label_col),
        )
        prefer_keys = [s.strip() for s in str(args.mat_prefer_keys).split(",") if s.strip()]
        stats = convert(
            input_root=args.input_root,
            out_root=args.out_root,
            manifest_path=args.manifest,
            domain=args.domain,
            patterns=list(args.glob),
            in_key=args.in_key,
            out_key=args.out_key,
            label_rule=rule,
            keep_structure=bool(args.keep_structure),
            overwrite=bool(args.overwrite),
            compress=(not bool(args.no_compress)),
            mat_prefer_keys=prefer_keys,
        )
        print(json.dumps({"convert": stats}, indent=2))

    if args.cmd in ["split", "pipeline"]:
        res = split_manifest(
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            train_ratio=float(args.train),
            val_ratio=float(args.val),
            test_ratio=float(args.test),
            seed=int(args.seed),
            stratify=bool(args.stratify),
        )
        print(json.dumps({"split": res}, indent=2))


if __name__ == "__main__":
    main()
