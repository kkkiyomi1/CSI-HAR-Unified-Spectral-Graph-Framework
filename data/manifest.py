from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class ManifestItem:
    path: str
    label: int
    meta: Dict[str, Any]


def _parse_line_as_json(ln: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(ln)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def load_manifest(path: str) -> List[ManifestItem]:
    """
    Loads a manifest file.

    Supported formats:
    1) Manifest v2 JSONL:
       {"path":"...","label":3,"subject":"u1","domain":"roomA",...}

    2) Legacy plain text:
       /path/to/sample.mat
       /path/to/sample.mat 3   (optional label at end)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    items: List[ManifestItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue

            obj = _parse_line_as_json(ln)
            if obj is not None:
                p = str(obj["path"])
                y = int(obj["label"])
                meta = {k: v for k, v in obj.items() if k not in ("path", "label")}
                items.append(ManifestItem(path=p, label=y, meta=meta))
                continue

            # Legacy: "path" or "path label"
            if " " in ln:
                p, y_str = ln.rsplit(" ", 1)
                try:
                    y = int(y_str)
                except Exception:
                    p, y = ln, -1
            else:
                p, y = ln, -1

            items.append(ManifestItem(path=p, label=y, meta={}))

    return items


def manifest_paths_and_labels(items: List[ManifestItem]) -> Tuple[List[str], List[int]]:
    return [it.path for it in items], [int(it.label) for it in items]


def save_manifest(path: str, items: List[ManifestItem]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            obj = {"path": it.path, "label": int(it.label), **(it.meta or {})}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
