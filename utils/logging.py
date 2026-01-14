from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def setup_logger(run_dir: str, name: str = "csihar") -> logging.Logger:
    """
    Creates a console + file logger.
    """
    os.makedirs(run_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class JSONLWriter:
    """
    Append-only JSONL writer for metrics logging.
    Each call to write() appends one line.
    """

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def write(self, obj: Dict[str, Any]) -> None:
        obj = dict(obj)
        obj.setdefault("time", datetime.utcnow().isoformat(timespec="seconds") + "Z")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
