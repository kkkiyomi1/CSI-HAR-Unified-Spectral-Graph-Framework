from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: random seed.
        deterministic: if True, enables deterministic mode (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 2.x deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
