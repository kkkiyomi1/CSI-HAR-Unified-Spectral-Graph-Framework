from __future__ import annotations

import platform
from typing import Any, Dict

import torch


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
    else:
        info["cuda_device_count"] = 0

    return info
