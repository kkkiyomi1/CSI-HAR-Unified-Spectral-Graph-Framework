from __future__ import annotations

from typing import Any, Callable, Dict, List, Type

from torch.utils.data import Dataset


_DATASET_REGISTRY: Dict[str, Type[Dataset]] = {}


def register_dataset(name: str) -> Callable[[Type[Dataset]], Type[Dataset]]:
    """
    Decorator to register a Dataset class under a string name.
    """

    def _wrap(cls: Type[Dataset]) -> Type[Dataset]:
        if name in _DATASET_REGISTRY:
            raise KeyError(f"Dataset '{name}' already registered by {_DATASET_REGISTRY[name]}")
        _DATASET_REGISTRY[name] = cls
        return cls

    return _wrap


def build_dataset(name: str, paths: List[str], labels: List[int], params: Dict[str, Any]) -> Dataset:
    """
    Instantiate a dataset adapter by name.
    """
    if name not in _DATASET_REGISTRY:
        known = ", ".join(sorted(_DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Registered: [{known}]")
    cls = _DATASET_REGISTRY[name]
    return cls(paths=paths, labels=labels, **(params or {}))


def list_datasets() -> List[str]:
    return sorted(_DATASET_REGISTRY.keys())
