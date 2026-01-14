from .manifest import ManifestItem, load_manifest, manifest_paths_and_labels, save_manifest
from .registry import build_dataset, list_datasets, register_dataset

__all__ = [
    "ManifestItem",
    "load_manifest",
    "manifest_paths_and_labels",
    "save_manifest",
    "register_dataset",
    "build_dataset",
    "list_datasets",
]

# Note:
# Actual dataset adapters (widar_bvp, generic_csi) will be added in the next batch and
# will register themselves via @register_dataset("widar3_bvp") and @register_dataset("generic_csi").
