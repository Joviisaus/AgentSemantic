"""Dataset registry for benchmark runners.

Each dataset module exposes the same small interface as ``benchmark.compat200``:
``load_subset``, ``manifest_by_key``, ``load_gt``, and ``RESULTS_DIR``.
"""

from __future__ import annotations

from types import ModuleType

DATASET_CHOICES = ("compat200", "shapenetpart")


def get_dataset(name: str) -> ModuleType:
    """Return a benchmark dataset module by CLI name."""
    normalized = (name or "compat200").strip().lower().replace("-", "_")
    if normalized in {"compat200", "compat_200", "3dcompat200", "3d_compat200"}:
        from benchmark import compat200

        return compat200
    if normalized in {"shapenetpart", "shapenet_part", "shapenet-part"}:
        from benchmark import shapenetpart

        return shapenetpart
    choices = ", ".join(DATASET_CHOICES)
    raise ValueError(f"unknown dataset '{name}' (choices: {choices})")

