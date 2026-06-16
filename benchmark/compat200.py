"""3D CoMPaT-200 coarse dataset discovery, subsets and GT access.

This module is intentionally torch-free: it only scans the filesystem and reads
the numpy ``.npz`` ground-truth produced by :mod:`benchmark.export_gt`.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# --- canonical locations (Windows paths, adjust here if the data moves) ---
BUNDLE_DIR = Path(r"E:\data\d3compat_coarse_semantic_bundle")
COSMO_DATA_ROOT = Path(r"E:\Proj\CoSMo3D\dataset\test_3dcompat200")

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
GT_DIR = DATA_DIR / "gt"
SUBSET_DIR = DATA_DIR / "subsets"
MANIFEST_PATH = DATA_DIR / "manifest.json"
RESULTS_DIR = BENCH_DIR / "results"

NUM_POINTS = 5000
IGNORE_LABEL = -1


@dataclass
class InstanceMeta:
    """One CoMPaT-200 coarse shape."""

    key: str               # safe, filesystem-friendly id (no quotes)
    category: str          # object class, e.g. "air_conditioning_unit"
    instance: str          # raw instance dir name, e.g. "coarse_b'2a_01d'"
    proxy_ply: str         # bundle mesh path (AgentSematic input)
    gt_dir: str            # CoSMo3D instance dir with points.pt/mask2points.pt
    part_names: List[str]  # coarse part names, line order == GT label index


def safe_key(category: str, instance: str) -> str:
    """Filesystem-friendly key (drops quotes/specials from the instance id)."""
    raw = f"{category}__{instance}"
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "" for c in raw)


def _read_part_names(label_path: Path) -> List[str]:
    with open(label_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def discover_instances() -> List[InstanceMeta]:
    """Scan the bundle's proxy meshes and pair each with its CoSMo3D GT dir.

    Only instances that have *both* a proxy mesh and the GT ``mask2points.pt``
    on disk are returned.
    """
    out: List[InstanceMeta] = []
    proxy_dir = BUNDLE_DIR / "mesh_proxy"
    bundle_labels = BUNDLE_DIR / "labels"
    for ply in sorted(proxy_dir.glob("*.ply")):
        stem = ply.stem  # "air_conditioning_unit__coarse_b'2a_01d'"
        if "__" not in stem:
            continue
        category, instance = stem.split("__", 1)
        gt_dir = COSMO_DATA_ROOT / category / instance
        if not (gt_dir / "mask2points.pt").is_file():
            continue
        # prefer GT mask_labels.txt (canonical w.r.t. mask2points order);
        # fall back to bundle labels/*.txt
        label_path = gt_dir / "mask_labels.txt"
        if not label_path.is_file():
            label_path = bundle_labels / f"{stem}.txt"
        if not label_path.is_file():
            continue
        part_names = _read_part_names(label_path)
        if not part_names:
            continue
        out.append(
            InstanceMeta(
                key=safe_key(category, instance),
                category=category,
                instance=instance,
                proxy_ply=str(ply),
                gt_dir=str(gt_dir),
                part_names=part_names,
            )
        )
    return out


def write_manifest(instances: List[InstanceMeta]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"num_instances": len(instances), "instances": [asdict(i) for i in instances]}
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return MANIFEST_PATH


def load_manifest() -> List[InstanceMeta]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found; run `python -m benchmark.export_gt` first."
        )
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [InstanceMeta(**i) for i in data["instances"]]


def manifest_by_key() -> Dict[str, InstanceMeta]:
    return {m.key: m for m in load_manifest()}


# --------------------------------------------------------------------------- #
# Subsets
# --------------------------------------------------------------------------- #
def build_subset(
    name: str,
    instances: List[InstanceMeta],
    *,
    seed: int = 0,
    tiny_categories: int = 15,
    per_category: int = 1,
) -> List[str]:
    """Return a deterministic list of instance keys for a named subset.

    - ``tiny``  : 1 shape from each of the first ``tiny_categories`` categories
    - ``subset``: ``per_category`` shapes from every category
    - ``full``  : every instance
    """
    by_cat: Dict[str, List[InstanceMeta]] = {}
    for m in instances:
        by_cat.setdefault(m.category, []).append(m)
    for cat in by_cat:
        by_cat[cat].sort(key=lambda m: m.instance)

    rng = random.Random(seed)
    cats = sorted(by_cat)

    keys: List[str] = []
    if name == "full":
        for cat in cats:
            keys.extend(m.key for m in by_cat[cat])
    elif name == "tiny":
        for cat in cats[:tiny_categories]:
            keys.append(by_cat[cat][0].key)
    elif name == "subset":
        for cat in cats:
            pool = by_cat[cat]
            take = pool[: min(per_category, len(pool))]
            keys.extend(m.key for m in take)
    else:
        raise ValueError(f"unknown subset '{name}' (use tiny|subset|full)")
    rng.shuffle(keys)
    return keys


def save_subset(name: str, keys: List[str]) -> Path:
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBSET_DIR / f"{name}.txt"
    path.write_text("\n".join(keys) + "\n", encoding="utf-8")
    return path


def load_subset(name: str) -> List[str]:
    path = SUBSET_DIR / f"{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"subset '{name}' not found at {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# Ground truth access (numpy .npz)
# --------------------------------------------------------------------------- #
@dataclass
class GroundTruth:
    key: str
    category: str
    instance: str
    points: np.ndarray      # (N, 3) float32
    labels: np.ndarray      # (N,) int, 0-based part id, -1 = unlabeled
    part_names: List[str]


def gt_path(key: str) -> Path:
    return GT_DIR / f"{key}.npz"


def load_gt(key: str) -> GroundTruth:
    p = gt_path(key)
    if not p.is_file():
        raise FileNotFoundError(f"GT npz missing for '{key}': {p}")
    z = np.load(p, allow_pickle=True)
    return GroundTruth(
        key=key,
        category=str(z["category"]),
        instance=str(z["instance"]),
        points=z["points"].astype(np.float32),
        labels=z["labels"].astype(np.int64),
        part_names=[str(x) for x in z["part_names"].tolist()],
    )
