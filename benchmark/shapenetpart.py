"""ShapeNet-Part point-cloud dataset adapter for supplementary benchmark runs.

CoMPaT-200 remains the default benchmark. ShapeNet-Part uses separate
``data_shapenetpart`` and ``results_shapenetpart`` directories.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data_shapenetpart"
GT_DIR = DATA_DIR / "gt"
SUBSET_DIR = DATA_DIR / "subsets"
MANIFEST_PATH = DATA_DIR / "manifest.json"
RESULTS_DIR = BENCH_DIR / "results_shapenetpart"

SHAPENETPART_ROOT = Path(
    os.environ.get("SHAPENETPART_ROOT", r"E:\data\shapenetpart")
)
IGNORE_LABEL = -1


@dataclass
class InstanceMeta:
    """One ShapeNet-Part point-cloud shape."""

    key: str
    category: str
    instance: str
    source_path: str
    part_names: List[str]


@dataclass
class GroundTruth:
    key: str
    category: str
    instance: str
    points: np.ndarray
    labels: np.ndarray
    part_names: List[str]


def safe_key(category: str, instance: str) -> str:
    raw = f"{category}__{instance}"
    return "".join(c if (c.isalnum() or c in ("-", "_", ".")) else "_" for c in raw)


def write_manifest(instances: List[InstanceMeta]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"num_instances": len(instances), "instances": [asdict(i) for i in instances]}
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return MANIFEST_PATH


def load_manifest() -> List[InstanceMeta]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found; run `python -m benchmark.export_shapenetpart_gt` first."
        )
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [InstanceMeta(**item) for item in data["instances"]]


def manifest_by_key() -> Dict[str, InstanceMeta]:
    return {m.key: m for m in load_manifest()}


def build_subset(
    name: str,
    instances: List[InstanceMeta],
    *,
    seed: int = 0,
    tiny_categories: int = 15,
    per_category: int = 1,
) -> List[str]:
    by_cat: Dict[str, List[InstanceMeta]] = {}
    for meta in instances:
        by_cat.setdefault(meta.category, []).append(meta)
    for metas in by_cat.values():
        metas.sort(key=lambda m: m.instance)

    rng = random.Random(seed)
    categories = sorted(by_cat)
    keys: List[str] = []
    if name == "full":
        for cat in categories:
            keys.extend(meta.key for meta in by_cat[cat])
    elif name == "tiny":
        for cat in categories[:tiny_categories]:
            keys.append(by_cat[cat][0].key)
    elif name == "subset":
        for cat in categories:
            keys.extend(meta.key for meta in by_cat[cat][: min(per_category, len(by_cat[cat]))])
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
        raise FileNotFoundError(f"ShapeNet-Part subset '{name}' not found at {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def gt_path(key: str) -> Path:
    return GT_DIR / f"{key}.npz"


def load_gt(key: str) -> GroundTruth:
    path = gt_path(key)
    if not path.is_file():
        raise FileNotFoundError(f"ShapeNet-Part GT npz missing for '{key}': {path}")
    z = np.load(path, allow_pickle=True)
    return GroundTruth(
        key=key,
        category=str(z["category"]),
        instance=str(z["instance"]),
        points=z["points"].astype(np.float32),
        labels=z["labels"].astype(np.int64),
        part_names=[str(x) for x in z["part_names"].tolist()],
    )

