"""Unified per-point mIoU metrics for 3D CoMPaT-200 coarse.

Matches CoSMo3D's evaluation semantics
(``model/evaluation/core.py:compute_3d_iou_upsample``):

    per-sample mIoU = mean over the sample's parts of I/U,
                      parts whose union is 0 are *skipped* (not counted as 1.0).

We also expose the ShapeNet/PointCLIP convention (empty union -> IoU 1.0) for
reference, and a plain point accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

IGNORE_LABEL = -1


@dataclass
class SampleScore:
    key: str
    category: str
    miou_skip: float          # CoSMo3D convention (skip empty-union parts)
    miou_empty1: float        # ShapeNet convention (empty union -> 1.0)
    accuracy: float
    num_parts: int
    part_ious: Dict[str, float] = field(default_factory=dict)


def sample_scores(
    pred: np.ndarray,
    gt: np.ndarray,
    part_names: Sequence[str],
    *,
    key: str = "",
    category: str = "",
) -> SampleScore:
    """Score one shape. ``pred``/``gt`` are 0-based part ids (-1 = unlabeled)."""
    pred = np.asarray(pred).astype(np.int64)
    gt = np.asarray(gt).astype(np.int64)
    num_parts = len(part_names)

    # accuracy over labeled GT points
    labeled = gt != IGNORE_LABEL
    acc = float(np.mean(pred[labeled] == gt[labeled])) if labeled.any() else 0.0

    ious_skip: List[float] = []
    ious_e1: List[float] = []
    part_ious: Dict[str, float] = {}
    for p in range(num_parts):
        inter = int(np.sum((pred == p) & (gt == p)))
        union = int(np.sum((pred == p) | (gt == p)))
        if union == 0:
            ious_e1.append(1.0)
            part_ious[part_names[p]] = float("nan")
            continue
        iou = inter / float(union)
        ious_skip.append(iou)
        ious_e1.append(iou)
        part_ious[part_names[p]] = iou

    miou_skip = float(np.mean(ious_skip)) if ious_skip else 0.0
    miou_e1 = float(np.mean(ious_e1)) if ious_e1 else 0.0
    return SampleScore(
        key=key,
        category=category,
        miou_skip=miou_skip,
        miou_empty1=miou_e1,
        accuracy=acc,
        num_parts=num_parts,
        part_ious=part_ious,
    )


@dataclass
class AggregateScore:
    num_samples: int
    instance_miou_skip: float       # mean of per-sample miou (micro over shapes)
    instance_miou_empty1: float
    category_miou_skip: float       # macro: mean over categories (CoSMo3D headline)
    instance_accuracy: float
    per_category: Dict[str, float] = field(default_factory=dict)


def aggregate(samples: Sequence[SampleScore]) -> AggregateScore:
    if not samples:
        return AggregateScore(0, 0.0, 0.0, 0.0, 0.0, {})
    inst_skip = float(np.mean([s.miou_skip for s in samples]))
    inst_e1 = float(np.mean([s.miou_empty1 for s in samples]))
    inst_acc = float(np.mean([s.accuracy for s in samples]))

    by_cat: Dict[str, List[float]] = {}
    for s in samples:
        by_cat.setdefault(s.category, []).append(s.miou_skip)
    per_cat = {c: float(np.mean(v)) for c, v in by_cat.items()}
    cat_macro = float(np.mean(list(per_cat.values()))) if per_cat else 0.0

    return AggregateScore(
        num_samples=len(samples),
        instance_miou_skip=inst_skip,
        instance_miou_empty1=inst_e1,
        category_miou_skip=cat_macro,
        instance_accuracy=inst_acc,
        per_category=per_cat,
    )
