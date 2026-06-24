"""AgentSematic point-cloud adapter for the CoMPaT-200 coarse benchmark.

Unlike :mod:`benchmark.methods.ours`, this runs the dedicated point-cloud
pipeline directly on the exported GT points and writes per-point predictions.

Run under the **agentsematic** env::

    python -m benchmark.run_method --method ours_pointcloud --subset tiny --provider qwen
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmark import compat200 as C
from benchmark.methods.ours import _combine_face_labels, _humanize, build_client
from benchmark.pointcloud_gt_semantic import describe_gt_part_pointcloud
from pipeline.pointcloud_pipeline import PointCloudRefinementConfig, PointCloudRefinementPipeline

LOGGER = logging.getLogger("benchmark.ours_pointcloud")


def run_instance(
    meta: C.InstanceMeta,
    client,
    work_dir: Path,
    *,
    dataset_module=None,
    prompt_template: str = "{part}",
    max_depth: int = 6,
    max_seeds: int = 4,
    min_block_size: int = 60,
    model: Optional[str] = None,
    select_best_view: bool = True,
    parallel_vlm: int = 1,
) -> Dict:
    """Run point-cloud refinement for one shape and write predictions."""
    D = dataset_module or C
    t0 = time.perf_counter()
    gt = D.load_gt(meta.key)
    points = np.asarray(gt.points, dtype=np.float32)
    part_names = gt.part_names

    part_point_ids: List[List[int]] = []
    per_part_meta: List[Dict] = []
    for p_idx, part in enumerate(part_names):
        out_dir = work_dir / meta.key / f"{p_idx:02d}_{_humanize(part).replace(' ', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        semantic, gt_panel, gt_raw = describe_gt_part_pointcloud(
            client,
            points,
            gt.labels,
            p_idx,
            out_dir,
            object_category=_humanize(meta.category),
            target_part_name=_humanize(part),
            model=model,
        )
        LOGGER.info(
            "[%s] pointcloud part %d/%d gt='%s' -> VLM semantic='%s'",
            meta.key,
            p_idx + 1,
            len(part_names),
            part,
            semantic,
        )
        if not semantic:
            LOGGER.warning(
                "[%s] pointcloud part %d/%d '%s': empty GT-derived semantic; skip",
                meta.key,
                p_idx + 1,
                len(part_names),
                part,
            )
            part_point_ids.append([])
            per_part_meta.append(
                {
                    "part": part,
                    "semantic": "",
                    "gt_semantic_panel": str(gt_panel) if gt_panel else None,
                    "gt_semantic_raw": gt_raw,
                    "num_full_points": 0,
                }
            )
            continue

        try:
            config = PointCloudRefinementConfig(
                semantic=semantic,
                max_depth=max_depth,
                root_seed_count=max_seeds,
                max_seeds=max_seeds,
                min_block_size=min_block_size,
                classification_model=model,
                seed_model=model,
                split_strategy_model=model,
                view_select_model=model,
                select_best_view=select_best_view,
                # Keep the benchmark comparable and cheaper: classification still
                # uses VLM, while split strategy falls back to deterministic geometry.
                ask_split_strategy=False,
                split_strategy="geodesic_fps",
                split_method="geodesic_fps",
                parallel_vlm_workers=max(1, int(parallel_vlm)),
                live_report=False,
            )
            pipeline = PointCloudRefinementPipeline(client, config)
            result = pipeline.run(points, out_dir)
            full_ids = list(result.full_point_ids)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[%s] pointcloud part '%s' failed: %s", meta.key, part, exc)
            full_ids = []

        part_point_ids.append(full_ids)
        per_part_meta.append(
            {
                "part": part,
                "semantic": semantic,
                "semantic_source": "vlm_gt_pointcloud_highlight",
                "gt_semantic_panel": str(gt_panel) if gt_panel else None,
                "gt_semantic_raw": gt_raw,
                "num_full_points": len(full_ids),
            }
        )

    point_pred = _combine_face_labels(len(points), part_point_ids)

    pred_dir = D.RESULTS_DIR / "ours_pointcloud"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{meta.key}.npy", point_pred.astype(np.int64))

    summary = {
        "key": meta.key,
        "category": meta.category,
        "part_names": part_names,
        "n_points": int(points.shape[0]),
        "per_part": per_part_meta,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (pred_dir / f"{meta.key}.meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


__all__ = ["build_client", "run_instance"]
