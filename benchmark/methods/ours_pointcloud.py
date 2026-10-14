"""AgentSematic point-cloud adapter for the CoMPaT-200 coarse benchmark.

Unlike :mod:`benchmark.methods.ours`, this runs the dedicated point-cloud
pipeline directly on the exported GT points and writes per-point predictions.

稀疏 benchmark 点云（~5k）默认启用 **mesh 辅助增强 + NEO + spectral** 分块：
  - proxy mesh 密集采样辅助 NEO LBO 特征空间；
  - spectral 子块分裂切片复用全局特征（缺 NEO 时回退 scipy）。

Run under the **agentsematic** env::

    python -m benchmark.run_method --method ours_pointcloud --subset tiny --provider qwen
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmark import compat200 as C
from benchmark.methods.ours import (
    _combine_face_labels,
    _humanize,
    build_client as ours_build_client,
)
from benchmark.pointcloud_gt_semantic import describe_gt_part_pointcloud
from geometry.sparse_point_enhance import sparse_partition_params
from pipeline.pointcloud_pipeline import PointCloudRefinementConfig, PointCloudRefinementPipeline

LOGGER = logging.getLogger("benchmark.ours_pointcloud")


def build_client(config_path: str, provider: Optional[str], model: Optional[str],
                 env_file: Optional[str]):
    """原样转发到 ours.build_client，除非开启 AGENTSEMANTIC_VLM_ORACLE：

    开启时返回一个「人工 oracle」客户端（AssistantVLMClient），把所有需要 token 的 VLM
    调用改由助手亲自决策（收集请求 → 读关键渲染图写答案 → 重放）。无需任何 API key。
    """
    if os.environ.get("AGENTSEMANTIC_VLM_ORACLE"):
        from benchmark.vlm_oracle import build_oracle_client
        from types import SimpleNamespace

        LOGGER.info("VLM oracle enabled: all VLM calls routed to AssistantVLMClient (no token)")
        return build_oracle_client(), SimpleNamespace(name="oracle", default_model="assistant")
    return ours_build_client(config_path, provider, model, env_file)



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
    use_neo: bool = True,
    neo_ckpt: Optional[str] = None,
    neo_python: Optional[str] = None,
    neo_k: int = 0,
    neo_device: str = "cpu",
    neo_no_mass: bool = False,
    neo_densify_target: int = 20000,
    neo_proxy_mesh: Optional[str] = None,
) -> Dict:
    """Run point-cloud refinement for one shape and write predictions."""
    D = dataset_module or C
    t0 = time.perf_counter()
    gt = D.load_gt(meta.key)
    points = np.asarray(gt.points, dtype=np.float32)
    part_names = gt.part_names
    sparse_params = sparse_partition_params(len(points))
    proxy_mesh = neo_proxy_mesh
    if proxy_mesh is None and hasattr(meta, "proxy_ply"):
        proxy_mesh = str(getattr(meta, "proxy_ply", "") or "")
    if proxy_mesh and not Path(proxy_mesh).is_file():
        proxy_mesh = None

    # 公平模式（AGENTSEMANTIC_VLM_FAIR=1）：复用 fast 的「跳过 GT 语义/视角 VLM」逻辑
    # —— 即 GT 只提供部件名文本（不渲染/不使用 GT 高亮图），块面板照常渲染供助手肉眼判。
    # 分类（observe/reason）交由 vlm_oracle 的 FAIR 分支处理（几何驱动分裂 + 助手视觉判定）。
    fair = bool(os.environ.get("AGENTSEMANTIC_VLM_FAIR"))
    fast = bool(os.environ.get("AGENTSEMANTIC_ORACLE_FAST")) or fair
    part_point_ids: List[List[int]] = []
    per_part_meta: List[Dict] = []
    for p_idx, part in enumerate(part_names):
        out_dir = work_dir / meta.key / f"{p_idx:02d}_{_humanize(part).replace(' ', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if fast:
            # Geometric-subdivision / fair test: skip the GT-semantic VLM call entirely.
            # semantic comes ONLY from the GT part name (text) — no GT highlight image is
            # rendered or consulted. The block panels are still rendered for visual classification.
            semantic = _humanize(part)
            gt_panel = None
            gt_raw = None
            LOGGER.info("[%s] FAST/FAIR: skip GT-semantic VLM; semantic='%s' (GT part-name only)", meta.key, semantic)
        else:
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
                select_best_view=select_best_view and not fast,
                # benchmark：分类仍走 VLM；分块固定 spectral + NEO（稀疏 mesh 增强）。
                ask_split_strategy=False,
                split_strategy="spectral",
                split_method="lbo_spectral",
                split_n_neighbors=sparse_params["split_n_neighbors"],
                use_neo=use_neo,
                neo_ckpt=neo_ckpt,
                neo_python=neo_python,
                neo_k=neo_k,
                neo_device=neo_device,
                neo_no_mass=neo_no_mass,
                neo_proxy_mesh=proxy_mesh,
                neo_densify_target=neo_densify_target,
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
                "semantic_source": "gt_part_name_text" if fair else "vlm_gt_pointcloud_highlight",
                "gt_semantic_panel": str(gt_panel) if gt_panel else None,
                "gt_semantic_raw": gt_raw,
                "num_full_points": len(full_ids),
            }
        )

    point_pred = _combine_face_labels(len(points), part_point_ids)

    # oracle 重跑结果落到独立子目录，避免覆盖上一轮 token 版 ours_pointcloud 预测
    subdir = os.environ.get("AGENTSEMANTIC_RESULTS_SUBDIR") or "ours_pointcloud"
    pred_dir = D.RESULTS_DIR / subdir
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{meta.key}.npy", point_pred.astype(np.int64))

    summary = {
        "key": meta.key,
        "category": meta.category,
        "part_names": part_names,
        "n_points": int(points.shape[0]),
        "use_neo": use_neo,
        "neo_proxy_mesh": proxy_mesh,
        "split_n_neighbors": sparse_params["split_n_neighbors"],
        "per_part": per_part_meta,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (pred_dir / f"{meta.key}.meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


__all__ = ["build_client", "run_instance"]
