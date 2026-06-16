"""AgentSematic ("ours") adapter for the CoMPaT-200 coarse benchmark.

For each shape we run the VLM semantic-refinement pipeline once per coarse part
name, collect the per-face "full" masks, resolve overlaps, and transfer the
resulting per-face labels onto the GT point cloud.

Run under the **agentsematic** env::

    conda run -n agentsematic python -m benchmark.run_method --method ours --subset tiny
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from benchmark import compat200 as C
from benchmark.bridge import face_labels_to_point_labels
from pipeline.cli import _load_mesh
from pipeline.pipeline import run_semantic_refinement
from vlm.client import build_vlm_client
from vlm.config import DEFAULT_CONFIG_PATH, get_provider_config, load_vlm_config

LOGGER = logging.getLogger("benchmark.ours")


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip()


def _semantic_for(part: str, category: str, template: str) -> str:
    return template.format(part=_humanize(part), category=_humanize(category))


def _combine_face_labels(
    n_faces: int,
    part_face_ids: List[List[int]],
) -> np.ndarray:
    """Resolve per-part face claims into a single per-face label array.

    Overlaps are resolved by *specificity*: the part claiming the fewest faces
    wins (a small specific part beats a large catch-all). Ties -> smaller index.
    Unclaimed faces stay -1.
    """
    labels = np.full(n_faces, C.IGNORE_LABEL, dtype=np.int64)
    totals = [len(ids) for ids in part_face_ids]
    # process parts from most-specific (fewest faces) to least, last write wins
    order = sorted(range(len(part_face_ids)), key=lambda p: (-totals[p], -p))
    for p in order:
        ids = np.asarray(part_face_ids[p], dtype=np.int64)
        if ids.size:
            labels[ids] = p
    return labels


def build_client(config_path: str, provider: Optional[str], model: Optional[str],
                 env_file: Optional[str]):
    cfg = load_vlm_config(
        config_path, load_env=True,
        env_paths=[env_file] if env_file else None,
    )
    provider_cfg = get_provider_config(cfg, provider, model=model)
    if not provider_cfg.has_key() and not provider_cfg.is_local():
        raise RuntimeError(
            f"provider '{provider_cfg.name}' missing API key; set it in .env"
        )
    LOGGER.info("vlm provider=%s model=%s", provider_cfg.name, provider_cfg.default_model)
    return build_vlm_client(provider_cfg), provider_cfg


def run_instance(
    meta: C.InstanceMeta,
    client,
    work_dir: Path,
    *,
    prompt_template: str = "{part}",
    max_depth: int = 6,
    max_seeds: int = 4,
    min_block_size: int = 20,
    model: Optional[str] = None,
    select_best_view: bool = True,
    parallel_vlm: int = 1,
) -> Dict:
    """Run all parts for one shape; return summary + write predictions npy."""
    t0 = time.perf_counter()
    vertices, faces = _load_mesh(Path(meta.proxy_ply))
    gt = C.load_gt(meta.key)
    part_names = gt.part_names  # canonical order (GT label index)

    part_face_ids: List[List[int]] = []
    per_part_meta: List[Dict] = []
    for p_idx, part in enumerate(part_names):
        semantic = _semantic_for(part, meta.category, prompt_template)
        out_dir = work_dir / meta.key / f"{p_idx:02d}_{_humanize(part).replace(' ', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("[%s] part %d/%d '%s' -> semantic='%s'",
                    meta.key, p_idx + 1, len(part_names), part, semantic)
        try:
            res = run_semantic_refinement(
                client, vertices, faces, [0] * len(faces), out_dir,
                semantic=semantic,
                max_depth=max_depth,
                max_seeds=max_seeds,
                min_block_size=min_block_size,
                classification_model=model,
                seed_model=model,
                split_strategy_model=model,
                view_select_model=model,
                select_best_view=select_best_view,
                live_report=False,
                parallel_vlm_workers=parallel_vlm,
                object_category=_humanize(meta.category),
                candidate_parts=[_humanize(p) for p in part_names],
            )
            full_ids = list(res.full_face_ids)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[%s] part '%s' failed: %s", meta.key, part, exc)
            full_ids = []
        part_face_ids.append(full_ids)
        per_part_meta.append({
            "part": part, "semantic": semantic, "num_full_faces": len(full_ids),
        })

    face_labels = _combine_face_labels(len(faces), part_face_ids)
    point_pred = face_labels_to_point_labels(vertices, faces, face_labels, gt.points)

    pred_dir = C.RESULTS_DIR / "ours"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{meta.key}.npy", point_pred.astype(np.int64))

    summary = {
        "key": meta.key,
        "category": meta.category,
        "part_names": part_names,
        "n_faces": int(len(faces)),
        "n_points": int(gt.points.shape[0]),
        "per_part": per_part_meta,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (pred_dir / f"{meta.key}.meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
