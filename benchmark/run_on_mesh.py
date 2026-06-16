"""Run AgentSematic ("ours") on a **real mesh** and export CoSMo3D-comparable metrics.

This is the *fair* mesh-native evaluation: your method runs on the mesh's own
faces (no point-cloud reconstruction). The resulting per-face labels are
transferred to the GT points by the **closest point on the actual surface**
(exact, because the GT points were sampled from that surface), then scored with
CoSMo3D's exact mIoU convention. If a CoSMo3D prediction exists for the same
shape, it is scored on the same GT and printed side-by-side.

Example (single shape)::

    conda run -n agentsematic python -m benchmark.run_on_mesh ^
        --mesh "E:\\path\\to\\original_airplane.glb" ^
        --gt-key airplane__coarse_b00_004 ^
        --provider qwen --align bbox

``--gt-key`` refers to the GT exported by ``benchmark.export_gt`` (points +
0-based per-point labels + part names). Use ``--align bbox`` if the mesh is not
already in the GT point frame; the script prints the mean surface-residual so
you can tell whether the frames match.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

from benchmark import compat200 as C
from benchmark import metrics as M
from benchmark.methods.ours import (
    _combine_face_labels,
    _humanize,
    _semantic_for,
    build_client,
)
from pipeline.cli import _load_mesh
from pipeline.pipeline import run_semantic_refinement
from vlm.config import DEFAULT_CONFIG_PATH

LOGGER = logging.getLogger("benchmark.run_on_mesh")


def _align_to_points(vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Center + uniform-scale the mesh so its AABB matches the GT points' AABB."""
    v_min, v_max = vertices.min(0), vertices.max(0)
    p_min, p_max = points.min(0), points.max(0)
    v_ext = (v_max - v_min).max()
    p_ext = (p_max - p_min).max()
    if v_ext <= 1e-12:
        return vertices
    scale = p_ext / v_ext
    v_center = (v_min + v_max) / 2.0
    p_center = (p_min + p_max) / 2.0
    return (vertices - v_center) * scale + p_center


def _faces_to_points_on_surface(
    vertices: np.ndarray, faces: np.ndarray, face_labels: np.ndarray, points: np.ndarray
):
    """For each GT point, take the label of the nearest mesh face (by centroid).

    Fast and dependency-light (cKDTree on face centroids). Returns
    (point_labels, mean_residual_distance); the residual is the mean
    point->nearest-centroid distance and is a coarse misalignment indicator.
    """
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float64)
    centroids = vertices[faces].mean(axis=1)
    tree = cKDTree(centroids)
    dist, idx = tree.query(np.asarray(points, dtype=np.float64), k=1)
    return face_labels[idx], float(np.mean(dist))


def run_on_mesh(
    mesh_path: Path,
    gt: C.GroundTruth,
    client,
    work_dir: Path,
    *,
    prompt_template: str = "{part}",
    max_depth: int = 4,
    max_seeds: int = 4,
    min_block_size: int = 20,
    model: Optional[str] = None,
    parallel_vlm: int = 3,
    align: str = "none",
) -> dict:
    t0 = time.perf_counter()
    vertices, faces = _load_mesh(mesh_path)
    if align == "bbox":
        vertices = _align_to_points(vertices, gt.points)
    part_names = gt.part_names

    part_face_ids: List[List[int]] = []
    per_part_meta: List[dict] = []
    for p_idx, part in enumerate(part_names):
        semantic = _semantic_for(part, gt.category, prompt_template)
        out_dir = work_dir / gt.key / f"{p_idx:02d}_{_humanize(part).replace(' ', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("[%s] part %d/%d '%s' -> semantic='%s'",
                    gt.key, p_idx + 1, len(part_names), part, semantic)
        try:
            res = run_semantic_refinement(
                client, vertices, faces, [0] * len(faces), out_dir,
                semantic=semantic,
                max_depth=max_depth, max_seeds=max_seeds, min_block_size=min_block_size,
                classification_model=model, seed_model=model,
                split_strategy_model=model, view_select_model=model,
                select_best_view=True, live_report=False,
                parallel_vlm_workers=parallel_vlm,
                object_category=_humanize(gt.category),
                candidate_parts=[_humanize(p) for p in part_names],
            )
            full_ids = list(res.full_face_ids)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[%s] part '%s' failed: %s", gt.key, part, exc)
            full_ids = []
        part_face_ids.append(full_ids)
        per_part_meta.append({"part": part, "num_full_faces": len(full_ids)})

    face_labels = _combine_face_labels(len(faces), part_face_ids)
    point_pred, residual = _faces_to_points_on_surface(vertices, faces, face_labels, gt.points)

    pred_dir = C.RESULTS_DIR / "ours_mesh"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{gt.key}.npy", point_pred.astype(np.int64))

    return {
        "key": gt.key, "category": gt.category, "n_faces": int(len(faces)),
        "surface_residual": residual, "per_part": per_part_meta,
        "point_pred": point_pred, "elapsed_seconds": time.perf_counter() - t0,
    }


def _fmt_scores(name: str, s: M.SampleScore) -> str:
    return (f"{name:10} mIoU(skip)={s.miou_skip:.4f}  "
            f"mIoU(empty=1)={s.miou_empty1:.4f}  acc={s.accuracy:.4f}")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Run ours on a real mesh -> CoSMo3D-comparable mIoU")
    ap.add_argument("--mesh", required=True, help="path to the original mesh (.glb/.obj/.ply)")
    ap.add_argument("--gt-key", required=True, help="GT key exported by benchmark.export_gt")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--provider", default="qwen")
    ap.add_argument("--model", default=None)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--align", choices=["none", "bbox"], default="none",
                    help="bbox = center+scale mesh to GT-point AABB before transfer")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--parallel-vlm", type=int, default=3)
    ap.add_argument("--work-dir", default=str(C.RESULTS_DIR / "ours_mesh_work"))
    args = ap.parse_args(argv)

    gt = C.load_gt(args.gt_key)
    client, _ = build_client(args.config, args.provider, args.model, args.env_file)

    print(f"[run] mesh={args.mesh}")
    print(f"[run] gt={args.gt_key} category={gt.category} parts={gt.part_names}")
    out = run_on_mesh(
        Path(args.mesh), gt, client, Path(args.work_dir),
        model=args.model, max_depth=args.max_depth,
        parallel_vlm=args.parallel_vlm, align=args.align,
    )
    print(f"[run] surface residual (mean point->mesh dist) = {out['surface_residual']:.5f}"
          "  (large => mesh/GT frames not aligned; try --align bbox)")

    ours_score = M.sample_scores(out["point_pred"], gt.labels, gt.part_names,
                                 key=gt.key, category=gt.category)

    print("\n=== metrics (CoSMo3D convention) ===")
    print(_fmt_scores("ours(mesh)", ours_score))

    report = {
        "key": gt.key, "category": gt.category, "mesh": str(args.mesh),
        "align": args.align, "surface_residual": out["surface_residual"],
        "provider": args.provider, "model": args.model,
        "ours": {
            "miou_skip": ours_score.miou_skip, "miou_empty1": ours_score.miou_empty1,
            "accuracy": ours_score.accuracy, "part_ious": ours_score.part_ious,
        },
        "per_part_faces": out["per_part"],
        "elapsed_seconds": out["elapsed_seconds"],
    }

    cosmo_pp = C.RESULTS_DIR / "cosmo3d" / f"{gt.key}.npy"
    if cosmo_pp.is_file():
        cosmo_pred = np.load(cosmo_pp)
        if cosmo_pred.shape[0] == gt.labels.shape[0]:
            cosmo_score = M.sample_scores(cosmo_pred, gt.labels, gt.part_names,
                                          key=gt.key, category=gt.category)
            print(_fmt_scores("cosmo3d", cosmo_score))
            print(f"{'delta':10} mIoU(skip)={ours_score.miou_skip - cosmo_score.miou_skip:+.4f}")
            report["cosmo3d"] = {
                "miou_skip": cosmo_score.miou_skip, "miou_empty1": cosmo_score.miou_empty1,
                "accuracy": cosmo_score.accuracy, "part_ious": cosmo_score.part_ious,
            }
    else:
        print(f"[note] no CoSMo3D prediction at {cosmo_pp} (run benchmark.methods.run_cosmo3d_infer)")

    print("\n--- per-part IoU (ours) ---")
    for part, iou in ours_score.part_ious.items():
        print(f"  {part:24} {iou if iou == iou else float('nan'):.4f}")

    out_dir = C.RESULTS_DIR / "mesh_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{gt.key}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] {out_dir / (gt.key + '.json')}")
    print(f"[pred]   {C.RESULTS_DIR / 'ours_mesh' / (gt.key + '.npy')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
