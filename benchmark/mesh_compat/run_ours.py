"""Stage 2a (run in the **agentsematic** conda env).

Run the AgentSematic semantic-refinement pipeline (``pipeline/cli.py`` ->
``run_semantic_refinement``) natively on each cached mesh, and emit unified
``PredRecord`` JSONL that ``scripts/eval_unified.py`` scores with the same
metrics as every other method.

Per shape, per coarse part name we run the pipeline once on the mesh's own
faces, collect ``full_face_ids``, resolve overlaps by specificity (smallest
part wins), map faces -> global coarse class ids, and project to the GT points
via the cached ``target_face_idx`` (point->triangle). Faces no part claims get
a sentinel class outside the vocabulary, so they don't pollute IoU (identical
convention to ``evaluation/spectral_predictor.py``).

Example::

    D:\\...\\envs\\agentsematic\\python.exe ^
        E:\\Proj\\AgentSematic\\benchmark\\mesh_compat\\run_ours.py ^
        --cache E:\\Proj\\AgentSematic\\benchmark\\results\\mesh_cache ^
        --out   E:\\Proj\\AgentSematic\\benchmark\\results\\preds_ours_agent_3dcompat.jsonl ^
        --provider qwen --max-depth 4 --parallel-vlm 3
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

AGENTSEMATIC_ROOT = Path(__file__).resolve().parents[2]
if str(AGENTSEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTSEMATIC_ROOT))

# MeshReductionAgent provides the *canonical* per-shape scorer so the live
# numbers here match the authoritative Stage-3 eval_unified report.
MRA_ROOT = Path(r"E:\Proj\MeshReductionAgent")
if str(MRA_ROOT) not in sys.path:
    sys.path.insert(0, str(MRA_ROOT))

from benchmark.methods.ours import (  # noqa: E402
    _combine_face_labels,
    _humanize,
    build_client,
)
from benchmark.mesh_compat.gt_semantic import (  # noqa: E402
    describe_gt_part,
    face_gt_labels,
    part_highlight_face_ids,
)
from pipeline.pipeline import run_semantic_refinement  # noqa: E402
from rendering.render_component import (  # noqa: E402
    mesh_face_connected_components,
    render_full_coverage_panel,
    render_initial_segmentation_panel,
    render_semantic_mask_panel,
)
from vlm.config import DEFAULT_CONFIG_PATH  # noqa: E402

try:
    from evaluation.seg_metrics import _shape_part_iou, _shape_boundary_iou
    _HAVE_METRICS = True
except Exception:  # noqa: BLE001
    _HAVE_METRICS = False

LOGGER = logging.getLogger("benchmark.mesh_compat.ours")

SENTINEL_OFFSET = 10000


def _print_visual_summary(sid: str, work_dir: Path, per_part: List[dict]) -> None:
    """After each shape, list GT/ours renders and VLM semantics for quick inspection."""
    shape_dir = work_dir / sid
    print(f"\n--- visual outputs: {sid} ---", flush=True)
    for name, path in (
        ("整模连通块", shape_dir / "mesh_all_components.png"),
        ("GT 全部件", shape_dir / "shape_gt_all_parts.png"),
        ("Ours 全部件", shape_dir / "shape_ours_all_parts.png"),
    ):
        print(f"  [{name}] {path}" + ("  [ok]" if path.is_file() else "  [missing]"),
              flush=True)
    for q_i, part in enumerate(per_part):
        cid = part.get("class_id", "?")
        out_dir = work_dir / sid / f"{q_i:02d}_cls{cid}"
        gt_png = out_dir / "gt_highlight.png"
        ours_png = out_dir / "ours_highlight.png"
        gt_name = part.get("gt_name", "?")
        semantic = part.get("vlm_semantic", "") or "(empty)"
        n_full = part.get("num_full_faces", 0)
        n_islands = part.get("n_gt_islands", "?")
        print(
            f"  [{q_i:02d}_cls{cid}] gt_name={gt_name!r}  "
            f"VLM semantic={semantic!r}  gt_islands={n_islands}  ours_faces={n_full}",
            flush=True,
        )
        print(f"    GT   : {gt_png}" + ("  [ok]" if gt_png.is_file() else "  [missing]"),
              flush=True)
        print(f"    Ours : {ours_png}" + ("  [ok]" if ours_png.is_file() else "  [missing]"),
              flush=True)
    print(flush=True)


def _render_shape_summaries(
    sid: str,
    work_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_gt: np.ndarray,
    part_names: List[str],
    part_class_ids: List[int],
    part_face_ids: List[List[int]],
) -> None:
    """Per-shape panels: all mesh CCs + GT all-parts + ours all-parts."""
    shape_dir = work_dir / sid
    shape_dir.mkdir(parents=True, exist_ok=True)
    n_faces = int(len(faces))
    cc = mesh_face_connected_components(vertices, faces)
    n_cc = int(np.unique(cc).size)

    try:
        render_initial_segmentation_panel(
            vertices, faces, cc.tolist(),
            shape_dir / "mesh_all_components.png",
            title=f"整模连通块 n={n_cc}（每块一色，含全部非连通岛）",
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[%s] mesh_all_components render failed: %s", sid, exc)

    gt_idx = np.full(n_faces, -1, dtype=np.int64)
    for q_i, cid in enumerate(part_class_ids):
        gt_idx[face_gt == int(cid)] = q_i
    try:
        render_full_coverage_panel(
            vertices, faces, gt_idx.tolist(), part_names,
            shape_dir / "shape_gt_all_parts.png",
            title=f"GT 全部件（整模 {n_cc} 连通块均可见）",
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[%s] shape_gt_all_parts render failed: %s", sid, exc)

    ours_idx = np.full(n_faces, -1, dtype=np.int64)
    for q_i, ids in enumerate(part_face_ids):
        if ids:
            ours_idx[np.asarray(ids, dtype=np.int64)] = q_i
    try:
        render_full_coverage_panel(
            vertices, faces, ours_idx.tolist(), part_names,
            shape_dir / "shape_ours_all_parts.png",
            title=f"Ours 全部件（整模 {n_cc} 连通块均可见）",
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[%s] shape_ours_all_parts render failed: %s", sid, exc)


def _live_metrics(rec: dict, part_class_ids) -> dict:
    """Per-shape insmIoU (shapenet policy) / OA / boundary, matching eval_unified."""
    pred = np.asarray(rec["pred_part_ids"], dtype=np.int64)
    gt = np.asarray(rec["gt_part_ids"], dtype=np.int64)
    oa = float((pred == gt).mean()) if pred.size else float("nan")
    if _HAVE_METRICS and part_class_ids:
        miou, _ = _shape_part_iou(pred, gt, part_class_ids, empty_union_policy="shapenet")
        try:
            b = _shape_boundary_iou(pred, gt, np.asarray(rec["gt_xyz"], dtype=np.float32),
                                    part_class_ids)
        except Exception:  # noqa: BLE001
            b = float("nan")
    else:
        miou, b = float("nan"), float("nan")
    return {"insmIoU": miou, "OA": oa, "boundary_insmIoU": b}


def _record_to_json(shape_id, category_id, category_name, gt_part_ids,
                    pred_part_ids, gt_xyz, per_query_scores,
                    inference_seconds, vlm_calls, extra) -> dict:
    """Match MeshReductionAgent ``seg_metrics.record_to_json`` schema exactly."""
    return {
        "shape_id": shape_id,
        "category_id": int(category_id),
        "category_name": category_name,
        "gt_part_ids": np.asarray(gt_part_ids, dtype=np.int64).tolist(),
        "pred_part_ids": np.asarray(pred_part_ids, dtype=np.int64).tolist(),
        "gt_xyz": np.asarray(gt_xyz, dtype=float).tolist(),
        "per_query_scores": (
            {int(k): np.asarray(v, dtype=float).tolist()
             for k, v in per_query_scores.items()}
            if per_query_scores else None
        ),
        "inference_seconds": inference_seconds,
        "vlm_calls": vlm_calls,
        "extra": extra,
    }


def run_one_shape(sid, meta, npz, client, work_dir, *, prompt_template,
                  max_depth, max_seeds, min_block_size, model, parallel_vlm) -> dict:
    vertices = np.asarray(npz["vertices"], dtype=np.float64)
    faces = np.asarray(npz["faces"], dtype=np.int64)
    gt_xyz = np.asarray(npz["gt_xyz"], dtype=np.float32)
    gt_part_ids = np.asarray(npz["gt_part_ids"], dtype=np.int64)
    target_face_idx = np.asarray(npz["target_face_idx"], dtype=np.int64)

    part_names: List[str] = meta["part_names"]
    part_class_ids: List[int] = [int(x) for x in meta["part_class_ids"]]
    category = meta["category_name"]
    n_faces = int(len(faces))
    K = len(part_names)

    # Dense per-face GT label (each face -> nearest GT point's class), used to
    # paint the GT region for the VLM's semantic-from-visual step.
    face_gt = face_gt_labels(
        vertices, faces, gt_xyz, gt_part_ids, target_face_idx=target_face_idx,
    )
    n_mesh_cc = int(np.unique(mesh_face_connected_components(vertices, faces)).size)

    t0 = time.perf_counter()
    part_face_ids: List[List[int]] = []
    per_part_meta: List[dict] = []
    for q_i, part in enumerate(part_names):
        cid = part_class_ids[q_i]
        out_dir = work_dir / sid / f"{q_i:02d}_cls{cid}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- semantic comes from the GT *visual*, NOT the dataset name ---
        gt_face_ids, n_islands, _ = part_highlight_face_ids(
            vertices, faces, face_gt, cid,
        )
        semantic, _panel, _raw = describe_gt_part(
            client, vertices, faces, gt_face_ids, out_dir,
            object_category=_humanize(category), model=model,
            n_islands=n_islands, n_mesh_components=n_mesh_cc,
        )
        if not semantic:
            LOGGER.warning("[%s] part %d/%d (cls=%d): empty VLM semantic; skip",
                           sid, q_i + 1, K, cid)
            part_face_ids.append([])
            per_part_meta.append({"class_id": cid, "gt_name": part,
                                  "vlm_semantic": "", "num_full_faces": 0,
                                  "n_gt_islands": n_islands})
            continue
        LOGGER.info("[%s] part %d/%d cls=%d gt_name='%s' -> VLM semantic='%s' "
                    "(gt_faces=%d islands=%d mesh_cc=%d)", sid, q_i + 1, K, cid,
                    part, semantic, int(gt_face_ids.size), n_islands, n_mesh_cc)
        try:
            res = run_semantic_refinement(
                client, vertices, faces, [0] * n_faces, out_dir,
                semantic=semantic, max_depth=max_depth, max_seeds=max_seeds,
                min_block_size=min_block_size,
                classification_model=model, seed_model=model,
                split_strategy_model=model, view_select_model=model,
                select_best_view=True, live_report=False,
                parallel_vlm_workers=parallel_vlm,
                object_category=_humanize(category),
                candidate_parts=None,  # do NOT leak the GT part vocabulary
            )
            full_ids = list(res.full_face_ids)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[%s] part '%s' failed: %s", sid, part, exc)
            full_ids = []
        part_face_ids.append(full_ids)
        ours_png = out_dir / "ours_highlight.png"
        mask_png = out_dir / "semantic_mask.png"
        if mask_png.is_file():
            shutil.copy2(mask_png, ours_png)
        elif full_ids:
            try:
                render_semantic_mask_panel(
                    vertices, faces, full_ids, ours_png,
                    title=(
                        f"Ours 语义={semantic!r}  full={len(full_ids)}  "
                        f"整模连通块={n_mesh_cc}"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("[%s] ours_highlight render failed: %s", sid, exc)
        per_part_meta.append({"class_id": cid, "gt_name": part,
                              "vlm_semantic": semantic,
                              "num_full_faces": len(full_ids),
                              "n_gt_islands": n_islands})

    # Resolve overlaps -> per-face local part index (-1 = unclaimed).
    face_local = _combine_face_labels(n_faces, part_face_ids)  # -1 or 0..K-1
    sentinel = (max(part_class_ids) + SENTINEL_OFFSET) if part_class_ids else SENTINEL_OFFSET
    class_ids = np.asarray(part_class_ids, dtype=np.int64)

    face_pred_global = np.full(n_faces, sentinel, dtype=np.int64)
    claimed = face_local >= 0
    face_pred_global[claimed] = class_ids[face_local[claimed]]

    # Binary per-face per-query score (1.0 where that query claimed the face).
    face_score = np.zeros((n_faces, K), dtype=np.float32)
    for q_i in range(K):
        ids = np.asarray(part_face_ids[q_i], dtype=np.int64)
        if ids.size:
            face_score[ids, q_i] = 1.0

    # Project to GT points via point->triangle face index.
    tfi = np.clip(target_face_idx, 0, n_faces - 1)
    pred_pts_global = face_pred_global[tfi].astype(np.int64)
    per_query_scores = {
        int(class_ids[q_i]): face_score[tfi, q_i].astype(np.float32)
        for q_i in range(K)
    }

    elapsed = time.perf_counter() - t0
    n_claimed_faces = int(claimed.sum())
    LOGGER.info("[%s] done t=%.1fs claimed_faces=%d/%d", sid, elapsed,
                n_claimed_faces, n_faces)
    _render_shape_summaries(
        sid, work_dir, vertices, faces, face_gt,
        part_names, part_class_ids, part_face_ids,
    )
    return _record_to_json(
        sid, meta["category_id"], category, gt_part_ids, pred_pts_global, gt_xyz,
        per_query_scores, float(elapsed), None,
        {"method": "ours_agent", "n_faces": n_faces,
         "n_claimed_faces": n_claimed_faces, "sentinel_class": int(sentinel),
         "per_part": per_part_meta, "uses_target_face_idx": True},
    )


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="AgentSematic pipeline -> unified JSONL")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--provider", default="qwen")
    ap.add_argument("--model", default=None)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--prompt-template", default="{part}")
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--max-seeds", type=int, default=4)
    ap.add_argument("--min-block-size", type=int, default=20)
    ap.add_argument("--parallel-vlm", type=int, default=3)
    ap.add_argument("--shape-ids", default="", help="comma list to restrict")
    ap.add_argument("--limit", type=int, default=0, help="process first N shapes")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args(argv)

    index = json.loads((args.cache / "index.json").read_text(encoding="utf-8"))
    sids = list(index.keys())
    if args.shape_ids.strip():
        want = {s.strip() for s in args.shape_ids.split(",") if s.strip()}
        sids = [s for s in sids if s in want]
    if args.limit > 0:
        sids = sids[: args.limit]
    if not sids:
        print("[ours] no shapes to run", file=sys.stderr)
        return 1

    work_dir = Path(args.work_dir) if args.work_dir else (args.cache / "_ours_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    client, _ = build_client(args.config, args.provider, args.model, args.env_file)

    print(f"[ours] {len(sids)} shapes (fewest-components-first) -> {args.out}",
          flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0
    run_miou: List[float] = []
    run_oa: List[float] = []
    with open(args.out, "w", encoding="utf-8") as fout:
        for pos, sid in enumerate(sids, 1):
            meta = index[sid]
            print(f"\n=== [{pos}/{len(sids)}] shape {sid} ({meta['category_name']}, "
                  f"components={meta.get('n_components', '?')}) ===", flush=True)
            npz = np.load(args.cache / f"{sid}.npz")
            try:
                rec = run_one_shape(
                    sid, meta, npz, client, work_dir,
                    prompt_template=args.prompt_template,
                    max_depth=args.max_depth, max_seeds=args.max_seeds,
                    min_block_size=args.min_block_size, model=args.model,
                    parallel_vlm=args.parallel_vlm,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("[%s] shape failed: %s", sid, exc)
                continue
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            n_done += 1

            m = _live_metrics(rec, [int(x) for x in meta["part_class_ids"]])
            if m["insmIoU"] == m["insmIoU"]:
                run_miou.append(m["insmIoU"])
            if m["OA"] == m["OA"]:
                run_oa.append(m["OA"])
            run_ins = float(np.mean(run_miou)) if run_miou else float("nan")
            run_oacc = float(np.mean(run_oa)) if run_oa else float("nan")
            print(
                f"  >>> [ours live] {sid}: insmIoU={m['insmIoU']:.4f} "
                f"OA={m['OA']:.4f} bIoU={m['boundary_insmIoU']:.4f}  ||  "
                f"running(n={n_done}) insmIoU={run_ins:.4f} OA={run_oacc:.4f}",
                flush=True,
            )
            per_part = (rec.get("extra") or {}).get("per_part") or []
            if per_part:
                _print_visual_summary(sid, work_dir, per_part)

    print(f"\n[ours] wrote {n_done}/{len(sids)} records -> {args.out}")
    if run_miou:
        print(f"[ours] FINAL running insmIoU={float(np.mean(run_miou)):.4f} "
              f"OA={float(np.mean(run_oa)):.4f} over {n_done} shapes "
              f"(authoritative table from eval_unified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
