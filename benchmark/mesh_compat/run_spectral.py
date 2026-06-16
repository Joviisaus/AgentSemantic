"""Stage 2b (run in the **MeshReduction** conda env).

Run the referenced LLM_Spectral_Search method (``evaluation/spectral_predictor.py``
-> ``predict_one_shape``) on the **same** cached meshes + GT that the
AgentSematic predictor consumed, and emit unified ``PredRecord`` JSONL.

Because both methods read the identical ``<cache>/<sid>.npz`` (same mesh, same
GT points, same ``target_face_idx``), the resulting metrics are a strictly fair
head-to-head.

Example::

    D:\\...\\envs\\MeshReduction\\python.exe ^
        E:\\Proj\\AgentSematic\\benchmark\\mesh_compat\\run_spectral.py ^
        --cache E:\\Proj\\AgentSematic\\benchmark\\results\\mesh_cache ^
        --out   E:\\Proj\\AgentSematic\\benchmark\\results\\preds_spectral_3dcompat.jsonl ^
        --provider qwen --num-modes 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

MRA_ROOT = Path(r"E:\Proj\MeshReductionAgent")
for p in (MRA_ROOT, MRA_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluation.seg_metrics import PredRecord, record_to_json  # noqa: E402
from evaluation.spectral_predictor import predict_one_shape  # noqa: E402
from evaluation.vlm_errors import VLMUnauthorizedError, TooManyComponentsError  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spectral baseline on shared cache -> JSONL")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--provider", default="qwen")
    ap.add_argument("--num-modes", type=int, default=5)
    ap.add_argument("--score-normalization", default="per_query_max",
                    choices=["per_query_max", "per_query_p95", "none"])
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--max-components-per-shape", type=int, default=50)
    ap.add_argument("--shape-ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import trimesh

    index = json.loads((args.cache / "index.json").read_text(encoding="utf-8"))
    sids = list(index.keys())
    if args.shape_ids.strip():
        want = {s.strip() for s in args.shape_ids.split(",") if s.strip()}
        sids = [s for s in sids if s in want]
    if args.limit > 0:
        sids = sids[: args.limit]
    if not sids:
        print("[spectral] no shapes to run", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[spectral] {len(sids)} shapes -> {args.out}", flush=True)
    n_done = 0
    aborted_401 = False
    with open(args.out, "w", encoding="utf-8") as fout:
        for sid in sids:
            meta = index[sid]
            print(f"\n=== shape {sid} ({meta['category_name']}) ===", flush=True)
            npz = np.load(args.cache / f"{sid}.npz")
            vertices = np.asarray(npz["vertices"], dtype=np.float64)
            faces = np.asarray(npz["faces"], dtype=np.int64)
            gt_xyz = np.asarray(npz["gt_xyz"], dtype=np.float32)
            gt_part_ids = np.asarray(npz["gt_part_ids"], dtype=np.int64)
            target_face_idx = np.asarray(npz["target_face_idx"], dtype=np.int64)

            part_names: List[str] = meta["part_names"]
            part_class_ids: List[int] = [int(x) for x in meta["part_class_ids"]]
            part_name_to_class_id: Dict[str, int] = dict(zip(part_names, part_class_ids))

            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            try:
                pred = predict_one_shape(
                    mesh=mesh,
                    shape_class_name=meta["category_name"],
                    part_name_to_class_id=part_name_to_class_id,
                    target_points=gt_xyz.astype(np.float64),
                    target_face_idx=target_face_idx,
                    score_normalization=args.score_normalization,
                    score_threshold=float(args.score_threshold),
                    provider_name=args.provider,
                    num_modes=args.num_modes,
                    work_dir=args.cache / "_spectral_work" / sid,
                    max_components_per_shape=int(args.max_components_per_shape),
                    shape_id_for_logs=sid,
                )
            except TooManyComponentsError as e:
                print(f"  [skip] {e}", flush=True)
                continue
            except VLMUnauthorizedError as e:
                print(f"\n[ABORT] VLM 401 while processing {sid}: {e}", flush=True)
                aborted_401 = True
                break
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR predict: {e}", flush=True)
                traceback.print_exc()
                continue

            rec = PredRecord(
                shape_id=sid,
                category_id=int(meta["category_id"]),
                category_name=meta["category_name"],
                gt_part_ids=gt_part_ids,
                pred_part_ids=pred.pred_pts_global,
                gt_xyz=gt_xyz,
                per_query_scores=pred.per_query_scores,
                inference_seconds=pred.inference_seconds,
                vlm_calls=pred.vlm_calls,
                extra={"method": "spectral", "tool_meta": pred.tool_meta},
            )
            fout.write(json.dumps(record_to_json(rec)) + "\n")
            fout.flush()
            n_done += 1
            print(f"  done: t={pred.inference_seconds:.1f}s vlm~{pred.vlm_calls} "
                  f"(n_done={n_done})", flush=True)

    print(f"\n[spectral] wrote {n_done}/{len(sids)} records -> {args.out}")
    if aborted_401:
        print("[exit=2] aborted on VLM 401", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
