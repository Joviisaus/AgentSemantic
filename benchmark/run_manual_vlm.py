"""Run the point-cloud "ours" pipeline on a single CoMPaT-200 shape with a
**file-based manual VLM** instead of an API model.

Every place the pipeline would call a VLM (GT->semantic, view selection, block
classification observe/reason, seed-count, ...) is turned into a small on-disk
RPC: the request (prompt text + panel image paths) is written to
``<pred>/ours_manual/_manual_vlm/requests/<id>.json`` and the process blocks
until a matching ``responses/<id>.txt`` (the raw JSON reply) appears. A human or
agent acting as the VLM looks at the panel image and writes that reply.

This lets *the agent* personally be the VLM for an apples-to-apples comparison
against cosmo3d / pointclip (whose predictions already exist).

Run under the **agentsematic** env::

    conda run -n agentsematic python -m benchmark.run_manual_vlm --key balance_scale__coarse_b2d_026
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from benchmark import compat200 as C
from benchmark.methods.ours import _combine_face_labels, _humanize
from benchmark.pointcloud_gt_semantic import describe_gt_part_pointcloud
from pipeline.pointcloud_pipeline import (
    PointCloudRefinementConfig,
    PointCloudRefinementPipeline,
)
from vlm.client import VLMResponse

LOGGER = logging.getLogger("benchmark.run_manual_vlm")


def _classify_kind(system: str, user: str, n_images: int) -> str:
    """Best-effort tag so the human/agent server knows what is being asked."""
    s = (system or "") + "\n" + (user or "")
    sl = s.lower()
    if "red_covers" in sl or "search query" in sl or "search phrase" in sl:
        return "gt_semantic"
    if "best_index" in sl or "classify_view_indices" in sl:
        return "view_select"
    if "seed_count" in sl:
        return "seed_count"
    if '"label"' in s and "partial" in sl:
        return "classify_reason"
    if "covered_parts" in sl:
        return "classify_observe"
    if "keep_components" in sl:
        return "component_split"
    if "operations" in sl:
        return "postprocess_ops"
    if "chosen_block_id" in sl:
        return "revive"
    return f"unknown(imgs={n_images})"


class ManualVLMClient:
    """Drop-in replacement for :class:`vlm.client.VLMClient` that serves every
    request from disk so a human/agent can answer it by viewing the panel."""

    def __init__(self, rpc_dir: Path, *, poll_seconds: float = 1.0,
                 timeout_seconds: float = 7200.0):
        self.rpc_dir = Path(rpc_dir)
        self.req_dir = self.rpc_dir / "requests"
        self.resp_dir = self.rpc_dir / "responses"
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.resp_dir.mkdir(parents=True, exist_ok=True)
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self._counter = 0

    def chat(
        self,
        user_prompt: str,
        image_paths: Sequence[str | Path] = (),
        *,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        model: Optional[str] = None,
    ) -> VLMResponse:
        self._counter += 1
        rid = f"{self._counter:05d}"
        imgs = [str(Path(p).resolve()) for p in image_paths]
        kind = _classify_kind(system, user_prompt, len(imgs))

        request = {
            "id": rid,
            "kind": kind,
            "model": model,
            "json_mode": json_mode,
            "system": system,
            "user_prompt": user_prompt,
            "image_paths": imgs,
            "ts": time.time(),
        }
        (self.req_dir / f"{rid}.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # convenience: always-current pending pointer
        (self.rpc_dir / "pending.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        img_str = imgs[0] if imgs else "<none>"
        print(f"MANUAL_VLM_WAIT id={rid} kind={kind} n_imgs={len(imgs)} img={img_str}",
              flush=True)

        resp_path = self.resp_dir / f"{rid}.txt"
        t0 = time.perf_counter()
        while not resp_path.is_file():
            if time.perf_counter() - t0 > self.timeout_seconds:
                raise RuntimeError(f"manual VLM timeout waiting for {resp_path}")
            time.sleep(self.poll_seconds)

        text = resp_path.read_text(encoding="utf-8")
        print(f"MANUAL_VLM_GOT id={rid} kind={kind} chars={len(text)}", flush=True)
        return VLMResponse(
            text=text,
            raw={"manual": True, "id": rid},
            provider="manual",
            model=model or "agent",
            elapsed_seconds=time.perf_counter() - t0,
        )


def run_one(
    key: str,
    *,
    method_name: str = "ours_manual",
    max_depth: int = 3,
    max_seeds: int = 4,
    min_block_size: int = 150,
    poll_seconds: float = 1.0,
) -> Dict[str, Any]:
    gt = C.load_gt(key)
    points = np.asarray(gt.points, dtype=np.float32)
    part_names = gt.part_names

    pred_dir = C.RESULTS_DIR / method_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    work_dir = pred_dir / "_work" / key
    work_dir.mkdir(parents=True, exist_ok=True)

    rpc_dir = pred_dir / "_manual_vlm"
    client = ManualVLMClient(rpc_dir, poll_seconds=poll_seconds)

    print(f"MANUAL_VLM_START key={key} parts={part_names} n_points={len(points)}",
          flush=True)

    part_point_ids: List[List[int]] = []
    per_part_meta: List[Dict[str, Any]] = []

    for p_idx, part in enumerate(part_names):
        out_dir = work_dir / f"{p_idx:02d}_{_humanize(part).replace(' ', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"MANUAL_VLM_PART idx={p_idx} part={part}", flush=True)

        semantic, gt_panel, gt_raw = describe_gt_part_pointcloud(
            client, points, gt.labels, p_idx, out_dir,
            object_category=_humanize(gt.category),
            target_part_name=_humanize(part),
        )
        print(f"MANUAL_VLM_SEMANTIC idx={p_idx} part={part} semantic={semantic!r}",
              flush=True)

        if not semantic:
            part_point_ids.append([])
            per_part_meta.append({"part": part, "semantic": "", "num_full_points": 0})
            continue

        config = PointCloudRefinementConfig(
            semantic=semantic,
            max_depth=max_depth,
            root_seed_count=max_seeds,
            max_seeds=max_seeds,
            min_block_size=min_block_size,
            ask_split_strategy=False,
            split_strategy="geodesic_fps",
            split_method="geodesic_fps",
            select_best_view=True,
            select_classify_views=True,
            parallel_vlm_workers=1,
            postprocess=False,
            revive_when_empty=False,
            live_report=False,
        )
        pipeline = PointCloudRefinementPipeline(client, config)
        result = pipeline.run(points, out_dir)
        full_ids = list(result.full_point_ids)
        print(f"MANUAL_VLM_PART_DONE idx={p_idx} part={part} full_points={len(full_ids)}",
              flush=True)

        part_point_ids.append(full_ids)
        per_part_meta.append({
            "part": part,
            "semantic": semantic,
            "num_full_points": len(full_ids),
        })

    point_pred = _combine_face_labels(len(points), part_point_ids)
    np.save(pred_dir / f"{key}.npy", point_pred.astype(np.int64))
    summary = {
        "key": key,
        "category": gt.category,
        "part_names": part_names,
        "n_points": int(points.shape[0]),
        "per_part": per_part_meta,
    }
    (pred_dir / f"{key}.meta.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"MANUAL_VLM_ALL_DONE key={key} pred={pred_dir / (key + '.npy')}", flush=True)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run ours pipeline with a manual (agent) VLM")
    ap.add_argument("--key", default="balance_scale__coarse_b2d_026")
    ap.add_argument("--method-name", default="ours_manual")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-seeds", type=int, default=4)
    ap.add_argument("--min-block-size", type=int, default=150)
    ap.add_argument("--poll-seconds", type=float, default=1.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("pipeline.pointcloud_pipeline").setLevel(logging.INFO)

    run_one(
        args.key,
        method_name=args.method_name,
        max_depth=args.max_depth,
        max_seeds=args.max_seeds,
        min_block_size=args.min_block_size,
        poll_seconds=args.poll_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
