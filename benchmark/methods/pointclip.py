"""PointCLIP V2 adapter for the ShapeNet-Part benchmark.

This module wires the official PointCLIP V2 zero-shot part-segmentation code
under ``third_party/PointCLIP_V2/zeroshot_seg`` into the local benchmark schema.
It runs directly on the exported ShapeNet-Part GT points and writes
``benchmark/results_shapenetpart/pointclip/<key>.npy`` as category-local per-point
part ids aligned with the evaluator.

Run under the ``PointCLIPV2`` conda env, for example::

    conda run -n PointCLIPV2 python -m benchmark.run_method \
        --dataset shapenetpart --method pointclip --subset tiny --overwrite
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("benchmark.pointclip")

BENCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_DIR.parent
DEFAULT_PCV2_SEG_ROOT = REPO_ROOT / "third_party" / "PointCLIP_V2" / "zeroshot_seg"
DEFAULT_MODEL_NAME = "ViT-B/16"
PC_NUM = 2048


@dataclass
class _PointCLIPRuntime:
    """Imported PointCLIP V2 modules and loaded CLIP model."""

    model_name: str
    device: str
    torch: object
    clip_module: object
    best_prompt: Dict[str, List[str]]
    best_vweight: Dict[str, List[float]]
    cat2id: Dict[str, int]
    feat_dim: int
    extractor: object
    clip_model: object


def _dataset_name(dataset_module) -> str:
    return getattr(dataset_module, "__name__", "").rsplit(".", 1)[-1].lower()


def _is_shapenetpart(dataset_module) -> bool:
    return _dataset_name(dataset_module) == "shapenetpart"


def _resolve_pcv2_root() -> Path:
    root = Path(os.environ.get("POINTCLIPV2_SEG_ROOT", str(DEFAULT_PCV2_SEG_ROOT)))
    if not root.is_dir():
        raise FileNotFoundError(
            "PointCLIP V2 zeroshot_seg root not found. Expected either "
            f"{DEFAULT_PCV2_SEG_ROOT} or POINTCLIPV2_SEG_ROOT."
        )
    return root.resolve()


def _setup_import_path(pcv2_root: Path) -> None:
    root_s = str(pcv2_root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def _build_extractor(torch, clip_model, pcv2_root: Path, device: str):
    """Create a lightweight copy of PointCLIP V2's Extractor.

    The official ``main.py`` always writes intermediate feature-map caches for a
    whole category. For benchmark subsets we only need selected shapes, so this
    class reuses the official Realistic_Projection implementation in-process.
    """

    _setup_import_path(pcv2_root)
    from realistic_projection import Realistic_Projection  # type: ignore

    class Extractor(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.pc_views = Realistic_Projection(gpu=device)
            self.get_img = self.pc_views.get_img

        def mv_proj(self, pc):
            img, is_seen, point_loc_in_img = self.get_img(pc)
            img = img[:, :, 20:204, 20:204]
            point_loc_in_img = torch.ceil((point_loc_in_img - 20) * 224.0 / 184.0)
            img = torch.nn.functional.interpolate(
                img, size=(224, 224), mode="bilinear", align_corners=True
            )
            return img, is_seen, point_loc_in_img

        def forward(self, pc):
            img, is_seen, point_loc_in_img = self.mv_proj(pc)
            _, x = self.model.encode_image(img)
            x = x / x.norm(dim=-1, keepdim=True)
            b, l, c = x.shape
            x = x.reshape(b, 14, 14, c).permute(0, 3, 1, 2)
            return is_seen, point_loc_in_img, x

    extractor = Extractor(clip_model).to(device)
    extractor.eval()
    return extractor


def _load_runtime(model_name: Optional[str] = None) -> _PointCLIPRuntime:
    pcv2_root = _resolve_pcv2_root()
    _setup_import_path(pcv2_root)

    import torch  # type: ignore
    from clip import clip as pc_clip  # type: ignore
    from best_param import best_prompt, best_vweight  # type: ignore
    from post_search import cat2id, feat_dims  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PointCLIP V2 official projection code requires CUDA in this adapter. "
            "Run on a machine with CUDA or use precomputed pointclip predictions."
        )

    chosen_model = model_name or os.environ.get("POINTCLIPV2_MODEL", DEFAULT_MODEL_NAME)
    if chosen_model not in feat_dims:
        raise ValueError(f"unsupported PointCLIP V2 model '{chosen_model}'")

    device = "cuda:0"
    LOGGER.info("loading PointCLIP V2 CLIP model=%s device=%s", chosen_model, device)
    clip_model, _ = pc_clip.load(chosen_model, device=device)
    clip_model.eval()
    extractor = _build_extractor(torch, clip_model, pcv2_root, device)
    return _PointCLIPRuntime(
        model_name=chosen_model,
        device=device,
        torch=torch,
        clip_module=pc_clip,
        best_prompt=best_prompt,
        best_vweight=best_vweight,
        cat2id=cat2id,
        feat_dim=int(feat_dims[chosen_model]),
        extractor=extractor,
        clip_model=clip_model,
    )


def _encode_text(runtime: _PointCLIPRuntime, category: str):
    torch = runtime.torch
    prompts = runtime.best_prompt[category]
    tokens = torch.cat([runtime.clip_module.tokenize(p) for p in prompts]).to(runtime.device)
    with torch.no_grad():
        text_feat = runtime.clip_model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat, prompts


def _predict_one(runtime: _PointCLIPRuntime, points: np.ndarray, category: str) -> Tuple[np.ndarray, Dict]:
    """Return category-local PointCLIP V2 predictions for one 2048-point shape."""

    torch = runtime.torch
    if points.shape != (PC_NUM, 3):
        raise ValueError(f"PointCLIP V2 expects points shape {(PC_NUM, 3)}, got {points.shape}")
    if category not in runtime.best_prompt:
        raise ValueError(f"unsupported ShapeNet-Part category for PointCLIP V2: {category}")

    text_feat, prompts = _encode_text(runtime, category)
    part_num = int(text_feat.shape[0])
    view_weights = torch.tensor(runtime.best_vweight[category], device=runtime.device).view(1, -1, 1, 1)

    pc = torch.from_numpy(points.astype(np.float32))[None, :, :].to(runtime.device)
    with torch.no_grad():
        is_seen, point_loc, feat = runtime.extractor(pc)

        # Match the official PointCLIP V2 post_search.py cache path: saved
        # feature maps are reshaped directly from (B, 10, C, 14, 14) to
        # (B, 10, 196, C). This reproduces the released zero-shot segmentation
        # implementation rather than an idealized patch-major variant.
        feat = feat[None, :, :, :, :].reshape(1, 10, 196, runtime.feat_dim)
        b, nv, hw, c = feat.size(0), feat.size(1), feat.size(2), feat.size(3)
        feat = feat.reshape(b * nv, hw, c)
        point_loc = point_loc.reshape(b * nv, -1, 2)
        is_seen = is_seen.reshape(b * nv, -1, 1)

        logits = 100.0 * feat.half() @ text_feat.half().t()
        output = logits.float().permute(0, 2, 1).reshape(-1, part_num, int(hw**0.5), int(hw**0.5))

        avgpool = torch.nn.AvgPool2d(6, 1, 0)
        padding = torch.nn.ReplicationPad2d([2, 3, 2, 3])
        output = avgpool(padding(output))
        output = torch.nn.functional.interpolate(output, size=(224, 224), mode="bilinear", align_corners=False)

        nbatch = torch.repeat_interleave(
            torch.arange(0, nv * b, device=runtime.device)[:, None], PC_NUM
        ).view(-1).long()
        yy = point_loc[:, :, 0].view(-1).long().clamp(0, 223)
        xx = point_loc[:, :, 1].view(-1).long().clamp(0, 223)
        point_logits = output[nbatch, :, yy, xx].view(b, nv, PC_NUM, part_num)
        point_logits = torch.sum(point_logits * view_weights * is_seen.reshape(b, nv, PC_NUM, 1), dim=1)
        pred = point_logits.argmax(dim=-1).reshape(-1).cpu().numpy().astype(np.int64)

    meta = {
        "model_name": runtime.model_name,
        "device": runtime.device,
        "prompts": prompts,
        "view_weights": [float(x) for x in runtime.best_vweight[category]],
        "num_text_parts": part_num,
    }
    return pred, meta


def _write_meta(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_subset(
    keys: Iterable[str],
    by_key: Dict[str, object],
    *,
    dataset_module=None,
    overwrite: bool = False,
) -> int:
    """Run PointCLIP V2 on a ShapeNet-Part subset.

    ``dataset_module`` must be ``benchmark.shapenetpart``. The signature keeps
    compatibility with the benchmark dispatcher while allowing the default
    CoMPaT-200 code path to remain unsupported for this ShapeNet-Part baseline.
    """

    if dataset_module is None:
        from benchmark import shapenetpart as dataset_module  # type: ignore

    if not _is_shapenetpart(dataset_module):
        raise NotImplementedError(
            "PointCLIP V2 adapter is wired for ShapeNet-Part only. "
            "Use --dataset shapenetpart."
        )

    keys = list(keys)
    pred_dir = dataset_module.RESULTS_DIR / "pointclip"
    pred_dir.mkdir(parents=True, exist_ok=True)

    pending = [key for key in keys if overwrite or not (pred_dir / f"{key}.npy").is_file()]
    if not pending:
        print(f"[pointclip] all {len(keys)} predictions already exist; use --overwrite to recompute")
        return 0

    runtime = _load_runtime()
    t0_all = time.perf_counter()
    for i, key in enumerate(keys):
        meta = by_key[key]
        pred_path = pred_dir / f"{key}.npy"
        meta_path = pred_dir / f"{key}.meta.json"
        if pred_path.is_file() and not overwrite:
            print(f"[{i+1}/{len(keys)}] skip {key} (exists)")
            continue

        gt = dataset_module.load_gt(key)
        t0 = time.perf_counter()
        print(f"[{i+1}/{len(keys)}] pointclip {key} ({gt.category}, {len(gt.part_names)} parts)", flush=True)
        pred, pcv2_meta = _predict_one(runtime, gt.points, gt.category)
        elapsed = time.perf_counter() - t0
        np.save(pred_path, pred.astype(np.int64))
        _write_meta(
            meta_path,
            {
                "key": key,
                "category": gt.category,
                "instance": gt.instance,
                "part_names": gt.part_names,
                "source_path": getattr(meta, "source_path", None),
                "n_points": int(gt.points.shape[0]),
                "elapsed_seconds": elapsed,
                "pointclip_v2": pcv2_meta,
            },
        )
        print(f"    done in {elapsed:.1f}s -> {pred_path}", flush=True)

    print(f"[pointclip] subset complete in {time.perf_counter() - t0_all:.1f}s")
    return 0
