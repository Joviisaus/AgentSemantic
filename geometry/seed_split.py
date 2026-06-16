"""
分块的"种子分裂"：
- 询问 VLM 在这个分块内需要多少子块（seed_count ≥ 2）
- 几何上做分裂；默认走「测地最远点采样 (FPS) + 测地 Voronoi」（限定在分块的子 mesh 内），
  如果 FPS 失败再回退到原本的谱聚类。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from geometry.partition import (
    SubBlockSplit,
    VALID_STRATEGIES,
    split_block_by_connected_components,
    split_block_by_geodesic_fps,
    split_block_by_spectral_seeds,
)
from geometry.partition.common import extract_submesh as _extract_submesh
from geometry.partition.scale import (
    scaled_contraction_iters,
    scaled_lloyd_iter,
    scaled_num_rays,
)
from vlm.client import VLMClient, parse_json_response
from vlm.prompts import build_seed_prompt, build_split_strategy_prompt

LOGGER = logging.getLogger("geometry.seed_split")


@dataclass
class SeedCountResponse:
    seed_count: int
    reason: str = ""
    raw_text: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitStrategyResponse:
    strategy: str = "geodesic_fps"
    reason: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedAndStrategyResponse:
    seed_count: int = 2
    strategy: str = "geodesic_fps"
    reason: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)


def ask_seed_count(
    client: VLMClient,
    image_path: str | Path,
    semantic: str,
    *,
    max_seeds: int = 8,
    fallback: int = 2,
    model: Optional[str] = None,
    covered_parts: Optional[List[str]] = None,
    semantic_parts: Optional[List[str]] = None,
    extra_parts: Optional[List[str]] = None,
    block_faces: int = 0,
) -> SeedCountResponse:
    """让 VLM 给出 seed_count，失败时回退到 fallback。

    传入 covered_parts / semantic_parts / extra_parts 时，VLM 会把它们当作
    上一阶段的"先验事实"，输出更稳的 seed_count（通常会更激进地多切几刀）。
    """
    prompt = build_seed_prompt(
        semantic,
        max_seeds=max_seeds,
        covered_parts=covered_parts,
        semantic_parts=semantic_parts,
        extra_parts=extra_parts,
        block_faces=block_faces,
    )
    try:
        response = client.chat(
            user_prompt=prompt.user,
            image_paths=[Path(image_path)],
            system=prompt.system,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 种子数查询失败: %s", exc)
        return SeedCountResponse(seed_count=fallback, error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
        raw_count = int(payload.get("seed_count", fallback))
        reason = str(payload.get("reason") or "")
    except (ValueError, TypeError) as exc:
        LOGGER.warning("VLM 种子数响应解析失败: %s", text[:200])
        return SeedCountResponse(
            seed_count=fallback,
            reason=text.strip()[:200],
            raw_text=text,
            error=str(exc),
            raw_response=response.raw,
        )

    seed_count = max(2, min(int(max_seeds), int(raw_count)))
    return SeedCountResponse(
        seed_count=seed_count,
        reason=reason,
        raw_text=text,
        raw_response=response.raw,
    )


def _clamp_split_params(strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """把 VLM 返回的参数裁剪到安全范围。"""
    out: Dict[str, Any] = {}
    strat = str(strategy or "geodesic_fps").strip().lower()
    if strat == "graph_cut":
        out["lambda_smooth"] = float(np.clip(
            float(params.get("lambda_smooth", 1.0)), 0.1, 3.0
        ))
        out["lambda_concave"] = float(np.clip(
            float(params.get("lambda_concave", 0.85)), 0.0, 2.0
        ))
    elif strat == "spectral":
        # 兼容旧版 scalar_weight → curvature_weight
        if "scalar_weight" in params and "curvature_weight" not in params:
            params = {**params, "curvature_weight": params["scalar_weight"]}
        out["geodesic_weight"] = float(
            np.clip(float(params.get("geodesic_weight", 0.20)), 0.05, 0.80)
        )
        out["curvature_weight"] = float(
            np.clip(float(params.get("curvature_weight", 0.15)), 0.05, 0.80)
        )
        out["concavity_weight"] = float(
            np.clip(float(params.get("concavity_weight", 0.70)), 0.05, 0.90)
        )
        if "graph_vertex_budget" in params and params["graph_vertex_budget"] is not None:
            budget = int(params["graph_vertex_budget"])
            if budget > 0:
                out["graph_vertex_budget"] = int(np.clip(budget, 4_000, 20_000))
    elif strat == "cvt":
        out["lloyd_iter"] = int(np.clip(
            int(params.get("lloyd_iter", 4)), 2, 12
        ))
    elif strat == "sdf":
        out["num_rays"] = int(np.clip(int(params.get("num_rays", 8)), 4, 24))
        out["cone_angle_deg"] = float(np.clip(
            float(params.get("cone_angle_deg", 60.0)), 15.0, 120.0
        ))
    elif strat == "skeleton":
        out["contraction_iters"] = int(np.clip(
            int(params.get("contraction_iters", 5)), 2, 12
        ))
        out["lloyd_iter"] = int(np.clip(
            int(params.get("lloyd_iter", 0)), 0, 12
        ))
        if "graph_vertex_budget" in params and params["graph_vertex_budget"] is not None:
            budget = int(params["graph_vertex_budget"])
            if budget > 0:
                out["graph_vertex_budget"] = int(np.clip(budget, 4_000, 20_000))
    elif strat == "geodesic_fps":
        out["lloyd_iter"] = int(np.clip(
            int(params.get("lloyd_iter", 0)), 0, 12
        ))
        if "graph_vertex_budget" in params and params["graph_vertex_budget"] is not None:
            budget = int(params["graph_vertex_budget"])
            if budget > 0:
                out["graph_vertex_budget"] = int(np.clip(budget, 4_000, 20_000))
    return out


def ask_split_strategy(
    client: VLMClient,
    image_path: str | Path | None,
    *,
    semantic_context: str = "",
    seed_count: int = 2,
    covered_parts: Optional[List[str]] = None,
    semantic_parts: Optional[List[str]] = None,
    extra_parts: Optional[List[str]] = None,
    matched_parts: Optional[List[str]] = None,
    block_faces: int = 0,
    split_stage: str = "main",
    strategy_hint: str = "",
    fallback: str = "spectral",
    model: Optional[str] = None,
) -> SplitStrategyResponse:
    """让 VLM 根据语义、面数规模与可选 CLI 提示选择分块策略与参数。"""
    prompt = build_split_strategy_prompt(
        semantic_context=semantic_context,
        seed_count=seed_count,
        covered_parts=covered_parts,
        semantic_parts=semantic_parts,
        extra_parts=extra_parts,
        matched_parts=matched_parts,
        block_faces=block_faces,
        split_stage=split_stage,
        strategy_hint=strategy_hint,
    )
    image_paths = [Path(image_path)] if image_path else []
    try:
        response = client.chat(
            user_prompt=prompt.user,
            image_paths=image_paths,
            system=prompt.system,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 分块策略查询失败: %s", exc)
        return SplitStrategyResponse(strategy=fallback, error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
        raw_strat = str(payload.get("strategy") or fallback).strip().lower()
        reason = str(payload.get("reason") or "")
        raw_params = payload.get("params") or {}
        if not isinstance(raw_params, dict):
            raw_params = {}
    except (ValueError, TypeError) as exc:
        LOGGER.warning("VLM 分块策略响应解析失败: %s", text[:200])
        return SplitStrategyResponse(
            strategy=fallback,
            reason=text.strip()[:200],
            raw_text=text,
            error=str(exc),
            raw_response=response.raw,
        )

    strategy = raw_strat if raw_strat in VALID_STRATEGIES else fallback
    params = _clamp_split_params(strategy, raw_params)
    return SplitStrategyResponse(
        strategy=strategy,
        reason=reason,
        params=params,
        raw_text=text,
        raw_response=response.raw,
    )


_LARGE_FACE_THRESHOLD = 80_000
_HUGE_FACE_THRESHOLD = 200_000


def _default_strategy_params(
    strategy: str, *, block_faces: int = 0, n_vertices: int = 0
) -> Dict[str, Any]:
    """各策略的默认参数（geodesic_fps/skeleton 默认不池化、不跑 Lloyd）。"""
    faces = int(block_faces)
    verts = int(n_vertices) if n_vertices > 0 else faces // 2
    if strategy == "spectral":
        return {
            "geodesic_weight": 0.20,
            "curvature_weight": 0.15,
            "concavity_weight": 0.70,
        }
    if strategy == "graph_cut":
        return {"lambda_smooth": 1.0, "lambda_concave": 0.85}
    if strategy == "cvt":
        return {"lloyd_iter": scaled_lloyd_iter(faces, 4)}
    if strategy == "sdf":
        return {
            "num_rays": scaled_num_rays(faces, 8),
            "cone_angle_deg": 60.0,
        }
    if strategy == "skeleton":
        return {
            "contraction_iters": scaled_contraction_iters(verts, 5),
            "lloyd_iter": 0,
        }
    if strategy == "geodesic_fps":
        return {"lloyd_iter": 0}
    return {}


def infer_split_strategy_heuristic(
    *,
    block_faces: int = 0,
    strategy_hint: str = "spectral",
    seed_count: int = 2,
) -> SplitStrategyResponse:
    """按面数规模与 CLI 提示快速选策略（不调 VLM）。

    大块不再强制 geodesic_fps；测地类策略会在内部自动池化图以控制耗时。
    """
    _ = seed_count
    hint = str(strategy_hint or "auto").strip().lower()
    faces = int(block_faces)
    est_vertices = max(faces // 2, 1)

    if hint != "auto" and hint in VALID_STRATEGIES:
        strat = hint
        reason = f"CLI 指定 {hint}（{faces} face"
        if faces >= _LARGE_FACE_THRESHOLD:
            reason += "，大图可考虑 VLM 指定 graph_vertex_budget"
        reason += "）"
    else:
        strat = "spectral"
        reason = f"block_faces={faces}，spectral（测地距离特征+谱嵌入初始化+Voronoi）"
        if faces >= _LARGE_FACE_THRESHOLD:
            reason += "；大图可由 VLM 指定 graph_vertex_budget 池化"

    params = _clamp_split_params(
        strat,
        _default_strategy_params(strat, block_faces=faces, n_vertices=est_vertices),
    )
    return SplitStrategyResponse(
        strategy=strat,
        reason=reason,
        params=params,
    )


def infer_split_strategy_retry(
    *,
    block_faces: int = 0,
    tried_strategies: Optional[List[str]] = None,
    failure_reason: str = "",
    seed_count: int = 2,
) -> SplitStrategyResponse:
    """上一策略失败/超时后的启发式换策略（不调 VLM）。"""
    tried = {str(s).strip().lower() for s in (tried_strategies or [])}
    faces = int(block_faces)
    est_vertices = max(faces // 2, 1)
    order = ["spectral", "skeleton", "graph_cut", "sdf", "cvt", "geodesic_fps"]
    for strat in order:
        if strat in tried:
            continue
        params = _clamp_split_params(
            strat,
            _default_strategy_params(strat, block_faces=faces, n_vertices=est_vertices),
        )
        reason = f"重试换策略 {strat}"
        if failure_reason:
            reason += f"（因 {failure_reason}）"
        if faces >= _LARGE_FACE_THRESHOLD:
            reason += "；大图可显式指定 graph_vertex_budget 池化"
        return SplitStrategyResponse(
            strategy=strat,
            reason=reason,
            params=params,
        )
    return infer_split_strategy_heuristic(
        block_faces=faces,
        strategy_hint="spectral",
        seed_count=seed_count,
    )


def ask_split_strategy_retry(
    client: VLMClient,
    image_path: str | Path | None,
    *,
    semantic_context: str = "",
    seed_count: int = 2,
    matched_parts: Optional[List[str]] = None,
    block_faces: int = 0,
    split_stage: str = "main",
    strategy_hint: str = "",
    previous_strategy: str = "",
    failure_reason: str = "",
    tried_strategies: Optional[List[str]] = None,
    fallback: str = "geodesic_fps",
    model: Optional[str] = None,
) -> SplitStrategyResponse:
    """上一策略失败/超时后，让 VLM 换用不同分块策略。"""
    tried = list(tried_strategies or [])
    ctx_parts = [semantic_context or "全覆盖语义细分"]
    if previous_strategy:
        ctx_parts.append(f"上一策略「{previous_strategy}」未成功")
    if failure_reason:
        ctx_parts.append(f"失败原因：{failure_reason}")
    if tried:
        ctx_parts.append(f"已尝试过：{'、'.join(tried)}（请选不同 strategy）")
    ctx_parts.append("优先选能在当前面数规模下快速完成且能切出多个子块的方法")
    if matched_parts:
        ctx_parts.append(
            f"切分后子块并集必须覆盖全部 matched_parts（{ '、'.join(matched_parts) }），"
            "禁止小部位（如头、手）被吞并；凹陷明显时优先 graph_cut（高 lambda_concave）"
            "或 spectral（高 concavity_weight）；避免 cvt 与均匀 geodesic_fps"
        )
    retry_ctx = "。".join(ctx_parts)
    resp = ask_split_strategy(
        client,
        image_path,
        semantic_context=retry_ctx,
        seed_count=seed_count,
        matched_parts=matched_parts,
        block_faces=block_faces,
        split_stage=split_stage,
        strategy_hint=strategy_hint,
        fallback=fallback,
        model=model,
    )
    if resp.strategy in tried and len(tried) >= 1:
        fb = infer_split_strategy_retry(
            block_faces=block_faces,
            tried_strategies=tried,
            failure_reason=failure_reason or "vlm repeated strategy",
            seed_count=seed_count,
        )
        fb.reason = f"VLM 重复 {resp.strategy}；{fb.reason}"
        return fb
    return resp


def infer_split_strategy_fallback(
    *,
    seed_count: int = 2,
    block_faces: int = 0,
    strategy_hint: str = "spectral",
    reason: str = "",
) -> SplitStrategyResponse:
    """无 VLM 时的兜底策略。"""
    resp = infer_split_strategy_heuristic(
        block_faces=block_faces,
        strategy_hint=strategy_hint,
        seed_count=seed_count,
    )
    if reason:
        resp.reason = reason
    return resp


def ask_seed_and_strategy(
    client: VLMClient,
    image_path: str | Path,
    semantic: str,
    *,
    max_seeds: int = 8,
    fallback_seed: int = 2,
    fallback_strategy: str = "geodesic_fps",
    covered_parts: Optional[List[str]] = None,
    semantic_parts: Optional[List[str]] = None,
    extra_parts: Optional[List[str]] = None,
    block_faces: int = 0,
    model: Optional[str] = None,
) -> SeedAndStrategyResponse:
    """一次 VLM 调用同时返回 seed_count + 分块策略（合并 prompt）。"""
    prompt = build_seed_prompt(
        semantic,
        max_seeds=max_seeds,
        covered_parts=covered_parts,
        semantic_parts=semantic_parts,
        extra_parts=extra_parts,
        block_faces=block_faces,
    )
    try:
        response = client.chat(
            user_prompt=prompt.user,
            image_paths=[Path(image_path)],
            system=prompt.system,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM seed+strategy 查询失败: %s", exc)
        return SeedAndStrategyResponse(
            seed_count=fallback_seed,
            strategy=fallback_strategy,
            error=str(exc),
        )

    text = response.text or ""
    try:
        payload = parse_json_response(text)
        raw_count = int(payload.get("seed_count", fallback_seed))
        raw_strat = str(payload.get("strategy") or fallback_strategy).strip().lower()
        reason = str(payload.get("reason") or "")
        raw_params = payload.get("params") or {}
        if not isinstance(raw_params, dict):
            raw_params = {}
    except (ValueError, TypeError) as exc:
        LOGGER.warning("VLM seed+strategy 解析失败: %s", text[:200])
        return SeedAndStrategyResponse(
            seed_count=fallback_seed,
            strategy=fallback_strategy,
            reason=text.strip()[:200],
            raw_text=text,
            error=str(exc),
            raw_response=response.raw,
        )

    seed_count = max(2, min(int(max_seeds), int(raw_count)))
    strategy = raw_strat if raw_strat in VALID_STRATEGIES else fallback_strategy
    return SeedAndStrategyResponse(
        seed_count=seed_count,
        strategy=strategy,
        reason=reason,
        params=_clamp_split_params(strategy, raw_params),
        raw_text=text,
        raw_response=response.raw,
    )


__all__ = [
    "SeedCountResponse",
    "SplitStrategyResponse",
    "SeedAndStrategyResponse",
    "SubBlockSplit",
    "VALID_STRATEGIES",
    "ask_seed_count",
    "ask_split_strategy",
    "ask_seed_and_strategy",
    "infer_split_strategy_heuristic",
    "infer_split_strategy_retry",
    "ask_split_strategy_retry",
    "infer_split_strategy_fallback",
    "split_block_by_connected_components",
    "split_block_by_geodesic_fps",
    "split_block_by_spectral_seeds",
    "_extract_submesh",
]
