"""Dispatch block partition by strategy name."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any, Dict, Optional, Sequence, Tuple

from .common import SubBlockSplit
from .cvt import split_block_by_cvt
from .geodesic_fps import split_block_by_geodesic_fps
from .graph_cut import split_block_by_graph_cut
from .sdf import split_block_by_sdf
from .skeleton import split_block_by_skeleton
from .spectral import split_block_by_spectral_seeds

LOGGER = logging.getLogger("geometry.partition.strategy")

VALID_STRATEGIES = (
    "graph_cut",
    "spectral",
    "cvt",
    "sdf",
    "skeleton",
    "geodesic_fps",
)

# 这些策略在子进程里跑收益不大，且避免 pickle 整 mesh + 误杀；超时仍由上层重试控制
_INPROCESS_STRATEGIES = frozenset({"spectral", "geodesic_fps", "cvt", "skeleton"})


def split_block_by_strategy(
    vertices,
    faces,
    face_ids: Sequence[int],
    seed_count: int,
    strategy: str,
    *,
    lambda_smooth: float = 1.0,
    lambda_concave: float = 0.85,
    scalar_weight: float = 0.35,
    geodesic_weight: Optional[float] = None,
    curvature_weight: Optional[float] = None,
    concavity_weight: Optional[float] = None,
    lloyd_iter: int = 0,
    num_rays: int = 16,
    cone_angle_deg: float = 60.0,
    contraction_iters: int = 5,
    graph_vertex_budget: Optional[int] = None,
    vertex_scalars: Optional[Sequence[float]] = None,
) -> SubBlockSplit:
    strat = str(strategy or "geodesic_fps").strip().lower()
    if strat not in VALID_STRATEGIES:
        LOGGER.warning("未知分块策略 %r，回退 geodesic_fps", strategy)
        strat = "geodesic_fps"

    common = dict(
        vertices=vertices,
        faces=faces,
        face_ids=face_ids,
        seed_count=seed_count,
        vertex_scalars=vertex_scalars,
    )

    if strat == "graph_cut":
        return split_block_by_graph_cut(
            **common,
            lambda_smooth=lambda_smooth,
            lambda_concave=lambda_concave,
            scalar_weight=scalar_weight,
        )
    if strat == "spectral":
        return split_block_by_spectral_seeds(
            **common,
            geodesic_weight=geodesic_weight,
            curvature_weight=curvature_weight,
            concavity_weight=concavity_weight,
            graph_vertex_budget=graph_vertex_budget,
        )
    if strat == "cvt":
        return split_block_by_cvt(
            **common,
            lloyd_iter=lloyd_iter,
            scalar_weight=scalar_weight,
        )
    if strat == "sdf":
        return split_block_by_sdf(
            **common,
            num_rays=num_rays,
            cone_angle_deg=cone_angle_deg,
            scalar_weight=scalar_weight,
        )
    if strat == "skeleton":
        return split_block_by_skeleton(
            **common,
            contraction_iters=contraction_iters,
            lloyd_iter=lloyd_iter,
            graph_vertex_budget=graph_vertex_budget,
            scalar_weight=scalar_weight,
        )
    return split_block_by_geodesic_fps(
        **common,
        lloyd_iter=lloyd_iter,
        graph_vertex_budget=graph_vertex_budget,
        scalar_weight=scalar_weight,
    )


def _split_worker(kwargs: Dict[str, Any]) -> SubBlockSplit:
    """ProcessPool 子进程入口（须为模块级函数以便 pickle）。"""
    return split_block_by_strategy(**kwargs)


def split_needs_strategy_retry(
    split: Optional[SubBlockSplit],
    *,
    error: Optional[str] = None,
) -> Tuple[bool, str]:
    """判断几何分裂是否应换策略重试。"""
    if error:
        return True, str(error)
    if split is None:
        return True, "split_returned_none"
    if int(split.num_clusters) <= 1:
        return True, "degenerate_single_cluster"
    return False, ""


def split_block_by_strategy_timed(
    timeout_seconds: float,
    *,
    vertices,
    faces,
    face_ids: Sequence[int],
    seed_count: int,
    strategy: str,
    lambda_smooth: float = 1.0,
    lambda_concave: float = 0.85,
    scalar_weight: float = 0.35,
    geodesic_weight: Optional[float] = None,
    curvature_weight: Optional[float] = None,
    concavity_weight: Optional[float] = None,
    lloyd_iter: int = 0,
    num_rays: int = 16,
    cone_angle_deg: float = 60.0,
    contraction_iters: int = 5,
    graph_vertex_budget: Optional[int] = None,
    vertex_scalars: Optional[Sequence[float]] = None,
) -> Tuple[Optional[SubBlockSplit], Optional[str]]:
    """带超时执行分裂；超时或异常时返回 (None, reason)。"""
    kwargs: Dict[str, Any] = dict(
        vertices=vertices,
        faces=faces,
        face_ids=list(face_ids),
        seed_count=int(seed_count),
        strategy=strategy,
        lambda_smooth=lambda_smooth,
        lambda_concave=lambda_concave,
        scalar_weight=scalar_weight,
        geodesic_weight=geodesic_weight,
        curvature_weight=curvature_weight,
        concavity_weight=concavity_weight,
        lloyd_iter=lloyd_iter,
        num_rays=num_rays,
        cone_angle_deg=cone_angle_deg,
        contraction_iters=contraction_iters,
        graph_vertex_budget=graph_vertex_budget,
        vertex_scalars=vertex_scalars,
    )
    timeout = float(timeout_seconds)
    strat = str(strategy or "geodesic_fps").strip().lower()
    if timeout <= 0 or strat in _INPROCESS_STRATEGIES:
        try:
            return split_block_by_strategy(**kwargs), None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_split_worker, kwargs)
            try:
                return fut.result(timeout=timeout), None
            except FuturesTimeout:
                pool.shutdown(wait=False, cancel_futures=True)
                return None, "timeout"
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("timed split 进程异常，回退同线程执行: %s", exc)
        try:
            return split_block_by_strategy(**kwargs), None
        except Exception as exc2:  # noqa: BLE001
            return None, str(exc2)
