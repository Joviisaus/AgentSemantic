"""
点云分块策略调度 — 与 mesh ``geometry.partition.strategy`` 对齐的策略名与参数。

NEO 仅作为 ``spectral`` 策略的可选 LBO 特征空间加速器（``precomputed_eigvecs``），
不覆盖其它策略。

``ask_point_split_strategy`` 让 VLM 看分块高亮图，挑分块策略 + 参数（曲率指标、
凸包/凹凸指标等），与 mesh ``geometry.seed_split.ask_split_strategy`` 对齐，只是
策略集换成 ``VALID_POINT_STRATEGIES``、prompt 走点云版（``media_type="pointcloud"``）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.pointcloud_cluster import (
    DEFAULT_CURVATURE_WEIGHT,
    DEFAULT_KNN,
    PointSubSplit,
    _finalize_kmeans_labels,
    _groups_from_labels,
    _kmeans,
    build_point_knn_graph,
    compute_point_normals_and_curvature,
    enforce_per_label_point_connectivity,
    feature_spectral_cluster_points,
    split_points_by_geodesic_fps,
)

LOGGER = logging.getLogger("geometry.pointcloud_partition")

# graph_cut 已禁用（边缘平滑但不保持形状）：不再作为可选策略，遇到时统一改用 spectral。
VALID_POINT_STRATEGIES = (
    "spectral",
    "cvt",
    "sdf",
    "skeleton",
    "geodesic_fps",
)


def split_needs_strategy_retry(
    split: Optional[PointSubSplit],
    *,
    error: Optional[str] = None,
) -> Tuple[bool, str]:
    if error:
        return True, str(error)
    if split is None:
        return True, "split_returned_none"
    if int(split.num_clusters) <= 1:
        return True, "degenerate_single_cluster"
    return False, ""


def _sub_block(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    ids = np.asarray(list(point_ids), dtype=np.int64)
    seed_count = max(2, int(seed_count))
    all_pts = np.asarray(points, dtype=np.float64)
    sub_pts = all_pts[ids]
    return ids, sub_pts, seed_count


def _too_small(ids: np.ndarray, seed_count: int) -> Optional[PointSubSplit]:
    if ids.size == 0:
        return PointSubSplit([], np.zeros(0, dtype=np.int64), 0)
    if ids.size <= seed_count:
        return PointSubSplit(
            sub_point_groups=[ids.tolist()],
            sub_labels=np.zeros(ids.size, dtype=np.int64),
            num_clusters=1,
            debug={"reason": "子点集太小，跳过分裂"},
        )
    return None


def _compute_point_sdf(
    points: np.ndarray,
    normals: np.ndarray,
    *,
    num_rays: int = 16,
    cone_angle_deg: float = 60.0,
) -> np.ndarray:
    """点云近似 SDF：沿 -法向锥内射线，在 kNN 邻域内找穿透厚度。"""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    nrm = np.asarray(normals, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    tree = cKDTree(pts)
    bbox = float(np.linalg.norm(pts - pts.mean(axis=0), axis=1).max())
    search_r = max(bbox * 0.5, 1e-6)
    cone_rad = np.deg2rad(float(cone_angle_deg))
    rng = np.random.default_rng(0)
    center = pts.mean(axis=0)
    sdf = np.zeros(n, dtype=np.float64)

    for i in range(n):
        origin = pts[i] - nrm[i] * 1e-4
        inward = -nrm[i]
        inward /= max(float(np.linalg.norm(inward)), 1e-12)
        directions: List[np.ndarray] = [inward]
        if num_rays > 1:
            aux = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(aux, inward))) > 0.9:
                aux = np.array([0.0, 1.0, 0.0])
            u = np.cross(inward, aux)
            u /= max(float(np.linalg.norm(u)), 1e-12)
            v = np.cross(inward, u)
            for _ in range(int(num_rays) - 1):
                theta = rng.uniform(0, 2 * np.pi)
                phi = rng.uniform(0, cone_rad)
                d = (
                    inward * np.cos(phi)
                    + u * np.sin(phi) * np.cos(theta)
                    + v * np.sin(phi) * np.sin(theta)
                )
                d /= max(float(np.linalg.norm(d)), 1e-12)
                directions.append(d)

        thicknesses: List[float] = []
        cand_idx = [j for j in tree.query_ball_point(origin, r=search_r) if j != i]
        if not cand_idx:
            sdf[i] = float(np.linalg.norm(pts[i] - center))
            continue
        vecs = pts[cand_idx] - origin
        for d in directions:
            proj = vecs @ d
            valid = proj > 1e-5
            if np.any(valid):
                thicknesses.append(float(np.min(proj[valid])))

        if thicknesses:
            sdf[i] = float(np.mean(thicknesses))
        else:
            sdf[i] = float(np.linalg.norm(pts[i] - center))

    sdf = sdf / max(float(np.percentile(sdf, 95)), 1e-6)
    return sdf


def split_points_by_graph_cut(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
    *,
    lambda_smooth: float = 1.0,
    lambda_concave: float = 0.85,
    n_neighbors: int = DEFAULT_KNN,
    random_state: int = 0,
) -> PointSubSplit:
    """点云版 graph_cut：kNN 图 + 凹凸感知边权 + alpha-expansion。"""
    from geometry.partition.graph_cut import (
        _alpha_expansion_labels,
        _edge_concavity_weights,
        _face_geodesic_to_seeds,
    )

    ids, sub_pts, seed_count = _sub_block(points, point_ids, seed_count)
    tiny = _too_small(ids, seed_count)
    if tiny is not None:
        return tiny

    m = len(sub_pts)
    graph = build_point_knn_graph(sub_pts, n_neighbors=n_neighbors)
    normals, _ = compute_point_normals_and_curvature(sub_pts, n_neighbors=n_neighbors)
    centroids = sub_pts

    center = centroids.mean(axis=0)
    start = int(np.argmax(np.linalg.norm(centroids - center, axis=1)))
    seeds: List[int] = [start]
    dist_to_seeds = np.linalg.norm(centroids - centroids[start], axis=1)
    for _ in range(1, min(seed_count, m)):
        nxt = int(np.argmax(dist_to_seeds))
        if dist_to_seeds[nxt] <= 0:
            break
        seeds.append(nxt)
        dist_to_seeds = np.minimum(
            dist_to_seeds, np.linalg.norm(centroids - centroids[nxt], axis=1)
        )

    geo_dists = _face_geodesic_to_seeds(graph, seeds)
    unary = geo_dists.T.copy()
    for col in range(len(seeds)):
        col_data = unary[:, col]
        finite = col_data[np.isfinite(col_data)]
        scale = float(np.median(finite)) if finite.size else 1.0
        unary[:, col] = col_data / max(scale, 1e-6)

    er, ec, ew = _edge_concavity_weights(
        centroids, normals, graph,
        lambda_smooth=float(lambda_smooth),
        lambda_concave=float(lambda_concave),
    )
    labels = _alpha_expansion_labels(unary, er, ec, ew)
    labels = enforce_per_label_point_connectivity(graph, labels)

    return _groups_from_labels(
        ids, labels,
        strategy="graph_cut",
    )


def split_points_by_spectral(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
    *,
    point_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
    curvature_weight: Optional[float] = None,
    concavity_weight: Optional[float] = None,
    normal_weight: float = 1.0,
    n_neighbors: int = DEFAULT_KNN,
    enforce_connectivity: bool = True,
    precomputed_eigvecs: Optional[np.ndarray] = None,
    random_state: int = 0,
) -> PointSubSplit:
    """点云版 spectral：默认 feature_spectral；有 NEO 特征向量时走 LBO 嵌入。"""
    ids, sub_pts, seed_count = _sub_block(points, point_ids, seed_count)
    tiny = _too_small(ids, seed_count)
    if tiny is not None:
        return tiny

    cur_w = DEFAULT_CURVATURE_WEIGHT if curvature_weight is None else float(curvature_weight)
    conv_w = 0.0 if concavity_weight is None else float(concavity_weight)

    sub_scalars = None
    if point_scalars is not None and len(point_scalars) == len(points):
        sub_scalars = np.asarray(point_scalars, dtype=np.float64)[ids]

    if precomputed_eigvecs is not None:
        ev = np.asarray(precomputed_eigvecs, dtype=np.float64)
        if ev.shape[0] >= int(ids.max()) + 1 and ev.shape[1] > 1:
            emb = ev[ids][:, 1 : seed_count + 1]
            if emb.shape[1] > 0:
                labels = _kmeans(emb, seed_count, random_state=random_state)
                return _finalize_kmeans_labels(
                    ids, sub_pts, labels,
                    strategy="spectral_neo_lbo",
                    enforce_connectivity=enforce_connectivity,
                    n_neighbors=n_neighbors,
                )

    try:
        labels, _k, _dbg = feature_spectral_cluster_points(
            sub_pts,
            n_clusters=seed_count,
            point_scalars=sub_scalars,
            scalar_weight=scalar_weight,
            curvature_weight=cur_w,
            convexity_weight=conv_w,
            normal_weight=normal_weight,
            n_neighbors=n_neighbors,
            min_clusters=2,
            max_clusters=max(seed_count, 2),
            random_state=random_state,
        )
        return _finalize_kmeans_labels(
            ids, sub_pts, labels,
            strategy="spectral",
            enforce_connectivity=enforce_connectivity,
            n_neighbors=n_neighbors,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("spectral 失败，回退坐标 KMeans: %s", exc)
        return _finalize_kmeans_labels(
            ids, sub_pts, _kmeans(sub_pts, seed_count, random_state=random_state),
            strategy="spectral_kmeans_fallback",
            enforce_connectivity=enforce_connectivity,
            n_neighbors=n_neighbors,
        )


def split_points_by_cvt(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
    *,
    lloyd_iter: int = 4,
    n_neighbors: int = DEFAULT_KNN,
    random_state: int = 0,
) -> PointSubSplit:
    """点云 CVT：geodesic_fps + Lloyd 松弛（与 mesh cvt 对齐）。"""
    split = split_points_by_geodesic_fps(
        points, point_ids, seed_count,
        n_neighbors=n_neighbors,
        lloyd_iter=int(lloyd_iter),
        random_state=random_state,
    )
    dbg = dict(split.debug) if split.debug else {}
    dbg["strategy"] = "cvt"
    dbg["lloyd_iter"] = int(lloyd_iter)
    return PointSubSplit(
        sub_point_groups=split.sub_point_groups,
        sub_labels=split.sub_labels,
        num_clusters=split.num_clusters,
        debug=dbg,
    )


def split_points_by_sdf(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
    *,
    num_rays: int = 16,
    cone_angle_deg: float = 60.0,
    scalar_weight: float = 0.35,
    n_neighbors: int = DEFAULT_KNN,
    enforce_connectivity: bool = True,
    random_state: int = 0,
) -> PointSubSplit:
    """点云 SDF：近似厚度 + 位置/法向特征 → KMeans。"""
    _ = scalar_weight
    ids, sub_pts, seed_count = _sub_block(points, point_ids, seed_count)
    tiny = _too_small(ids, seed_count)
    if tiny is not None:
        return tiny

    normals, _ = compute_point_normals_and_curvature(sub_pts, n_neighbors=n_neighbors)
    sdf = _compute_point_sdf(
        sub_pts, normals, num_rays=int(num_rays), cone_angle_deg=float(cone_angle_deg),
    )
    center = sub_pts.mean(axis=0)
    pos = (sub_pts - center) / max(
        float(np.linalg.norm(sub_pts - center, axis=1).max()), 1e-6
    )
    nrm = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    features = np.hstack([sdf.reshape(-1, 1), pos, nrm * 0.35])

    try:
        labels = _kmeans(features.astype(np.float64), seed_count, random_state=random_state)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("SDF KMeans 失败 (%s)，回退 geodesic_fps", exc)
        fb = split_points_by_geodesic_fps(
            points, point_ids, seed_count, n_neighbors=n_neighbors, random_state=random_state,
        )
        dbg = dict(fb.debug or {})
        dbg["fallback"] = "geodesic_fps_after_sdf_error"
        dbg["sdf_error"] = str(exc)
        return PointSubSplit(
            sub_point_groups=fb.sub_point_groups,
            sub_labels=fb.sub_labels,
            num_clusters=fb.num_clusters,
            debug=dbg,
        )

    out = _finalize_kmeans_labels(
        ids, sub_pts, labels,
        strategy="sdf",
        enforce_connectivity=enforce_connectivity,
        n_neighbors=n_neighbors,
    )
    out.debug.update({
        "num_rays": int(num_rays),
        "cone_angle_deg": float(cone_angle_deg),
        "sdf_range": [float(sdf.min()), float(sdf.max())],
    })
    return out


def split_points_by_skeleton(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
    *,
    contraction_iters: int = 5,
    lloyd_iter: int = 0,
    n_neighbors: int = DEFAULT_KNN,
    random_state: int = 0,
) -> PointSubSplit:
    """点云 skeleton：kNN 图 Laplacian 收缩 + 骨架 FPS + 测地 Voronoi。"""
    from geometry.partition.geodesic_fps import geodesic_voronoi_assign, lloyd_relax_geodesic
    from geometry.partition.skeleton import (
        _skeleton_candidate_indices,
        laplacian_contract_graph,
        skeleton_fps_seeds,
    )

    ids, sub_pts, seed_count = _sub_block(points, point_ids, seed_count)
    tiny = _too_small(ids, seed_count)
    if tiny is not None:
        return tiny

    graph = build_point_knn_graph(sub_pts, n_neighbors=n_neighbors)
    skel_error: Optional[str] = None
    strategy = "laplacian_contract_skeleton_voronoi"

    try:
        contracted = laplacian_contract_graph(
            graph, sub_pts, n_iters=int(contraction_iters),
        )
        cand = _skeleton_candidate_indices(sub_pts, contracted)
        fps_seeds = skeleton_fps_seeds(
            graph, sub_pts, contracted, seed_count, candidate_idx=cand,
        )
        if len(fps_seeds) < 2:
            raise RuntimeError(f"骨架种子仅 {len(fps_seeds)} 个")
        if int(lloyd_iter) > 0:
            _seeds_final, labels = lloyd_relax_geodesic(
                graph, sub_pts, list(fps_seeds), n_iter=int(lloyd_iter),
            )
        else:
            labels = geodesic_voronoi_assign(graph, fps_seeds)
        labels = enforce_per_label_point_connectivity(graph, labels)
    except Exception as exc:  # noqa: BLE001
        skel_error = str(exc)
        LOGGER.warning("骨架分块失败 (%s)，回退 geodesic_fps", exc)
        fb = split_points_by_geodesic_fps(
            points, point_ids, seed_count,
            n_neighbors=n_neighbors, lloyd_iter=int(lloyd_iter), random_state=random_state,
        )
        dbg = dict(fb.debug or {})
        dbg["fallback"] = "geodesic_fps_after_skeleton_error"
        dbg["skeleton_error"] = skel_error
        return PointSubSplit(
            sub_point_groups=fb.sub_point_groups,
            sub_labels=fb.sub_labels,
            num_clusters=fb.num_clusters,
            debug=dbg,
        )

    out = _groups_from_labels(ids, labels, strategy=strategy)
    out.debug.update({
        "contraction_iters": int(contraction_iters),
        "lloyd_iter": int(lloyd_iter),
        "skeleton_error": skel_error,
        "num_skeleton_candidates": int(len(cand)),
        "skeleton_seed_idx": fps_seeds,
    })
    return out


def split_points_by_strategy(
    points: np.ndarray,
    point_ids: Sequence[int],
    seed_count: int,
    strategy: str,
    *,
    lambda_smooth: float = 1.0,
    lambda_concave: float = 0.85,
    scalar_weight: float = 0.35,
    geodesic_weight: Optional[float] = None,
    curvature_weight: Optional[float] = None,
    concavity_weight: Optional[float] = None,
    normal_weight: float = 1.0,
    lloyd_iter: int = 0,
    num_rays: int = 16,
    cone_angle_deg: float = 60.0,
    contraction_iters: int = 5,
    graph_vertex_budget: Optional[int] = None,
    point_scalars: Optional[np.ndarray] = None,
    n_neighbors: int = DEFAULT_KNN,
    enforce_connectivity: bool = True,
    precomputed_eigvecs: Optional[np.ndarray] = None,
    random_state: int = 0,
) -> PointSubSplit:
    """按策略名分块（与 mesh ``split_block_by_strategy`` 参数对齐）。"""
    _ = geodesic_weight, graph_vertex_budget
    strat = str(strategy or "geodesic_fps").strip().lower()
    if strat == "graph_cut":
        LOGGER.info("graph_cut 已禁用，改用 spectral")
        strat = "spectral"
    if strat not in VALID_POINT_STRATEGIES:
        LOGGER.warning("未知点云分块策略 %r，回退 geodesic_fps", strategy)
        strat = "geodesic_fps"

    if strat == "spectral":
        return split_points_by_spectral(
            points, point_ids, seed_count,
            point_scalars=point_scalars,
            scalar_weight=scalar_weight,
            curvature_weight=curvature_weight,
            concavity_weight=concavity_weight,
            normal_weight=normal_weight,
            n_neighbors=n_neighbors,
            enforce_connectivity=enforce_connectivity,
            precomputed_eigvecs=precomputed_eigvecs,
            random_state=random_state,
        )
    if strat == "cvt":
        return split_points_by_cvt(
            points, point_ids, seed_count,
            lloyd_iter=lloyd_iter if lloyd_iter > 0 else 4,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
    if strat == "sdf":
        return split_points_by_sdf(
            points, point_ids, seed_count,
            num_rays=num_rays,
            cone_angle_deg=cone_angle_deg,
            scalar_weight=scalar_weight,
            n_neighbors=n_neighbors,
            enforce_connectivity=enforce_connectivity,
            random_state=random_state,
        )
    if strat == "skeleton":
        return split_points_by_skeleton(
            points, point_ids, seed_count,
            contraction_iters=contraction_iters,
            lloyd_iter=lloyd_iter,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
    return split_points_by_geodesic_fps(
        points, point_ids, seed_count,
        n_neighbors=n_neighbors,
        lloyd_iter=lloyd_iter,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# VLM 选策略（点云版，对齐 mesh geometry.seed_split.ask_split_strategy）
# ---------------------------------------------------------------------------


def clamp_point_split_params(strategy: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """把 VLM 返回的点云分块参数裁剪到安全范围（与 mesh _clamp_split_params 对齐）。"""
    out: Dict[str, Any] = {}
    strat = str(strategy or "geodesic_fps").strip().lower()
    params = params or {}
    if strat == "graph_cut":
        out["lambda_smooth"] = float(np.clip(float(params.get("lambda_smooth", 1.0)), 0.1, 3.0))
        out["lambda_concave"] = float(np.clip(float(params.get("lambda_concave", 0.85)), 0.0, 2.0))
    elif strat == "spectral":
        out["curvature_weight"] = float(
            np.clip(float(params.get("curvature_weight", DEFAULT_CURVATURE_WEIGHT)), 0.0, 2.0)
        )
        out["concavity_weight"] = float(
            np.clip(float(params.get("concavity_weight", 0.0)), 0.0, 2.0)
        )
        if "normal_weight" in params:
            out["normal_weight"] = float(np.clip(float(params["normal_weight"]), 0.0, 2.0))
        if "scalar_weight" in params:
            out["scalar_weight"] = float(np.clip(float(params["scalar_weight"]), 0.0, 1.0))
    elif strat == "cvt":
        out["lloyd_iter"] = int(np.clip(int(params.get("lloyd_iter", 4)), 2, 12))
    elif strat == "sdf":
        out["num_rays"] = int(np.clip(int(params.get("num_rays", 8)), 4, 24))
        out["cone_angle_deg"] = float(np.clip(float(params.get("cone_angle_deg", 60.0)), 15.0, 120.0))
    elif strat == "skeleton":
        out["contraction_iters"] = int(np.clip(int(params.get("contraction_iters", 5)), 2, 12))
        out["lloyd_iter"] = int(np.clip(int(params.get("lloyd_iter", 0)), 0, 12))
    elif strat == "geodesic_fps":
        out["lloyd_iter"] = int(np.clip(int(params.get("lloyd_iter", 0)), 0, 12))
    return out


def default_point_split_params(strategy: str) -> Dict[str, Any]:
    """各点云策略的默认参数（无 VLM 时用）。"""
    return clamp_point_split_params(strategy, {})


def ask_point_split_strategy(
    client: Any,
    image_path: "str | Path | None",
    *,
    semantic_context: str = "",
    seed_count: int = 2,
    covered_parts: Optional[List[str]] = None,
    semantic_parts: Optional[List[str]] = None,
    extra_parts: Optional[List[str]] = None,
    matched_parts: Optional[List[str]] = None,
    block_points: int = 0,
    split_stage: str = "main",
    strategy_hint: str = "",
    fallback: str = "spectral",
    model: Optional[str] = None,
) -> "Any":
    """让 VLM 看分块高亮图，选点云分块策略 + 参数（曲率指标 / 凸包指标等）。

    返回 ``geometry.seed_split.SplitStrategyResponse``（复用同一数据类）。
    """
    from geometry.seed_split import SplitStrategyResponse
    from vlm.client import parse_json_response
    from vlm.prompts import build_split_strategy_prompt

    prompt = build_split_strategy_prompt(
        semantic_context=semantic_context,
        seed_count=seed_count,
        covered_parts=covered_parts,
        semantic_parts=semantic_parts,
        extra_parts=extra_parts,
        matched_parts=matched_parts,
        block_faces=int(block_points),
        split_stage=split_stage,
        strategy_hint=strategy_hint,
        media_type="pointcloud",
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
        LOGGER.warning("VLM 点云分块策略查询失败: %s", exc)
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
        LOGGER.warning("VLM 点云分块策略响应解析失败: %s", text[:200])
        return SplitStrategyResponse(
            strategy=fallback,
            reason=text.strip()[:200],
            raw_text=text,
            error=str(exc),
            raw_response=response.raw,
        )

    strategy = raw_strat if raw_strat in VALID_POINT_STRATEGIES else fallback
    params = clamp_point_split_params(strategy, raw_params)
    return SplitStrategyResponse(
        strategy=strategy,
        reason=reason,
        params=params,
        raw_text=text,
        raw_response=response.raw,
    )
