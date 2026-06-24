"""
最终分块「几何后处理」—— VLM 指标驱动的语义掩码清洗（点云 / mesh 通用）。

主循环（``pointcloud_pipeline`` / ``pipeline``）跑完后，最终 ``full`` 选区往往带有
几何瑕疵：边缘锯齿、孤立小岛、离群点、内部空洞、把相邻部件「兜」进凸包等。本模块
对「最终选区」做一轮 **几何指标体检 + VLM 决策 + 形态学清洗**：

1. ``compute_selection_metrics``：对 (centroids, selected_mask) 算一组与媒介无关的
   几何质量指标——
     - 凸包：hull_volume / hull_area / **solidity**（选区点数 / 落在凸包内的总点数，
       越低说明把别的部件兜进来了或很空）；
     - 边缘平滑度：boundary_ratio（边界点占比，越高越薄/越碎）、boundary_roughness
       （边界点「邻域选中比」的离散度，越大越锯齿）、edge_smoothness（综合，越大越光滑）；
     - 曲率：选区平均曲率、边界平均曲率（PCA surface variation）；
     - 碎片：连通分量数、最大分量占比、小岛数与占比；
     - 空洞：被选区包住的未选点数（hole_fraction）；
     - 离群：到最近同类点距离异常大的点占比（outlier_fraction）。
2. ``ask_postprocess_ops``：把指标 JSON + 一张高亮渲染图喂给 VLM，让它从一组**纯几何
   清洗算子**里挑要执行哪些（带参数），或判定「已足够干净」。
3. ``apply_ops``：在元素的 kNN 图上执行形态学清洗算子（开运算/闭运算去锯齿、去小岛、
   去离群、填洞、保留最大连通块），返回优化后的 mask。

「与媒介无关」：只依赖元素质心坐标 ``centroids (N,3)`` 与布尔选区 ``mask (N,)``。
  - 点云：centroids = 点坐标，元素 = point_id；
  - mesh ：centroids = 面质心 = vertices[faces].mean(1)，元素 = face_id。
所有 VLM / 渲染失败都不阻断主流程，安全回退到原始 mask。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.pointcloud_cluster import (
    DEFAULT_KNN,
    build_point_knn_graph,
    compute_point_normals_and_curvature,
)

LOGGER = logging.getLogger("pipeline.geometry_postprocess")

# VLM 可选的几何清洗算子名（apply_ops 支持的全集）。
VALID_POSTPROCESS_OPS: Tuple[str, ...] = (
    "remove_small_components",
    "keep_largest_component",
    "remove_outliers",
    "smooth_boundary",
    "fill_holes",
)


# ---------------------------------------------------------------------------
# 邻接 / 工具
# ---------------------------------------------------------------------------


def build_neighbor_index(centroids: np.ndarray, n_neighbors: int = DEFAULT_KNN) -> np.ndarray:
    """对所有元素质心建 kNN 邻居索引 (N, k)（不含自身）。

    形态学算子（膨胀/腐蚀/填洞/边界识别）都在这张「全元素」邻居表上做，向量化、可复用。
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(centroids, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64)
    k = int(min(max(n_neighbors + 1, 2), n))
    tree = cKDTree(pts)
    _dist, idx = tree.query(pts, k=k)
    idx = np.atleast_2d(idx)
    if idx.shape[1] >= 2:
        # 第 0 列通常是自身，去掉
        idx = idx[:, 1:]
    return idx.astype(np.int64)


def _neighbor_selected_ratio(mask: np.ndarray, nbr_idx: np.ndarray) -> np.ndarray:
    """每个元素的「kNN 邻居被选中比例」（0~1），向量化。"""
    if nbr_idx.size == 0:
        return np.zeros(len(mask), dtype=np.float64)
    return mask[nbr_idx].mean(axis=1)


def _selected_components(
    centroids: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """在「仅选区」子云的 kNN 图上求连通分量。

    返回 (selected_idx, comp_labels)：selected_idx 是被选元素的全局下标，
    comp_labels 与之对齐，给出每个选中元素的分量号。无选中元素时返回空数组。
    """
    from scipy.sparse.csgraph import connected_components

    sel_idx = np.flatnonzero(mask)
    if sel_idx.size <= 1:
        return sel_idx, np.zeros(sel_idx.size, dtype=np.int64)
    sub = np.asarray(centroids, dtype=np.float64)[sel_idx]
    graph = build_point_knn_graph(sub, n_neighbors=n_neighbors)
    _ncc, cc = connected_components(graph, directed=False)
    return sel_idx, cc.astype(np.int64)


@dataclass
class ComponentInfo:
    """选区里一个连通分量的信息（rank 0 = 最大块）。"""

    rank: int
    size: int
    fraction: float          # size / num_selected
    indices: np.ndarray      # 该分量元素的全局下标 (M,)


def analyze_selected_components(
    centroids: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = DEFAULT_KNN,
) -> List[ComponentInfo]:
    """把选区拆成连通分量，按大小降序返回每块的 (rank, size, fraction, 全局下标)。"""
    mask = np.asarray(mask, dtype=bool)
    sel_idx, comp = _selected_components(centroids, mask, n_neighbors)
    if comp.size == 0:
        return []
    num_selected = int(sel_idx.size)
    ids, sizes = np.unique(comp, return_counts=True)
    order = np.argsort(sizes)[::-1]
    out: List[ComponentInfo] = []
    for rank, j in enumerate(order):
        cid = int(ids[j])
        member = sel_idx[comp == cid]
        out.append(
            ComponentInfo(
                rank=rank,
                size=int(sizes[j]),
                fraction=float(int(sizes[j]) / num_selected) if num_selected else 0.0,
                indices=member.astype(np.int64),
            )
        )
    return out


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


@dataclass
class SelectionMetrics:
    num_total: int
    num_selected: int
    selected_fraction: float
    num_components: int
    largest_component_fraction: float
    second_component_fraction: float
    num_small_islands: int
    island_point_fraction: float
    boundary_ratio: float
    boundary_roughness: float
    edge_smoothness: float
    hole_fraction: float
    outlier_fraction: float
    solidity: float
    hull_volume: float
    hull_area: float
    mean_curvature: float
    boundary_mean_curvature: float

    def to_dict(self) -> Dict[str, Any]:
        def r(x: float, nd: int = 4) -> float:
            try:
                return round(float(x), nd)
            except (TypeError, ValueError):
                return 0.0

        return {
            "num_total": int(self.num_total),
            "num_selected": int(self.num_selected),
            "selected_fraction": r(self.selected_fraction),
            "num_components": int(self.num_components),
            "largest_component_fraction": r(self.largest_component_fraction),
            "second_component_fraction": r(self.second_component_fraction),
            "num_small_islands": int(self.num_small_islands),
            "island_point_fraction": r(self.island_point_fraction),
            "boundary_ratio": r(self.boundary_ratio),
            "boundary_roughness": r(self.boundary_roughness),
            "edge_smoothness": r(self.edge_smoothness),
            "hole_fraction": r(self.hole_fraction),
            "outlier_fraction": r(self.outlier_fraction),
            "solidity": r(self.solidity),
            "hull_volume": r(self.hull_volume, 6),
            "hull_area": r(self.hull_area, 6),
            "mean_curvature": r(self.mean_curvature),
            "boundary_mean_curvature": r(self.boundary_mean_curvature),
        }


def _convex_hull_metrics(
    centroids: np.ndarray,
    sel_idx: np.ndarray,
) -> Tuple[float, float, float]:
    """选区凸包的 (solidity, volume, area)。

    solidity = 选区元素数 / 落在选区凸包内的全部元素数；越低 = 凸包把别的部件兜了进来
    （选区不够「实」/有大凹陷），越接近 1 = 选区充满其凸包。数值/退化时返回保守值。
    """
    try:
        from scipy.spatial import ConvexHull, Delaunay
    except ImportError:
        return 1.0, 0.0, 0.0

    sub = np.asarray(centroids, dtype=np.float64)[sel_idx]
    if len(sub) < 4:
        return 1.0, 0.0, 0.0
    # 退化（共面/共线）时凸包会抛 QhullError
    try:
        hull = ConvexHull(sub)
    except Exception:  # noqa: BLE001 — QhullError 等
        return 1.0, 0.0, 0.0

    volume = float(hull.volume)
    area = float(hull.area)
    try:
        tri = Delaunay(sub[hull.vertices])
        inside = tri.find_simplex(np.asarray(centroids, dtype=np.float64)) >= 0
        n_inside = int(np.count_nonzero(inside))
        solidity = float(len(sel_idx)) / float(max(n_inside, 1))
    except Exception:  # noqa: BLE001
        solidity = 1.0
    return float(np.clip(solidity, 0.0, 1.0)), volume, area


def compute_selection_metrics(
    centroids: np.ndarray,
    mask: np.ndarray,
    *,
    n_neighbors: int = DEFAULT_KNN,
    nbr_idx: Optional[np.ndarray] = None,
    curvature: Optional[np.ndarray] = None,
    island_fraction: float = 0.1,
    hole_threshold: float = 0.7,
    outlier_std_ratio: float = 3.0,
) -> SelectionMetrics:
    """计算最终选区的几何质量指标（凸包 / 边缘平滑度 / 曲率 / 碎片 / 空洞 / 离群）。"""
    centroids = np.asarray(centroids, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    n = len(centroids)
    sel_idx = np.flatnonzero(mask)
    num_selected = int(sel_idx.size)
    empty = SelectionMetrics(
        num_total=n, num_selected=num_selected,
        selected_fraction=(num_selected / n if n else 0.0),
        num_components=0, largest_component_fraction=0.0,
        second_component_fraction=0.0,
        num_small_islands=0, island_point_fraction=0.0,
        boundary_ratio=0.0, boundary_roughness=0.0, edge_smoothness=1.0,
        hole_fraction=0.0, outlier_fraction=0.0,
        solidity=1.0, hull_volume=0.0, hull_area=0.0,
        mean_curvature=0.0, boundary_mean_curvature=0.0,
    )
    if num_selected == 0 or n == 0:
        return empty

    if nbr_idx is None:
        nbr_idx = build_neighbor_index(centroids, n_neighbors=n_neighbors)
    if curvature is None:
        _normals, curvature = compute_point_normals_and_curvature(
            centroids, n_neighbors=n_neighbors
        )
    curvature = np.asarray(curvature, dtype=np.float64)

    ratio = _neighbor_selected_ratio(mask, nbr_idx)

    # ---- 边界 / 边缘平滑度 ----
    boundary = mask & (ratio < 0.999)
    boundary_count = int(np.count_nonzero(boundary))
    boundary_ratio = boundary_count / num_selected
    if boundary_count > 0:
        # 边界点的「邻域选中比」离散度：锯齿边界 ratio 起伏大 → roughness 大
        boundary_roughness = float(np.std(ratio[boundary]))
    else:
        boundary_roughness = 0.0
    # 综合边缘平滑度：边界越少、起伏越小越光滑（归一到 ~[0,1]）
    edge_smoothness = float(np.clip(1.0 - 0.5 * boundary_ratio - boundary_roughness, 0.0, 1.0))

    # ---- 空洞：未选但被选区包住（邻域选中比高）----
    holes = (~mask) & (ratio >= hole_threshold)
    hole_fraction = float(np.count_nonzero(holes)) / num_selected

    # ---- 连通分量 / 小岛 ----
    s_idx, comp = _selected_components(centroids, mask, n_neighbors)
    if comp.size:
        _ids, sizes = np.unique(comp, return_counts=True)
        sizes = np.sort(sizes)[::-1]
        num_components = int(len(sizes))
        largest = int(sizes[0])
        largest_component_fraction = largest / num_selected
        second_component_fraction = (int(sizes[1]) / num_selected) if len(sizes) > 1 else 0.0
        island_thresh = max(1, int(island_fraction * largest))
        island_mask = sizes < island_thresh
        num_small_islands = int(np.count_nonzero(island_mask))
        island_point_fraction = float(sizes[island_mask].sum()) / num_selected
    else:
        num_components = 0
        largest_component_fraction = 0.0
        second_component_fraction = 0.0
        num_small_islands = 0
        island_point_fraction = 0.0

    # ---- 离群点：到最近同类点距离异常 ----
    outlier_fraction = 0.0
    if num_selected >= 4:
        try:
            from scipy.spatial import cKDTree

            sub = centroids[sel_idx]
            tree = cKDTree(sub)
            d, _ = tree.query(sub, k=2)
            nn = d[:, 1]
            med = float(np.median(nn))
            std = float(np.std(nn))
            thr = med + outlier_std_ratio * std
            outlier_fraction = float(np.count_nonzero(nn > thr)) / num_selected
        except (ImportError, ValueError):
            outlier_fraction = 0.0

    # ---- 凸包 ----
    solidity, hull_volume, hull_area = _convex_hull_metrics(centroids, sel_idx)

    # ---- 曲率 ----
    mean_curv = float(np.mean(curvature[sel_idx])) if num_selected else 0.0
    if boundary_count > 0:
        boundary_mean_curv = float(np.mean(curvature[boundary]))
    else:
        boundary_mean_curv = mean_curv

    return SelectionMetrics(
        num_total=n,
        num_selected=num_selected,
        selected_fraction=num_selected / n,
        num_components=num_components,
        largest_component_fraction=largest_component_fraction,
        second_component_fraction=second_component_fraction,
        num_small_islands=num_small_islands,
        island_point_fraction=island_point_fraction,
        boundary_ratio=boundary_ratio,
        boundary_roughness=boundary_roughness,
        edge_smoothness=edge_smoothness,
        hole_fraction=hole_fraction,
        outlier_fraction=outlier_fraction,
        solidity=solidity,
        hull_volume=hull_volume,
        hull_area=hull_area,
        mean_curvature=mean_curv,
        boundary_mean_curvature=boundary_mean_curv,
    )


# ---------------------------------------------------------------------------
# 几何清洗算子（在 kNN 图上做形态学）
# ---------------------------------------------------------------------------


def op_keep_largest_component(
    centroids: np.ndarray, mask: np.ndarray, n_neighbors: int
) -> np.ndarray:
    sel_idx, comp = _selected_components(centroids, mask, n_neighbors)
    if comp.size == 0:
        return mask.copy()
    ids, sizes = np.unique(comp, return_counts=True)
    keep = int(ids[int(np.argmax(sizes))])
    out = np.zeros_like(mask)
    out[sel_idx[comp == keep]] = True
    return out


def op_keep_components(
    centroids: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int,
    keep_ranks: Sequence[int],
) -> np.ndarray:
    """只保留指定 rank 的连通分量（rank 0 = 最大块），其余置 False。

    供「两个都很大的连通域」经 VLM 判定后，丢弃不属于该语义的那一块用。
    keep_ranks 为空或越界全部丢弃时回退原 mask（不做改动）。
    """
    comps = analyze_selected_components(centroids, mask, n_neighbors)
    if not comps:
        return mask.copy()
    keep = {int(r) for r in keep_ranks}
    out = np.zeros_like(mask)
    for comp in comps:
        if comp.rank in keep:
            out[comp.indices] = True
    if not out.any():
        return mask.copy()
    return out


def op_remove_small_components(
    centroids: np.ndarray, mask: np.ndarray, n_neighbors: int, *, min_fraction: float = 0.1
) -> np.ndarray:
    sel_idx, comp = _selected_components(centroids, mask, n_neighbors)
    if comp.size == 0:
        return mask.copy()
    ids, sizes = np.unique(comp, return_counts=True)
    largest = int(sizes.max())
    thr = max(1, int(float(min_fraction) * largest))
    out = np.zeros_like(mask)
    for cid, sz in zip(ids, sizes):
        if sz >= thr:
            out[sel_idx[comp == cid]] = True
    # 全被判为小块时（理论上不会，largest>=thr）兜底保留最大块
    if not out.any():
        return op_keep_largest_component(centroids, mask, n_neighbors)
    return out


def op_remove_outliers(
    centroids: np.ndarray, mask: np.ndarray, *, std_ratio: float = 3.0
) -> np.ndarray:
    sel_idx = np.flatnonzero(mask)
    if sel_idx.size < 4:
        return mask.copy()
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return mask.copy()
    sub = np.asarray(centroids, dtype=np.float64)[sel_idx]
    tree = cKDTree(sub)
    d, _ = tree.query(sub, k=2)
    nn = d[:, 1]
    med = float(np.median(nn))
    std = float(np.std(nn))
    thr = med + float(std_ratio) * std
    out = mask.copy()
    out[sel_idx[nn > thr]] = False
    return out


def op_fill_holes(
    mask: np.ndarray, nbr_idx: np.ndarray, *, threshold: float = 0.7, iters: int = 2
) -> np.ndarray:
    out = mask.copy()
    for _ in range(max(1, int(iters))):
        ratio = _neighbor_selected_ratio(out, nbr_idx)
        add = (~out) & (ratio >= float(threshold))
        if not add.any():
            break
        out[add] = True
    return out


def _erode(mask: np.ndarray, nbr_idx: np.ndarray, keep_thresh: float) -> np.ndarray:
    ratio = _neighbor_selected_ratio(mask, nbr_idx)
    out = mask.copy()
    out[mask & (ratio < keep_thresh)] = False
    return out


def _dilate(mask: np.ndarray, nbr_idx: np.ndarray, add_thresh: float) -> np.ndarray:
    ratio = _neighbor_selected_ratio(mask, nbr_idx)
    out = mask.copy()
    out[(~mask) & (ratio >= add_thresh)] = True
    return out


def op_smooth_boundary(
    mask: np.ndarray,
    nbr_idx: np.ndarray,
    *,
    mode: str = "both",
    iters: int = 1,
    keep_thresh: float = 0.4,
    add_thresh: float = 0.6,
) -> np.ndarray:
    """kNN 图上的形态学边界平滑。

    - ``open`` = 先腐蚀后膨胀：削掉锯齿尖刺 / 离群细丝；
    - ``close`` = 先膨胀后腐蚀：补小缺口 / 平滑凹陷；
    - ``both`` = 先 close 再 open，整体磨平边界。
    """
    out = mask.copy()
    iters = max(1, int(iters))
    mode = str(mode or "both").strip().lower()

    def _open(m: np.ndarray) -> np.ndarray:
        for _ in range(iters):
            m = _erode(m, nbr_idx, keep_thresh)
        for _ in range(iters):
            m = _dilate(m, nbr_idx, add_thresh)
        return m

    def _close(m: np.ndarray) -> np.ndarray:
        for _ in range(iters):
            m = _dilate(m, nbr_idx, add_thresh)
        for _ in range(iters):
            m = _erode(m, nbr_idx, keep_thresh)
        return m

    if mode == "open":
        out = _open(out)
    elif mode == "close":
        out = _close(out)
    else:
        out = _open(_close(out))
    # 形态学把选区清空时（极端参数）放弃，保留原 mask
    if not out.any():
        return mask.copy()
    return out


# ---------------------------------------------------------------------------
# 参数裁剪 + 算子调度
# ---------------------------------------------------------------------------


def clamp_op_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = params or {}
    out: Dict[str, Any] = {}
    if name == "remove_small_components":
        out["min_fraction"] = float(np.clip(float(params.get("min_fraction", 0.1)), 0.01, 0.5))
    elif name == "remove_outliers":
        out["std_ratio"] = float(np.clip(float(params.get("std_ratio", 3.0)), 1.5, 6.0))
    elif name == "smooth_boundary":
        mode = str(params.get("mode", "both")).strip().lower()
        out["mode"] = mode if mode in ("open", "close", "both") else "both"
        out["iters"] = int(np.clip(int(params.get("iters", 1)), 1, 3))
    elif name == "fill_holes":
        out["iters"] = int(np.clip(int(params.get("iters", 2)), 1, 4))
        out["threshold"] = float(np.clip(float(params.get("threshold", 0.7)), 0.5, 0.95))
    return out


def apply_ops(
    centroids: np.ndarray,
    mask: np.ndarray,
    ops: Sequence[Dict[str, Any]],
    *,
    n_neighbors: int = DEFAULT_KNN,
    nbr_idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """按顺序执行 VLM 选定的几何清洗算子，返回 (优化后 mask, 每步效果日志)。"""
    centroids = np.asarray(centroids, dtype=np.float64)
    out = np.asarray(mask, dtype=bool).copy()
    if nbr_idx is None:
        nbr_idx = build_neighbor_index(centroids, n_neighbors=n_neighbors)

    log: List[Dict[str, Any]] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        name = str(op.get("name") or op.get("op") or "").strip().lower()
        if name in ("", "none"):
            continue
        if name not in VALID_POSTPROCESS_OPS:
            LOGGER.warning("跳过未知后处理算子: %r", name)
            continue
        params = clamp_op_params(name, op.get("params") or {})
        before = int(np.count_nonzero(out))
        try:
            if name == "keep_largest_component":
                out = op_keep_largest_component(centroids, out, n_neighbors)
            elif name == "remove_small_components":
                out = op_remove_small_components(
                    centroids, out, n_neighbors, min_fraction=params["min_fraction"]
                )
            elif name == "remove_outliers":
                out = op_remove_outliers(centroids, out, std_ratio=params["std_ratio"])
            elif name == "fill_holes":
                out = op_fill_holes(
                    out, nbr_idx, threshold=params["threshold"], iters=params["iters"]
                )
            elif name == "smooth_boundary":
                out = op_smooth_boundary(
                    out, nbr_idx,
                    mode=params["mode"], iters=params["iters"],
                )
        except Exception as exc:  # noqa: BLE001 — 单个算子失败不影响其余
            LOGGER.warning("后处理算子 %s 执行失败: %s", name, exc)
            log.append({"name": name, "params": params, "error": str(exc)})
            continue
        after = int(np.count_nonzero(out))
        log.append({"name": name, "params": params, "before": before, "after": after,
                    "delta": after - before})
        LOGGER.info("postprocess op %s: %d -> %d (%+d)", name, before, after, after - before)
    return out, log


# ---------------------------------------------------------------------------
# VLM 决策
# ---------------------------------------------------------------------------


@dataclass
class PostProcessDecision:
    ops: List[Dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    raw_text: str = ""
    error: Optional[str] = None


def ask_postprocess_ops(
    client: Any,
    image_path: "str | Path | None",
    semantic: str,
    metrics: Dict[str, Any],
    *,
    media_type: str = "pointcloud",
    n_views: int = 12,
    model: Optional[str] = None,
) -> PostProcessDecision:
    """把几何指标 + 高亮图喂给 VLM，让它挑要执行的几何清洗算子（带参数）。

    任何失败都返回空 ops（= 不改动），不阻断主流程。
    """
    from vlm.client import parse_json_response
    from vlm.prompts import build_postprocess_prompt

    prompt = build_postprocess_prompt(
        semantic,
        metrics=metrics,
        media_type="pointcloud" if media_type == "pointcloud" else "mesh",
        n_views=n_views,
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
        LOGGER.warning("VLM 后处理决策失败: %s", exc)
        return PostProcessDecision(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except (ValueError, TypeError) as exc:
        LOGGER.warning("VLM 后处理响应解析失败: %s", text[:200])
        return PostProcessDecision(reason=text.strip()[:200], raw_text=text, error=str(exc))

    raw_ops = payload.get("operations") or payload.get("ops") or []
    ops: List[Dict[str, Any]] = []
    if isinstance(raw_ops, list):
        for item in raw_ops:
            if isinstance(item, str):
                name = item.strip().lower()
                if name in VALID_POSTPROCESS_OPS:
                    ops.append({"name": name, "params": clamp_op_params(name, {})})
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("op") or "").strip().lower()
                if name in VALID_POSTPROCESS_OPS:
                    ops.append({"name": name, "params": clamp_op_params(name, item.get("params") or {})})
    reason = str(payload.get("reason") or "")
    return PostProcessDecision(ops=ops, reason=reason, raw_text=text)


# ---------------------------------------------------------------------------
# 大连通域拆分判定（两个都很大的连通域 → 结合语义 + 渲染请 VLM 判断）
# ---------------------------------------------------------------------------


@dataclass
class ComponentSplitDecision:
    """VLM 对「多个大连通域」的去留判定。

    - ``keep_ranks``：判定属于该语义、应保留的分量 rank（0 = 最大块）；
    - ``split``：True 表示确实要丢弃部分分量（keep_ranks 是全部分量的真子集）。
    keep_ranks 为空时视为「全部保留」（不改动）。
    """

    keep_ranks: List[int] = field(default_factory=list)
    split: bool = False
    reason: str = ""
    raw_text: str = ""
    error: Optional[str] = None


def ask_component_split_decision(
    client: Any,
    image_path: "str | Path | None",
    semantic: str,
    components: Sequence[ComponentInfo],
    metrics: Dict[str, Any],
    *,
    media_type: str = "pointcloud",
    n_views: int = 12,
    model: Optional[str] = None,
) -> ComponentSplitDecision:
    """选区分裂成多个大连通域时，把分色渲染图 + 语义 + 各块占比喂给 VLM，
    让它判断哪些连通域真正属于该语义（保留），哪些是溢出到别的部件（丢弃）。

    任何失败都返回空判定（= 全部保留，不改动），不阻断主流程。
    """
    from vlm.client import parse_json_response
    from vlm.prompts import build_component_split_prompt

    comp_summary = [
        {"id": int(c.rank), "fraction": round(float(c.fraction), 4), "size": int(c.size)}
        for c in components
    ]
    n_comp = len(comp_summary)
    prompt = build_component_split_prompt(
        semantic,
        components=comp_summary,
        metrics=metrics,
        media_type="pointcloud" if media_type == "pointcloud" else "mesh",
        n_views=n_views,
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
        LOGGER.warning("VLM 连通域拆分判定失败: %s", exc)
        return ComponentSplitDecision(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except (ValueError, TypeError) as exc:
        LOGGER.warning("VLM 连通域拆分响应解析失败: %s", text[:200])
        return ComponentSplitDecision(reason=text.strip()[:200], raw_text=text, error=str(exc))

    raw_keep = payload.get("keep_components")
    if raw_keep is None:
        raw_keep = payload.get("keep")
    keep: List[int] = []
    if isinstance(raw_keep, list):
        for v in raw_keep:
            try:
                iv = int(v)
            except (ValueError, TypeError):
                continue
            if 0 <= iv < n_comp:
                keep.append(iv)
    keep = sorted(set(keep))
    # 没给/给全 = 不拆分（全部保留）。
    split = bool(keep) and len(keep) < n_comp
    reason = str(payload.get("reason") or "")
    return ComponentSplitDecision(keep_ranks=keep, split=split, reason=reason, raw_text=text)


__all__ = [
    "VALID_POSTPROCESS_OPS",
    "SelectionMetrics",
    "PostProcessDecision",
    "ComponentInfo",
    "ComponentSplitDecision",
    "build_neighbor_index",
    "compute_selection_metrics",
    "analyze_selected_components",
    "apply_ops",
    "clamp_op_params",
    "ask_postprocess_ops",
    "ask_component_split_decision",
    "op_keep_largest_component",
    "op_keep_components",
    "op_remove_small_components",
    "op_remove_outliers",
    "op_fill_holes",
    "op_smooth_boundary",
]
