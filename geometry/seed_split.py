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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from geometry.spectral_cluster import (
    spectral_cluster_vertices,
    vertex_labels_to_face_labels,
)

from vlm.client import VLMClient, parse_json_response
from vlm.prompts import build_seed_prompt

LOGGER = logging.getLogger("geometry.seed_split")


@dataclass
class SeedCountResponse:
    seed_count: int
    reason: str = ""
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


# ---------------------------------------------------------------------------
# 子 mesh 谱聚类
# ---------------------------------------------------------------------------


@dataclass
class SubBlockSplit:
    sub_face_groups: List[List[int]]  # 每个子块对应的全局 face_ids
    sub_labels: np.ndarray            # (M,) 子标签，按 face_ids 顺序
    num_clusters: int
    debug: Dict[str, Any] = field(default_factory=dict)


def _extract_submesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从原 mesh 抽出仅包含 face_ids 的子 mesh，并把顶点 / 面索引重新编号。
    返回 (sub_vertices, sub_faces, global_vertex_index)。
    """
    face_ids_arr = np.asarray(list(face_ids), dtype=np.int64)
    sub_face = faces[face_ids_arr]
    unique_v, inverse = np.unique(sub_face.reshape(-1), return_inverse=True)
    new_faces = inverse.reshape(sub_face.shape).astype(np.int64)
    sub_vertices = vertices[unique_v]
    return sub_vertices, new_faces, unique_v


# ---------------------------------------------------------------------------
# 测地最远点采样 (FPS) + 测地 Voronoi
# ---------------------------------------------------------------------------


def _build_submesh_vertex_graph(
    sub_vertices: np.ndarray, sub_faces: np.ndarray
) -> "Any":
    """从子 mesh 三角形构建顶点邻接图，边权 = 欧氏边长。
    返回 scipy CSR 稀疏矩阵；离群顶点会自动出现在图中（行/列）但无邻居。
    """
    from scipy.sparse import csr_matrix  # 局部导入，避免顶层依赖

    n_v = int(sub_vertices.shape[0])
    if sub_faces.size == 0:
        return csr_matrix((n_v, n_v))

    tri = sub_faces.astype(np.int64)
    a = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    b = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    # 双向边
    rows = np.concatenate([a, b])
    cols = np.concatenate([b, a])
    diff = sub_vertices[rows] - sub_vertices[cols]
    weights = np.linalg.norm(diff, axis=1)
    # 防止零权造成路径退化
    weights = np.maximum(weights, 1e-9)
    return csr_matrix((weights, (rows, cols)), shape=(n_v, n_v))


def _geodesic_fps_seeds(
    graph: "Any",
    sub_vertices: np.ndarray,
    k: int,
    *,
    start_idx: Optional[int] = None,
) -> List[int]:
    """在图上做测地最远点采样，返回 k 个种子顶点索引（按取得顺序）。"""
    from scipy.sparse.csgraph import dijkstra

    n_v = int(graph.shape[0])
    if k <= 0 or n_v == 0:
        return []
    k = min(k, n_v)

    # 起点：默认取离子 mesh 几何中心最远的顶点（更稳，比随机起点能复现）。
    if start_idx is None:
        center = sub_vertices.mean(axis=0)
        start_idx = int(np.argmax(np.linalg.norm(sub_vertices - center, axis=1)))

    seeds: List[int] = [int(start_idx)]
    dist = dijkstra(graph, indices=start_idx, directed=False)
    # 不可达写为 -inf 防止反复挑同一个无穷远点
    dist = np.where(np.isfinite(dist), dist, -np.inf)

    for _ in range(1, k):
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0.0:
            break  # 没有更远的可达顶点
        seeds.append(nxt)
        new_dist = dijkstra(graph, indices=nxt, directed=False)
        new_dist = np.where(np.isfinite(new_dist), new_dist, -np.inf)
        # 取「到任一已有种子的最近距离」，再从中选最远 → 经典 FPS
        dist = np.minimum(dist, new_dist)

    return seeds


def _geodesic_voronoi_assign(graph: "Any", seeds: Sequence[int]) -> np.ndarray:
    """从 K 个种子并行跑 Dijkstra，按最近种子做 Voronoi 划分。"""
    from scipy.sparse.csgraph import dijkstra

    if not seeds:
        return np.zeros(graph.shape[0], dtype=np.int64)

    indices = np.asarray(list(seeds), dtype=np.int64)
    dists = dijkstra(graph, indices=indices, directed=False)  # (k, n_v)
    # 不可达 -> +inf 已是 numpy 默认行为
    return np.argmin(dists, axis=0).astype(np.int64)


def _lloyd_relax_geodesic(
    graph: "Any",
    sub_vertices: np.ndarray,
    seeds: List[int],
    *,
    n_iter: int = 5,
) -> Tuple[List[int], np.ndarray]:
    """
    Lloyd 风格迭代：
      ① Voronoi 划分
      ② 每个簇取「簇内 3D 重心最近的顶点」作为新种子
    重复 n_iter 次直至稳定。FPS 的初始种子常常是几何外凸的 tip，单次 Voronoi 极度不均衡；
    Lloyd 迭代能把种子拉到簇心，让 Voronoi 切片大小相近。

    返回 (final_seeds, final_labels)。
    """
    if not seeds:
        return [], np.zeros(graph.shape[0], dtype=np.int64)

    seeds = list(seeds)
    labels = _geodesic_voronoi_assign(graph, seeds)
    for _ in range(max(1, n_iter)):
        new_seeds: List[int] = []
        changed = False
        for k, cur in enumerate(seeds):
            mask = labels == k
            if not np.any(mask):
                new_seeds.append(cur)
                continue
            cluster_pts = sub_vertices[mask]
            cluster_mean = cluster_pts.mean(axis=0)
            # 在该簇内挑离 3D 重心最近的顶点（限制 mask 内）
            d3 = np.linalg.norm(sub_vertices - cluster_mean, axis=1)
            d3_masked = np.where(mask, d3, np.inf)
            new_seed = int(np.argmin(d3_masked))
            if new_seed != cur:
                changed = True
            new_seeds.append(new_seed)
        if not changed:
            break
        seeds = new_seeds
        labels = _geodesic_voronoi_assign(graph, seeds)
    return seeds, labels


def _build_face_adjacency_csr(faces: np.ndarray) -> sp.csr_matrix:
    """构建面邻接图（共享一条边 = 邻居）。返回 (M, M) 稀疏对称矩阵。"""
    m = int(len(faces))
    if m == 0:
        return sp.csr_matrix((0, 0), dtype=np.int8)
    f = np.asarray(faces, dtype=np.int64)
    a0, a1, a2 = f[:, 0], f[:, 1], f[:, 2]
    fid = np.arange(m, dtype=np.int64)
    edges = np.vstack([
        np.column_stack([np.minimum(a0, a1), np.maximum(a0, a1), fid]),
        np.column_stack([np.minimum(a1, a2), np.maximum(a1, a2), fid]),
        np.column_stack([np.minimum(a2, a0), np.maximum(a2, a0), fid]),
    ])
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    es = edges[order]
    same = (es[1:, 0] == es[:-1, 0]) & (es[1:, 1] == es[:-1, 1])
    if not np.any(same):
        return sp.csr_matrix((m, m), dtype=np.int8)
    src = es[:-1, 2][same]
    dst = es[1:, 2][same]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.int8)
    return sp.csr_matrix((data, (rows, cols)), shape=(m, m))


def enforce_per_label_face_connectivity(
    faces: np.ndarray,
    face_labels: np.ndarray,
    *,
    max_iter: int = 8,
) -> np.ndarray:
    """
    把面标签清洗成「每个 label 只对应一片连通区域」。

    顶点→面投票常常在边界产生孤立小片，或者 KMeans-fallback 输出本身就跨连通片。
    这里：
      ① 在面邻接图上找出每个 label 的所有连通分量
      ② 每个 label 只保留最大的那片
      ③ 其余碎片按「跨 label 邻接边数最多」的邻居 label 重新分配
      ④ 迭代直到稳定（或达到 max_iter）

    返回清洗后的 face_labels（不修改输入）。
    """
    labels = np.asarray(face_labels, dtype=np.int64).copy()
    m = int(len(faces))
    if m <= 1:
        return labels

    adj = _build_face_adjacency_csr(faces)
    if adj.nnz == 0:
        return labels
    indptr = adj.indptr
    indices = adj.indices

    for _it in range(max_iter):
        row_idx = np.repeat(np.arange(m, dtype=np.int64), np.diff(indptr))
        col_idx = indices.astype(np.int64)
        same_label = labels[row_idx] == labels[col_idx]
        if np.any(same_label):
            sub = sp.csr_matrix(
                (
                    np.ones(int(same_label.sum()), dtype=np.int8),
                    (row_idx[same_label], col_idx[same_label]),
                ),
                shape=(m, m),
            )
            _n_cc, cc_labels = connected_components(sub, directed=False)
        else:
            cc_labels = np.arange(m, dtype=np.int64)
        cc_labels = cc_labels.astype(np.int64)

        unique_labs = np.unique(labels)
        problematic: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for lab in unique_labs:
            mask = labels == lab
            if not mask.any():
                continue
            sub_cc = cc_labels[mask]
            cc_ids, cc_sizes = np.unique(sub_cc, return_counts=True)
            if len(cc_ids) <= 1:
                continue
            problematic.append((int(lab), cc_ids, cc_sizes))

        if not problematic:
            break

        reassign = np.full(m, -1, dtype=np.int64)
        for lab, cc_ids, cc_sizes in problematic:
            largest_cc = int(cc_ids[int(np.argmax(cc_sizes))])
            for cc_id in cc_ids:
                cc_id_int = int(cc_id)
                if cc_id_int == largest_cc:
                    continue
                frag_idx = np.where(cc_labels == cc_id_int)[0]
                if frag_idx.size == 0:
                    continue
                neighbor_labs: List[int] = []
                for fi in frag_idx:
                    nbrs = indices[indptr[fi]:indptr[fi + 1]]
                    if nbrs.size == 0:
                        continue
                    out_mask = cc_labels[nbrs] != cc_id_int
                    if not np.any(out_mask):
                        continue
                    neighbor_labs.extend(labels[nbrs[out_mask]].tolist())
                if not neighbor_labs:
                    continue
                vals, counts = np.unique(
                    np.asarray(neighbor_labs, dtype=np.int64), return_counts=True
                )
                new_lab = int(vals[int(np.argmax(counts))])
                reassign[frag_idx] = new_lab

        changed_mask = reassign >= 0
        if not np.any(changed_mask):
            break
        labels[changed_mask] = reassign[changed_mask]

    return labels


def split_block_by_connected_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    *,
    min_component_faces: int = 1,
) -> Optional[List[List[int]]]:
    """
    把 block 拆成多个连通分量；返回每个 CC 对应的 face_id 列表。

    单 CC 时返回 None（调用方继续走 FPS+谱聚类）。
    `min_component_faces`：低于该面数的 CC 会被合并到最大的相邻 CC 里以避免噪声小簇。
    （这里"相邻"按 CC 编号近邻处理，简化实现：直接合到最大 CC。）
    """
    face_ids_list = list(face_ids)
    if len(face_ids_list) <= 1:
        return None

    sub_v, sub_f, _ = _extract_submesh(vertices, faces, face_ids_list)
    if len(sub_v) == 0 or len(sub_f) == 0:
        return None

    try:
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        return None

    graph = _build_submesh_vertex_graph(sub_v, sub_f)
    n_cc, v_labels = connected_components(graph, directed=False)
    if n_cc <= 1:
        return None

    # 顶点 label → 面 label（一个面的 3 顶点必同 CC，因为它们由该面边相连）
    sub_face_labels = vertex_labels_to_face_labels(v_labels, sub_f)
    sub_face_labels_arr = np.asarray(sub_face_labels, dtype=np.int64)

    # 收集每个 CC 的面
    raw_groups: List[List[int]] = []
    for k in range(n_cc):
        idx = np.flatnonzero(sub_face_labels_arr == k)
        if idx.size == 0:
            continue
        raw_groups.append([face_ids_list[i] for i in idx.tolist()])

    if len(raw_groups) <= 1:
        return None

    # 合并过小的 CC 到最大 CC
    raw_groups.sort(key=len, reverse=True)
    if min_component_faces > 1 and len(raw_groups) > 1:
        big = raw_groups[0]
        kept: List[List[int]] = [big]
        for g in raw_groups[1:]:
            if len(g) < min_component_faces:
                big.extend(g)
            else:
                kept.append(g)
        raw_groups = kept

    if len(raw_groups) <= 1:
        return None
    return raw_groups


def split_block_by_geodesic_fps(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    seed_count: int,
    *,
    vertex_scalars: Optional[np.ndarray] = None,  # 保留签名兼容，未使用
    scalar_weight: float = 0.35,                  # 保留签名兼容，未使用
    lloyd_iter: int = 8,
) -> SubBlockSplit:
    """
    限定在分块内做测地划分（不走谱聚类）：
      ① 子 mesh 顶点图（边权 = 欧氏边长）
      ② 测地最远点采样 (FPS) → K 个种子顶点（重心）
      ③ Lloyd 迭代：Voronoi 划分 → 每簇取「簇内 3D 重心最近顶点」作新种子，
         重复 lloyd_iter 次直至稳定。FPS 的初始种子常常落在几何外凸的 tip，
         单次 Voronoi 极度不均衡；Lloyd 把种子拉到簇心，让 Voronoi 切片大小相近。
      ④ 顶点 label → 面 label，再做一次连通性清洗（修补面投票边界孤立小片）。
    """
    face_ids_list = list(face_ids)
    seed_count = max(2, int(seed_count))
    if not face_ids_list:
        return SubBlockSplit(
            sub_face_groups=[], sub_labels=np.zeros(0, dtype=np.int64), num_clusters=0
        )

    sub_v, sub_f, _global_v_idx = _extract_submesh(vertices, faces, face_ids_list)
    if len(sub_v) <= seed_count or len(sub_f) <= 1:
        return SubBlockSplit(
            sub_face_groups=[list(face_ids_list)],
            sub_labels=np.zeros(len(face_ids_list), dtype=np.int64),
            num_clusters=1,
            debug={"reason": "submesh 太小，跳过 FPS 分裂"},
        )

    fps_seeds: List[int] = []
    seeds_final: List[int] = []
    fps_error: Optional[str] = None
    strategy = "geodesic_fps_voronoi_lloyd"
    try:
        graph = _build_submesh_vertex_graph(sub_v, sub_f)
        fps_seeds = _geodesic_fps_seeds(graph, sub_v, seed_count)
        if len(fps_seeds) < 2:
            raise RuntimeError(f"FPS 仅取得 {len(fps_seeds)} 个种子")
        seeds_final, v_labels = _lloyd_relax_geodesic(
            graph, sub_v, list(fps_seeds), n_iter=lloyd_iter
        )
    except Exception as exc:  # noqa: BLE001
        fps_error = str(exc)
        LOGGER.warning("测地 FPS+Voronoi 失败 (%s)，回退坐标 KMeans", exc)
        try:
            from geometry.spectral_cluster import _kmeans  # type: ignore

            v_labels = _kmeans(sub_v.astype(np.float64), seed_count, random_state=0)
            strategy = "coord_kmeans_fallback"
        except Exception as exc2:  # noqa: BLE001
            return SubBlockSplit(
                sub_face_groups=[list(face_ids_list)],
                sub_labels=np.zeros(len(face_ids_list), dtype=np.int64),
                num_clusters=1,
                debug={
                    "error": f"{exc} | {exc2}",
                    "fallback": "all_failed",
                    "fps_error": fps_error,
                },
            )

    sub_face_labels = vertex_labels_to_face_labels(v_labels, sub_f)
    sub_face_labels_arr = np.asarray(sub_face_labels, dtype=np.int64)
    sub_face_labels_arr = enforce_per_label_face_connectivity(sub_f, sub_face_labels_arr)

    used = sorted({int(v) for v in sub_face_labels_arr})
    groups: List[List[int]] = []
    for k in used:
        idx = np.flatnonzero(sub_face_labels_arr == k)
        group = [face_ids_list[i] for i in idx.tolist()]
        if group:
            groups.append(group)

    return SubBlockSplit(
        sub_face_groups=groups,
        sub_labels=sub_face_labels_arr,
        num_clusters=len(groups),
        debug={
            "strategy": strategy,
            "seed_count_requested": seed_count,
            "seed_count_actual": len(used),
            "fps_seed_local_v_idx": fps_seeds,
            "lloyd_seed_local_v_idx": seeds_final,
            "fps_error": fps_error,
            "lloyd_iter": lloyd_iter,
        },
    )


def split_block_by_spectral_seeds(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    seed_count: int,
    *,
    vertex_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
) -> SubBlockSplit:
    """
    在分块内部做一次谱聚类：把 face_ids 所覆盖的子 mesh 切成 seed_count 块。

    返回的 sub_face_groups 与 sub_labels 都按 face_ids 顺序对齐。
    """
    face_ids_list = list(face_ids)
    seed_count = max(2, int(seed_count))
    if not face_ids_list:
        return SubBlockSplit(sub_face_groups=[], sub_labels=np.zeros(0, dtype=np.int64), num_clusters=0)

    sub_v, sub_f, global_v_idx = _extract_submesh(vertices, faces, face_ids_list)
    if len(sub_v) <= seed_count or len(sub_f) <= 1:
        # 没有足够多顶点继续切，直接整体当一块
        sub_labels = np.zeros(len(face_ids_list), dtype=np.int64)
        return SubBlockSplit(
            sub_face_groups=[list(face_ids_list)],
            sub_labels=sub_labels,
            num_clusters=1,
            debug={"reason": "submesh 太小，跳过分裂"},
        )

    sub_vertex_scalars: Optional[np.ndarray] = None
    if vertex_scalars is not None and len(vertex_scalars) == len(vertices):
        sub_vertex_scalars = np.asarray(vertex_scalars, dtype=np.float64)[global_v_idx]

    fallback_debug: Dict[str, Any] = {}
    try:
        vlabels, k_actual, debug = spectral_cluster_vertices(
            sub_v,
            sub_f,
            n_clusters=seed_count,
            vertex_scalars=sub_vertex_scalars,
            scalar_weight=scalar_weight,
            min_clusters=2,
            max_clusters=max(seed_count, 2),
        )
    except (ValueError, ImportError) as exc:
        LOGGER.warning("子谱聚类失败，退回 1 块：%s", exc)
        return SubBlockSplit(
            sub_face_groups=[list(face_ids_list)],
            sub_labels=np.zeros(len(face_ids_list), dtype=np.int64),
            num_clusters=1,
            debug={"error": str(exc)},
        )
    except Exception as exc:  # ARPACK 不收敛 / 数值奇异等
        LOGGER.warning(
            "子谱聚类异常 (%s)，退化为坐标 KMeans seed_count=%d",
            type(exc).__name__,
            seed_count,
        )
        try:
            from geometry.spectral_cluster import _kmeans  # type: ignore
        except ImportError:
            return SubBlockSplit(
                sub_face_groups=[list(face_ids_list)],
                sub_labels=np.zeros(len(face_ids_list), dtype=np.int64),
                num_clusters=1,
                debug={"error": str(exc), "fallback": "kmeans_unavailable"},
            )
        vlabels = _kmeans(sub_v.astype(np.float64), seed_count, random_state=0)
        k_actual = int(seed_count)
        fallback_debug = {
            "spectral_error": f"{type(exc).__name__}: {exc}",
            "fallback": "coordinate_kmeans",
        }
        debug = fallback_debug

    sub_face_labels = vertex_labels_to_face_labels(vlabels, sub_f)
    sub_labels_arr = np.asarray(sub_face_labels, dtype=np.int64)

    groups: List[List[int]] = []
    for k in range(k_actual):
        idx = np.flatnonzero(sub_labels_arr == k)
        group = [face_ids_list[i] for i in idx.tolist()]
        if group:
            groups.append(group)

    return SubBlockSplit(
        sub_face_groups=groups,
        sub_labels=sub_labels_arr,
        num_clusters=len(groups),
        debug=debug,
    )
