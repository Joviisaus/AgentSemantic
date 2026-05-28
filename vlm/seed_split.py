"""
分块的"种子分裂"：
- 询问 VLM 在这个分块内需要多少子块（seed_count ≥ 2）
- 对该分块的子 mesh 做谱聚类，得到子分块列表
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spectral_cluster import spectral_cluster_vertices, vertex_labels_to_face_labels

from .client import VLMClient, parse_json_response
from .prompts import build_seed_prompt

LOGGER = logging.getLogger("vlm.seed_split")


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
) -> SeedCountResponse:
    """让 VLM 给出 seed_count，失败时回退到 fallback。"""
    prompt = build_seed_prompt(semantic, max_seeds=max_seeds)
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
