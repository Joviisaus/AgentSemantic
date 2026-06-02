"""
一次性分组分裂：把 partial 块**过分割**为 N 个候选小区域，让 VLM 直接选择
把哪些候选区合并到一起（每组带 label = full / partial / none）。

相比 ask_seed_count + 谱聚类二刀切：
  - VLM 不再只给"切几刀"，而是直接给"哪些候选拼成一组、组的语义结论"。
  - 每个 child block 已经带 VLM 判定的 label，可跳过下一轮的 classify 阶段。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.spectral_cluster import (
    spectral_cluster_vertices,
    vertex_labels_to_face_labels,
)

from vlm.client import VLMClient, parse_json_response
from vlm.prompts import LABEL_FULL, LABEL_NONE, LABEL_PARTIAL, build_group_prompt

from rendering.render_component import (
    CANDIDATE_PALETTE,
    render_candidate_panel,
)

from .seed_split import _extract_submesh

LOGGER = logging.getLogger("geometry.grouped_split")


@dataclass
class CandidateGroup:
    group_id: str
    candidate_ids: List[int]            # 1-indexed VLM 编号
    label: str                          # "full" | "partial" | "none"
    confidence: float = 0.0
    description: str = ""
    face_ids: List[int] = field(default_factory=list)  # 该组占据的全局 face_ids


@dataclass
class GroupedSplitResult:
    ok: bool
    n_candidates: int
    panel_path: Optional[Path] = None
    groups: List[CandidateGroup] = field(default_factory=list)
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    reason: str = ""
    raw_response: Dict[str, Any] = field(default_factory=dict)


def _farthest_point_seeds(
    points: np.ndarray, k: int, random_state: int = 0
) -> np.ndarray:
    """对 (N, 3) 点云做最远点采样得到 k 个种子的索引（保证空间分布均匀）。"""
    n = int(points.shape[0])
    if k >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    seeds = [int(rng.integers(0, n))]
    dists = np.linalg.norm(points - points[seeds[0]], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(dists))
        seeds.append(nxt)
        d_new = np.linalg.norm(points - points[nxt], axis=1)
        dists = np.minimum(dists, d_new)
    return np.asarray(seeds, dtype=np.int64)


def _balanced_face_kmeans(
    face_centers: np.ndarray, k: int, *, n_iter: int = 20
) -> np.ndarray:
    """对 face center 做基于 FPS 初始化的 KMeans。
    比谱聚类更均衡，candidate 大小相近，VLM 在过分割面板里更容易区分。
    """
    n = int(face_centers.shape[0])
    if n <= k:
        return np.arange(n, dtype=np.int64) % k
    seed_idx = _farthest_point_seeds(face_centers, k, random_state=0)
    centers = face_centers[seed_idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        # 距离矩阵 (n, k)
        d = np.linalg.norm(
            face_centers[:, None, :] - centers[None, :, :], axis=-1
        )
        new_labels = np.argmin(d, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                centers[ci] = face_centers[mask].mean(axis=0)
    return labels


def _over_segment_block(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    n_candidates: int,
    *,
    vertex_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
) -> Tuple[np.ndarray, int]:
    """
    在分块内做一次过分割（n_candidates 通常远多于最终需要的组数）。
    使用 FPS 初始化的面中心 KMeans —— 候选大小相对均衡，VLM 更容易区分。
    返回 (face_label_global, k_actual)：
      - face_label_global 长度 = faces.shape[0]，候选编号 0..k_actual-1，未在块内的为 -1
      - k_actual：实际得到的候选数
    """
    n_total_faces = int(faces.shape[0])
    face_label_global = np.full(n_total_faces, -1, dtype=np.int64)
    face_ids_list = list(face_ids)
    if not face_ids_list or n_candidates < 2:
        return face_label_global, 0
    if len(face_ids_list) < n_candidates:
        # 块太小，每个面单独成簇但限制到 ≥2
        for i, fid in enumerate(face_ids_list):
            face_label_global[fid] = i % max(2, len(face_ids_list))
        return face_label_global, max(1, len(face_ids_list))

    face_ids_arr = np.asarray(face_ids_list, dtype=np.int64)
    block_face_centers = vertices[faces[face_ids_arr]].mean(axis=1)  # (M, 3)

    try:
        sub_face_labels_arr = _balanced_face_kmeans(
            block_face_centers, n_candidates
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "FPS-KMeans 过分割异常 (%s)，回退为顺序切分", type(exc).__name__,
        )
        sub_face_labels_arr = np.arange(len(face_ids_list)) % n_candidates

    for i, fid in enumerate(face_ids_list):
        face_label_global[fid] = int(sub_face_labels_arr[i])
    used = sorted({int(v) for v in sub_face_labels_arr})
    return face_label_global, len(used)


def _build_candidate_table(n_candidates: int) -> str:
    """构造 VLM 友好的候选清单字符串。"""
    color_names = (
        "红", "蓝", "绿", "紫", "橙", "黄", "棕", "粉",
        "青", "巧克力", "天青", "橄榄",
    )
    lines: List[str] = []
    for i in range(n_candidates):
        rgb = CANDIDATE_PALETTE[i % len(CANDIDATE_PALETTE)]
        name = color_names[i] if i < len(color_names) else f"色{i+1}"
        lines.append(
            f"- 候选 **{i + 1}** ({name})  RGB=({rgb[0]},{rgb[1]},{rgb[2]})"
        )
    return "\n".join(lines)


def _normalize_group_label(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in (LABEL_FULL, LABEL_PARTIAL, LABEL_NONE):
        return s
    if "full" in s:
        return LABEL_FULL
    if "partial" in s or "局部" in s or "部分" in s:
        return LABEL_PARTIAL
    if "none" in s or "no" == s or "不" in s:
        return LABEL_NONE
    return LABEL_PARTIAL  # 默认偏保守，让它继续切


def split_block_by_vlm_grouping(
    client: VLMClient,
    vertices: np.ndarray,
    faces: np.ndarray,
    block_face_ids: Sequence[int],
    semantic: str,
    output_panel_path: str | Path,
    *,
    n_candidates: int = 8,
    max_groups: int = 4,
    image_size: int = 512,
    cols: int = 4,
    vertex_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
    model: Optional[str] = None,
) -> GroupedSplitResult:
    """
    端到端：过分割 → 渲染候选面板 → 调 VLM 一次性分组 → 返回每组 face_ids + label。

    失败时返回 ok=False 的结果，调用方可回退到经典 ask_seed_count + 谱聚类。
    """
    # 限制 n_candidates 到 palette 上限
    n_candidates = max(2, min(int(n_candidates), len(CANDIDATE_PALETTE)))
    max_groups = max(2, min(int(max_groups), n_candidates))

    # ---- ① 过分割 ----
    face_label_global, k_actual = _over_segment_block(
        vertices,
        faces,
        block_face_ids,
        n_candidates,
        vertex_scalars=vertex_scalars,
        scalar_weight=scalar_weight,
    )
    if k_actual < 2:
        return GroupedSplitResult(
            ok=False,
            n_candidates=k_actual,
            error="over_segment_too_small",
            reason=f"块过小，无法过分割成 {n_candidates} 候选",
        )

    # 重映射 candidate_id 为 0..k_actual-1（按 used_ids 顺序）
    used_ids = sorted({int(v) for v in face_label_global if v >= 0})
    remap = {old: new for new, old in enumerate(used_ids)}
    remapped_global = np.where(
        face_label_global >= 0,
        np.vectorize(lambda v: remap.get(int(v), -1))(face_label_global),
        -1,
    )

    # ---- ② 渲染候选面板 ----
    output_panel_path = Path(output_panel_path)
    try:
        render_candidate_panel(
            vertices,
            faces,
            remapped_global.tolist(),
            output_panel_path,
            title=f"VLM 候选分组  block_faces={len(block_face_ids)}",
            image_size=image_size,
            cols=cols,
        )
    except (OSError, ImportError, ValueError) as exc:
        LOGGER.warning("候选面板渲染失败：%s", exc)
        return GroupedSplitResult(
            ok=False,
            n_candidates=len(used_ids),
            error=f"render_failed: {exc}",
            reason="候选面板渲染失败",
        )

    # ---- ③ 调 VLM 一次性分组 ----
    table_str = _build_candidate_table(len(used_ids))
    prompt = build_group_prompt(
        semantic,
        n_candidates=len(used_ids),
        candidate_table=table_str,
        max_groups=max_groups,
    )
    try:
        response = client.chat(
            user_prompt=prompt.user,
            image_paths=[output_panel_path],
            system=prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 分组调用失败：%s", exc)
        return GroupedSplitResult(
            ok=False,
            n_candidates=len(used_ids),
            panel_path=output_panel_path,
            error=str(exc),
            reason="VLM 分组调用失败",
        )

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning("VLM 分组响应非 JSON：%s", text[:200])
        return GroupedSplitResult(
            ok=False,
            n_candidates=len(used_ids),
            panel_path=output_panel_path,
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            reason=text.strip()[:200],
            raw_response=response.raw,
        )

    raw_groups = payload.get("groups") or []
    if not isinstance(raw_groups, list):
        raw_groups = []
    overall_reason = str(payload.get("reason") or "").strip()

    # ---- ④ 把候选编号映射回 face_ids ----
    cand_to_faces: Dict[int, List[int]] = {i: [] for i in range(len(used_ids))}
    for fid, lab in enumerate(remapped_global.tolist()):
        if lab >= 0:
            cand_to_faces[int(lab)].append(int(fid))

    consumed = set()  # 避免重复分配候选
    groups: List[CandidateGroup] = []
    for raw in raw_groups:
        if not isinstance(raw, dict):
            continue
        gid = str(raw.get("group_id") or f"G{len(groups)+1}").strip()
        cand_ids_raw = raw.get("candidate_ids") or []
        if not isinstance(cand_ids_raw, list):
            cand_ids_raw = []
        cand_ids: List[int] = []
        for v in cand_ids_raw:
            try:
                ci = int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= ci <= len(used_ids) and ci not in consumed:
                cand_ids.append(ci)
                consumed.add(ci)
        if not cand_ids:
            continue
        face_ids: List[int] = []
        for ci in cand_ids:
            face_ids.extend(cand_to_faces.get(ci - 1, []))
        if not face_ids:
            continue
        groups.append(
            CandidateGroup(
                group_id=gid,
                candidate_ids=cand_ids,
                label=_normalize_group_label(raw.get("label")),
                confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0))),
                description=str(raw.get("description") or "").strip(),
                face_ids=face_ids,
            )
        )

    # 收尾：未被 VLM 提到的候选 → 兜底为 partial 组（继续切）
    leftover = [i for i in range(1, len(used_ids) + 1) if i not in consumed]
    if leftover:
        leftover_faces: List[int] = []
        for ci in leftover:
            leftover_faces.extend(cand_to_faces.get(ci - 1, []))
        if leftover_faces:
            groups.append(
                CandidateGroup(
                    group_id="LEFTOVER",
                    candidate_ids=leftover,
                    label=LABEL_PARTIAL,
                    confidence=0.3,
                    description=f"VLM 未明确分组的候选（{len(leftover)} 个）",
                    face_ids=leftover_faces,
                )
            )

    if not groups:
        return GroupedSplitResult(
            ok=False,
            n_candidates=len(used_ids),
            panel_path=output_panel_path,
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error="empty_groups",
            reason=overall_reason or "VLM 没有返回有效分组",
            raw_response=response.raw,
        )

    return GroupedSplitResult(
        ok=True,
        n_candidates=len(used_ids),
        panel_path=output_panel_path,
        groups=groups,
        raw_text=text,
        elapsed_seconds=response.elapsed_seconds,
        provider=response.provider,
        model=response.model,
        reason=overall_reason,
        raw_response=response.raw,
    )


__all__ = [
    "CandidateGroup",
    "GroupedSplitResult",
    "split_block_by_vlm_grouping",
]
