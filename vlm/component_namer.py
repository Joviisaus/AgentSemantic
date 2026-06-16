"""
同部位不连通块的 VLM 细分命名。

full_coverage 给出的语义清单是**粗粒度、对称实例合并**的（如「腿」），但几何上
一个「腿」常常是 4 个互不相连的块（四条狗腿）。本模块：

1. 用 ``geometry.connected_components_per_part`` 把每个部位切成连通块
   （与最终建骨同口径）；
2. 对**含 ≥2 个块**的部位，渲染一张"各块不同颜色 + 数字标签"的 12 视角拼图，
   交给 VLM，让它根据各块与主体的**邻接/骨架角色**决定**是否**给出更具体的名字
   （如 肢-1/肢-2）；禁止方位词；
3. 把 VLM 的命名落回 ``(semantic_parts, face_semantic_index)``，得到一份细分后的
   标签，供下游骨架/动画直接使用。

只做"命名细分"，不改几何分块本身——即便 VLM 决定不拆，建骨阶段仍会按连通块各自
建骨（互不牵连）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.semantic_rig import connected_components_per_part

from .client import VLMClient, VLMResponse, parse_json_response
from .prompts import build_component_name_prompt

LOGGER = logging.getLogger(__name__)


@dataclass
class ComponentNaming:
    """细分命名结果。"""

    semantic_parts: List[str] = field(default_factory=list)
    face_semantic_index: List[int] = field(default_factory=list)
    renames: Dict[str, List[str]] = field(default_factory=dict)  # 原名 -> [各块新名]
    panels: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    changed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "semantic_parts": list(self.semantic_parts),
            "renames": {k: list(v) for k, v in self.renames.items()},
            "panels": list(self.panels),
            "warnings": list(self.warnings),
            "changed": self.changed,
        }


def _sanitize(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", str(name)).strip("_")
    return s or "part"


def _describe_components(
    centroids: Sequence[np.ndarray],
    comp_face_counts: Sequence[int],
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> str:
    """给每个块生成一行拓扑/几何参考（不含方位词）。"""
    center = 0.5 * (bbox_min + bbox_max)
    span = np.maximum(bbox_max - bbox_min, 1e-9)
    total_faces = max(1, sum(comp_face_counts))
    lines: List[str] = []
    for i, c in enumerate(centroids):
        d = (np.asarray(c, dtype=np.float64) - center) / span
        face_ratio = 100.0 * comp_face_counts[i] / total_faces
        lines.append(
            f"- {i + 1}: 重心≈({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})  "
            f"归一化偏移≈({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})  "
            f"面数占比≈{face_ratio:.0f}%"
        )
    return "\n".join(lines)


def _parse_labels(payload: Any, n: int, part: str) -> Dict[int, str]:
    """解析 VLM 返回的 {id: label}（1-based）。缺失/为空的留空，由调用方兜底。"""
    out: Dict[int, str] = {}
    if not isinstance(payload, dict):
        return out
    comps = payload.get("components")
    if not isinstance(comps, list):
        return out
    for item in comps:
        if not isinstance(item, dict):
            continue
        try:
            cid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if not (1 <= cid <= n):
            continue
        label = str(item.get("label") or "").strip()
        if label:
            out[cid] = label
    return out


def refine_disconnected_parts(
    client: VLMClient,
    vertices: np.ndarray,
    faces: np.ndarray,
    semantic_parts: Sequence[str],
    face_semantic_index: Sequence[int],
    *,
    out_dir: Optional[str | Path] = None,
    image_size: int = 512,
    panel_cols: int = 4,
    model: Optional[str] = None,
    min_component_faces: int = 20,
    max_components: int = 12,
) -> ComponentNaming:
    """对含多个不连通块的部位，调 VLM 细分命名，返回细分后的标签。"""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    parts = list(semantic_parts)

    comp_of_face, comp_part = connected_components_per_part(
        vertices, faces, face_semantic_index, min_component_faces
    )
    n_comp = len(comp_part)

    result = ComponentNaming(
        semantic_parts=list(parts),
        face_semantic_index=[int(x) for x in face_semantic_index],
    )
    if n_comp == 0:
        result.warnings.append("无有效连通块，跳过细分命名")
        return result

    # 每个连通块的新名字，默认沿用原部位名
    new_name_of_comp: List[str] = [parts[comp_part[c]] for c in range(n_comp)]

    # 按部位分组
    part_comps: Dict[int, List[int]] = {}
    for c, p in enumerate(comp_part):
        part_comps.setdefault(p, []).append(c)

    face_centers = vertices[faces].mean(axis=1)
    bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)

    out_path = Path(out_dir) if out_dir else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    try:
        from rendering.render_component import render_candidate_panel
    except ImportError as exc:  # pragma: no cover
        result.warnings.append(f"缺少 pyvista/渲染依赖，跳过细分命名: {exc}")
        return result

    for p, comps in sorted(part_comps.items()):
        if len(comps) < 2:
            continue
        name = parts[p]
        if len(comps) > max_components:
            result.warnings.append(
                f"部位「{name}」有 {len(comps)} 个块，超过上限 {max_components}，跳过命名"
            )
            continue

        # 候选标签：本部位各块取 0..K-1，其余 -1
        cand = np.full(len(faces), -1, dtype=np.int64)
        centroids: List[np.ndarray] = []
        comp_face_counts: List[int] = []
        for local, c in enumerate(comps):
            mask = comp_of_face == c
            cand[mask] = local
            centroids.append(face_centers[mask].mean(axis=0))
            comp_face_counts.append(int(mask.sum()))

        panel_path = (out_path / f"refine_{_sanitize(name)}.png") if out_path else None
        if panel_path is None:
            result.warnings.append("未提供 out_dir，无法渲染细分面板")
            continue
        try:
            render_candidate_panel(
                vertices, faces, cand, panel_path,
                title=f"部位「{name}」断成 {len(comps)} 块",
                image_size=image_size, cols=panel_cols,
            )
        except Exception as exc:  # noqa: BLE001  渲染失败不致命
            result.warnings.append(f"部位「{name}」面板渲染失败: {exc}")
            continue
        result.panels.append(str(panel_path))

        table = _describe_components(centroids, comp_face_counts, bbox_min, bbox_max)
        prompt = build_component_name_prompt(name, len(comps), table)
        try:
            response: VLMResponse = client.chat(
                user_prompt=prompt.user,
                image_paths=[panel_path],
                system=prompt.system,
                temperature=0.0,
                json_mode=True,
                model=model,
            )
            payload = parse_json_response(response.text or "")
        except (RuntimeError, ValueError) as exc:
            result.warnings.append(f"部位「{name}」VLM 命名失败，沿用原名: {exc}")
            continue

        labels = _parse_labels(payload, len(comps), name)
        names_here: List[str] = []
        for local, c in enumerate(comps):
            lab = labels.get(local + 1, "").strip()
            new_name_of_comp[c] = lab or name  # 缺失则沿用原名
            names_here.append(new_name_of_comp[c])
        result.renames[name] = names_here
        n_distinct = len({nm for nm in names_here if nm != name})
        LOGGER.info("部位「%s」%d 块 → %s", name, len(comps), names_here)
        if n_distinct:
            result.changed = True

    # 落回 (semantic_parts, face_semantic_index)
    name_to_idx: Dict[str, int] = {}
    new_parts: List[str] = []
    comp_new_idx = np.full(max(n_comp, 1), -1, dtype=np.int64)
    for c in range(n_comp):
        nm = new_name_of_comp[c]
        if nm not in name_to_idx:
            name_to_idx[nm] = len(new_parts)
            new_parts.append(nm)
        comp_new_idx[c] = name_to_idx[nm]

    safe = np.clip(comp_of_face, 0, max(n_comp - 1, 0))
    mapped = comp_new_idx[safe]
    new_face_idx = np.where(comp_of_face >= 0, mapped, -1).astype(np.int64)

    result.semantic_parts = new_parts
    result.face_semantic_index = [int(x) for x in new_face_idx]
    return result


__all__ = ["ComponentNaming", "refine_disconnected_parts"]
