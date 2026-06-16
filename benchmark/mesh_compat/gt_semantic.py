"""Derive a concise search query from the *visual* GT instead of the part name.

Fairness fix: handing the dataset's part name (e.g. "body") straight to the
pipeline leaks the answer. Instead, for each GT part we:
  1. paint the GT region onto the mesh faces (nearest GT point per face),
  2. render a multi-view highlight panel (red = the part, grey = the rest),
  3. ask the VLM for a concise semantic phrase naming the red region.

The returned phrase is used as the segmentation query; evaluation still scores
the pipeline output against the same GT part, so the phrase only needs to point
the pipeline at the right region, not match the dataset string.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from rendering.render_component import (
    mesh_face_connected_components,
    render_highlighted_component_panel,
)
from vlm.client import VLMClient, VLMResponse, parse_json_response

LOGGER = logging.getLogger("benchmark.mesh_compat.gt_semantic")

_SYS_PROMPT = (
    "你是3D资产分析助手。给你的是同一个物体的多视角渲染拼图，其中**鲜艳红色**"
    "高亮的是该物体的某一个部件，其余部分为浅灰色。\n"
    "请只根据红色高亮区域所在的几何位置和形状，用一个**简洁的名词短语**命名这个"
    "部件（2-6个字，例如“桌面”“椅子腿”“瓶盖”）。\n"
    "要求：不要解释，不要提到“红色/高亮/颜色”，不要输出多个候选。\n"
    '只输出 JSON：{"semantic": "<部件名>"}'
)


def _align_vertices_to_gt(vertices: np.ndarray, gt_xyz: np.ndarray) -> np.ndarray:
    """Center + uniform-scale mesh vertices so their AABB matches the GT points."""
    v = np.asarray(vertices, dtype=np.float64)
    p = np.asarray(gt_xyz, dtype=np.float64)
    v_min, v_max = v.min(0), v.max(0)
    p_min, p_max = p.min(0), p.max(0)
    v_ext = float((v_max - v_min).max())
    p_ext = float((p_max - p_min).max())
    if v_ext <= 1e-12 or p_ext <= 1e-12:
        return v
    scale = p_ext / v_ext
    return (v - (v_min + v_max) / 2.0) * scale + (p_min + p_max) / 2.0


def _fill_unlabeled_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Propagate labels to unlabeled faces (label < 0) via nearest labeled face.

    Works entirely in the mesh's own coordinate system (face centroids), so it
    is immune to the mesh<->GT coordinate mismatch.
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    unl = labels < 0
    if not unl.any() or unl.all():
        return labels
    faces = np.asarray(faces, dtype=np.int64)
    fc = np.asarray(vertices, dtype=np.float64)[faces].mean(axis=1)
    known = ~unl
    tree = cKDTree(fc[known])
    _, nn = tree.query(fc[unl], k=1)
    labels[unl] = labels[known][nn]
    return labels


def face_gt_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    gt_xyz: np.ndarray,
    gt_part_ids: np.ndarray,
    target_face_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dense per-face GT label.

    Preferred path (when ``target_face_idx`` is given): use the authoritative
    point->face mapping computed at export time. Each face that owns one or more
    GT points takes the **majority** class among them; faces with no GT point are
    filled from the nearest labeled face *on the same mesh* (mesh-space KDTree).
    This avoids the speckled labels caused by the crude mesh<->GT bbox alignment.

    Fallback path: bbox-align the mesh to the GT bbox and take each face's
    nearest GT point's class.
    """
    gti = np.asarray(gt_part_ids, dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64)
    n_faces = len(faces)

    if target_face_idx is not None:
        tfi = np.clip(np.asarray(target_face_idx, dtype=np.int64), 0, n_faces - 1)
        # Per-face majority vote among the GT points that land on it.
        classes = np.unique(gti)
        cls_to_local = {int(c): i for i, c in enumerate(classes)}
        votes = np.zeros((n_faces, len(classes)), dtype=np.int64)
        local = np.array([cls_to_local[int(c)] for c in gti], dtype=np.int64)
        np.add.at(votes, (tfi, local), 1)
        has_vote = votes.sum(axis=1) > 0
        labels = np.full(n_faces, -1, dtype=np.int64)
        labels[has_vote] = classes[np.argmax(votes[has_vote], axis=1)]
        return _fill_unlabeled_faces(vertices, faces, labels)

    gt_xyz = np.asarray(gt_xyz, dtype=np.float64)
    verts = _align_vertices_to_gt(vertices, gt_xyz)
    fc = verts[faces].mean(axis=1)
    _, nn = cKDTree(gt_xyz).query(fc, k=1)
    return gti[nn]


def part_highlight_face_ids(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_gt: np.ndarray,
    class_id: int,
) -> Tuple[np.ndarray, int, int]:
    """All faces of ``class_id``, spanning every disconnected island on the mesh.

    Returns (face_ids, n_islands, n_mesh_components).
    """
    face_gt = np.asarray(face_gt, dtype=np.int64)
    base = np.flatnonzero(face_gt == int(class_id))
    if base.size == 0:
        cc = mesh_face_connected_components(vertices, faces)
        return base, 0, int(np.unique(cc).size) if cc.size else 0
    cc = mesh_face_connected_components(vertices, faces)
    n_mesh = int(np.unique(cc).size)
    islands = sorted({int(cc[fi]) for fi in base})
    # Union of this class on every touched island (keeps mixed CCs from over-expanding).
    mask = np.zeros(len(faces), dtype=bool)
    for island in islands:
        mask |= (cc == island) & (face_gt == int(class_id))
    ids = np.flatnonzero(mask)
    return ids, len(islands), n_mesh


def describe_gt_part(
    client: VLMClient,
    vertices: np.ndarray,
    faces: np.ndarray,
    highlight_face_ids: np.ndarray,
    out_dir: Path,
    *,
    object_category: str = "",
    model: Optional[str] = None,
    image_size: int = 512,
    n_islands: int = 0,
    n_mesh_components: int = 0,
) -> Tuple[str, Optional[Path], str]:
    """Render the GT-highlighted panel and ask the VLM for a concise semantic.

    The panel always shows the **full mesh** (all disconnected components in grey)
    with the target part highlighted in red across every island it occupies.

    Returns (semantic, panel_path, raw_text). ``semantic`` is "" on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = out_dir / "gt_highlight.png"
    ids = list(np.asarray(highlight_face_ids, dtype=np.int64))
    title = "部件高亮（红=GT部件，灰=整模含全部连通块）"
    if n_islands > 0 or n_mesh_components > 0:
        title += f"  部件岛={n_islands or '?'}  整模连通块={n_mesh_components or '?'}"
    try:
        render_highlighted_component_panel(
            vertices, faces, ids,
            panel, title=title, image_size=image_size, cols=4,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("GT panel render failed: %s", exc)
        return "", None, ""

    user = "请给出红色高亮部件的简洁语义名称。"
    if object_category:
        user = f"该物体是一个「{object_category}」。" + user
    try:
        resp: VLMResponse = client.chat(
            user_prompt=user, image_paths=[panel], system=_SYS_PROMPT,
            temperature=0.0, json_mode=True, model=model,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("GT-semantic VLM call failed: %s", exc)
        return "", panel, ""

    text = resp.text or ""
    semantic = ""
    try:
        payload = parse_json_response(text)
        semantic = str(payload.get("semantic") or "").strip()
    except Exception:  # noqa: BLE001
        semantic = text.strip().splitlines()[0].strip() if text.strip() else ""
    # strip stray quotes/punctuation
    semantic = semantic.strip().strip('“”"\'：: ')
    return semantic, panel, text
