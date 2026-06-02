"""
SemanticBlock 树结构与树工具函数。

每个 block 表示某个 mesh 子区域，其状态在迭代过程中演化：
    pending  -> 还没被 VLM 判定
    full     -> VLM 判定"完全符合语义"
    none     -> VLM 判定"完全不符合"
    partial  -> VLM 判定"局部符合"，待分裂
    split    -> 已经被分裂，由 children 接力
    abandoned-> 由于达到深度上限 / 太小不再分裂，并入"none"

block_id 形如 "C0", "C0.1", "C0.1.2"，便于 JSON 中追踪父子关系。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class BlockStatus(str, Enum):
    PENDING = "pending"
    FULL = "full"
    NONE = "none"
    PARTIAL = "partial"
    SPLIT = "split"
    ABANDONED = "abandoned"


@dataclass
class SemanticBlock:
    block_id: str
    face_ids: List[int]
    depth: int = 0
    parent_id: Optional[str] = None
    status: BlockStatus = BlockStatus.PENDING
    classification: Optional[Dict[str, Any]] = None
    seed_decision: Optional[Dict[str, Any]] = None
    seed_count: Optional[int] = None
    children_ids: List[str] = field(default_factory=list)
    panel_path: Optional[str] = None
    notes: str = ""
    # 由 review 触发的强制分裂：跳过 VLM classify，直接走 partial 的分裂路径。
    # 用于 review 判定 over_claimed 时把过大的 FULL 块降级回去再切。
    force_split: bool = False
    # 是否已做过 segmat 几何预分割（auto-K 谱聚类，基于特征值 gap 自动估 K）。
    # 单连通块的第一次几何切走 segmat；segmat 的子块再切才走 VLM 给 K 的 FPS-spectral。
    # CC 切出的新 CC 子块默认 False（每个 CC 都需要各自做一次 segmat）。
    segmat_done: bool = False

    @property
    def size(self) -> int:
        return len(self.face_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_id": self.block_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "status": self.status.value,
            "num_faces": self.size,
            "seed_count": self.seed_count,
            "children_ids": list(self.children_ids),
            "panel_path": self.panel_path,
            "notes": self.notes,
            "classification": self.classification,
            "seed_decision": self.seed_decision,
        }


def build_initial_blocks(
    face_labels: Sequence[int],
    *,
    min_block_size: int = 1,
    id_prefix: str = "C",
) -> List[SemanticBlock]:
    """
    从已有 face_labels 构造初始 block 列表（每个聚类 / 分块一个 block）。

    label < 0 的 face 视为背景，单独成一个 "BG" block（默认状态 PENDING，
    后续 VLM 也会评估它，但通常会被判为 none）。
    """
    groups: Dict[int, List[int]] = {}
    for face_id, label in enumerate(face_labels):
        groups.setdefault(int(label), []).append(face_id)

    blocks: List[SemanticBlock] = []
    counter = 0
    for label in sorted(groups.keys()):
        face_ids = groups[label]
        if len(face_ids) < max(1, min_block_size):
            continue
        block_id = f"{id_prefix}{counter}"
        counter += 1
        blocks.append(
            SemanticBlock(
                block_id=block_id,
                face_ids=face_ids,
                depth=0,
                parent_id=None,
                status=BlockStatus.PENDING,
                notes=f"initial label={label}",
            )
        )
    return blocks


def gather_face_status(
    blocks: Dict[str, SemanticBlock],
    n_faces: int,
) -> Dict[str, List[int]]:
    """
    汇总最终每个 face 的状态：
      full_face_ids / none_face_ids / pending_face_ids
    只看叶子节点（即 status != SPLIT 的 block），因为 SPLIT 已经把 face 交给 children。
    """
    full_ids: List[int] = []
    none_ids: List[int] = []
    pending_ids: List[int] = []

    for block in blocks.values():
        if block.status == BlockStatus.SPLIT:
            continue
        if block.status == BlockStatus.FULL:
            full_ids.extend(block.face_ids)
        elif block.status in (BlockStatus.NONE, BlockStatus.ABANDONED):
            none_ids.extend(block.face_ids)
        else:
            pending_ids.extend(block.face_ids)

    assigned = set(full_ids) | set(none_ids) | set(pending_ids)
    leftover = [i for i in range(n_faces) if i not in assigned]
    none_ids.extend(leftover)
    return {
        "full_face_ids": sorted(set(full_ids)),
        "none_face_ids": sorted(set(none_ids)),
        "pending_face_ids": sorted(set(pending_ids)),
    }


def export_tree_json(
    blocks: Dict[str, SemanticBlock],
    path: str | Path,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """把整棵 block 树写成 JSON，便于复盘和可视化。"""
    payload: Dict[str, Any] = {
        "num_blocks": len(blocks),
        "blocks": [blocks[k].to_dict() for k in sorted(blocks.keys())],
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path
