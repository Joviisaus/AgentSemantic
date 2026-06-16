"""
VLM 层级语义树：看整模型渲染图生成由粗到细的语义分区树。

调用方（``pipeline.semantic_tree_partition``）分两阶段：
1. 根节点切一级子块，保证全覆盖；
2. BFS 对已分块逐层细分。
用户可指定 refine_depth / refine_paths / refine_nodes 控制细分深度。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from .client import VLMClient, VLMResponse, parse_json_response
from .prompts import build_semantic_tree_prompt

LOGGER = logging.getLogger("vlm.semantic_tree")


@dataclass
class SemanticTreeNode:
    """语义树节点。"""

    name: str
    children: List["SemanticTreeNode"] = field(default_factory=list)
    path: str = ""  # 从根起的唯一路径，如 "上半身/手臂"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "children": [c.to_dict() for c in self.children],
        }

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def depth(self) -> int:
        """根「整体」depth=0。"""
        if not self.path:
            return 0
        return self.path.count("/") + 1

    def iter_nodes(self) -> List["SemanticTreeNode"]:
        out: List[SemanticTreeNode] = [self]
        for ch in self.children:
            out.extend(ch.iter_nodes())
        return out

    def collect_leaf_names(self) -> List[str]:
        if self.is_leaf:
            name = self.name.strip()
            return [name] if name and name != "整体" else []
        names: List[str] = []
        seen: Set[str] = set()
        for ch in self.children:
            for n in ch.collect_leaf_names():
                if n not in seen:
                    seen.add(n)
                    names.append(n)
        return names

    def find_by_path(self, path: str) -> Optional["SemanticTreeNode"]:
        target = str(path or "").strip().strip("/")
        if not target:
            return self
        for node in self.iter_nodes():
            if node.path == target:
                return node
        return None


@dataclass
class SemanticTreeResult:
    """VLM 生成的语义树。"""

    root: SemanticTreeNode = field(
        default_factory=lambda: SemanticTreeNode(name="整体")
    )
    object_kind: str = ""
    reason: str = ""
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.root.children) and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_kind": self.object_kind,
            "tree": self.root.to_dict(),
            "reason": self.reason,
            "elapsed_seconds": self.elapsed_seconds,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
        }


def _attach_paths(node: SemanticTreeNode, parent_path: str = "") -> None:
    name = node.name.strip()
    if parent_path:
        node.path = f"{parent_path}/{name}"
    else:
        node.path = name if name != "整体" else ""
    for ch in node.children:
        base = node.path if node.path else ""
        _attach_paths(ch, base)


def _parse_tree_node(raw: Any, *, seen_names: Set[str]) -> Optional[SemanticTreeNode]:
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    if name in seen_names and name != "整体":
        LOGGER.info("语义树：跳过重复节点名 '%s'", name)
        return None
    seen_names.add(name)
    children_raw = raw.get("children")
    children: List[SemanticTreeNode] = []
    if isinstance(children_raw, list):
        for item in children_raw:
            ch = _parse_tree_node(item, seen_names=seen_names)
            if ch is not None:
                children.append(ch)
    return SemanticTreeNode(name=name, children=children)


def parse_semantic_tree_payload(payload: Dict[str, Any]) -> SemanticTreeResult:
    """从 JSON dict 解析语义树（VLM 响应或用户文件）。"""
    tree_raw = payload.get("tree") or payload.get("root")
    if tree_raw is None and "name" in payload:
        tree_raw = payload
    if not isinstance(tree_raw, dict):
        return SemanticTreeResult(error="tree 字段缺失或不是 object")

    seen: Set[str] = set()
    root = _parse_tree_node(tree_raw, seen_names=seen)
    if root is None:
        return SemanticTreeResult(error="无法解析 tree 根节点")
    if root.name != "整体":
        root = SemanticTreeNode(name="整体", children=[root])
    _attach_paths(root)

    if not root.children:
        return SemanticTreeResult(
            root=root,
            error="根节点 children 为空",
        )

    return SemanticTreeResult(
        root=root,
        object_kind=str(payload.get("object_kind") or "").strip(),
        reason=str(payload.get("reason") or "").strip(),
    )


def load_semantic_tree_file(path: str | Path) -> SemanticTreeResult:
    """从 JSON 文件加载语义树。"""
    import json

    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 不是 JSON object")
    result = parse_semantic_tree_payload(data)
    if not result.ok:
        raise ValueError(result.error or "语义树解析失败")
    return result


def generate_semantic_tree(
    client: VLMClient,
    overview_panel: str | Path,
    *,
    hint: str = "",
    model: Optional[str] = None,
    max_depth: int = 5,
) -> SemanticTreeResult:
    """让 VLM 看整模型 12 视角拼图，生成层级语义树。"""
    overview_panel = Path(overview_panel)
    prompt = build_semantic_tree_prompt(hint=hint, max_depth=max_depth)
    try:
        response: VLMResponse = client.chat(
            user_prompt=prompt.user,
            image_paths=[overview_panel],
            system=prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 语义树生成失败: %s", exc)
        return SemanticTreeResult(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        return SemanticTreeResult(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            raw_response=response.raw,
        )

    result = parse_semantic_tree_payload(payload)
    result.raw_text = text
    result.elapsed_seconds = response.elapsed_seconds
    result.provider = response.provider
    result.model = response.model
    result.raw_response = response.raw
    return result


def should_refine_node(
    node: SemanticTreeNode,
    *,
    refine_depth: int = 1,
    refine_all: bool = False,
    refine_paths: Optional[Sequence[str]] = None,
    refine_nodes: Optional[Sequence[str]] = None,
) -> bool:
    """判断该节点是否应继续几何细分（需有 children）。"""
    if node.is_leaf:
        return False
    if refine_all:
        return True

    norm_paths = [
        str(p).strip().strip("/") for p in (refine_paths or []) if str(p).strip()
    ]
    norm_nodes = {str(x).strip() for x in (refine_nodes or []) if str(x).strip()}
    path = node.path.strip("/")

    # 根节点：指定路径/节点模式时，只要目标在子树内就必须先切根
    if node.name == "整体":
        if norm_paths or norm_nodes:
            return True
        return refine_depth >= 1

    if norm_nodes and node.name in norm_nodes:
        return True

    if norm_paths:
        for rp in norm_paths:
            if path == rp:
                return True
            if path and path.startswith(rp + "/"):
                return True
            if rp.startswith(path + "/") if path else False:
                return True
        return False

    return node.depth < refine_depth


__all__ = [
    "SemanticTreeNode",
    "SemanticTreeResult",
    "generate_semantic_tree",
    "load_semantic_tree_file",
    "parse_semantic_tree_payload",
    "should_refine_node",
]
