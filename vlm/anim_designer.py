"""
语义动画 agent：用户给一个动作名（如「走路」「摇尾巴」「挥手」），
LLM 根据骨架部位清单 + 层级，生成关键帧动画 spec。

与 ``rig_designer`` 解耦：rig 设计只管层级/关节轴；本模块只管动画。
输出格式与 rig-spec 的 ``animations`` 字段兼容，可直接交给
``geometry.semantic_rig.resolve_rig_spec`` 解析。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .client import VLMClient, VLMResponse, parse_json_response

LOGGER = logging.getLogger(__name__)


ANIM_DESIGNER_SYSTEM_PROMPT = (
    "你是一名 3D 角色动画师。用户会给你一个**语义动作名**（如走路、坐下、摇尾巴、"
    "挥手、抬头），以及模型的语义部位清单和骨架层级。\n"
    "你要设计一段**自然、可循环（若适用）**的简短动画，用关键帧描述各部位关节"
    "相对静止姿态的旋转角度（度）。\n"
    "约定：+Y 朝上、+X 朝右、+Z 朝前；角度绕各骨在 rig 里定义的旋转轴，"
    "正/负表示方向；幅度符合该关节现实活动范围，不要夸张到穿模。\n"
    "angles 的键**只能**用用户给出的部位名（封闭清单）；根/中枢部位一般保持 0°。\n"
    "只输出 JSON，不要解释。"
)

ANIM_DESIGNER_USER_TEMPLATE = (
    "物体类型：{object_kind_hint}\n\n"
    "请求的语义动作：{anim_names}\n"
    "动作补充说明：{anim_hint}\n\n"
    "语义部位（angles 键只能从这里选）：\n{parts_table}\n\n"
    "骨架层级（parent 表示父部位，动画从根向外驱动）：\n{hierarchy_table}\n\n"
    "请输出 JSON：\n"
    "{{\n"
    '  "animations": [\n'
    "    {{\n"
    '      "name": "<动作名，用用户给的语义名>",\n'
    '      "fps": 24,\n'
    '      "duration": <秒>,\n'
    '      "keyframes": [\n'
    '        {{"t": 0.0, "angles": {{}}}},\n'
    '        {{"t": <秒>, "angles": {{"<部位>": <角度>, "...": <角度>}}}}\n'
    "      ]\n"
    "    }}\n"
    "  ]\n"
    "}}\n\n"
    "要求：\n"
    "1. 每个请求的动作名各一段 animation（name 与用户给的语义名一致或极近）。\n"
    "2. 第一个关键帧 t=0 一般为静止（angles 空或全 0）。\n"
    "3. 关键帧 3~6 个为宜；周期性动作（走路、摇尾巴）首尾应能衔接。\n"
    "4. 四足走路：四条腿交替、躯干轻微起伏；挥手：手臂+可选头部；"
    "坐下：后腿弯曲、躯干下沉。\n"
    "5. duration 通常 1.0~3.0 秒；fps 固定 24。\n"
    "6. 若清单里已细分（如 左前腿/右前腿），请分别给角度；"
    "若仍是统称（如 腿），该键会同步驱动所有同名连通块。"
)


@dataclass
class AnimDesign:
    """VLM 生成的动画 spec。"""

    animations: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.animations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "animations": list(self.animations),
            "provider": self.provider,
            "model": self.model,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


def format_hierarchy_table(
    semantic_parts: Sequence[str],
    parent_map: Dict[str, str],
    *,
    root: str = "",
) -> str:
    """把 {部位: 父部位名或 root} 格式化成缩进树文本。"""
    parts = [str(p).strip() for p in semantic_parts if str(p).strip()]
    if not parts:
        return "（无）"
    children: Dict[str, List[str]] = {p: [] for p in parts}
    roots: List[str] = []
    for p in parts:
        pp = parent_map.get(p, "root")
        if pp in ("", "root", "ROOT", None) or pp not in parts:
            roots.append(p)
        else:
            children.setdefault(pp, []).append(p)
    if root and root in parts:
        roots = [root] + [r for r in roots if r != root]
    if not roots:
        roots = [parts[0]]

    lines: List[str] = []

    def walk(name: str, depth: int) -> None:
        prefix = "  " * depth + ("- " if depth else "- ")
        lines.append(f"{prefix}{name}" + (" (root)" if depth == 0 and name in roots else ""))
        for c in sorted(children.get(name, [])):
            walk(c, depth + 1)

    for r in roots:
        walk(r, 0)
    return "\n".join(lines)


def hierarchy_from_skeleton(skeleton) -> Dict[str, str]:
    """从已建 SemanticSkeleton 提取 {部位名: 父部位名或 root}。"""
    bones = skeleton.bones
    out: Dict[str, str] = {}
    for b in bones:
        if b.parent < 0:
            out[b.name] = "root"
        else:
            out[b.name] = bones[b.parent].name
    return out


def hierarchy_from_rig_spec(spec: Dict[str, Any]) -> Dict[str, str]:
    """从 rig-spec bones 段提取层级。"""
    out: Dict[str, str] = {}
    bones = spec.get("bones") or []
    if not isinstance(bones, list):
        return out
    for entry in bones:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        parent = entry.get("parent", "root")
        out[name] = str(parent or "root").strip()
    return out


def _coerce_animations(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    anims = payload.get("animations")
    if not isinstance(anims, list):
        if "keyframes" in payload:
            return [payload]
        return []
    return [a for a in anims if isinstance(a, dict) and a.get("keyframes")]


def design_animations(
    client: VLMClient,
    animation_names: Sequence[str],
    semantic_parts: Sequence[str],
    hierarchy_table: str,
    *,
    object_kind_hint: str = "",
    anim_hint: str = "",
    image_path: Optional[str | Path] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> AnimDesign:
    """根据语义动作名 + 骨架信息，让 LLM 生成关键帧动画。"""
    names = [str(n).strip() for n in animation_names if str(n).strip()]
    if not names:
        raise ValueError("animation_names 不能为空")
    parts = [str(p).strip() for p in semantic_parts if str(p).strip()]
    if not parts:
        raise ValueError("semantic_parts 不能为空")

    parts_table = "\n".join(f"- {p}" for p in parts)
    user = ANIM_DESIGNER_USER_TEMPLATE.format(
        object_kind_hint=object_kind_hint.strip() or "（无，按结构自行判断）",
        anim_names="、".join(f"「{n}」" for n in names),
        anim_hint=anim_hint.strip() or "（无）",
        parts_table=parts_table,
        hierarchy_table=hierarchy_table.strip() or "（未提供，按部位常识推断）",
    )
    images = [Path(image_path)] if image_path else []

    try:
        response: VLMResponse = client.chat(
            user_prompt=user,
            image_paths=images,
            system=ANIM_DESIGNER_SYSTEM_PROMPT,
            temperature=temperature,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("动画 agent 调用失败: %s", exc)
        return AnimDesign(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning("动画 agent 响应非 JSON: %s", text[:200])
        return AnimDesign(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            raw_response=response.raw,
        )

    animations = _coerce_animations(payload)
    if not animations:
        return AnimDesign(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error="返回缺少有效 animations",
            raw_response=response.raw,
        )

    if len(names) == 1 and len(animations) == 1:
        if not str(animations[0].get("name") or "").strip():
            animations[0]["name"] = names[0]

    return AnimDesign(
        animations=animations,
        raw_text=text,
        elapsed_seconds=response.elapsed_seconds,
        provider=response.provider,
        model=response.model,
        raw_response=response.raw,
    )


__all__ = [
    "AnimDesign",
    "design_animations",
    "format_hierarchy_table",
    "hierarchy_from_rig_spec",
    "hierarchy_from_skeleton",
]
