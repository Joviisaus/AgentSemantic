"""
让 LLM 给定骨架层级 + 关节活动（动画关键帧）。

输入：一个封闭的语义部位清单（来自 full_coverage 的 face_semantic_labels.json）
      + 部位之间的几何邻接关系（哪些部位互相接触），可选一张整模型渲染图。
输出：一份 rig-spec（JSON），描述
      - 骨架层级：每个部位的父部位（root 为静止/中枢部位，如 躯干 / 底座）
      - 每根骨的关节旋转轴（模型局部坐标系）
      - 1~N 段动画：每段给若干关键帧，关键帧给"哪些关节转多少度"

本模块只负责调 VLM 拿到并清洗成结构化 dict，不依赖 geometry。
spec 的语义化解析（名字→下标、建动画）由 ``geometry.semantic_rig.resolve_rig_spec``
负责，保持 vlm 层与几何层解耦。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import VLMClient, VLMResponse, parse_json_response

LOGGER = logging.getLogger(__name__)


RIG_DESIGNER_SYSTEM_PROMPT = (
    "你是一名 3D 角色绑定（rigging）与动画师。给定一个物体的语义部位清单和"
    "部位之间的邻接关系，你要：\n"
    "1) 设计一套合理的骨架层级：选一个静止/中枢部位作为根（如 躯干、身体、"
    "底座、主体），其余部位各指定唯一父部位，父子尽量沿邻接关系，形成一棵树；\n"
    "2) 为每根非根骨指定一个关节旋转轴（模型局部坐标系的单位向量，约定 +Y 朝上、"
    "+X 朝右、+Z 朝前；拿不准时给最像该关节弯曲方向的轴）；\n"
    "3) 设计 1~2 段简短、自然、能激活主要关节的动画，用关键帧描述：每个关键帧给"
    "时间 t（秒）和该时刻若干关节相对静止姿态的旋转角度（度，绕该骨的轴；正负表方向，"
    "幅度适中、符合该关节的现实活动范围）。\n"
    "只输出 JSON，不要解释。"
)

RIG_DESIGNER_USER_TEMPLATE = (
    "物体类型提示：{object_kind_hint}\n\n"
    "语义部位（封闭清单，parent / 动画里只能用这些名字）：\n{parts_table}\n\n"
    "部位邻接（互相接触，建议据此连父子）：\n{adjacency_table}\n\n"
    "请输出如下结构的 JSON：\n"
    "{{\n"
    '  "object_kind": "...",\n'
    '  "root": "<根部位名>",\n'
    '  "bones": [\n'
    '    {{"name":"<部位>", "parent":"<父部位名或 root>", "axis":[x,y,z]}}\n'
    "    // 每个部位一条；根部位 parent 用 \"root\"\n"
    "  ],\n"
    '  "animations": [\n'
    '    {{"name":"<动作名>", "fps":24, "duration":<秒>,\n'
    '      "keyframes":[\n'
    '        {{"t":0.0, "angles":{{}} }},\n'
    '        {{"t":<秒>, "angles":{{"<部位>":<角度>, "...":<角度>}} }}\n'
    "      ]}}\n"
    "  ]\n"
    "}}\n"
    "要求：root 的 parent 必须是 \"root\"；每个列出的部位都要出现在 bones 里；"
    "动画 angles 的键必须是上面清单里的部位名；起始关键帧一般为静止（空 angles）。"
)


@dataclass
class RigDesign:
    """VLM 设计出的 rig-spec（原始结构化结果，未做名字→下标解析）。"""

    spec: Dict[str, Any] = field(default_factory=dict)
    object_kind: str = ""
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.spec.get("bones"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec,
            "object_kind": self.object_kind,
            "provider": self.provider,
            "model": self.model,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


def _format_adjacency(adjacency_pairs: Sequence[Tuple[str, str]]) -> str:
    if not adjacency_pairs:
        return "（未提供邻接信息；请按物体常识推断层级）"
    return "\n".join(f"- {a} ↔ {b}" for a, b in adjacency_pairs)


def design_rig(
    client: VLMClient,
    semantic_parts: Sequence[str],
    adjacency_pairs: Sequence[Tuple[str, str]],
    *,
    object_kind_hint: str = "",
    image_path: Optional[str | Path] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> RigDesign:
    """调 VLM 设计骨架层级 + 关节轴 + 动画关键帧。

    参数
    ----
    semantic_parts   : 封闭部位清单
    adjacency_pairs  : [(部位A, 部位B), ...] 几何上相邻的部位对
    object_kind_hint : 物体类型提示（如「类人机甲」「四足动物」）
    image_path       : 可选，整模型渲染图，帮助 LLM 判断朝向/结构
    """
    parts = [str(p).strip() for p in semantic_parts if str(p).strip()]
    if not parts:
        raise ValueError("semantic_parts 不能为空")

    parts_table = "\n".join(f"- {p}" for p in parts)
    user = RIG_DESIGNER_USER_TEMPLATE.format(
        object_kind_hint=object_kind_hint.strip() or "（无，按结构自行判断）",
        parts_table=parts_table,
        adjacency_table=_format_adjacency(adjacency_pairs),
    )
    images = [Path(image_path)] if image_path else []

    try:
        response: VLMResponse = client.chat(
            user_prompt=user,
            image_paths=images,
            system=RIG_DESIGNER_SYSTEM_PROMPT,
            temperature=temperature,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("rig 设计 VLM 调用失败: %s", exc)
        return RigDesign(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        LOGGER.warning("rig 设计响应非 JSON: %s", text[:200])
        return RigDesign(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
            raw_response=response.raw,
        )

    if not isinstance(payload, dict):
        return RigDesign(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error="返回顶层不是 JSON 对象",
            raw_response=response.raw,
        )

    return RigDesign(
        spec=payload,
        object_kind=str(payload.get("object_kind") or object_kind_hint or "").strip(),
        raw_text=text,
        elapsed_seconds=response.elapsed_seconds,
        provider=response.provider,
        model=response.model,
        error=None if payload.get("bones") else "spec 缺少 bones",
        raw_response=response.raw,
    )
