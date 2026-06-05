"""
由 Agent 根据语义分块信息，为硬分块边界优化指定能量函数权重与目标。

几何侧使用三类指标：
  - 体积比（volume_ratio）：各骨段占总体积比例
  - 表面积比（area_ratio）：各骨段边界长度占总分块缝比例
  - 最大凹陷深度（max_concavity_depth）：骨段边界上凹槽深度峰值

Agent 输出 ``partition_energy_spec.json``，供 ``geometry.skin_optimize`` 消费。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import VLMClient, VLMResponse, parse_json_response

LOGGER = logging.getLogger(__name__)

PARTITION_ENERGY_SYSTEM = (
    "你是 3D 角色绑定专家。给定语义骨段清单、骨架层级、接缝邻接，以及当前"
    "分块的几何指标（体积比、表面积比、最大凹陷深度），请为**硬分块边界优化**"
    "设计能量函数参数。\n"
    "目标：让分块缝卡在解剖凹槽（颈、肩、腕等），同时各骨段体积/暴露边界比例"
    "符合该物体语义（头小、四肢细长等）。\n"
    "只输出 JSON，不要解释。"
)

PARTITION_ENERGY_USER_TEMPLATE = (
    "物体类型：{object_kind}\n\n"
    "骨段清单（封闭，只能用这些名字）：\n{bones_table}\n\n"
    "骨架父子（parent → child）：\n{hierarchy_table}\n\n"
    "接缝邻接（应优先把边界放进凹槽）：\n{seam_table}\n\n"
    "当前分块几何指标（建骨初始状态，供参考）：\n{metrics_table}\n\n"
    "请输出 JSON：\n"
    "{{\n"
    '  "object_kind": "...",\n'
    '  "global": {{\n'
    '    "w_volume": <体积比偏差惩罚权重>,\n'
    '    "w_area": <表面积比偏差惩罚权重>,\n'
    '    "w_concavity": <凹陷对齐奖励权重>,\n'
    '    "w_stick": <贴合初始分块权重>,\n'
    '    "w_smooth": <边界平滑权重>\n'
    "  }},\n"
    '  "bones": [\n'
    "    {{\n"
    '      "name": "<骨段名>",\n'
    '      "volume_ratio_target": <0~1>,\n'
    '      "area_ratio_target": <0~1>,\n'
    '      "concavity_pull": <0~3 越大越要把缝卡进凹槽>,\n'
    '      "notes": "<一句说明，如：颈根凹陷处切分>"\n'
    "    }}\n"
    "  ],\n"
    '  "seams": [\n'
    '    {{"bone_a": "<名>", "bone_b": "<名>", "w_concavity": <0~5>}}\n'
    "  ]\n"
    "}}\n"
    "要求：bones 覆盖清单中每个骨段；volume/area 目标之和不必为 1 但应合理；"
    "seams 覆盖主要邻接对；四肢末端 concavity_pull 可较低，躯干连接处较高。"
)


@dataclass
class PartitionEnergyDesign:
    spec: Dict[str, Any] = field(default_factory=dict)
    object_kind: str = ""
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None

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


def _format_hierarchy(
    bone_names: Sequence[str],
    parents: Sequence[int],
) -> str:
    lines = []
    for i, name in enumerate(bone_names):
        p = int(parents[i]) if i < len(parents) else -1
        pname = bone_names[p] if 0 <= p < len(bone_names) else "root"
        lines.append(f"- {pname} → {name}")
    return "\n".join(lines) if lines else "（无）"


def _format_seams(seam_pairs: Sequence[Tuple[str, str]]) -> str:
    if not seam_pairs:
        return "（未提供；按常识推断）"
    return "\n".join(f"- {a} ↔ {b}" for a, b in seam_pairs)


def _format_metrics_table(
    bone_names: Sequence[str],
    volume_ratio: Sequence[float],
    area_ratio: Sequence[float],
    max_concavity: Sequence[float],
) -> str:
    rows = []
    for i, name in enumerate(bone_names):
        vr = volume_ratio[i] if i < len(volume_ratio) else 0.0
        ar = area_ratio[i] if i < len(area_ratio) else 0.0
        mc = max_concavity[i] if i < len(max_concavity) else 0.0
        rows.append(
            f"- {name}: volume_ratio={vr:.4f}, area_ratio={ar:.4f}, "
            f"max_concavity_depth={mc:.4f}"
        )
    return "\n".join(rows)


def design_partition_energy(
    client: VLMClient,
    bone_names: Sequence[str],
    *,
    parents: Sequence[int] = (),
    seam_pairs: Sequence[Tuple[str, str]] = (),
    volume_ratio: Sequence[float] = (),
    area_ratio: Sequence[float] = (),
    max_concavity_depth: Sequence[float] = (),
    object_kind_hint: str = "",
    image_path: Optional[str | Path] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
) -> PartitionEnergyDesign:
    """调 Agent 设计分块边界能量参数。"""
    names = [str(n).strip() for n in bone_names if str(n).strip()]
    if not names:
        raise ValueError("bone_names 不能为空")

    user = PARTITION_ENERGY_USER_TEMPLATE.format(
        object_kind=object_kind_hint.strip() or "（无，按结构判断）",
        bones_table="\n".join(f"- {n}" for n in names),
        hierarchy_table=_format_hierarchy(names, parents),
        seam_table=_format_seams(seam_pairs),
        metrics_table=_format_metrics_table(
            names, volume_ratio, area_ratio, max_concavity_depth,
        ),
    )
    images = [Path(image_path)] if image_path else []

    try:
        response: VLMResponse = client.chat(
            user_prompt=user,
            image_paths=images,
            system=PARTITION_ENERGY_SYSTEM,
            temperature=temperature,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("partition energy VLM 调用失败: %s", exc)
        return PartitionEnergyDesign(error=str(exc))

    text = response.text or ""
    try:
        payload = parse_json_response(text)
    except ValueError as exc:
        return PartitionEnergyDesign(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error=f"JSON 解析失败: {exc}",
        )

    if not isinstance(payload, dict):
        return PartitionEnergyDesign(
            raw_text=text,
            elapsed_seconds=response.elapsed_seconds,
            provider=response.provider,
            model=response.model,
            error="返回顶层不是 JSON 对象",
        )

    return PartitionEnergyDesign(
        spec=payload,
        object_kind=str(payload.get("object_kind") or object_kind_hint or "").strip(),
        raw_text=text,
        elapsed_seconds=response.elapsed_seconds,
        provider=response.provider,
        model=response.model,
        error=None if payload.get("bones") else "spec 缺少 bones",
    )
