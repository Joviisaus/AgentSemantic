"""
VLM 提示模板：
- 三分类（完全符合 / 局部符合 / 完全不符合）
- 局部符合时估计种子数（用于谱聚类分裂）

约定 VLM 必须返回严格的 JSON，便于稳定解析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

LABEL_FULL: Final = "full"
LABEL_PARTIAL: Final = "partial"
LABEL_NONE: Final = "none"

VALID_LABELS: Final = (LABEL_FULL, LABEL_PARTIAL, LABEL_NONE)

CLASSIFY_SYSTEM_PROMPT = (
    "你是一个 3D 模型语义分析助手。"
    "用户会给你 12 个不同视角拼接的渲染图：图中只有一个分块用高亮颜色突出显示，"
    "其余区域用淡色作为参考。"
    "请严格按 JSON 格式回答，禁止解释多余内容、禁止使用 Markdown 代码块包裹。"
)

CLASSIFY_USER_TEMPLATE = """用户给定的语义区域：
「{semantic}」

判定准则（高亮分块 相对于 语义区域 的关系）：
- "full"：高亮分块的所有部分都属于语义区域，是语义区域的一部分（即使只覆盖语义区域的一小块也算）。
  例如：语义=「面部」，高亮=「眼睛」 → full。
- "none"：高亮分块与语义区域完全无关，没有任何交集。
  例如：语义=「面部」，高亮=「腿部」 → none。
- "partial"：高亮分块与语义区域有交集，但分块里同时包含明显不属于语义区域的部分。
  例如：语义=「眼睛」，高亮=「整个面部」 → partial。

请综合 12 个视角整体判断，避免被单一视角误导。
高亮颜色：鲜艳红/橙；非高亮：淡灰。

只能返回如下 JSON 对象，不要附加任何文字：
{{
  "label": "full" | "partial" | "none",
  "confidence": 0.0~1.0,
  "reason": "用中文一句话简述判断依据"
}}
"""


SEED_SYSTEM_PROMPT = (
    "你是一个 3D 模型语义分析助手。"
    "用户给你 12 个视角拼接的渲染图，高亮分块只局部符合用户语义，"
    "现在需要把它再切成若干子块继续判断。"
    "请严格按 JSON 格式回答。"
)

SEED_USER_TEMPLATE = """用户给定的语义区域：
「{semantic}」

高亮分块「局部符合」该语义。请估计需要把这个高亮分块继续切成多少个子块，
以便其中一部分能够"完全符合"语义、另一部分"完全不符合"。

约束：
- 子块数量必须是 ≥ 2 的整数；
- 数量越少越好（避免过度细分），通常 2~6 之间；
- 如果分块结构复杂（如同时包含很多不同语义部位），最多给到 {max_seeds}。

只能返回如下 JSON 对象：
{{
  "seed_count": <int ≥ 2>,
  "reason": "中文简述为什么需要这个粒度"
}}
"""


@dataclass
class ClassifyPrompt:
    system: str
    user: str


def build_classify_prompt(semantic: str) -> ClassifyPrompt:
    return ClassifyPrompt(
        system=CLASSIFY_SYSTEM_PROMPT,
        user=CLASSIFY_USER_TEMPLATE.format(semantic=semantic.strip()),
    )


def build_seed_prompt(semantic: str, max_seeds: int = 8) -> ClassifyPrompt:
    return ClassifyPrompt(
        system=SEED_SYSTEM_PROMPT,
        user=SEED_USER_TEMPLATE.format(
            semantic=semantic.strip(), max_seeds=max(2, int(max_seeds))
        ),
    )
