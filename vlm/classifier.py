"""
分块三分类：调用 VLM 把单个分块判定为
    full     完全符合语义
    partial  局部符合语义
    none     完全不符合语义
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import VLMClient, VLMResponse, parse_json_response
from .prompts import (
    LABEL_FULL,
    LABEL_NONE,
    LABEL_PARTIAL,
    VALID_LABELS,
    build_classify_prompt,
    build_observe_prompt,
    build_reason_prompt,
)

LOGGER = logging.getLogger("vlm.classifier")


@dataclass
class ComponentClassification:
    label: str  # full / partial / none
    confidence: float = 0.0
    reason: str = ""
    raw_text: str = ""
    elapsed_seconds: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
    # —— 结构化推理字段（新 prompt 引入；旧 prompt 下保持空） ——
    covered_parts: List[str] = field(default_factory=list)
    dominant_region: str = ""  # 观察阶段对红色"主集中位置"的一句话描述
    semantic_parts: List[str] = field(default_factory=list)
    extra_parts: List[str] = field(default_factory=list)
    missing_overlap: Optional[bool] = None
    overridden_from: Optional[str] = None  # 若发生服务端兜底改写，记录原始 label
    # —— 主导边判定（用于 max-depth 强制收尾决策）——
    semantic_area_ratio: Optional[float] = None  # VLM 估的"语义面积 / 红色总面积"
    dominant_side: str = ""        # "semantic" | "extra" | "balanced" | ""
    committed_if_forced: str = ""  # "full" | "none" | "" — VLM 在被迫收尾时的最终承诺

    @property
    def is_full(self) -> bool:
        return self.label == LABEL_FULL

    @property
    def is_partial(self) -> bool:
        return self.label == LABEL_PARTIAL

    @property
    def is_none(self) -> bool:
        return self.label == LABEL_NONE


def _normalize_label(value: Any) -> str:
    if not isinstance(value, str):
        return LABEL_NONE
    s = value.strip().lower()
    # 容错：中文回答
    cn_map = {
        "完全符合": LABEL_FULL,
        "完全符合语义": LABEL_FULL,
        "局部符合": LABEL_PARTIAL,
        "局部符合语义": LABEL_PARTIAL,
        "部分符合": LABEL_PARTIAL,
        "完全不符合": LABEL_NONE,
        "完全不符合语义": LABEL_NONE,
        "不符合": LABEL_NONE,
    }
    if s in cn_map:
        return cn_map[s]
    if s in VALID_LABELS:
        return s
    if "full" in s and "partial" not in s:
        return LABEL_FULL
    if "partial" in s or "部分" in s or "局部" in s:
        return LABEL_PARTIAL
    if "none" in s or "no" == s or "不" in s:
        return LABEL_NONE
    return LABEL_NONE


def classify_component(
    client: VLMClient,
    image_path: str | Path,
    semantic: str,
    *,
    model: Optional[str] = None,
) -> ComponentClassification:
    """
    两阶段分类：

      1. **观察阶段**：把高亮拼图发给 VLM，让它**只**列出红色覆盖到的部位
         （prompt 里完全不出现 semantic，规避确认偏差）。
      2. **推理阶段**：把 covered_parts 当作纯文本，连同 semantic 发回给 VLM，
         让它做集合比较得到 label（这一步不带图）。

    任何阶段失败时退化成 label="none" + error。
    """
    image_path = Path(image_path)

    # ---------- 阶段 1：观察 ----------
    obs_prompt = build_observe_prompt()
    try:
        obs_response: VLMResponse = client.chat(
            user_prompt=obs_prompt.user,
            image_paths=[image_path],
            system=obs_prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 观察阶段调用失败 (%s): %s", image_path.name, exc)
        return ComponentClassification(
            label=LABEL_NONE,
            reason="VLM 观察阶段调用失败",
            error=str(exc),
        )

    obs_text = obs_response.text or ""
    try:
        obs_payload = parse_json_response(obs_text)
    except ValueError as exc:
        LOGGER.warning("VLM 观察阶段响应非 JSON (%s): %s", image_path.name, obs_text[:200])
        return ComponentClassification(
            label=LABEL_NONE,
            reason=obs_text.strip()[:200],
            raw_text=obs_text,
            elapsed_seconds=obs_response.elapsed_seconds,
            provider=obs_response.provider,
            model=obs_response.model,
            error=f"观察阶段 JSON 解析失败: {exc}",
            raw_response=obs_response.raw,
        )

    covered = _coerce_str_list(obs_payload.get("covered_parts"))
    dominant_region = str(obs_payload.get("dominant_region") or "").strip()

    # 红色完全没覆盖任何部位 → 直接 none，不必再问推理阶段
    if not covered:
        return ComponentClassification(
            label=LABEL_NONE,
            confidence=0.9,
            reason="观察阶段未发现任何鲜艳红色覆盖部位",
            raw_text=obs_text,
            elapsed_seconds=obs_response.elapsed_seconds,
            provider=obs_response.provider,
            model=obs_response.model,
            raw_response=obs_response.raw,
            covered_parts=[],
            dominant_region=dominant_region,
            semantic_parts=[],
            extra_parts=[],
            missing_overlap=True,
        )

    # ---------- 阶段 2：推理 ----------
    rsn_prompt = build_reason_prompt(semantic, covered)
    try:
        rsn_response: VLMResponse = client.chat(
            user_prompt=rsn_prompt.user,
            image_paths=[],  # 关键：不带图，避免再次幻觉
            system=rsn_prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM 推理阶段调用失败: %s", exc)
        return ComponentClassification(
            label=LABEL_NONE,
            reason="VLM 推理阶段调用失败",
            covered_parts=covered,
            raw_text=obs_text,
            elapsed_seconds=obs_response.elapsed_seconds,
            provider=obs_response.provider,
            model=obs_response.model,
            error=str(exc),
            raw_response=obs_response.raw,
        )

    rsn_text = rsn_response.text or ""
    try:
        rsn_payload = parse_json_response(rsn_text)
    except ValueError as exc:
        LOGGER.warning("VLM 推理阶段响应非 JSON: %s", rsn_text[:200])
        return ComponentClassification(
            label=_normalize_label(rsn_text),
            reason=rsn_text.strip()[:200],
            covered_parts=covered,
            raw_text=obs_text + "\n\n---REASON---\n" + rsn_text,
            elapsed_seconds=obs_response.elapsed_seconds + rsn_response.elapsed_seconds,
            provider=rsn_response.provider,
            model=rsn_response.model,
            error=f"推理阶段 JSON 解析失败: {exc}",
            raw_response=rsn_response.raw,
        )

    label = _normalize_label(rsn_payload.get("label"))
    confidence = float(rsn_payload.get("confidence", 0.0) or 0.0)
    reason = str(rsn_payload.get("reason") or "").strip()
    semantic_parts = _coerce_str_list(rsn_payload.get("semantic_parts"))
    extra = _coerce_str_list(rsn_payload.get("extra_parts"))
    missing = _coerce_bool(rsn_payload.get("missing_overlap"))
    # 主导边 + 强制承诺
    raw_ratio = rsn_payload.get("semantic_area_ratio")
    semantic_area_ratio: Optional[float]
    try:
        semantic_area_ratio = (
            max(0.0, min(1.0, float(raw_ratio))) if raw_ratio is not None else None
        )
    except (TypeError, ValueError):
        semantic_area_ratio = None
    dominant_side = str(rsn_payload.get("dominant_side") or "").strip().lower()
    if dominant_side not in ("semantic", "extra", "balanced", ""):
        dominant_side = ""
    # 如果 VLM 没显式给 dominant_side，但给了 ratio，就推一个
    if not dominant_side and semantic_area_ratio is not None:
        if semantic_area_ratio >= 0.7:
            dominant_side = "semantic"
        elif semantic_area_ratio <= 0.3:
            dominant_side = "extra"
        else:
            dominant_side = "balanced"
    committed_if_forced = (
        str(rsn_payload.get("committed_if_forced") or "").strip().lower()
    )
    if committed_if_forced not in ("full", "none", ""):
        committed_if_forced = ""
    # 兜底：未给 committed_if_forced 时按 dominant_side 推
    if not committed_if_forced:
        if dominant_side == "semantic":
            committed_if_forced = LABEL_FULL
        elif dominant_side in ("extra", "balanced"):
            committed_if_forced = LABEL_NONE

    # 服务端兜底：以结构化字段为准
    overridden_from: Optional[str] = None

    # —— 内部矛盾检测 —— #
    # VLM 偶尔会把"明显不属于语义"的部位塞进 covered_parts，但偷偷在 extra_parts 留空，
    # 让 label 直接走 full（如：covered=[头部整体, 右大腿, 左小腿] / semantic=[头部] / extra=[]）。
    # 这里基于 covered\semantic 重算 inferred_extra；若 inferred_extra 非空但 declared extra 为空，
    # 视为 VLM 自相矛盾的过度声称——demote 到 partial 让 pipeline 再切一刀，把幻觉部位剥出去。
    inferred_extra = _infer_extra_parts(
        covered=covered, semantic_parts=semantic_parts, semantic=semantic,
    )
    contradiction = bool(inferred_extra) and not extra
    if contradiction:
        LOGGER.warning(
            "classifier contradiction: covered=%s, semantic_parts=%s, declared_extra=[], "
            "inferred_extra=%s — demote to partial",
            covered, semantic_parts, inferred_extra,
        )
        # 用推算出的 extra 填回字段，让下游 / 报告看到真实情况
        extra = inferred_extra
        # 既然 covered 里混着非语义部位，semantic_area_ratio 不可能 ≥0.7
        if semantic_area_ratio is not None and semantic_area_ratio > 0.5:
            semantic_area_ratio = 0.5
        # dominant 强制改成 balanced，强制承诺改成 none（防止收尾时被认作 full）
        dominant_side = "balanced"
        committed_if_forced = LABEL_NONE

    if missing is True:
        forced = LABEL_NONE
    elif extra:
        forced = LABEL_PARTIAL
    elif semantic_parts:
        forced = LABEL_FULL
    else:
        forced = label
    if forced != label:
        LOGGER.info(
            "classifier override: label %s -> %s (extra=%s, missing=%s, contradiction=%s)",
            label, forced, extra, missing, contradiction,
        )
        overridden_from = label
        label = forced

    combined_text = obs_text + "\n\n---REASON---\n" + rsn_text
    return ComponentClassification(
        label=label,
        confidence=max(0.0, min(1.0, confidence)),
        reason=reason,
        raw_text=combined_text,
        elapsed_seconds=obs_response.elapsed_seconds + rsn_response.elapsed_seconds,
        provider=rsn_response.provider,
        model=rsn_response.model,
        raw_response={"observe": obs_response.raw, "reason": rsn_response.raw},
        covered_parts=covered,
        dominant_region=dominant_region,
        semantic_parts=semantic_parts,
        extra_parts=extra,
        missing_overlap=missing,
        overridden_from=overridden_from,
        semantic_area_ratio=semantic_area_ratio,
        dominant_side=dominant_side,
        committed_if_forced=committed_if_forced,
    )


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return []


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


# ---------------------------------------------------------------------------
# 内部矛盾检测：从 covered_parts \ semantic_parts 推算"应当出现"的 extra_parts
# ---------------------------------------------------------------------------

# 部位名常见的修饰后缀，匹配前先剥掉它们以做模糊比较。
_PART_NAME_SUFFIXES: tuple[str, ...] = (
    "整体", "全部", "区域", "部分", "部位", "一带", "一圈",
    "上半段", "下半段", "上部", "下部",
)


def _normalize_part_name(name: str) -> str:
    """把部位名归一化：去掉括号补充、空白、常见笼统后缀。"""
    if not name:
        return ""
    s = name.strip()
    # 去掉括号内补充说明（中英括号）
    for lp, rp in (("（", "）"), ("(", ")"), ("[", "]"), ("【", "】")):
        while lp in s and rp in s:
            i = s.find(lp)
            j = s.find(rp, i + 1)
            if i < 0 or j < 0:
                break
            s = (s[:i] + s[j + 1:]).strip()
    # 去掉笼统后缀（"头部整体" → "头部"）
    changed = True
    while changed:
        changed = False
        for suf in _PART_NAME_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                s = s[: -len(suf)]
                changed = True
    return s.strip()


def _part_matches(a: str, b: str) -> bool:
    """两个部位名是否「指同一个东西」：归一化后只要一方是另一方子串就算。"""
    na, nb = _normalize_part_name(a), _normalize_part_name(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


def _infer_extra_parts(
    *,
    covered: List[str],
    semantic_parts: List[str],
    semantic: str,
) -> List[str]:
    """
    根据 covered_parts 和 semantic_parts（以及用户输入 semantic 关键词）推算 extra_parts。

    判定一个 covered 项 `p` 属于 extra 的规则：
      ① p 与任何 semantic_part 都「指不到一起」（_part_matches=False）
      ② p 自身也不直接命中 user-side semantic（substring 匹配）
    满足这两条 → p 是真实的 extra（VLM 不能假装它不存在）。
    """
    if not covered:
        return []
    sem_norm = _normalize_part_name(semantic)
    extras: List[str] = []
    for p in covered:
        if not isinstance(p, str) or not p.strip():
            continue
        # 命中 semantic_parts 视作语义内
        if any(_part_matches(p, sp) for sp in semantic_parts):
            continue
        # 命中用户原始 semantic 关键词也算语义内（处理 semantic_parts 留空的情况）
        if sem_norm and (sem_norm in p or p in sem_norm):
            continue
        extras.append(p.strip())
    return extras
