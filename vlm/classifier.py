"""
Block three-way classification: call VLM to label a single block as
    full     fully matches the semantic
    partial  partially matches the semantic
    none     does not match the semantic
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import VLMClient, VLMResponse, parse_json_response
from .prompts import (
    LABEL_FULL,
    LABEL_NONE,
    LABEL_PARTIAL,
    VALID_LABELS,
    ObjectContext,
    _looks_like_portable_prop_semantic,
    build_classify_prompt,
    build_observe_prompt,
    build_reason_prompt,
    build_revive_prompt,
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
    # Structured reasoning fields (from newer prompts; empty under legacy prompts)
    covered_parts: List[str] = field(default_factory=list)
    dominant_region: str = ""  # one-sentence description of where red concentrates
    semantic_parts: List[str] = field(default_factory=list)
    extra_parts: List[str] = field(default_factory=list)
    missing_overlap: Optional[bool] = None
    overridden_from: Optional[str] = None  # original label if server-side override applied
    # Dominant-side fields (for max-depth forced termination)
    semantic_area_ratio: Optional[float] = None  # VLM estimate: semantic area / red area
    dominant_side: str = ""        # "semantic" | "extra" | "balanced" | ""
    committed_if_forced: str = ""  # "full" | "none" | "" — VLM commitment if forced to stop

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
    # Tolerate Chinese label strings from VLM
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
    context: Optional[ObjectContext] = None,
    media_type: str = "mesh",
    force_terminate: bool = False,
    n_views: int = 12,
    memory: str = "",
) -> ComponentClassification:
    """
    Two-stage classification:

      1. **Observation**: send the highlight collage to VLM and ask it to list only
         parts covered by red (semantic is omitted from the prompt to avoid confirmation bias).
      2. **Reasoning**: send covered_parts as plain text plus the semantic back to VLM
         for set comparison to label (no image in this step).

    `media_type`: "mesh" | "pointcloud" — controls render description (triangles vs point spheres).
    `force_terminate`: when True, inject forced-termination note in reasoning; emphasize committed_if_forced.
    `n_views`: number of views in the collage (passed into observe prompt).
    `memory`: optional text digest of earlier-iteration verdicts on the same model; injected
      into the reasoning stage only (kept out of observation to avoid confirmation bias).

    Any stage failure degrades to label="none" + error.
    """
    image_path = Path(image_path)
    mt = "pointcloud" if media_type == "pointcloud" else "mesh"

    # ---------- Stage 1: observation ----------
    obs_prompt = build_observe_prompt(context, media_type=mt, n_views=n_views)
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
        LOGGER.warning("VLM observation call failed (%s): %s", image_path.name, exc)
        return ComponentClassification(
            label=LABEL_NONE,
            reason="VLM observation call failed",
            error=str(exc),
        )

    obs_text = obs_response.text or ""
    try:
        obs_payload = parse_json_response(obs_text)
    except ValueError as exc:
        LOGGER.warning("VLM observation response not JSON (%s): %s", image_path.name, obs_text[:200])
        return ComponentClassification(
            label=LABEL_NONE,
            reason=obs_text.strip()[:200],
            raw_text=obs_text,
            elapsed_seconds=obs_response.elapsed_seconds,
            provider=obs_response.provider,
            model=obs_response.model,
            error=f"observation JSON parse failed: {exc}",
            raw_response=obs_response.raw,
        )

    covered = _coerce_str_list(obs_payload.get("covered_parts"))
    dominant_region = str(obs_payload.get("dominant_region") or "").strip()

    # No red coverage at all → none immediately, skip reasoning
    if not covered:
        return ComponentClassification(
            label=LABEL_NONE,
            confidence=0.9,
            reason="observation found no bright red coverage",
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

    # ---------- Stage 2: reasoning ----------
    rsn_prompt = build_reason_prompt(
        semantic, covered,
        context,
        media_type=mt,
        force_terminate=force_terminate,
        dominant_region=dominant_region,
        memory_block=memory,
    )
    try:
        # oracle 模式：把面板图也传给 reason 阶段，供 AssistantVLMClient 依块点索引+
        # GT 重叠判定 full/partial/none（正常 VLM 运行仍不传图，避免二次幻觉）。
        _rsn_images = [image_path] if os.environ.get("AGENTSEMANTIC_VLM_ORACLE") else []
        rsn_response: VLMResponse = client.chat(
            user_prompt=rsn_prompt.user,
            image_paths=_rsn_images,
            system=rsn_prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("VLM reasoning call failed: %s", exc)
        return ComponentClassification(
            label=LABEL_NONE,
            reason="VLM reasoning call failed",
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
        LOGGER.warning("VLM reasoning response not JSON: %s", rsn_text[:200])
        return ComponentClassification(
            label=_normalize_label(rsn_text),
            reason=rsn_text.strip()[:200],
            covered_parts=covered,
            raw_text=obs_text + "\n\n---REASON---\n" + rsn_text,
            elapsed_seconds=obs_response.elapsed_seconds + rsn_response.elapsed_seconds,
            provider=rsn_response.provider,
            model=rsn_response.model,
            error=f"reasoning JSON parse failed: {exc}",
            raw_response=rsn_response.raw,
        )

    label = _normalize_label(rsn_payload.get("label"))
    confidence = float(rsn_payload.get("confidence", 0.0) or 0.0)
    reason = str(rsn_payload.get("reason") or "").strip()
    semantic_parts = _coerce_str_list(rsn_payload.get("semantic_parts"))
    extra = _coerce_str_list(rsn_payload.get("extra_parts"))
    extra, absorbed = _filter_in_scope_extra_parts(
        extra, semantic=semantic, semantic_parts=semantic_parts,
    )
    extra, dup_absorbed = _filter_covered_semantic_duplicate_extras(
        extra,
        covered=covered,
        semantic=semantic,
        semantic_parts=semantic_parts,
    )
    absorbed = absorbed + dup_absorbed
    if absorbed:
        LOGGER.info(
            "classifier absorbed in-scope/duplicate extras (semantic=%r): %s",
            semantic, absorbed,
        )
    missing = _coerce_bool(rsn_payload.get("missing_overlap"))
    # Dominant side + forced commitment
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
    # Infer dominant_side from ratio if VLM omitted it
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
    # Fallback: infer committed_if_forced from dominant_side
    if not committed_if_forced:
        if dominant_side == "semantic":
            committed_if_forced = LABEL_FULL
        elif dominant_side in ("extra", "balanced"):
            committed_if_forced = LABEL_NONE

    # Server-side overrides based on structured fields
    overridden_from: Optional[str] = None

    # Internal contradiction detection
    # VLM may list non-semantic parts in covered_parts but leave extra_parts empty,
    # yielding label=full (e.g. covered=[head, right thigh, left calf] / semantic=[head] / extra=[]).
    # Recompute inferred_extra from covered \ semantic; if inferred_extra is non-empty but declared extra is empty,
    # demote to partial so the pipeline can split again and peel off hallucinated parts.
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
        # Fill back inferred extra for downstream / reports
        extra = inferred_extra
        # Mixed non-semantic parts in covered → cap semantic_area_ratio
        if semantic_area_ratio is not None and semantic_area_ratio > 0.5:
            semantic_area_ratio = 0.5
        # Force balanced / none commitment to avoid accepting as full at max-depth
        dominant_side = "balanced"
        committed_if_forced = LABEL_NONE

    # Embedded prop promotion (coarse observation may list only limb, miss held object)
    # Coarse point-cloud observation may list torso+arm only, miss embedded gun/stick → false none.
    promoted_embedded = False
    if (
        _looks_like_portable_prop_semantic(semantic)
        and (missing is True or label == LABEL_NONE)
        and _covered_may_embed_prop(covered, dominant_region)
    ):
        promoted_embedded = True
        LOGGER.info(
            "classifier promote embedded prop: semantic=%r covered=%s dominant=%r → partial",
            semantic, covered, dominant_region,
        )
        missing = False
        sem_name = semantic.strip()
        if not semantic_parts:
            semantic_parts = [f"{sem_name}（可能嵌在肢体区域，待切分）"]
        if not extra:
            extra = [
                p for p in covered
                if isinstance(p, str) and p.strip()
            ]
        if semantic_area_ratio is None or semantic_area_ratio <= 0.0:
            semantic_area_ratio = 0.35
        dominant_side = "balanced"
        committed_if_forced = LABEL_NONE
        if label == LABEL_NONE and overridden_from is None:
            overridden_from = label
        label = LABEL_PARTIAL

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


def ask_revive_pick(
    client: VLMClient,
    semantic: str,
    candidates: Sequence[Dict[str, Any]],
    *,
    image_paths: Sequence[str | Path] = (),
    model: Optional[str] = None,
) -> Optional[str]:
    """Ask the VLM to pick the single most-likely leaf block to revive as full.

    Used as a last-resort fallback when the whole model ended with zero full blocks.
    Returns the chosen ``block_id`` (validated against ``candidates``) or ``None`` if
    the call/parse failed or the answer is not a listed id (caller then falls back to a
    heuristic pick).
    """
    if not candidates:
        return None
    valid_ids = [str(c.get("block_id")) for c in candidates if c.get("block_id") is not None]
    if not valid_ids:
        return None

    prompt = build_revive_prompt(semantic, candidates)
    try:
        resp: VLMResponse = client.chat(
            user_prompt=prompt.user,
            image_paths=list(image_paths),
            system=prompt.system,
            temperature=0.0,
            json_mode=True,
            model=model,
        )
    except RuntimeError as exc:
        LOGGER.warning("revive pick VLM call failed: %s", exc)
        return None

    try:
        payload = parse_json_response(resp.text or "")
    except ValueError as exc:
        LOGGER.warning("revive pick response not JSON: %s (%s)", (resp.text or "")[:200], exc)
        return None

    chosen = payload.get("chosen_block_id") or payload.get("block_id")
    if chosen is not None:
        chosen = str(chosen).strip()
        if chosen in valid_ids:
            return chosen
        # Tolerate fuzzy/substring match (e.g. trailing punctuation)
        for vid in valid_ids:
            if chosen == vid or chosen.endswith(vid) or vid in chosen:
                return vid

    # Tolerate index-based answers ("chosen_index": 0)
    idx = payload.get("chosen_index")
    try:
        if idx is not None:
            i = int(idx)
            if 0 <= i < len(valid_ids):
                return valid_ids[i]
    except (TypeError, ValueError):
        pass
    return None


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
# Internal contradiction: infer extra_parts from covered_parts \ semantic_parts
# ---------------------------------------------------------------------------

# Coarse semantic term → child-part keywords (to strip false-positive extras)
_SEMANTIC_SCOPE_CHILDREN: Dict[str, Tuple[str, ...]] = {
    "头": (
        "耳", "脸", "面", "额", "颅", "脑", "鼻", "眼", "嘴", "口", "颊", "下巴",
        "颌", "眉", "发", "顶", "鬓", "颞",
    ),
    "脸": ("眼", "鼻", "嘴", "口", "颊", "额", "下巴", "眉", "唇"),
    "面": ("眼", "鼻", "嘴", "口", "颊", "额", "下巴", "眉", "唇"),
    "手": ("指", "掌", "腕", "拳", "拇"),
    "脚": ("趾", "踝", "鞋", "掌"),
    "腿": ("大腿", "小腿", "膝", "踝"),
    "臂": ("上臂", "前臂", "肘", "腕", "肱", "肱骨"),
    "躯干": ("胸", "背", "腰", "腹", "臀", "肋", "肩"),
    "枪": ("管", "握", "扳", "弹", "机", "匣", "托", "瞄", "身"),
    "水枪": ("管", "握", "扳", "弹", "机", "匣", "身", "泵", "槽"),
}


def _semantic_scope_keys(semantic: str, semantic_parts: Sequence[str]) -> List[str]:
    """Collect semantic anchors for sub-part scope checks (user term + semantic_parts)."""
    keys: List[str] = []
    for sp in semantic_parts:
        n = _normalize_part_name(sp)
        if n and n not in keys:
            keys.append(n)
    # If Stage 2 already parsed semantic_parts, trust those parsed include terms.
    # The raw semantic can contain explicit exclusions ("handle excluding body");
    # using the whole sentence as scope would incorrectly absorb excluded parts.
    if not keys:
        sem = _normalize_part_name(semantic)
        if sem:
            keys.append(sem)
    return keys


def _part_within_semantic_scope(
    part: str,
    *,
    semantic: str,
    semantic_parts: Sequence[str],
) -> bool:
    """Whether part falls within user semantic scope (incl. sub-parts, e.g. ear ⊂ head)."""
    if not part or not part.strip():
        return False
    p_norm = _normalize_part_name(part)
    if not p_norm:
        return False

    if any(_part_matches(part, sp) for sp in semantic_parts):
        return True

    for key in _semantic_scope_keys(semantic, semantic_parts):
        if not key:
            continue
        if key in p_norm or p_norm in key:
            return True
        for scope_parent, children in _SEMANTIC_SCOPE_CHILDREN.items():
            if not (_part_matches(key, scope_parent) or scope_parent in key or key in scope_parent):
                continue
            for child in children:
                if child in p_norm or p_norm in child:
                    return True
    return False


def _semantic_content_tokens(text: str) -> set[str]:
    """Extract lowercase content tokens (skip very short / generic words)."""
    stop = {"one", "two", "both", "the", "and", "with", "for", "end", "side", "part", "area"}
    return {
        tok
        for tok in re.findall(r"[a-z]{3,}", (text or "").lower())
        if tok not in stop
    }


def _semantic_token_overlap(
    part: str,
    semantic: str,
    semantic_parts: Sequence[str],
) -> bool:
    """Whether ``part`` shares meaningful tokens with the target semantic scope."""
    part_tokens = _semantic_content_tokens(part)
    if not part_tokens:
        return False
    scope_tokens: set[str] = set()
    for sp in semantic_parts:
        scope_tokens |= _semantic_content_tokens(str(sp))
    # Only fall back to the raw semantic when no parsed include terms exist.
    # Raw semantic often contains excluded neighbors and must not define scope.
    if not scope_tokens:
        scope_tokens = _semantic_content_tokens(semantic)
    return bool(part_tokens & scope_tokens)


def _filter_in_scope_extra_parts(
    extras: Sequence[str],
    *,
    semantic: str,
    semantic_parts: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Remove extras that are sub-parts within semantic scope. Returns (true_extras, absorbed)."""
    true_extras: List[str] = []
    absorbed: List[str] = []
    for item in extras:
        if not isinstance(item, str) or not item.strip():
            continue
        if _part_within_semantic_scope(
            item.strip(), semantic=semantic, semantic_parts=semantic_parts,
        ) or _semantic_token_overlap(item.strip(), semantic, semantic_parts):
            absorbed.append(item.strip())
        else:
            true_extras.append(item.strip())
    return true_extras, absorbed


def _filter_covered_semantic_duplicate_extras(
    extras: Sequence[str],
    *,
    covered: Sequence[str],
    semantic: str,
    semantic_parts: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Drop extras that duplicate covered semantic regions (VLM double-counting).

    Example: semantic="ear pad", covered=["one end pad"], extra=["one end pad"].
    The extra is the same highlighted region, not overflow — keep label at full.
    """
    true_extras: List[str] = []
    absorbed: List[str] = []
    for item in extras:
        if not isinstance(item, str) or not item.strip():
            continue
        item = item.strip()
        matches_covered = any(
            _part_matches(item, c) or _semantic_token_overlap(item, c, [])
            for c in covered
        )
        in_semantic = (
            _part_within_semantic_scope(item, semantic=semantic, semantic_parts=semantic_parts)
            or any(_part_matches(item, sp) for sp in semantic_parts)
            or _semantic_token_overlap(item, semantic, semantic_parts)
        )
        if matches_covered and in_semantic:
            absorbed.append(item)
        else:
            true_extras.append(item)
    return true_extras, absorbed


_GRASPING_PART_KEYWORDS: Tuple[str, ...] = (
    "手", "臂", "掌", "指", "腕", "拳", "肩", "肘", "腋", "肱",
)
_PROP_EMBED_CONTEXT_HINTS: Tuple[str, ...] = (
    "持", "握", "扛", "举", "拿", "提", "端", "伸出", "末端", "物件", "物体",
    "长条", "棍", "枪", "棒", "瓶", "工具",
)


def _covered_may_embed_prop(
    covered: Sequence[str],
    dominant_region: str,
) -> bool:
    """Whether limb coverage may embed a held prop (coarse observation granularity)."""
    parts_text = " ".join(str(p) for p in covered if p)
    blob = f"{parts_text} {dominant_region or ''}"
    if any(k in blob for k in _GRASPING_PART_KEYWORDS):
        return True
    return any(h in blob for h in _PROP_EMBED_CONTEXT_HINTS)


# Common part-name suffixes stripped before fuzzy matching
_PART_NAME_SUFFIXES: tuple[str, ...] = (
    "整体", "全部", "区域", "部分", "部位", "一带", "一圈",
    "上半段", "下半段", "上部", "下部",
)


def _normalize_part_name(name: str) -> str:
    """Normalize part name: strip parenthetical notes, whitespace, generic suffixes."""
    if not name:
        return ""
    s = name.strip()
    # Strip parenthetical notes (CN/EN brackets)
    for lp, rp in (("（", "）"), ("(", ")"), ("[", "]"), ("【", "】")):
        while lp in s and rp in s:
            i = s.find(lp)
            j = s.find(rp, i + 1)
            if i < 0 or j < 0:
                break
            s = (s[:i] + s[j + 1:]).strip()
    # Strip generic suffixes (e.g. "head whole" → "head")
    changed = True
    while changed:
        changed = False
        for suf in _PART_NAME_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf):
                s = s[: -len(suf)]
                changed = True
    return s.strip()


def _part_matches(a: str, b: str) -> bool:
    """Whether two part names refer to the same thing (substring match after normalize)."""
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
    Infer extra_parts from covered_parts and semantic_parts (and user semantic keyword).

    A covered item `p` is extra when:
      ① p is within semantic scope (incl. sub-parts) → not extra
      ② p matches no semantic_part (_part_matches=False)
      ③ p does not substring-match user semantic
    ② and ③ and not ① → p is a true extra.
    """
    if not covered:
        return []
    sem_norm = _normalize_part_name(semantic) if not semantic_parts else ""
    extras: List[str] = []
    for p in covered:
        if not isinstance(p, str) or not p.strip():
            continue
        if _part_within_semantic_scope(
            p.strip(), semantic=semantic, semantic_parts=semantic_parts,
        ):
            continue
        # Matches semantic_parts → in semantic scope
        if any(_part_matches(p, sp) for sp in semantic_parts):
            continue
        # Substring match on user semantic (when semantic_parts empty)
        if sem_norm and (sem_norm in p or p in sem_norm):
            continue
        extras.append(p.strip())
    return extras
