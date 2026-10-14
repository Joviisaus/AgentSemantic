"""人工 oracle VLM 客户端：把需要 token 的真实 VLM 换成「由助手亲自决策」。

工作模式（三阶段）：
  1. collect：跑 benchmark，每次 VLM 调用都把 (system + prompt + 渲染图) 落盘到
     ``requests/<id>.json``，并写一份 manifest ``requests.jsonl`` + ``tally.json``。
     没有人工答案时，返回确定性「规则」答案，保证流程 token-free 跑通。
  2. decide：助手读取 semantic / view 两类请求的渲染图，亲自写答案到
     ``answers/<id>.json``（内容即该 VLM 调用本应返回的 JSON 文本）。
  3. replay：用同样命令重跑；客户端优先返回 ``answers/<id>.json`` 里的人工答案，
     其余仍用规则答案。最终预测落到 ``results/ours_pointcloud_oracle/``。

ID 稳定性：semantic / view 用「<key>::<part_idx>::<tag>」结构化 id（从渲染图路径解析
key 与 part_idx），与规则内容无关，保证 collect 与 replay 阶段 id 一致、人工答案可命中。
其余调用（classify 两阶段 / seed / postprocess / revive）用内容哈希 id，仅走规则、不做人工。

注意：classify / seed / postprocess 三类「机械决策」按约定由确定性规则替代（无视觉输入时
无法区分目标部件，规则为：observe 返回高亮区域被覆盖、reason 返回 full、seed 返回 2、
postprocess 返回空操作）。这是「ours 方法去掉 VLM 视觉判断」的 ablation，会在报告里写明。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vlm.client import VLMClient, VLMResponse, parse_json_response

try:
    from rendering.heatmap_renderer import DEFAULT_OVERVIEW_VIEW, VIEW_GRID
except Exception:  # pragma: no cover
    VIEW_GRID = [(22.5, 45.0)] * 12
    DEFAULT_OVERVIEW_VIEW = (22.5, 45.0)

LOGGER = logging.getLogger("benchmark.vlm_oracle")

ORACLE_ROOT = Path(__file__).resolve().parent / "vlm_oracle"
REQUESTS_DIR = ORACLE_ROOT / "requests"
ANSWERS_DIR = ORACLE_ROOT / "answers"
MANIFEST = ORACLE_ROOT / "requests.jsonl"
TALLY = ORACLE_ROOT / "tally.json"

_SEQ = {"n": 0}
_TALLY: Dict[str, int] = {}

# 公平模式（AGENTSEMANTIC_VLM_FAIR=1）：
#   - GT 仅用于提供部件名（语义文本），分类阶段绝不读 GT 掩码/高亮图。
#   - 分裂由几何（block 大小）驱动：块 >= FAIR_MIN_BLOCK 点 → partial（继续分裂），
#     否则 → full（终止叶子，最终由助手肉眼看图判定该叶子是否属于目标部件）。
#   - observe/reason 的确定性规则在 FAIR 下不加载 GT；仅有助手写的 answers/<id>.json
#     （逐叶子块看渲染面板做出的视觉判定）会被采纳。
FAIR = bool(os.environ.get("AGENTSEMANTIC_VLM_FAIR"))
FAIR_MIN_BLOCK = int(os.environ.get("AGENTSEMANTIC_FAIR_MIN_BLOCK", "20"))


def _fallback_view_index() -> int:
    target = tuple(DEFAULT_OVERVIEW_VIEW)  # type: ignore[assignment]
    best, best_d = 0, float("inf")
    for i, (elev, azim) in enumerate(VIEW_GRID):
        d = (elev - target[0]) ** 2 + (azim - target[1]) ** 2
        if d < best_d:
            best_d, best = d, i
    return best


def _extract_field(text: str, key: str) -> str:
    m = re.search(re.escape(key) + r"\s*=\s*([^\n]+)", text)
    return m.group(1).strip() if m else ""


def _infer_tag(system: str, user: str, image_paths: Sequence[Path]) -> str:
    s = (system or "").lower()
    u = (user or "").lower()
    if "point-cloud asset analyst" in s or "asset analyst" in s:
        return "semantic"
    if "视图选择器" in s or "view selector" in s:
        return "view"
    if "post-processing expert" in s or "segmentation post-processing" in s:
        return "postprocess"
    if "revive" in s or "revive" in u:
        return "revive"
    # 注意顺序：reason 的 system 文案里含有 "from the visual observer"，必须先在
    # reason 关键字（covered_parts / set comparison / classification (label)）命中，
    # 否则会被 "visual observer" 误判成 observe，导致 reason 阶段套用 observe 规则
    # （缺少 semantic_parts）→ classifier 矛盾降级 → partial 爆炸。
    if "covered_parts" in s or "set comparison" in s or "classification (label)" in s:
        return "reason"
    if "visual observer" in s:
        return "observe"
    if "seed_count" in s or "split into sub-blocks" in s or "partially matches the user semantic" in s:
        return "seed"
    if "split strategy" in s or "split_strategy" in u:
        return "split_strategy"
    return "observe" if image_paths else "other"


def _parse_key_part(image_paths: Sequence[Path]):
    for p in image_paths:
        name = Path(p).name
        if name in ("gt_semantic_highlight.png", "view_thumbnails.png"):
            parts = Path(p).parts
            if len(parts) >= 3:
                nn_part = parts[-2]
                key = parts[-3]
                m = re.match(r"(\d+)", nn_part)
                pid = int(m.group(1)) if m else 0
                return key, pid
    return None, None


def _structured_id(tag: str, image_paths: Sequence[Path]) -> Optional[str]:
    if tag in ("semantic", "view"):
        key, pid = _parse_key_part(image_paths)
        if key:
            # 用 __ 而非 :: ：Windows 文件名不允许冒号
            return f"{key}__{pid:02d}__{tag}"
    return None


def _hash_id(system: str, user: str, image_paths: Sequence[Path]) -> str:
    blob = "\n###\n".join(
        [system or "", user or "", "|".join(str(p) for p in image_paths)]
    )
    return "h" + hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 确定性规则答案（无视觉输入时的兜底）
# ---------------------------------------------------------------------------

def _rule_semantic(user_prompt: str) -> Dict[str, Any]:
    cat = _extract_field(user_prompt, "object_model")
    part = _extract_field(user_prompt, "target_part_label")
    cat = cat or "object"
    part = part or "part"
    semantic = f"the {part} of the {cat}"
    return {
        "object_model": cat,
        "target_location": "",
        "red_covers": [part],
        "grey_only": [],
        "include": [part],
        "exclude": [],
        "semantic": semantic,
    }


def _rule_view() -> Dict[str, Any]:
    idx = _fallback_view_index()
    return {
        "best_index": idx + 1,
        "confidence": 0.5,
        "classify_view_indices": [idx + 1],
        "reason": "rule: fallback overview view + pipeline-expanded diverse classify views",
    }


def _rule_observe(panel_path: Optional[str] = None) -> Dict[str, Any]:
    """公平模式：不读 GT，仅声明块自身几何被高亮（覆盖面由助手看图决定）。

    非公平（oracle）模式：GT 重叠规则（ours_pointcloud_oracle 的核心 ablation）：

    分类面板把「当前部件」以红色高亮渲染在 block 上；真实 VLM 读红区判断该块是否
    含有目标部件。这里用 GT 部件掩膜与该块点集的重叠分数直接模拟这一观察判断：

      - 块内只要有目标部件点（frac > 0）   → 红区存在，observe 报 covered
      - 块内完全没有目标部件点（frac == 0） → 无红，observe 报 none（直接丢弃）

    full / partial / none 的判定交给 reason 阶段（见 _rule_reason）。因此 mixed 块
    （0 < frac < 0.5）会在 reason 被判 partial，pipeline 继续谱聚类分裂，直到隔离出
    纯目标块（full）或纯非目标块（none）——构造一个确定性的、GT 派生的上界 ablation，
    而不会因为初始聚类无法隔离细长部件（如包带）而退化成低 IoU。
    """
    if FAIR:
        return {
            "covered_parts": ["the highlighted region"],
            "dominant_region": "the highlighted region",
            "notes": "fair: block's own geometry highlighted; coverage/part-membership "
                     "decided visually by assistant from the panel (no GT consulted)",
        }
    key, pidx, bid = _parse_panel_key_part(panel_path)
    if key is None or pidx is None:
        return {
            "covered_parts": ["the highlighted region"],
            "dominant_region": "the highlighted region",
            "notes": "rule: cannot locate block from panel path, assume covered",
        }
    labels = _get_gt_labels(key)
    face_ids = _load_block_points(panel_path)
    if labels is None or face_ids is None or not len(face_ids):
        return {
            "covered_parts": ["the highlighted region"],
            "dominant_region": "the highlighted region",
            "notes": "rule: missing GT/pts sidecar, assume covered",
        }
    blk = np.asarray(face_ids, dtype=np.int64)
    blk = blk[(blk >= 0) & (blk < len(labels))]
    if not blk.size:
        return {
            "covered_parts": [],
            "dominant_region": "",
            "notes": "rule: empty block, none",
        }
    frac = float(np.mean(labels[blk] == pidx))
    if frac > 0.0:
        return {
            "covered_parts": ["the highlighted region"],
            "dominant_region": "the highlighted region",
            "notes": f"rule: GT overlap={frac:.2f}>0 -> red present (covered); reason decides full/partial/none",
        }
    return {
        "covered_parts": [],
        "dominant_region": "",
        "notes": f"rule: GT overlap={frac:.2f}==0 -> no red, none",
    }


def _rule_reason(panel_path: Optional[str] = None) -> Dict[str, Any]:
    """公平模式：几何驱动分裂（不读 GT）。

    块大小（来自 _render_block_panel 落盘的 .pts.json sidecar）：
      - size >= FAIR_MIN_BLOCK  → partial（pipeline 继续谱聚类分裂；构造候选叶子集）
      - size <  FAIR_MIN_BLOCK  → full（终止叶子；最终由助手肉眼看图判 full/none）

    非公平（oracle）模式：GT 重叠驱动的 full/partial/none 判定（observe 已确认块内存在红区）：

      - frac >= 0.5            → full（块主要是目标部件）
      - 0 < frac < 0.5         → partial（块混合，pipeline 继续分裂；
                                  到 max_depth 时按 committed_if_forced 落 full/none）
      - frac == 0              → none（防御性，observe 一般已拦截）

    返回 extra_parts=["other geometry"] 表示 mixed，触发 classifier 的 partial 强约束
    （forced=partial），避免被「有 semantic_parts 即 full」的兜底逻辑覆盖成 full、
    从而真的进行分裂、提升对细长部件的捕获率。
    """
    if FAIR:
        face_ids = _load_block_points(panel_path)
        size = len(face_ids) if face_ids else 0
        if size >= FAIR_MIN_BLOCK:
            return {
                "label": "partial",
                "confidence": 0.5,
                "reason": f"fair: block size={size}>={FAIR_MIN_BLOCK} -> keep splitting (geometry, no GT)",
                "semantic_parts": ["the highlighted region"],
                "extra_parts": ["other geometry"],
                "missing_overlap": False,
                "dominant_side": "balanced",
                "committed_if_forced": "full",
            }
        return {
            "label": "full",
            "confidence": 0.5,
            "reason": f"fair: block size={size}<{FAIR_MIN_BLOCK} -> terminal leaf (assistant classifies visually)",
            "semantic_parts": ["the highlighted region"],
            "extra_parts": [],
            "missing_overlap": False,
            "dominant_side": "semantic",
            "committed_if_forced": "full",
        }
    key, pidx, bid = _parse_panel_key_part(panel_path)
    if key is not None and pidx is not None:
        labels = _get_gt_labels(key)
        face_ids = _load_block_points(panel_path)
        if labels is not None and face_ids is not None and len(face_ids):
            blk = np.asarray(face_ids, dtype=np.int64)
            blk = blk[(blk >= 0) & (blk < len(labels))]
            if blk.size:
                frac = float(np.mean(labels[blk] == pidx))
                if frac >= 0.5:
                    return {
                        "label": "full",
                        "confidence": round(min(1.0, 0.5 + frac / 2.0), 2),
                        "reason": f"rule: GT overlap={frac:.2f}>=0.5 -> full",
                        "semantic_parts": ["the highlighted region"],
                        "extra_parts": [],
                        "missing_overlap": False,
                        "dominant_side": "semantic",
                        "committed_if_forced": "full",
                    }
                if frac > 0.0:
                    committed = "full" if frac >= 0.3 else "none"
                    dominant = (
                        "semantic" if frac >= 0.5 else ("extra" if frac < 0.3 else "balanced")
                    )
                    return {
                        "label": "partial",
                        "confidence": round(max(0.3, min(0.7, frac)), 2),
                        "reason": f"rule: GT overlap={frac:.2f} in (0,0.5) -> partial (split)",
                        "semantic_parts": ["the highlighted region"],
                        "extra_parts": ["other geometry"],
                        "missing_overlap": False,
                        "dominant_side": dominant,
                        "committed_if_forced": committed,
                    }
                return {
                    "label": "none",
                    "confidence": 0.9,
                    "reason": "rule: GT overlap==0 -> none",
                    "semantic_parts": [],
                    "extra_parts": [],
                    "missing_overlap": True,
                    "committed_if_forced": "none",
                }
    # 兜底：拿不到面板/GT 时不分裂，直接接受为 full（与旧 ablation 行为一致，保证流程跑通）
    return {
        "label": "full",
        "confidence": 0.9,
        "reason": "rule: no panel overlap info, accept as full (no vision)",
        "semantic_parts": ["the highlighted region"],
        "extra_parts": [],
        "missing_overlap": False,
        "dominant_side": "semantic",
        "committed_if_forced": "full",
    }


# ---------------------------------------------------------------------------
# GT / 块点索引辅助（供 _rule_observe 的 GT 重叠判定）
# ---------------------------------------------------------------------------

_ORACLE_GT_CACHE: Dict[str, Any] = {}


def _get_gt_labels(key: str) -> Optional[np.ndarray]:
    if key in _ORACLE_GT_CACHE:
        return _ORACLE_GT_CACHE[key]
    try:
        from benchmark import shapenetpart as _D

        gt = _D.load_gt(key)
        labels = np.asarray(gt.labels, dtype=np.int64)
        _ORACLE_GT_CACHE[key] = labels
        return labels
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("oracle GT load failed %s: %s", key, exc)
        return None


def _parse_panel_key_part(panel_path: Optional[str]):
    """从 block 面板路径解析 (key, part_idx, block_id)。

    面板路径形如：<work_dir>/<key>/<NN_partname>/panels/block_<block_id>.png
    """
    if not panel_path:
        return None, None, None
    p = Path(panel_path)
    if not (p.name.startswith("block_") and p.parent.name == "panels"):
        return None, None, None
    part_dir = p.parent.parent
    key_dir = part_dir.parent
    key = key_dir.name
    m = re.match(r"(\d+)", part_dir.name)
    pidx = int(m.group(1)) if m else 0
    bid = p.stem[len("block_"):]
    return key, pidx, bid


def _load_block_points(panel_path: Optional[str]) -> Optional[List[int]]:
    """读取 _render_block_panel 落盘的块→点索引 sidecar。"""
    if not panel_path:
        return None
    pts_path = Path(panel_path).with_suffix(".pts.json")
    try:
        if pts_path.is_file():
            return json.loads(pts_path.read_text(encoding="utf-8")).get("face_ids")
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("oracle block pts load failed %s: %s", panel_path, exc)
    return None


def _rule_seed() -> Dict[str, Any]:
    return {"seed_count": 2, "reason": "rule: default seed_count=2"}


def _rule_postprocess() -> Dict[str, Any]:
    return {"operations": [], "reason": "rule: keep selection as-is (no vision)"}


def _rule_revive() -> Dict[str, Any]:
    return {"chosen_block_id": "", "reason": "rule: no revive"}


def _rule_for(tag: str, user_prompt: str, image_paths: Sequence[str] = ()) -> Dict[str, Any]:
    if tag == "semantic":
        return _rule_semantic(user_prompt)
    if tag == "view":
        return _rule_view()
    if tag == "observe":
        panel = image_paths[0] if image_paths else None
        return _rule_observe(panel)
    if tag == "reason":
        panel = image_paths[0] if image_paths else None
        return _rule_reason(panel)
    if tag == "seed":
        return _rule_seed()
    if tag == "postprocess":
        return _rule_postprocess()
    if tag == "revive":
        return _rule_revive()
    return {"note": "rule: passthrough", "label": "full"}


class AssistantVLMClient(VLMClient):
    """人工 oracle：优先返回 answers/<id>.json，否则返回确定性规则答案。"""

    def __init__(self, config=None):
        cfg = config or SimpleNamespace(
            name="oracle",
            default_model="assistant",
            max_retries=1,
            temperature=0.0,
            max_tokens=2048,
        )
        super().__init__(cfg)

    # 绕过基类 HTTP 重试，直接走 oracle 逻辑
    def chat(
        self,
        user_prompt: str,
        image_paths: Sequence[str | Path] = (),
        *,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        model: Optional[str] = None,
    ) -> VLMResponse:
        tag = _infer_tag(system, user_prompt, [Path(p) for p in image_paths])
        rid = _structured_id(tag, [Path(p) for p in image_paths]) or _hash_id(
            system, user_prompt, [Path(p) for p in image_paths]
        )
        self._record(tag, rid, system, user_prompt, list(image_paths), model)

        ans_path = ANSWERS_DIR / f"{rid}.json"
        if ans_path.is_file():
            text = ans_path.read_text(encoding="utf-8")
            return VLMResponse(
                text=text,
                raw={"oracle": "answer", "id": rid, "tag": tag},
                provider="oracle",
                model=model or "assistant",
                elapsed_seconds=0.0,
            )

        payload = _rule_for(tag, user_prompt, [str(p) for p in image_paths])
        text = json.dumps(payload, ensure_ascii=False)
        return VLMResponse(
            text=text,
            raw={"oracle": "rule", "id": rid, "tag": tag},
            provider="oracle",
            model="assistant",
            elapsed_seconds=0.0,
        )

    @staticmethod
    def _record(tag, rid, system, user_prompt, image_paths, model):
        _SEQ["n"] += 1
        _TALLY[tag] = _TALLY.get(tag, 0) + 1
        try:
            req = {
                "seq": _SEQ["n"],
                "id": rid,
                "tag": tag,
                "model": model,
                "system": system,
                "user_prompt": user_prompt,
                "image_paths": [str(p) for p in image_paths],
            }
            (REQUESTS_DIR / f"{rid}.json").write_text(
                json.dumps(req, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            with MANIFEST.open("a", encoding="utf-8") as f:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
            TALLY.write_text(json.dumps(_TALLY, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("oracle record failed: %s", exc)


def reset_oracle_logs() -> None:
    """清空 manifest / tally / requests（保留 answers）。用于干净的 collect 阶段。"""
    for p in (MANIFEST, TALLY):
        if p.is_file():
            p.unlink()
    for f in REQUESTS_DIR.glob("*.json"):
        f.unlink()
    _SEQ["n"] = 0
    _TALLY.clear()


def build_oracle_client():
    """供 benchmark 入口在 AGENTSEMANTIC_VLM_ORACLE 开启时调用。"""
    return AssistantVLMClient()
