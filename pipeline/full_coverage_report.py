"""
把一次 VLM 全覆盖语义分割结果（``output/<模型>/vlm_full/``）渲染成单文件
HTML 报告。

报告内容：
- 顶部摘要：object_kind / 迭代轮数 / 初始→叶 block / max_depth /
  min_block_size / total faces / assigned / background / pending
- 语义图例（每个 ``semantic_part`` 对应 ``FULL_COVERAGE_PALETTE`` 中的一个颜色
  + 面数 + 占比 + 进度条），与 ``full_coverage_mask.png`` 配色完全一致
- 整模型枚举图 + VLM 选定的展示视角 + 12 缩略图
- 迭代时间线（单视角 · 透明背景）
- 最终结果：full_coverage_mask.png / leaf_blocks_overview.png /
  initial_segmentation.png
- **按 depth 分组**列出每个 block 的完整决策卡片：
    - 12 视角高亮拼图
    - matched_parts（彩色 badge，颜色取自对应语义在调色板中的色块）
    - dominant、confidences、is_background、provider/model/elapsed
    - notes（自动解析 assigned / split_failed / max_depth_reached 等关键事件）
    - seed_decision（mode / k_actual / 子块 sizes / target / error）
- 底部：完整 ``block_tree.json``（折叠）

用法：
    python -m pipeline.full_coverage_report output/<模型>/vlm_full/
    # 默认写到 <output_dir>/report.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rendering.render_component import (
    FULL_COVERAGE_BG_RGB,
    FULL_COVERAGE_PALETTE,
    FULL_COVERAGE_PENDING_RGB,
)

# ---------------------------------------------------------------------------
# 常量 / 工具
# ---------------------------------------------------------------------------


STATUS_ZH = {
    "full": "已归类",
    "none": "背景 / 不匹配",
    "split": "已分裂",
    "abandoned": "已舍弃",
    "pending": "未决",
    "partial": "局部命中",
}

STATUS_COLOR = {
    "full": "#16a34a",       # 绿
    "none": "#737373",       # 灰
    "split": "#0891b2",      # 青
    "abandoned": "#9ca3af",  # 浅灰
    "pending": "#d97706",    # 橙
    "partial": "#d97706",    # 橙
}


def _esc(s: Any) -> str:
    return html.escape(str(s) if s is not None else "")


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _safe_palette_rgb(idx: int) -> Tuple[int, int, int]:
    if not FULL_COVERAGE_PALETTE:
        return (180, 180, 180)
    return tuple(FULL_COVERAGE_PALETTE[idx % len(FULL_COVERAGE_PALETTE)])


def _badge(text: str, color: str = "#6366f1", strong: bool = False) -> str:
    """彩色 badge；strong=True 会用更深的填充表示重点（如 dominant）。"""
    if strong:
        return (
            f'<span class="badge" style="background:{color};color:#fff;'
            f'border:1px solid {color};font-weight:700">'
            f'{_esc(text)}</span>'
        )
    return (
        f'<span class="badge" style="background:{color}22;'
        f'color:{color};border:1px solid {color}66">{_esc(text)}</span>'
    )


def _semantic_badge(
    name: str,
    part_color_map: Dict[str, str],
    *,
    strong: bool = False,
) -> str:
    """按 semantic_part 的固定调色板颜色渲染 badge。"""
    color = part_color_map.get(name, "#475569")
    return _badge(name, color, strong=strong)


def _list_semantic_badges(
    items: List[str],
    part_color_map: Dict[str, str],
    *,
    dominant: str = "",
) -> str:
    if not items:
        return '<span class="muted">[ 空 ]</span>'
    return " ".join(
        _semantic_badge(x, part_color_map, strong=(x == dominant and dominant))
        for x in items
    )


def _confidences_bar(
    confidences: Dict[str, float],
    part_color_map: Dict[str, str],
    *,
    dominant: str = "",
) -> str:
    """每个 matched part 一行 mini 进度条。"""
    if not confidences:
        return ""
    rows: List[str] = []
    for name, val in sorted(
        confidences.items(), key=lambda kv: -float(kv[1])
    ):
        try:
            pct = max(0.0, min(1.0, float(val))) * 100.0
        except (TypeError, ValueError):
            continue
        color = part_color_map.get(name, "#475569")
        star = ' ★' if name == dominant else ''
        rows.append(
            f'<div class="conf-row">'
            f'<span class="conf-name" style="color:{color}">'
            f'{_esc(name)}{star}</span>'
            f'<span class="conf-bar-outer">'
            f'<span class="conf-bar-inner" '
            f'style="width:{pct:.1f}%;background:{color}">'
            f'</span></span>'
            f'<span class="conf-val">{pct:.0f}%</span>'
            f'</div>'
        )
    return f'<div class="conf-grid">{"".join(rows)}</div>'


def _rel_image(panel_path: str, html_dir: Path) -> str:
    """把 block_tree.json 里的绝对 / 反斜杠路径转成相对 HTML 的 posix 路径。"""
    if not panel_path:
        return ""
    p = Path(panel_path.replace("\\", "/"))
    parts = p.parts
    if "panels" in parts:
        idx = parts.index("panels")
        return "/".join(parts[idx:])
    try:
        return p.relative_to(html_dir).as_posix()
    except ValueError:
        return p.as_posix()


def _rel_or_empty(p: Path, html_dir: Path) -> str:
    if not p.is_file():
        return ""
    try:
        return p.relative_to(html_dir).as_posix()
    except ValueError:
        return p.as_posix()


def _summary_card(title: str, value: Any, sub: str = "") -> str:
    sub_html = f'<div class="sub muted">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div class="summary-card"><b>{_esc(title)}</b>'
        f'<span>{_esc(value)}</span>{sub_html}</div>'
    )


# ---------------------------------------------------------------------------
# 渲染单个 block 卡片
# ---------------------------------------------------------------------------


def _parse_assigned_from_notes(notes: str) -> Optional[str]:
    """从 ``notes`` 字符串里抠出 ``assigned=XXX``（取最后一个）。"""
    if not notes:
        return None
    marker = "assigned="
    last = notes.rfind(marker)
    if last < 0:
        return None
    tail = notes[last + len(marker):]
    # 截到下一个 " | " 或 "(" 或行尾之前
    for sep in (" | ", " (", " ("):
        i = tail.find(sep)
        if i >= 0:
            tail = tail[:i]
    return tail.strip() or None


def _render_block_card(
    block: Dict[str, Any],
    html_dir: Path,
    part_color_map: Dict[str, str],
    assigned_semantic: Dict[str, str],
) -> str:
    block_id = block.get("block_id", "")
    parent_id = block.get("parent_id")
    depth = block.get("depth", 0)
    n_faces = block.get("num_faces", 0)
    children = block.get("children_ids", []) or []
    notes = block.get("notes", "") or ""
    panel_rel = _rel_image(block.get("panel_path", ""), html_dir)
    status = block.get("status", "")

    cls = block.get("classification") or {}
    matched = cls.get("matched_parts") or []
    is_bg = bool(cls.get("is_background", False))
    dominant = cls.get("dominant") or ""
    confidences = cls.get("confidences") or {}
    vlm_notes = cls.get("notes") or ""
    provider = cls.get("provider") or ""
    model = cls.get("model") or ""
    elapsed = cls.get("elapsed_seconds")
    err = cls.get("error")

    seed = block.get("seed_decision") or {}

    # ----- header -----
    status_color = STATUS_COLOR.get(status, "#6366f1")
    status_text = STATUS_ZH.get(status, status)
    # 给 status pill 后面加一个 assigned 大徽章（仅 FULL）
    assigned_name = assigned_semantic.get(block_id) or (
        _parse_assigned_from_notes(notes) if status == "full" else None
    )
    assigned_html = ""
    if assigned_name and status == "full":
        color = part_color_map.get(assigned_name, "#16a34a")
        assigned_html = (
            f'<span class="assigned-pill" style="background:{color};'
            f'color:#fff;border:1px solid {color};font-weight:700">'
            f'→ {_esc(assigned_name)}</span>'
        )

    parent_link = (
        f'<a href="#block-{_esc(parent_id)}">{_esc(parent_id)}</a>'
        if parent_id
        else '<span class="muted">root</span>'
    )
    children_links = (
        " ".join(
            f'<a href="#block-{_esc(c)}" class="child-link">{_esc(c)}</a>'
            for c in children
        )
        if children
        else '<span class="muted">—</span>'
    )

    elapsed_html = (
        f'<span class="muted">{float(elapsed):.1f}s</span>'
        if elapsed is not None
        else ""
    )
    provider_html = (
        f'<span class="muted">{_esc(provider)}/{_esc(model)}</span>'
        if provider else ""
    )

    img_html = (
        f'<a href="{_esc(panel_rel)}" target="_blank">'
        f'<img loading="lazy" src="{_esc(panel_rel)}" '
        f'alt="block {_esc(block_id)}" /></a>'
        if panel_rel
        else '<div class="muted">[无 panel]</div>'
    )

    # ----- matched / dominant / confidences -----
    matched_html = _list_semantic_badges(
        matched, part_color_map, dominant=dominant
    )
    dom_html = (
        _semantic_badge(dominant, part_color_map, strong=True)
        if dominant else '<span class="muted">—</span>'
    )
    conf_html = _confidences_bar(
        confidences, part_color_map, dominant=dominant
    )

    bg_badge_html = (
        '<span class="badge" style="background:#7373731f;color:#737373;'
        'border:1px solid #73737366">is_background</span>'
        if is_bg else ""
    )

    # ----- VLM notes -----
    vlm_reason_html = (
        f'<div class="reason">{_esc(vlm_notes)}</div>'
        if vlm_notes else '<span class="muted">[VLM 未给 notes]</span>'
    )

    # ----- seed_decision -----
    seed_block = ""
    if seed:
        mode = seed.get("mode", "")
        k_actual = seed.get("k_actual")
        sizes = seed.get("component_sizes") or seed.get("sizes") or []
        target = seed.get("seed_count_target")
        seed_err = seed.get("error")
        debug = seed.get("debug") or {}
        rows: List[str] = []
        rows.append(
            f'<div class="kv"><b>mode</b>: <code>{_esc(mode)}</code>'
            f' &nbsp; <b>target K</b>: <code>{_esc(target)}</code>'
            f' &nbsp; <b>actual K</b>: <code>'
            f'{_esc(k_actual if k_actual is not None else "—")}</code></div>'
        )
        if sizes:
            sizes_html = ", ".join(str(int(s)) for s in sizes)
            rows.append(
                f'<div class="kv"><b>子块 face 数</b>: '
                f'<code>[{_esc(sizes_html)}]</code></div>'
            )
        num_clusters = seed.get("num_clusters")
        if num_clusters is not None and "num_clusters" not in {"k_actual"}:
            rows.append(
                f'<div class="kv"><b>num_clusters</b>: '
                f'<code>{_esc(num_clusters)}</code></div>'
            )
        if debug:
            strategy = debug.get("strategy") or ""
            if strategy:
                rows.append(
                    f'<div class="kv"><b>strategy</b>: '
                    f'<code>{_esc(strategy)}</code></div>'
                )
            fps_err = debug.get("fps_error")
            if fps_err:
                rows.append(
                    f'<div class="kv" style="color:#f87171"><b>fps_error</b>: '
                    f'<code>{_esc(fps_err)}</code></div>'
                )
        if seed_err:
            rows.append(
                f'<div class="kv" style="color:#f87171"><b>error</b>: '
                f'<code>{_esc(seed_err)}</code></div>'
            )
        seed_block = (
            f'<div class="seed">'
            f'<b>分裂决策</b>'
            f'{"".join(rows)}'
            f'</div>'
        )

    # ----- notes 高亮 -----
    notes_block = ""
    if notes:
        # 高亮 max_depth_reached / split_failed / too_small / bg(...)
        notes_html = _esc(notes)
        for tag, color in (
            ("max_depth_reached", "#fb923c"),
            ("split_failed", "#f87171"),
            ("too_small", "#9ca3af"),
            ("assigned=", "#16a34a"),
            ("bg(", "#737373"),
            ("cc_init", "#0891b2"),
        ):
            esc_tag = _esc(tag)
            notes_html = notes_html.replace(
                esc_tag,
                f'<span style="color:{color};font-weight:600">{esc_tag}</span>',
            )
        notes_block = (
            f'<div class="kv"><b>备注</b>: '
            f'<code class="notes-code">{notes_html}</code></div>'
        )

    # ----- error (classify) -----
    err_html = ""
    if err:
        err_html = (
            f'<div class="kv" style="color:#f87171"><b>classify error</b>: '
            f'<code>{_esc(err)}</code></div>'
        )

    return f"""
<article class="block-card" id="block-{_esc(block_id)}">
  <header class="card-header">
    <div class="card-title">
      <span class="block-id">{_esc(block_id)}</span>
      <span class="status-pill" style="background:{status_color}22;
            color:{status_color};border:1px solid {status_color}55">
        {_esc(status_text)}
      </span>
      {assigned_html}
      <span class="muted">depth={_esc(depth)} · faces={_esc(n_faces)}</span>
      {bg_badge_html}
    </div>
    <div class="card-meta">
      parent: {parent_link} · children: {children_links}
      &nbsp; {provider_html} {elapsed_html}
    </div>
  </header>
  <div class="card-body">
    <div class="panel-img">{img_html}</div>
    <div class="card-fields">
      <div class="kv">
        <b>matched_parts</b>
        <div class="badge-row">{matched_html}</div>
      </div>
      <div class="kv">
        <b>dominant</b>: {dom_html}
      </div>
      {f'<div class="kv"><b>confidences</b>{conf_html}</div>' if conf_html else ''}
      <div class="kv reason-kv">
        <b>VLM 解释</b>:
        {vlm_reason_html}
      </div>
      {err_html}
      {seed_block}
      {notes_block}
    </div>
  </div>
</article>
"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------


CSS = """
:root {
  --bg: #0f172a;
  --bg-card: #1e293b;
  --bg-card-2: #334155;
  --fg: #e2e8f0;
  --fg-muted: #94a3b8;
  --accent: #38bdf8;
  --border: #334155;
}
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
               "Microsoft YaHei", sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.5;
}
.container { max-width: 1400px; margin: 0 auto; padding: 24px; }
header.top {
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
  margin-bottom: 24px;
}
header.top h1 { margin: 0 0 8px; color: var(--accent); font-size: 24px; }
header.top h1 .object-kind {
  color: #fde68a; font-style: normal; margin-left: 6px;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.summary-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
}
.summary-card b { color: var(--accent); display: block; font-size: 12px;
                  text-transform: uppercase; letter-spacing: 0.5px; }
.summary-card span { font-size: 18px; font-weight: 600; }
.summary-card .sub { font-size: 11px; margin-top: 2px; }

h2.section-title {
  color: var(--accent); font-size: 18px;
  margin: 28px 0 10px 0;
  padding-bottom: 6px; border-bottom: 1px solid var(--border);
}

/* 语义图例 */
.legend-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 10px;
  margin: 12px 0;
}
.legend-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
}
.legend-row1 { display: flex; align-items: center; gap: 10px; }
.legend-swatch {
  width: 18px; height: 18px; border-radius: 4px;
  border: 1px solid #ffffff22; flex-shrink: 0;
}
.legend-name { flex: 1; font-weight: 600; }
.legend-count { color: var(--fg-muted);
                font-variant-numeric: tabular-nums; }
.legend-bar-outer {
  width: 100%; height: 6px; background: #ffffff10;
  border-radius: 3px; overflow: hidden; margin-top: 6px;
}
.legend-bar-inner { height: 100%; border-radius: 3px; }
.legend-pct { color: var(--fg-muted); font-size: 11px;
              margin-top: 2px; text-align: right; }

/* 枚举区块 */
.enum-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 18px;
  margin: 14px 0;
}
.enum-card .enum-meta {
  display: grid;
  grid-template-columns: minmax(280px, 1fr) 1fr;
  gap: 18px; align-items: start;
}
@media (max-width: 900px) {
  .enum-card .enum-meta { grid-template-columns: 1fr; }
}
.enum-card img {
  width: 100%; max-height: 360px; object-fit: contain;
  border-radius: 6px; border: 1px solid var(--border);
  background: #fff;
}

/* 视角选择 */
.viewcard {
  display: grid; grid-template-columns: minmax(280px, 360px) 1fr;
  gap: 18px; align-items: start; margin: 18px 0;
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 18px;
}
@media (max-width: 900px) { .viewcard { grid-template-columns: 1fr; } }
.viewcard img.overview { width: 100%; }
.viewcard .vmeta { font-size: 14px; line-height: 1.6; }
.viewcard .vmeta b { color: var(--accent); }
.viewcard .vreason { margin-top: 8px; padding: 8px 12px;
                     background: var(--bg); border-left: 3px solid var(--accent);
                     border-radius: 6px; }
.thumb-strip { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
.thumb-strip img { width: 100%; max-height: 240px; object-fit: contain;
                   border: 1px solid var(--border); border-radius: 6px;
                   background: #fff; }

/* 透明背景棋盘衬底 */
.transparent-img {
  border-radius: 8px;
  border: 1px solid var(--border);
  background-color: #fff;
  background-image:
    linear-gradient(45deg, #d6d6dc 25%, transparent 25%),
    linear-gradient(-45deg, #d6d6dc 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, #d6d6dc 75%),
    linear-gradient(-45deg, transparent 75%, #d6d6dc 75%);
  background-size: 20px 20px;
  background-position: 0 0, 0 10px, 10px -10px, -10px 0;
}

/* 迭代时间线 */
.timeline-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin: 14px 0 24px;
}
.timeline-grid figure {
  margin: 0; background: var(--bg-card);
  border: 1px solid var(--border); border-radius: 8px;
  padding: 8px;
}
.timeline-grid figure img { width: 100%; display: block; border-radius: 6px; }
.timeline-grid figcaption { color: var(--fg-muted);
                            font-size: 12px; text-align: center; margin-top: 6px; }

/* 最终结果图 */
img.final-mask {
  width: 100%; border-radius: 8px;
  border: 1px solid var(--border);
  background: #fff;
}
.final-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 14px; margin-top: 12px;
}
.final-cell {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px;
}
.final-cell .muted { margin-bottom: 6px; }

/* 按 depth 分组 */
.depth-section {
  margin: 28px 0 12px;
}
.depth-section h2 { color: var(--accent); margin: 0 0 8px;
                    font-size: 18px; }
.cards { display: grid; grid-template-columns: 1fr; gap: 16px; }

/* block 卡片 */
.block-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 18px;
  scroll-margin-top: 12px;
}
.card-header { border-bottom: 1px solid var(--border);
               padding-bottom: 8px; margin-bottom: 10px; }
.card-title { display: flex; align-items: center; gap: 10px;
              flex-wrap: wrap; }
.block-id { font-weight: 700; font-size: 17px; color: var(--fg);
            font-family: ui-monospace, monospace; }
.status-pill { font-size: 12px; padding: 2px 10px; border-radius: 999px; }
.assigned-pill { font-size: 12px; padding: 2px 10px; border-radius: 999px;
                 letter-spacing: 0.2px; }
.card-meta { font-size: 12px; color: var(--fg-muted); margin-top: 4px; }
.card-meta a { color: var(--accent); text-decoration: none; }
.card-meta a:hover { text-decoration: underline; }
.card-meta .child-link { margin-right: 6px; }

.card-body {
  display: grid;
  grid-template-columns: minmax(280px, 1fr) 2fr;
  gap: 18px;
  align-items: start;
}
@media (max-width: 960px) {
  .card-body { grid-template-columns: 1fr; }
}
.panel-img img { width: 100%; max-width: 480px; border-radius: 6px;
                 border: 1px solid var(--border); display: block;
                 background: #fff; }

.kv { padding: 6px 0; }
.kv b { display: inline-block; min-width: 0; color: var(--accent);
        font-size: 13px; }
.kv code { font-size: 12px; background: var(--bg); padding: 2px 6px;
           border-radius: 4px; }
.notes-code { font-size: 12px; }

.badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.badge { font-size: 12px; padding: 2px 8px; border-radius: 999px;
         font-weight: 500; display: inline-block; }

.muted { color: var(--fg-muted); font-size: 12px; }
.reason { color: var(--fg); margin-top: 4px;
          background: var(--bg); padding: 8px 12px; border-radius: 6px;
          border-left: 3px solid var(--accent); font-size: 13px; }
.reason-kv .reason { font-size: 14px; }
.seed {
  margin-top: 8px; padding: 8px 12px;
  background: #0c4a6e22; border: 1px solid #0c4a6e;
  border-radius: 6px;
}
.seed > b { color: #38bdf8; }

/* confidences mini bar */
.conf-grid { display: grid; gap: 4px; margin-top: 6px; }
.conf-row { display: grid;
            grid-template-columns: minmax(80px, 110px) 1fr 40px;
            gap: 8px; align-items: center; font-size: 12px; }
.conf-name { font-weight: 600; }
.conf-bar-outer { height: 8px; background: #ffffff10;
                  border-radius: 4px; overflow: hidden; }
.conf-bar-inner { display: block; height: 100%; border-radius: 4px; }
.conf-val { color: var(--fg-muted); text-align: right;
            font-variant-numeric: tabular-nums; }

details { background: var(--bg-card); border: 1px solid var(--border);
          border-radius: 8px; padding: 8px 16px; margin-top: 24px; }
details summary { cursor: pointer; color: var(--accent);
                  font-weight: 600; }
pre { background: var(--bg); padding: 12px; border-radius: 6px;
      overflow-x: auto; font-size: 12px; }
"""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def generate_full_coverage_report(
    output_dir: str | Path,
    html_path: Optional[Path] = None,
) -> Path:
    """读 ``<output_dir>/block_tree.json`` 等产物，写 ``report.html``。"""
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"目录不存在：{output_dir}")
    tree_json = output_dir / "block_tree.json"
    if not tree_json.is_file():
        raise FileNotFoundError(
            f"找不到 block_tree.json（请先跑 pipeline.full_coverage_cli）："
            f"{tree_json}"
        )

    with open(tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    mode = tree.get("mode", "")
    if mode and mode != "full_coverage":
        # 不阻断，仅提示
        print(
            f"[warn] block_tree.json 的 mode='{mode}'，不是 'full_coverage'；"
            f"继续按 full_coverage 渲染。",
            file=sys.stderr,
        )

    blocks: List[Dict[str, Any]] = list(tree.get("blocks", []))
    semantic_parts: List[str] = list(tree.get("semantic_parts", []))
    num_iters = int(tree.get("num_iterations", 0))
    n_init = int(tree.get("num_initial_blocks", 0))
    n_final = int(tree.get("num_final_blocks", 0))
    max_depth = tree.get("max_depth", "")
    min_block_size = tree.get("min_block_size", "")
    max_seeds_per_split = tree.get("max_seeds_per_split", "")
    assigned_semantic: Dict[str, str] = dict(
        tree.get("assigned_semantic", {})
    )
    bg_face_count = int(tree.get("background_face_count", 0))
    pending_face_count = int(tree.get("pending_face_count", 0))
    sem_face_counts: Dict[str, int] = dict(tree.get("semantic_face_counts", {}))
    enumeration = tree.get("enumeration") or {}
    object_kind = enumeration.get("object_kind", "") or ""

    # 总面数：所有 semantic + bg + pending
    total_faces = (
        sum(int(v) for v in sem_face_counts.values())
        + bg_face_count + pending_face_count
    )

    # 每个 semantic_part 的固定颜色（与 full_coverage_mask.png 配色一致）
    part_color_map: Dict[str, str] = {}
    for k, name in enumerate(semantic_parts):
        part_color_map[name] = _rgb_to_hex(_safe_palette_rgb(k))

    if html_path is None:
        html_path = output_dir / "report.html"
    html_dir = html_path.parent

    # ---- 各种附属图路径 ----
    semantics_overview_rel = _rel_or_empty(
        output_dir / "semantics_overview.png", html_dir
    )
    coverage_mask_rel = _rel_or_empty(
        output_dir / "full_coverage_mask.png", html_dir
    )
    leaf_rel = _rel_or_empty(
        output_dir / "leaf_blocks_overview.png", html_dir
    )
    init_rel = _rel_or_empty(
        output_dir / "initial_segmentation.png", html_dir
    )
    overview_rel = _rel_or_empty(
        output_dir / "best_view_overview.png", html_dir
    )
    thumbs_rel = _rel_or_empty(
        output_dir / "view_thumbnails.png", html_dir
    )

    # 视角选择 JSON
    best_view_data: Dict[str, Any] = {}
    bv_json = output_dir / "best_view.json"
    if bv_json.is_file():
        try:
            with open(bv_json, "r", encoding="utf-8") as f:
                best_view_data = json.load(f) or {}
        except (OSError, ValueError) as exc:
            print(f"[warn] best_view.json 解析失败: {exc}", file=sys.stderr)

    # 迭代时间线
    iter_dir = output_dir / "iterations"
    iter_pngs: List[Path] = (
        sorted(iter_dir.glob("iter_*.png")) if iter_dir.is_dir() else []
    )

    # ---- 顶部摘要卡片 ----
    n_assigned_faces = sum(int(v) for v in sem_face_counts.values())
    n_populated_parts = sum(1 for v in sem_face_counts.values() if int(v) > 0)
    summary_cards_html = "".join([
        _summary_card("识别对象", object_kind or "—"),
        _summary_card(
            "语义部位",
            f"{n_populated_parts} / {len(semantic_parts)}",
            "populated / total",
        ),
        _summary_card("迭代轮数", num_iters),
        _summary_card("初始 → 叶 block", f"{n_init} → {n_final}"),
        _summary_card("max_depth", max_depth),
        _summary_card("min_block_size", min_block_size),
        _summary_card("max_seeds_per_split", max_seeds_per_split),
        _summary_card("total faces", f"{total_faces:,}"),
        _summary_card(
            "已分配 face",
            f"{n_assigned_faces:,}",
            (
                f"{(100.0 * n_assigned_faces / max(1, total_faces)):.1f}%"
                if total_faces else ""
            ),
        ),
        _summary_card("background", f"{bg_face_count:,}"),
        _summary_card("pending", f"{pending_face_count:,}"),
    ])

    # ---- 语义图例 ----
    legend_cards: List[str] = []
    # semantic_parts 按面数从多到少排序展示，但保持原索引颜色
    sorted_parts = sorted(
        semantic_parts,
        key=lambda n: -int(sem_face_counts.get(n, 0)),
    )
    for name in sorted_parts:
        color = part_color_map.get(name, "#475569")
        cnt = int(sem_face_counts.get(name, 0))
        pct = (100.0 * cnt / max(1, total_faces)) if total_faces else 0.0
        legend_cards.append(
            f'<div class="legend-card">'
            f'<div class="legend-row1">'
            f'<span class="legend-swatch" style="background:{color}"></span>'
            f'<span class="legend-name">{_esc(name)}</span>'
            f'<span class="legend-count">{cnt:,}</span>'
            f'</div>'
            f'<div class="legend-bar-outer">'
            f'<span class="legend-bar-inner" '
            f'style="width:{pct:.1f}%;background:{color};display:block">'
            f'</span></div>'
            f'<div class="legend-pct">{pct:.1f}%</div>'
            f'</div>'
        )
    # 追加 bg / pending
    bg_pct = (100.0 * bg_face_count / max(1, total_faces)) if total_faces else 0.0
    legend_cards.append(
        f'<div class="legend-card">'
        f'<div class="legend-row1">'
        f'<span class="legend-swatch" style="background:'
        f'{_rgb_to_hex(FULL_COVERAGE_BG_RGB)};border-color:#00000044">'
        f'</span>'
        f'<span class="legend-name">background</span>'
        f'<span class="legend-count">{bg_face_count:,}</span>'
        f'</div>'
        f'<div class="legend-bar-outer">'
        f'<span class="legend-bar-inner" '
        f'style="width:{bg_pct:.1f}%;background:'
        f'{_rgb_to_hex(FULL_COVERAGE_BG_RGB)};display:block">'
        f'</span></div>'
        f'<div class="legend-pct">{bg_pct:.1f}%</div>'
        f'</div>'
    )
    if pending_face_count > 0:
        p_pct = 100.0 * pending_face_count / max(1, total_faces)
        legend_cards.append(
            f'<div class="legend-card">'
            f'<div class="legend-row1">'
            f'<span class="legend-swatch" style="background:'
            f'{_rgb_to_hex(FULL_COVERAGE_PENDING_RGB)}"></span>'
            f'<span class="legend-name">pending</span>'
            f'<span class="legend-count">{pending_face_count:,}</span>'
            f'</div>'
            f'<div class="legend-bar-outer">'
            f'<span class="legend-bar-inner" '
            f'style="width:{p_pct:.1f}%;background:'
            f'{_rgb_to_hex(FULL_COVERAGE_PENDING_RGB)};display:block">'
            f'</span></div>'
            f'<div class="legend-pct">{p_pct:.1f}%</div>'
            f'</div>'
        )
    legend_grid_html = f'<div class="legend-grid">{"".join(legend_cards)}</div>'

    # ---- 枚举区块（object_kind + 12视角 + reason）----
    enum_section_html = ""
    if enumeration or semantics_overview_rel:
        enum_reason = (enumeration.get("reason") or "").strip()
        enum_provider = enumeration.get("provider") or ""
        enum_model = enumeration.get("model") or ""
        enum_elapsed = enumeration.get("elapsed_seconds")
        enum_err = enumeration.get("error")
        parts_badges = " ".join(
            _semantic_badge(n, part_color_map) for n in semantic_parts
        ) or '<span class="muted">[无]</span>'
        meta_lines: List[str] = []
        if object_kind:
            meta_lines.append(
                f'<div class="kv"><b>识别对象</b>: '
                f'<code>{_esc(object_kind)}</code></div>'
            )
        meta_lines.append(
            f'<div class="kv"><b>枚举到的部位（{len(semantic_parts)}）</b>'
            f'<div class="badge-row">{parts_badges}</div></div>'
        )
        if enum_reason:
            meta_lines.append(
                f'<div class="kv reason-kv"><b>VLM 理由</b>'
                f'<div class="reason">{_esc(enum_reason)}</div></div>'
            )
        if enum_provider or enum_elapsed is not None:
            extras: List[str] = []
            if enum_provider:
                extras.append(
                    f'<b>by</b> <code>'
                    f'{_esc(enum_provider)}/{_esc(enum_model)}</code>'
                )
            if enum_elapsed is not None:
                try:
                    extras.append(
                        f'<b>elapsed</b> <code>{float(enum_elapsed):.1f}s</code>'
                    )
                except (TypeError, ValueError):
                    pass
            meta_lines.append(
                f'<div class="muted">{" &nbsp;·&nbsp; ".join(extras)}</div>'
            )
        if enum_err:
            meta_lines.append(
                f'<div class="kv" style="color:#f87171">'
                f'<b>error</b>: <code>{_esc(enum_err)}</code></div>'
            )

        img_block = (
            f'<a href="{_esc(semantics_overview_rel)}" target="_blank">'
            f'<img src="{_esc(semantics_overview_rel)}" '
            f'alt="semantics_overview" /></a>'
            if semantics_overview_rel
            else '<div class="muted">[无 semantics_overview.png]</div>'
        )
        enum_section_html = f"""
<h2 class="section-title">语义枚举（VLM 输入：整模型 12 视角）</h2>
<div class="enum-card">
  <div class="enum-meta">
    <div>{img_block}</div>
    <div>{''.join(meta_lines)}</div>
  </div>
</div>
"""

    # ---- 视角选择区块 ----
    best_view_section = ""
    if best_view_data or overview_rel or thumbs_rel:
        bv_idx = best_view_data.get("index_1based", "—")
        elev = best_view_data.get("elev", None)
        azim = best_view_data.get("azim", None)
        bv_reason = best_view_data.get("reason", "") or ""
        bv_fallback = bool(best_view_data.get("fallback", False))
        bv_confidence = best_view_data.get("confidence")
        bv_provider = best_view_data.get("provider", "") or ""
        bv_model = best_view_data.get("model", "") or ""
        conf_html_bv = (
            f' · <b>conf</b>=<code>{float(bv_confidence):.2f}</code>'
            if bv_confidence is not None else ""
        )
        provider_html_bv = (
            f' · <b>by</b> <code>'
            f'{_esc(bv_provider)}/{_esc(bv_model)}</code>'
            if bv_provider else ""
        )
        elev_html = (
            f'<code>{float(elev):.1f}°</code>' if elev is not None else "—"
        )
        azim_html = (
            f'<code>{float(azim):.1f}°</code>' if azim is not None else "—"
        )
        tag = (
            '<span class="badge" style="background:#c2410c22;color:#fb923c;'
            'border:1px solid #fb923c55">默认 fallback</span>'
            if bv_fallback
            else '<span class="badge" style="background:#1d4ed822;'
                 'color:#60a5fa;border:1px solid #60a5fa55">VLM 选定</span>'
        )
        overview_img_html = (
            f'<a href="{_esc(overview_rel)}" target="_blank">'
            f'<img class="transparent-img overview" '
            f'src="{_esc(overview_rel)}" alt="best view overview" /></a>'
            if overview_rel
            else '<div class="muted">[未生成 overview]</div>'
        )
        thumbs_html = (
            f'<div class="muted" style="margin-top:10px">候选 12 视角</div>'
            f'<div class="thumb-strip"><a href="{_esc(thumbs_rel)}" '
            f'target="_blank"><img src="{_esc(thumbs_rel)}" '
            f'alt="thumbnails" /></a></div>'
            if thumbs_rel else ""
        )
        best_view_section = f"""
<h2 class="section-title">展示视角（VLM 选定）</h2>
<div class="viewcard">
  <div>{overview_img_html}</div>
  <div class="vmeta">
    <b>视角编号</b> #<code>{_esc(bv_idx)}</code> {tag}
    <br/><b>elev</b> {elev_html} &nbsp; <b>azim</b> {azim_html}
    {conf_html_bv}{provider_html_bv}
    <div class="vreason"><b>VLM 理由：</b>{_esc(bv_reason) or '—'}</div>
    {thumbs_html}
  </div>
</div>
"""

    # ---- 迭代时间线 ----
    timeline_section = ""
    if iter_pngs:
        figs: List[str] = []
        for png in iter_pngs:
            try:
                rel = png.relative_to(html_dir).as_posix()
            except ValueError:
                rel = png.as_posix()
            n = png.stem.split("_")[-1]  # iter_03 → "03"
            figs.append(
                f'<figure><a href="{_esc(rel)}" target="_blank">'
                f'<img class="transparent-img" src="{_esc(rel)}" '
                f'alt="iter {_esc(n)}" /></a>'
                f'<figcaption>iter {_esc(n)}</figcaption></figure>'
            )
        timeline_section = f"""
<h2 class="section-title">迭代时间线（单视角 · 透明背景）</h2>
<div class="muted">每轮迭代结束时按 VLM 选定视角渲染；颜色与上方语义图例一致。</div>
<div class="timeline-grid">{''.join(figs)}</div>
"""

    # ---- 最终结果区 ----
    final_cells: List[str] = []
    if coverage_mask_rel:
        final_cells.append(
            f'<div class="final-cell">'
            f'<div class="muted">full_coverage_mask.png — '
            f'按部位上色的最终全覆盖语义掩膜</div>'
            f'<a href="{_esc(coverage_mask_rel)}" target="_blank">'
            f'<img class="final-mask" src="{_esc(coverage_mask_rel)}" '
            f'alt="full_coverage_mask" /></a></div>'
        )
    if leaf_rel:
        final_cells.append(
            f'<div class="final-cell">'
            f'<div class="muted">leaf_blocks_overview.png — '
            f'每个叶分块独立随机色，便于核对切分粒度</div>'
            f'<a href="{_esc(leaf_rel)}" target="_blank">'
            f'<img class="final-mask" src="{_esc(leaf_rel)}" '
            f'alt="leaf_blocks_overview" /></a></div>'
        )
    if init_rel:
        final_cells.append(
            f'<div class="final-cell">'
            f'<div class="muted">initial_segmentation.png — '
            f'初始分块（连通分量展开后）</div>'
            f'<a href="{_esc(init_rel)}" target="_blank">'
            f'<img class="final-mask" src="{_esc(init_rel)}" '
            f'alt="initial_segmentation" /></a></div>'
        )
    final_section_html = ""
    if final_cells:
        final_section_html = (
            '<h2 class="section-title">最终结果</h2>'
            f'<div class="final-grid">{"".join(final_cells)}</div>'
        )

    # ---- 按 depth 分组的 block 卡片 ----
    by_depth: Dict[int, List[Dict[str, Any]]] = {}
    for b in blocks:
        by_depth.setdefault(int(b.get("depth", 0)), []).append(b)
    for d in by_depth:
        by_depth[d].sort(key=lambda x: x.get("block_id", ""))

    depth_sections: List[str] = []
    for depth in sorted(by_depth.keys()):
        bs = by_depth[depth]
        # 各 status 计数
        cnt: Dict[str, int] = {}
        for b in bs:
            cnt[b.get("status", "")] = cnt.get(b.get("status", ""), 0) + 1
        cnt_html = " · ".join(
            f'<span style="color:{STATUS_COLOR.get(k, "#6366f1")}">'
            f'{_esc(STATUS_ZH.get(k, k))} ×{v}</span>'
            for k, v in sorted(cnt.items())
        )
        cards_html = "\n".join(
            _render_block_card(b, html_dir, part_color_map, assigned_semantic)
            for b in bs
        )
        depth_sections.append(
            f"""
<section class="depth-section">
  <h2>depth = {depth} · {len(bs)} 个 block</h2>
  <div class="muted">{cnt_html}</div>
  <div class="cards" style="margin-top:10px">{cards_html}</div>
</section>
"""
        )

    title_obj_kind = (
        f' · <span class="object-kind">「{_esc(object_kind)}」</span>'
        if object_kind else ""
    )

    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>VLM 全覆盖语义分割报告{f' · {object_kind}' if object_kind else ''}</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<header class="top">
  <h1>VLM 全覆盖语义分割报告{title_obj_kind}</h1>
  <div class="muted">输出目录：<code>{_esc(output_dir)}</code></div>
  <div class="summary-grid">{summary_cards_html}</div>
</header>

<h2 class="section-title">语义图例（与 full_coverage_mask.png 配色一致）</h2>
{legend_grid_html}

{enum_section_html}

{best_view_section}

{timeline_section}

{final_section_html}

<h2 class="section-title">每个 block 的决策（按 depth 分组）</h2>
<div class="muted">
  跳转可点击 <code>parent</code> / <code>children</code>；matched_parts 中
  <b>加深填充</b>的是 dominant；FULL 块后跟随 <b>→ assigned</b> 大徽章。
</div>
{''.join(depth_sections)}

<details>
  <summary>原始 block_tree.json（点击展开）</summary>
  <pre>{_esc(json.dumps(tree, ensure_ascii=False, indent=2))}</pre>
</details>

</div>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(body, encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.full_coverage_report",
        description=(
            "把一次 VLM 全覆盖语义分割的输出"
            "（output/<模型>/vlm_full/）渲染成单文件 HTML 报告。"
        ),
    )
    p.add_argument(
        "output_dir",
        type=str,
        help="full_coverage 输出目录，例如 output/<模型>/vlm_full/",
    )
    p.add_argument(
        "-o",
        "--out",
        type=str,
        default=None,
        help="HTML 输出路径（默认 <output_dir>/report.html）",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out = Path(args.out) if args.out else None
    path = generate_full_coverage_report(Path(args.output_dir), out)
    print(f"[ok] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
