"""
把一次 VLM 语义细化结果（output/<模型>/vlm/<语义>/）渲染成单文件 HTML 报告。

报告内容：
- 顶部摘要（mesh / labels / 语义 / 配置 / 最终统计）
- 最终 semantic_mask.png 大图
- **按 depth 分组**列出每个 block 的完整决策：
  - 12 视角高亮拼图
  - VLM 观察阶段：covered_parts + dominant_region
  - VLM 推理阶段：semantic_parts / extra_parts / missing_overlap / label / reason
  - 如果分裂：seed_decision + 子块跳转链接
  - 如果在 max_depth 处接受为 best-effort full：注解
- 底部：完整 responses_log（折叠）

用法：
    python -m vlm.report output/<模型>/vlm/<语义>/
    # 默认写到 <output_dir>/report.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


STATUS_ZH = {
    "full": "完全符合",
    "none": "完全不符合",
    "partial": "局部符合",
    "split": "已分裂",
    "abandoned": "已舍弃",
}

STATUS_COLOR = {
    "full": "#16a34a",       # 绿
    "none": "#737373",       # 灰
    "partial": "#d97706",    # 橙
    "split": "#0891b2",      # 青
    "abandoned": "#9ca3af",  # 浅灰
    "full*": "#0d9488",      # max-depth best-effort full
}


def _esc(s: Any) -> str:
    return html.escape(str(s) if s is not None else "")


def _badge(text: str, color: str = "#6366f1") -> str:
    return (
        f'<span class="badge" style="background:{color}22;'
        f'color:{color};border:1px solid {color}55">{_esc(text)}</span>'
    )


def _list_badges(items: List[str], color: str) -> str:
    if not items:
        return '<span class="muted">[ 空 ]</span>'
    return " ".join(_badge(x, color) for x in items)


def _rel_image(panel_path: str, html_dir: Path) -> str:
    """
    把 block_tree.json 里写的 panel_path 转成相对 HTML 文件目录的相对路径。
    panel_path 通常长这样：output\\uploads_files_<...>\\vlm\\头发\\panels\\block_C0.png
    """
    if not panel_path:
        return ""
    p = Path(panel_path.replace("\\", "/"))
    # 优先匹配 panels/<...>，因为 panels/ 一定就在 output_dir 下
    parts = p.parts
    if "panels" in parts:
        idx = parts.index("panels")
        return "/".join(parts[idx:])
    # 兜底：尝试相对到 html_dir
    try:
        return p.relative_to(html_dir).as_posix()
    except ValueError:
        return p.as_posix()


def _block_status_label(block: Dict[str, Any]) -> str:
    status = block.get("status", "")
    notes = block.get("notes", "") or ""
    if status == "full" and "accepted_partial_at_max_depth" in notes:
        return "full*"
    return status


# ---------------------------------------------------------------------------
# 渲染单个 block 卡片
# ---------------------------------------------------------------------------


def _render_block_card(block: Dict[str, Any], html_dir: Path) -> str:
    block_id = block.get("block_id", "")
    parent_id = block.get("parent_id")
    depth = block.get("depth", 0)
    n_faces = block.get("num_faces", 0)
    children = block.get("children_ids", []) or []
    notes = block.get("notes", "") or ""
    panel_rel = _rel_image(block.get("panel_path", ""), html_dir)

    status_label = _block_status_label(block)
    status_color = STATUS_COLOR.get(status_label, "#6366f1")
    status_text = STATUS_ZH.get(status_label, status_label) + (
        "（best-effort）" if status_label == "full*" else ""
    )

    cls = block.get("classification") or {}
    covered = cls.get("covered_parts") or []
    dominant_region = cls.get("dominant_region") or ""
    semantic_parts = cls.get("semantic_parts") or []
    extra = cls.get("extra_parts") or []
    missing_overlap = cls.get("missing_overlap")
    label = cls.get("label") or ""
    confidence = cls.get("confidence")
    reason = cls.get("reason") or ""
    overridden_from = cls.get("overridden_from")
    elapsed = cls.get("elapsed_seconds")
    semantic_area_ratio = cls.get("semantic_area_ratio")
    dominant_side = cls.get("dominant_side") or ""
    committed_if_forced = cls.get("committed_if_forced") or ""

    seed = block.get("seed_decision") or {}
    seed_count = seed.get("seed_count")
    seed_reason = seed.get("reason") or ""

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

    # confidence
    conf_html = (
        f'<span class="muted">conf={float(confidence):.2f}</span>'
        if confidence is not None
        else ""
    )
    elapsed_html = (
        f'<span class="muted">{float(elapsed):.1f}s</span>'
        if elapsed is not None
        else ""
    )

    overridden_html = (
        f'<div class="kv"><b>服务端兜底</b>: '
        f'原始 label="{_esc(overridden_from)}" → 实际 "{_esc(label)}" '
        f'（按 extra_parts/missing_overlap 重写）</div>'
        if overridden_from
        else ""
    )

    img_html = (
        f'<a href="{_esc(panel_rel)}" target="_blank">'
        f'<img loading="lazy" src="{_esc(panel_rel)}" '
        f'alt="block {_esc(block_id)}" /></a>'
        if panel_rel
        else '<div class="muted">[无 panel]</div>'
    )

    seed_block = ""
    if seed_count:
        seed_block = (
            f'<div class="seed">'
            f'<b>分裂决策</b> seed_count={_esc(seed_count)}'
            f'<div class="reason">{_esc(seed_reason)}</div>'
            f'</div>'
        )

    notes_block = (
        f'<div class="kv"><b>备注</b>: <code>{_esc(notes)}</code></div>'
        if notes
        else ""
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
      <span class="muted">depth={_esc(depth)} · faces={_esc(n_faces)}</span>
    </div>
    <div class="card-meta">
      parent: {parent_link} · children: {children_links}
      &nbsp;{conf_html} {elapsed_html}
    </div>
  </header>
  <div class="card-body">
    <div class="panel-img">{img_html}</div>
    <div class="card-fields">
      <div class="kv">
        <b>VLM 观察 — covered_parts</b>
        <div class="badge-row">{_list_badges(covered, "#475569")}</div>
        {f'<div class="dominant"><b>主集中区域：</b>{_esc(dominant_region)}</div>'
         if dominant_region else ''}
      </div>
      <div class="kv">
        <b>VLM 推理 — semantic_parts</b>
        <div class="badge-row">{_list_badges(semantic_parts, "#16a34a")}</div>
      </div>
      <div class="kv">
        <b>extra_parts（红色多出的部分）</b>
        <div class="badge-row">{_list_badges(extra, "#d97706")}</div>
      </div>
      <div class="kv">
        <b>missing_overlap</b>:
        <code>{_esc(missing_overlap)}</code>
        &nbsp; <b>语义占比</b>:
        <code>{(f'{float(semantic_area_ratio):.2f}'
              if semantic_area_ratio is not None else '—')}</code>
        &nbsp; <b>主导边</b>:
        <code>{_esc(dominant_side or '—')}</code>
        &nbsp; <b>强制承诺</b>:
        <code>{_esc(committed_if_forced or '—')}</code>
        &nbsp; <b>label</b>:
        <span style="color:{STATUS_COLOR.get(label, '#475569')};
              font-weight:600">{_esc(label)}</span>
      </div>
      {overridden_html}
      <div class="kv reason-kv">
        <b>VLM 总结</b>:
        <div class="reason">{_esc(reason)}</div>
      </div>
      {seed_block}
      {notes_block}
    </div>
  </div>
</article>
"""


# ---------------------------------------------------------------------------
# 主入口
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
.container { max-width: 1280px; margin: 0 auto; padding: 24px; }
header.top {
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
  margin-bottom: 24px;
}
header.top h1 { margin: 0 0 8px; color: var(--accent); font-size: 24px; }
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
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

.iteration-section {
  margin: 32px 0 12px;
  padding-bottom: 8px;
  border-bottom: 1px dashed var(--border);
}
.iteration-section h2 { color: var(--accent); margin: 0 0 8px;
                        font-size: 18px; }

.cards { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 1024px) {
  .cards { grid-template-columns: 1fr; }
}

.block-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 18px;
  scroll-margin-top: 12px;
}
.card-header { border-bottom: 1px solid var(--border);
               padding-bottom: 8px; margin-bottom: 10px; }
.card-title { display: flex; align-items: center; gap: 12px;
              flex-wrap: wrap; }
.block-id { font-weight: 700; font-size: 17px; color: var(--fg);
            font-family: ui-monospace, monospace; }
.status-pill { font-size: 12px; padding: 2px 10px; border-radius: 999px; }
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
                 border: 1px solid var(--border); display: block; }

.kv { padding: 6px 0; }
.kv b { display: inline-block; min-width: 0; color: var(--accent);
        font-size: 13px; }
.kv code { font-size: 13px; background: var(--bg); padding: 2px 6px;
           border-radius: 4px; }

.badge-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.badge { font-size: 12px; padding: 2px 8px; border-radius: 999px;
         font-weight: 500; }

.muted { color: var(--fg-muted); font-size: 12px; }
.dominant { margin-top: 6px; font-size: 13px; color: var(--fg-muted); }
.reason { color: var(--fg); margin-top: 4px;
          background: var(--bg); padding: 8px 12px; border-radius: 6px;
          border-left: 3px solid var(--accent); font-size: 13px; }
.reason-kv .reason { font-size: 14px; }
.seed {
  margin-top: 8px; padding: 8px 12px;
  background: #0c4a6e22; border: 1px solid #0c4a6e;
  border-radius: 6px;
}

img.final-mask { width: 100%; border-radius: 8px;
                 border: 1px solid var(--border); }

/* 视角选择 + 透明背景图（用 checkerboard 衬底） */
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
details { background: var(--bg-card); border: 1px solid var(--border);
          border-radius: 8px; padding: 8px 16px; margin-top: 24px; }
details summary { cursor: pointer; color: var(--accent);
                  font-weight: 600; }
pre { background: var(--bg); padding: 12px; border-radius: 6px;
      overflow-x: auto; font-size: 12px; }
"""


def _summary_card(title: str, value: Any) -> str:
    return (
        f'<div class="summary-card"><b>{_esc(title)}</b>'
        f'<span>{_esc(value)}</span></div>'
    )


def generate_report(output_dir: Path, html_path: Optional[Path] = None) -> Path:
    output_dir = Path(output_dir).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"目录不存在：{output_dir}")
    tree_json = output_dir / "block_tree.json"
    if not tree_json.is_file():
        raise FileNotFoundError(
            f"找不到 block_tree.json（请先跑 vlm.cli）：{tree_json}"
        )

    with open(tree_json, "r", encoding="utf-8") as f:
        tree = json.load(f)

    blocks: List[Dict[str, Any]] = list(tree.get("blocks", []))
    semantic = tree.get("semantic", "")
    num_iters = tree.get("num_iterations", 0)
    n_init = tree.get("num_initial_blocks", 0)
    n_final = tree.get("num_final_blocks", 0)
    terminated_by_vlm = bool(tree.get("terminated_by_vlm", False))

    # 加载每轮 VLM 自评（reviews_log.json）
    reviews_log: List[Dict[str, Any]] = []
    reviews_path = output_dir / "reviews_log.json"
    if reviews_path.is_file():
        try:
            with open(reviews_path, "r", encoding="utf-8") as f:
                reviews_log = json.load(f) or []
        except (OSError, ValueError) as exc:
            print(f"[warn] reviews_log.json 解析失败: {exc}")

    # 统计各 status 的 face 数
    status_face_counts: Dict[str, int] = {}
    accepted_at_max = 0
    accepted_faces = 0
    for b in blocks:
        st = _block_status_label(b)
        status_face_counts[st] = status_face_counts.get(st, 0) + int(b.get("num_faces", 0))
        if st == "full*":
            accepted_at_max += 1
            accepted_faces += int(b.get("num_faces", 0))

    full_faces = status_face_counts.get("full", 0) + status_face_counts.get("full*", 0)
    none_faces = status_face_counts.get("none", 0)
    abandoned_faces = status_face_counts.get("abandoned", 0)

    # 按 depth 分组
    by_depth: Dict[int, List[Dict[str, Any]]] = {}
    for b in blocks:
        by_depth.setdefault(int(b.get("depth", 0)), []).append(b)
    for d in by_depth:
        # 按 block_id 字典序，便于追踪兄弟关系
        by_depth[d].sort(key=lambda x: x.get("block_id", ""))

    if html_path is None:
        html_path = output_dir / "report.html"
    html_dir = html_path.parent

    # final mask 路径
    mask_rel = ""
    mask_path = output_dir / "semantic_mask.png"
    if mask_path.is_file():
        try:
            mask_rel = mask_path.relative_to(html_dir).as_posix()
        except ValueError:
            mask_rel = mask_path.as_posix()
    init_rel = ""
    init_path = output_dir / "initial_segmentation.png"
    if init_path.is_file():
        try:
            init_rel = init_path.relative_to(html_dir).as_posix()
        except ValueError:
            init_rel = init_path.as_posix()
    leaf_rel = ""
    leaf_path = output_dir / "leaf_blocks_overview.png"
    if leaf_path.is_file():
        try:
            leaf_rel = leaf_path.relative_to(html_dir).as_posix()
        except ValueError:
            leaf_rel = leaf_path.as_posix()

    # 视角选择产物（best_view.json / best_view_overview.png / view_thumbnails.png）
    best_view_data: Dict[str, Any] = {}
    bv_json = output_dir / "best_view.json"
    if bv_json.is_file():
        try:
            with open(bv_json, "r", encoding="utf-8") as f:
                best_view_data = json.load(f) or {}
        except (OSError, ValueError) as exc:
            print(f"[warn] best_view.json 解析失败: {exc}")

    def _rel_or_empty(p: Path) -> str:
        if not p.is_file():
            return ""
        try:
            return p.relative_to(html_dir).as_posix()
        except ValueError:
            return p.as_posix()

    overview_rel = _rel_or_empty(output_dir / "best_view_overview.png")
    thumbs_rel = _rel_or_empty(output_dir / "view_thumbnails.png")

    # 迭代时间线（pipeline 写到 output_dir/iterations/iter_NN.png）
    iter_dir = output_dir / "iterations"
    iter_pngs: List[Path] = []
    if iter_dir.is_dir():
        iter_pngs = sorted(iter_dir.glob("iter_*.png"))

    # ---- 渲染 ----
    sections = []
    for depth in sorted(by_depth.keys()):
        bs = by_depth[depth]
        # 统计该层 status 分布
        cnt: Dict[str, int] = {}
        for b in bs:
            cnt[_block_status_label(b)] = cnt.get(_block_status_label(b), 0) + 1
        cnt_html = " · ".join(
            f'<span style="color:{STATUS_COLOR.get(k, "#6366f1")}">{_esc(STATUS_ZH.get(k, k))} ×{v}</span>'
            for k, v in sorted(cnt.items())
        )
        cards_html = "\n".join(_render_block_card(b, html_dir) for b in bs)
        sections.append(
            f"""
<section class="iteration-section">
  <h2>第 {depth + 1} 轮 · depth={depth} · {len(bs)} 个 block</h2>
  <div class="muted">{cnt_html}</div>
  <div class="cards">{cards_html}</div>
</section>
"""
        )

    summary_cards = [
        _summary_card("语义", f"「{semantic}」"),
        _summary_card("迭代轮数", num_iters),
        _summary_card("初始 → 叶 block", f"{n_init} → {n_final}"),
        _summary_card("最大深度", tree.get("max_depth", "")),
        _summary_card("符合 face", f"{full_faces}"),
        _summary_card("不符合 face", f"{none_faces}"),
        _summary_card("舍弃 face", f"{abandoned_faces}"),
        _summary_card(
            "max-depth 接受", f"{accepted_at_max} 块 / {accepted_faces} face"
        ),
        _summary_card(
            "VLM 自评终止", "是" if terminated_by_vlm else "否（按队列自然结束）"
        ),
    ]

    # ---- 自评轨迹（每轮一张进度图 + VLM 评估）----
    review_section = ""
    if reviews_log:
        cards: List[str] = []
        for entry in reviews_log:
            it = entry.get("iteration", 0)
            review = entry.get("review") or {}
            progress_panel = entry.get("progress_panel", "")
            panel_rel = ""
            if progress_panel:
                p = Path(progress_panel.replace("\\", "/"))
                parts = p.parts
                if "reviews" in parts:
                    idx = parts.index("reviews")
                    panel_rel = "/".join(parts[idx:])
                else:
                    try:
                        panel_rel = p.relative_to(html_dir).as_posix()
                    except ValueError:
                        panel_rel = p.as_posix()
            full_f = entry.get("full_faces", 0)
            pending_f = entry.get("pending_faces", 0)
            abn_f = entry.get("abandoned_faces", 0)
            queue_after = entry.get("queue_after_iter", []) or []
            terminate = bool(review.get("should_terminate"))
            red_asm = review.get("red_assessment", "") or ""
            yel_asm = review.get("yellow_assessment", "") or ""
            red_parts = review.get("red_covered_parts", []) or []
            missing = review.get("missing_regions", "") or ""
            extra_r = review.get("extra_regions", "") or ""
            reason = review.get("reason", "") or ""
            confidence = review.get("confidence")
            elapsed = review.get("elapsed_seconds")
            err = review.get("error")

            decision_color = "#dc2626" if terminate else "#0891b2"
            decision_text = "TERMINATE（VLM 决定终止）" if terminate else "CONTINUE（继续下一轮）"
            conf_html = (
                f'<span class="muted">conf={float(confidence):.2f}</span>'
                if confidence is not None
                else ""
            )
            elapsed_html = (
                f'<span class="muted">{float(elapsed):.1f}s</span>'
                if elapsed is not None
                else ""
            )
            err_html = (
                f'<div class="kv" style="color:#f87171"><b>error</b>: {_esc(err)}</div>'
                if err
                else ""
            )
            img_html = (
                f'<a href="{_esc(panel_rel)}" target="_blank">'
                f'<img loading="lazy" src="{_esc(panel_rel)}" '
                f'alt="iter {it} progress" /></a>'
                if panel_rel
                else '<div class="muted">[无 progress 面板]</div>'
            )
            queue_html = (
                " ".join(
                    f'<a href="#block-{_esc(c)}" class="child-link">{_esc(c)}</a>'
                    for c in queue_after
                )
                if queue_after
                else '<span class="muted">—</span>'
            )

            cards.append(
                f"""
<article class="block-card">
  <header class="card-header">
    <div class="card-title">
      <span class="block-id">Iteration {_esc(it)} · 自评</span>
      <span class="status-pill" style="background:{decision_color}22;
            color:{decision_color};border:1px solid {decision_color}55">
        {_esc(decision_text)}
      </span>
      {conf_html} {elapsed_html}
    </div>
    <div class="card-meta">
      此轮结束后 full={full_f} · pending={pending_f} · abandoned={abn_f}
      &nbsp;|&nbsp; 下一轮排队: {queue_html}
    </div>
  </header>
  <div class="card-body">
    <div class="panel-img">{img_html}</div>
    <div class="card-fields">
      <div class="kv">
        <b>red_assessment</b>:
        <code>{_esc(red_asm)}</code> &nbsp;
        <b>yellow_assessment</b>: <code>{_esc(yel_asm)}</code>
      </div>
      <div class="kv">
        <b>red 当前覆盖</b>
        <div class="badge-row">{_list_badges(red_parts, "#dc2626")}</div>
      </div>
      <div class="kv">
        <b>缺漏部位 (missing)</b>:
        <code>{_esc(missing) if missing else '—'}</code>
      </div>
      <div class="kv">
        <b>多余部位 (extra)</b>:
        <code>{_esc(extra_r) if extra_r else '—'}</code>
      </div>
      <div class="kv reason-kv">
        <b>VLM 决定理由</b>:
        <div class="reason">{_esc(reason)}</div>
      </div>
      {err_html}
    </div>
  </div>
</article>
"""
            )
        review_section = f"""
<section>
  <h2 style="color:var(--accent)">VLM 全局自评（每轮一帧）</h2>
  <div class="muted">每轮迭代结束后给 VLM 看一张 12 视角的多色进度图，
  让它评估 red 是否准确并决定是否提前终止。</div>
  <div class="cards" style="margin-top:12px">{''.join(cards)}</div>
</section>
"""

    final_imgs = ""
    if mask_rel:
        final_imgs += (
            f'<div class="muted" style="margin-top:8px">'
            f'最终语义掩膜（红=full / 灰=non-full）'
            f'</div>'
            f'<a href="{_esc(mask_rel)}" target="_blank">'
            f'<img class="final-mask" src="{_esc(mask_rel)}" '
            f'alt="semantic_mask" /></a>'
        )
    if leaf_rel:
        final_imgs += (
            f'<div class="muted" style="margin-top:14px">'
            f'叶分块总览（每块独立随机色，便于核对最终切分粒度）'
            f'</div>'
            f'<a href="{_esc(leaf_rel)}" target="_blank">'
            f'<img class="final-mask" src="{_esc(leaf_rel)}" '
            f'alt="leaf_blocks_overview" /></a>'
        )
    if init_rel:
        final_imgs += (
            f'<div class="muted" style="margin-top:14px">'
            f'初始分块（按连通分量切分前的根集合）'
            f'</div>'
            f'<a href="{_esc(init_rel)}" target="_blank">'
            f'<img class="final-mask" src="{_esc(init_rel)}" '
            f'alt="initial_segmentation" /></a>'
        )

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
        conf_html = (
            f' · <b>conf</b>=<code>{float(bv_confidence):.2f}</code>'
            if bv_confidence is not None else ""
        )
        provider_html = (
            f' · <b>by</b> <code>{_esc(bv_provider)}/{_esc(bv_model)}</code>'
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
            else '<span class="badge" style="background:#1d4ed822;color:#60a5fa;'
            'border:1px solid #60a5fa55">VLM 选定</span>'
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
<section>
  <h2 style="color:var(--accent)">展示视角（VLM 选定）</h2>
  <div class="viewcard">
    <div>{overview_img_html}</div>
    <div class="vmeta">
      <b>视角编号</b> #<code>{_esc(bv_idx)}</code> {tag}
      <br/><b>elev</b> {elev_html} &nbsp; <b>azim</b> {azim_html}
      {conf_html}{provider_html}
      <div class="vreason"><b>VLM 理由：</b>{_esc(bv_reason) or '—'}</div>
      {thumbs_html}
    </div>
  </div>
</section>
"""

    # ---- 迭代时间线区块 ----
    timeline_section = ""
    if iter_pngs:
        figs: List[str] = []
        for png in iter_pngs:
            try:
                rel = png.relative_to(html_dir).as_posix()
            except ValueError:
                rel = png.as_posix()
            n = png.stem.split("_")[-1]  # iter_03 -> "03"
            figs.append(
                f'<figure><a href="{_esc(rel)}" target="_blank">'
                f'<img class="transparent-img" src="{_esc(rel)}" '
                f'alt="iter {_esc(n)}" /></a>'
                f'<figcaption>iter {_esc(n)}</figcaption></figure>'
            )
        timeline_section = f"""
<section>
  <h2 style="color:var(--accent)">迭代时间线（单视角 · 透明背景）</h2>
  <div class="muted">每轮迭代结束时按 VLM 选定视角渲染：
    <span style="color:#fff;background:#1d1d1f;padding:1px 6px;
      border-radius:4px;border:1px solid #fff;">none=白</span>
    &nbsp;
    <span style="color:#a16400;background:#ffd782;padding:1px 6px;
      border-radius:4px;">partial=浅黄</span>
    &nbsp;
    <span style="color:#fff;background:#8c1923;padding:1px 6px;
      border-radius:4px;">full=深红</span>
    &nbsp;
    <span style="color:#444;background:#ebebeb;padding:1px 6px;
      border-radius:4px;">abandoned=极浅灰</span>
  </div>
  <div class="timeline-grid">{''.join(figs)}</div>
</section>
"""

    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>VLM 语义细化报告 · 「{_esc(semantic)}」</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">

<header class="top">
  <h1>VLM 语义细化报告 · 「{_esc(semantic)}」</h1>
  <div class="muted">输出目录：<code>{_esc(output_dir)}</code></div>
  <div class="summary-grid">
    {''.join(summary_cards)}
  </div>
</header>

{best_view_section}

{timeline_section}

<section>
  <h2 style="color:var(--accent)">最终结果</h2>
  {final_imgs}
</section>

{review_section}

{''.join(sections)}

<details>
  <summary>原始 block_tree.json（点击展开）</summary>
  <pre>{_esc(json.dumps(tree, ensure_ascii=False, indent=2))}</pre>
</details>

</div>
</body>
</html>
"""
    html_path.write_text(body, encoding="utf-8")
    return html_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vlm.report",
        description=(
            "把一次 VLM 语义细化的输出（output/<模型>/vlm/<语义>/）"
            "渲染成单文件 HTML 报告。"
        ),
    )
    p.add_argument(
        "output_dir",
        type=str,
        help="vlm 输出目录，例如 output/<模型>/vlm/<语义>/",
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
    path = generate_report(Path(args.output_dir), out)
    print(f"[ok] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
