"""
点云语义细化流水线（与 mesh 版 ``SemanticRefinementPipeline`` 同构）。

与 mesh 版的唯一区别：
- 标签单元是 **point_id** 而不是 face_id；
- **没有初始分块**——整朵点云作为唯一的根 block，直接开始分裂；
- 分裂用迁移自 mesh ``spectral_cluster`` 的几何特征谱聚类（``split_points``，默认
  ``feature_spectral``）：归一化坐标 + PCA 法向 + 曲率（+ 可选凹凸/凸包指标）→ kNN
  高斯亲和图 → 归一化图拉普拉斯 → 最小特征向量 → KMeans，切分贴合几何部件边界。
  纯点云**默认不考虑连通性**（可选 geodesic_fps / CC 清洗，见 split_points）。

迭代逻辑与 mesh 版完全一致：
  每个 PENDING block 渲染 12 视角高亮图 → VLM 三分类：
    full     完全符合语义       → 接受（叶子）
    none     完全不符合语义     → 拒绝（叶子）
    partial  局部符合，需细化   → VLM 给 seed_count，谱聚类分裂出子块，继续迭代
  直到没有 partial 或到达 max_depth。

产物（与 mesh 版对齐，可直接喂给 pipeline.report.generate_report）：
  panels/                每个 block 的高亮 12 视角图
  responses/             每次 VLM 原始响应
  iterations/iter_NN.png 每轮迭代末尾的单视角进度图（透明背景）
  initial_segmentation.png 第一刀切完后的分块总览
  semantic_mask.png      最终 full 点高亮
  live_progress.png + live_report.html  实时进度
  block_tree.json        完整 block 树（report.generate_report 的输入）
  semantic_mask.json     最终 full point_ids
  responses_log.json     调用流水
"""

from __future__ import annotations

import html as _html
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from geometry.blocks import BlockStatus, SemanticBlock, export_tree_json
from geometry.pointcloud_cluster import (
    DEFAULT_CURVATURE_WEIGHT,
    DEFAULT_KNN,
    DEFAULT_NORMAL_WEIGHT,
    split_points,
)
from geometry.seed_split import ask_seed_count
from rendering.heatmap_renderer import DEFAULT_OVERVIEW_VIEW
from rendering.pointcloud_render import (
    STATUS_COLOR_MAP,
    render_pointcloud_highlight_panel,
    render_pointcloud_overview,
    render_pointcloud_segmentation_panel,
    render_pointcloud_status_view,
    render_pointcloud_thumbnails_panel,
)
from vlm.classifier import classify_component
from vlm.client import VLMClient
from vlm.prompts import LABEL_FULL, LABEL_NONE, LABEL_PARTIAL
from vlm.view_selector import BestViewSelection, select_best_view

LOGGER = logging.getLogger("pipeline.pointcloud_pipeline")


@dataclass
class PointCloudRefinementConfig:
    semantic: str
    max_depth: int = 6
    root_seed_count: int = 6          # 第一刀（无初始分块）切多少块
    max_seeds: int = 6                # partial 块每次最多切几刀
    min_block_size: int = 60          # 小于此点数的块不再继续切，但仍走 VLM 判 full/none
    # 几何分块（迁移自 mesh spectral_cluster / seed_split）。默认 feature_spectral：
    # 归一化坐标 + PCA 法向 + 曲率(+可选凹凸) → kNN 谱聚类。纯点云默认不考虑连通性。
    split_method: str = "feature_spectral"   # feature_spectral | geodesic_fps | lbo_spectral | kmeans
    split_n_neighbors: int = DEFAULT_KNN     # kNN 图 / PCA 法向曲率的邻居数
    split_lloyd_iter: int = 8                # geodesic_fps 的 Lloyd 迭代次数
    split_cc_first: bool = True              # 分裂前先按 kNN 连通分量拆（消除远距断岛）
    split_enforce_connectivity: bool = True    # KMeans 后逐 label 连通清洗（每块单连通）
    split_curvature_weight: float = DEFAULT_CURVATURE_WEIGHT  # 曲率指标权重(feature_spectral)
    split_convexity_weight: float = 0.0      # 凹凸/凸包指标权重(feature_spectral，>0 启用)
    split_normal_weight: float = DEFAULT_NORMAL_WEIGHT        # 法向权重(feature_spectral)
    # NEO 后端：开启后在 run() 开始时对整朵点云**一次性**前馈预测全局 LBO 特征空间，
    # 之后所有子块分裂走 lbo_spectral，直接切片复用全局特征向量 + KMeans，
    # 不再每块重算 scipy eigsh（大点云上快很多）。NEO 跑在独立 .conda-neo 环境。
    use_neo: bool = False
    neo_ckpt: Optional[str] = None           # 权重 .ckpt（或环境变量 NEO_CKPT）
    neo_python: Optional[str] = None         # NEO 环境 python（默认 .conda-neo/bin/python）
    neo_k: int = 0                           # 取多少个模式；0=模型默认 targ_dim_model
    neo_device: str = "cpu"                  # cpu/cuda/mps
    neo_no_mass: bool = False                # 消融 mass-aware attention
    image_size: int = 512
    panel_cols: int = 4
    point_size: Optional[float] = None
    classification_model: Optional[str] = None
    seed_model: Optional[str] = None
    save_responses: bool = True
    # max_depth 处遇到 partial 时是否当作 best-effort full 接受。
    accept_partial_at_max_depth: bool = False
    accept_partial_min_points: int = 400
    # 实时报告
    live_report: bool = True
    live_report_refresh_seconds: int = 10
    single_view_image_size: int = 720
    # 展示视角（单视角 live/timeline 用）。默认与 mesh overview 同系。
    display_elev: float = DEFAULT_OVERVIEW_VIEW[0]
    display_azim: float = DEFAULT_OVERVIEW_VIEW[1]
    # 视角选择：主循环前让 VLM 看 12 张缩略图挑「最能展示该语义」的视角，
    # 作为 live_report / report / 每轮 iter PNG 的统一单视角展示视角（与 mesh 版一致）。
    # 注意：只影响展示图，不影响分类（分类仍用 12 视角高亮拼图）。
    select_best_view: bool = True
    view_select_model: Optional[str] = None
    # VLM 三分类 / ask_seed_count 并发数。>1 时一批块并行调 VLM（HTTP IO），
    # 等待期间主线程继续处理 force_split 等几何分块（见 run() 三轮 pass）。
    parallel_vlm_workers: int = 4


@dataclass
class PointCloudRefinementResult:
    ok: bool
    output_dir: Path
    semantic: str
    full_point_ids: List[int] = field(default_factory=list)
    none_point_ids: List[int] = field(default_factory=list)
    pending_point_ids: List[int] = field(default_factory=list)
    num_final_blocks: int = 0
    num_iterations: int = 0
    tree_json: Optional[Path] = None
    mask_json: Optional[Path] = None
    semantic_panel: Optional[Path] = None
    report_html: Optional[Path] = None
    # 视角选择产物（VLM 选最佳展示视角 + 透明背景 overview）
    best_view: Optional[BestViewSelection] = None
    best_view_json: Optional[Path] = None
    best_view_overview: Optional[Path] = None
    best_view_thumbnails: Optional[Path] = None
    elapsed_seconds: float = 0.0
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class PointCloudRefinementPipeline:
    """点云版「VLM 三分类 + KMeans 分裂」迭代循环。"""

    _LIVE_COLOR_MAP = STATUS_COLOR_MAP

    def __init__(self, client: VLMClient, config: PointCloudRefinementConfig):
        self.client = client
        self.config = config
        self.points: np.ndarray = np.zeros((0, 3))
        self.blocks: Dict[str, SemanticBlock] = {}
        self._counter_per_parent: Dict[str, int] = {}
        self._iter_progress_pngs: List[Path] = []
        self._best_view: Optional[BestViewSelection] = None
        # NEO 全局特征空间（整朵点云一次性前馈得到，子块分裂复用）。None=未用 NEO。
        self._neo_eigvecs: Optional[np.ndarray] = None

    # ---------- 展示视角 ----------
    def _display_view(self) -> Tuple[float, float]:
        """单视角展示用 (elev, azim)：优先 VLM 选中的最佳视角，否则用默认 overview。"""
        if self._best_view is not None:
            return float(self._best_view.elev), float(self._best_view.azim)
        return float(self.config.display_elev), float(self.config.display_azim)

    # ---------- 树工具 ----------
    def _next_child_id(self, parent_id: str) -> str:
        n = self._counter_per_parent.get(parent_id, 0)
        self._counter_per_parent[parent_id] = n + 1
        return f"{parent_id}.{n}"

    def _register_block(self, block: SemanticBlock) -> None:
        self.blocks[block.block_id] = block

    # ---------- 点状态（用于进度图 / 掩码） ----------
    def _compute_point_status(self, n_points: int) -> List[str]:
        status = ["none"] * n_points
        for block in self.blocks.values():
            if block.status == BlockStatus.SPLIT:
                continue
            if block.status == BlockStatus.FULL:
                tag = "full"
            elif block.status == BlockStatus.PENDING:
                tag = "pending"
            elif block.status == BlockStatus.ABANDONED:
                tag = "abandoned"
            else:
                tag = "none"
            for pid in block.face_ids:
                if 0 <= pid < n_points:
                    status[pid] = tag
        return status

    # ---------- 渲染 ----------
    def _render_block_panel(self, block: SemanticBlock, panel_dir: Path) -> Path:
        # 标题不写 semantic，避免给 VLM 暗示（确认偏差）。
        title = f"[block {block.block_id}] depth={block.depth} pts={block.size}"
        panel_path = panel_dir / f"block_{block.block_id}.png"
        return render_pointcloud_highlight_panel(
            self.points,
            block.face_ids,
            panel_path,
            title=title,
            image_size=(self.config.image_size, self.config.image_size),
            cols=self.config.panel_cols,
            point_size=self.config.point_size,
        )

    def _dump_response(self, responses_dir: Path, block_id: str, payload: Dict[str, Any]) -> None:
        if not self.config.save_responses:
            return
        try:
            (responses_dir / f"block_{block_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            LOGGER.warning("写 response 失败 %s: %s", block_id, exc)

    # ---------- NEO 全局特征空间 ----------
    def _compute_neo_eigenspace(self) -> None:
        """对整朵点云调用一次 NEO，前馈预测全局低频 LBO 特征向量，存入
        self._neo_eigvecs (N, k)，供后续所有子块 lbo_spectral 切片复用。"""
        from geometry.neo_backend import neo_eigenspace

        LOGGER.info("调用 NEO 前馈全局特征空间（整朵点云 %d 点，一次性）...", len(self.points))
        t = time.perf_counter()
        eigvecs, _eigvals = neo_eigenspace(
            self.points,
            ckpt=self.config.neo_ckpt,
            neo_python=self.config.neo_python,
            k=self.config.neo_k,
            device=self.config.neo_device,
            no_mass=self.config.neo_no_mass,
        )
        self._neo_eigvecs = np.asarray(eigvecs, dtype=np.float64)
        LOGGER.info(
            "NEO 全局特征空间就绪: eigvecs=%s，耗时 %.1fs；后续分裂走 lbo_spectral 切片复用",
            self._neo_eigvecs.shape, time.perf_counter() - t,
        )

    # ---------- 分裂 ----------
    def _split_block(self, block: SemanticBlock, seed_count: int) -> List[SemanticBlock]:
        # NEO 开启且全局特征空间就绪时，强制走 lbo_spectral：子块直接切片全局
        # 特征向量 + KMeans（不再每块重算 eigsh）。否则用 config 指定的方法。
        if self._neo_eigvecs is not None:
            method = "lbo_spectral"
        else:
            method = self.config.split_method
        sub = split_points(
            self.points,
            block.face_ids,
            seed_count=seed_count,
            method=method,
            n_neighbors=self.config.split_n_neighbors,
            lloyd_iter=self.config.split_lloyd_iter,
            split_cc_first=self.config.split_cc_first,
            enforce_connectivity=self.config.split_enforce_connectivity,
            curvature_weight=self.config.split_curvature_weight,
            convexity_weight=self.config.split_convexity_weight,
            normal_weight=self.config.split_normal_weight,
            precomputed_eigvecs=self._neo_eigvecs,
            random_state=0,
        )
        children: List[SemanticBlock] = []
        for group in sub.sub_point_groups:
            if not group:
                continue
            child_id = self._next_child_id(block.block_id)
            child = SemanticBlock(
                block_id=child_id,
                face_ids=list(group),
                depth=block.depth + 1,
                parent_id=block.block_id,
                status=BlockStatus.PENDING,
                notes=f"split({sub.debug.get('strategy','')})",
            )
            self._register_block(child)
            block.children_ids.append(child_id)
            children.append(child)
        return children

    # ---------- 实时报告 ----------
    def _save_iteration_progress(self, out_dir: Path, *, iteration: int) -> Optional[Path]:
        n_points = len(self.points)
        status = self._compute_point_status(n_points)
        it_dir = out_dir / "iterations"
        it_dir.mkdir(parents=True, exist_ok=True)
        png = it_dir / f"iter_{iteration:02d}.png"
        elev, azim = self._display_view()
        try:
            render_pointcloud_status_view(
                self.points,
                status,
                png,
                elev=elev,
                azim=azim,
                image_size=self.config.single_view_image_size,
                color_map=self._LIVE_COLOR_MAP,
                background="transparent",
                point_size=self.config.point_size,
            )
        except (OSError, ValueError, ImportError) as exc:
            LOGGER.warning("iter %d 进度图渲染失败: %s", iteration, exc)
            return None
        self._iter_progress_pngs.append(png)
        return png

    def _update_live_report(
        self,
        out_dir: Path,
        *,
        iteration: int,
        queue_after: Sequence[str],
        finished: bool = False,
        extra_note: str = "",
    ) -> None:
        if not self.config.live_report:
            return
        n_points = len(self.points)
        status = self._compute_point_status(n_points)
        live_png = out_dir / "live_progress.png"
        rendered = False
        elev, azim = self._display_view()
        try:
            render_pointcloud_status_view(
                self.points,
                status,
                live_png,
                elev=elev,
                azim=azim,
                image_size=self.config.single_view_image_size,
                color_map=self._LIVE_COLOR_MAP,
                background="transparent",
                point_size=self.config.point_size,
            )
            rendered = True
        except (OSError, ValueError, ImportError) as exc:
            LOGGER.warning("live progress 渲染失败: %s", exc)

        block_counts = {st.value: 0 for st in BlockStatus}
        for b in self.blocks.values():
            block_counts[b.status.value] = block_counts.get(b.status.value, 0) + 1
        full_pts = sum(1 for s in status if s == "full")
        pending_pts = sum(1 for s in status if s == "pending")
        abandoned_pts = sum(1 for s in status if s == "abandoned")
        none_pts = n_points - full_pts - pending_pts - abandoned_pts

        def esc(x: Any) -> str:
            return _html.escape(str(x))

        refresh_meta = (
            ""
            if finished
            else f'<meta http-equiv="refresh" content="{int(self.config.live_report_refresh_seconds)}">'
        )
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        status_tag = (
            '<span class="tag tag-done">已完成</span>'
            if finished
            else '<span class="tag tag-live">运行中</span>'
        )
        img_html = (
            f'<img class="progress" src="live_progress.png?t={int(time.time())}" alt="live progress" />'
            if rendered
            else '<div class="muted">(尚未生成 live_progress.png)</div>'
        )
        extra_html = f'<div class="extra">{esc(extra_note)}</div>' if extra_note else ""

        view_card_html = ""
        if self._best_view is not None:
            bv = self._best_view
            tag = (
                f'#{bv.index + 1} · elev={bv.elev:.0f}° azim={bv.azim:.0f}°'
                + (' · <span class="muted">fallback 默认视角</span>' if bv.fallback
                   else f' · conf={bv.confidence:.2f}')
            )
            view_card_html = (
                '<section class="view-card">'
                f'<div class="view-meta"><b>展示视角</b> {tag}</div>'
                f'<div class="view-reason">{esc(bv.reason or "—")}</div>'
                '</section>'
            )

        timeline_html = ""
        if self._iter_progress_pngs:
            items: List[str] = []
            for png in self._iter_progress_pngs:
                try:
                    rel = png.relative_to(out_dir).as_posix()
                except ValueError:
                    rel = f"iterations/{png.name}"
                m = png.stem.split("_")[-1]
                items.append(
                    f'<figure class="timeline-item">'
                    f'<img src="{esc(rel)}?t={int(time.time())}" alt="iter {esc(m)}" />'
                    f'<figcaption>iter {esc(m)}</figcaption></figure>'
                )
            timeline_html = (
                f'<section class="timeline"><h2>迭代时间线（共 {len(items)} 帧）</h2>'
                f'<div class="timeline-row">{"".join(items)}</div></section>'
            )

        css = """
            body { font-family: -apple-system, "Segoe UI", "Microsoft Yahei", sans-serif;
                   background:#f5f5f7; color:#1d1d1f; margin:0; padding:24px; }
            h1 { margin:0 0 8px 0; font-size:22px; }
            h2 { font-size:16px; margin:20px 0 10px 0; }
            .row { display:flex; gap:16px; flex-wrap:wrap; margin:14px 0; }
            .card { background:#fff; border:1px solid #d2d2d7; border-radius:10px;
                    padding:14px 18px; min-width:200px; }
            .card b { display:block; color:#6e6e73; font-size:12px; font-weight:500; margin-bottom:4px; }
            .card span.v { font-size:20px; font-weight:600; }
            .legend span { display:inline-flex; align-items:center; gap:6px; margin-right:14px; font-size:13px; }
            .legend i { display:inline-block; width:14px; height:14px; border:1px solid #999; border-radius:3px; }
            .tag { padding:3px 10px; border-radius:999px; font-size:12px; font-weight:600; margin-left:8px; }
            .tag-live { background:#fff7d9; color:#a16400; border:1px solid #f1d27a; }
            .tag-done { background:#d6f4e0; color:#0a6c2c; border:1px solid #7dc99a; }
            .muted { color:#6e6e73; }
            .ts { color:#86868b; font-size:12px; margin-bottom:10px; }
            .progress, .timeline-item img {
              max-width:100%; border:1px solid #d2d2d7; border-radius:8px; background-color:#fff;
              background-image: linear-gradient(45deg, #f0f0f5 25%, transparent 25%),
                                linear-gradient(-45deg, #f0f0f5 25%, transparent 25%),
                                linear-gradient(45deg, transparent 75%, #f0f0f5 75%),
                                linear-gradient(-45deg, transparent 75%, #f0f0f5 75%);
              background-size: 18px 18px; background-position: 0 0, 0 9px, 9px -9px, -9px 0; }
            .progress { max-width:720px; display:block; margin:0 auto; }
            .timeline { margin-top:24px; }
            .timeline-row { display:flex; gap:10px; overflow-x:auto; padding:8px 0; }
            .timeline-item { margin:0; flex:0 0 auto; text-align:center; }
            .timeline-item img { width:180px; height:auto; display:block; }
            .timeline-item figcaption { color:#6e6e73; font-size:12px; margin-top:4px; }
            .extra { background:#fff7d9; border:1px solid #f1d27a; padding:10px 14px;
                     border-radius:8px; margin:10px 0; }
            .view-card { background:#fff; border:1px solid #d2d2d7; border-radius:10px;
                         padding:12px 16px; margin:12px 0; }
            .view-meta { font-size:14px; }
            .view-reason { color:#3a3a3c; font-size:13px; margin-top:6px; line-height:1.5; }
        """

        body = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">{refresh_meta}
<title>点云 VLM live · {esc(self.config.semantic)} · iter {iteration}</title>
<style>{css}</style></head><body>
<h1>点云语义细化 · 实时进度 {status_tag}</h1>
<div class="ts">更新时间: {esc(ts)} · semantic: <b>{esc(self.config.semantic)}</b>
 · iteration: <b>{iteration}</b> · 下轮队列: <b>{len(queue_after)}</b></div>
<div class="legend">
  <span><i style="background:#fff"></i>none（白）</span>
  <span><i style="background:#ffd782"></i>partial 待分裂（浅黄）</span>
  <span><i style="background:#8c1923"></i>full（深红高亮）</span>
  <span><i style="background:#ebebeb"></i>abandoned（极浅灰）</span>
</div>
{view_card_html}
<div class="row">
  <div class="card"><b>full points</b><span class="v">{full_pts}</span></div>
  <div class="card"><b>partial(pending) points</b><span class="v">{pending_pts}</span></div>
  <div class="card"><b>none points</b><span class="v">{none_pts}</span></div>
  <div class="card"><b>abandoned points</b><span class="v">{abandoned_pts}</span></div>
  <div class="card"><b>total points</b><span class="v">{n_points}</span></div>
</div>
<div class="row">
  <div class="card"><b>full blocks</b><span class="v">{block_counts.get('full', 0)}</span></div>
  <div class="card"><b>pending blocks</b><span class="v">{block_counts.get('pending', 0)}</span></div>
  <div class="card"><b>split blocks</b><span class="v">{block_counts.get('split', 0)}</span></div>
  <div class="card"><b>none blocks</b><span class="v">{block_counts.get('none', 0)}</span></div>
  <div class="card"><b>abandoned blocks</b><span class="v">{block_counts.get('abandoned', 0)}</span></div>
</div>
{extra_html}
<h2>当前进度图（单视角 · 透明背景）</h2>
{img_html}
{timeline_html}
</body></html>
"""
        try:
            (out_dir / "live_report.html").write_text(body, encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("live report 写文件失败: %s", exc)

    # ---------- 分类 ----------
    @staticmethod
    def _cls_dict_from_component(cls: Any) -> Dict[str, Any]:
        return {
            "label": cls.label,
            "confidence": cls.confidence,
            "reason": cls.reason,
            "covered_parts": cls.covered_parts,
            "dominant_region": cls.dominant_region,
            "semantic_parts": cls.semantic_parts,
            "extra_parts": cls.extra_parts,
            "missing_overlap": cls.missing_overlap,
            "semantic_area_ratio": cls.semantic_area_ratio,
            "dominant_side": cls.dominant_side,
            "committed_if_forced": cls.committed_if_forced,
            "elapsed_seconds": cls.elapsed_seconds,
            "provider": cls.provider,
            "model": cls.model,
            "error": cls.error,
        }

    def _classify(
        self,
        block: SemanticBlock,
        panel_path: Path,
        *,
        force_terminate: bool = False,
    ) -> Dict[str, Any]:
        cls = classify_component(
            self.client,
            panel_path,
            self.config.semantic,
            model=self.config.classification_model,
            media_type="pointcloud",
            force_terminate=force_terminate,
        )
        return self._cls_dict_from_component(cls)

    def _batch_classify(
        self,
        tasks: Sequence[Tuple[str, Path]],
        *,
        force_terminate_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """并行批量三分类；VLM 等待期间不阻塞其它几何分块（见 run 主循环）。"""
        results: Dict[str, Dict[str, Any]] = {}
        if not tasks:
            return results

        ft_ids = force_terminate_ids or set()

        def _do(bid: str, panel: Path) -> Tuple[str, Dict[str, Any]]:
            return bid, self._classify(
                self.blocks[bid],
                panel,
                force_terminate=(bid in ft_ids),
            )

        workers = max(1, int(self.config.parallel_vlm_workers))
        if workers > 1 and len(tasks) > 1:
            n = min(workers, len(tasks))
            LOGGER.info("parallel classify: %d blocks, %d workers", len(tasks), n)
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = {ex.submit(_do, bid, panel): bid for bid, panel in tasks}
                for fut in as_completed(futs):
                    bid = futs[fut]
                    try:
                        _, cls = fut.result()
                        results[bid] = cls
                    except (RuntimeError, ValueError, OSError) as exc:
                        LOGGER.warning("parallel classify failed %s: %s", bid, exc)
        else:
            for bid, panel in tasks:
                try:
                    _, cls = _do(bid, panel)
                    results[bid] = cls
                except (RuntimeError, ValueError, OSError) as exc:
                    LOGGER.warning("serial classify failed %s: %s", bid, exc)
        return results

    def _batch_ask_seed_count(
        self,
        tasks: Sequence[Tuple[str, Path, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """并行批量 ask_seed_count（partial 块）。"""
        from geometry.seed_split import SeedCountResponse

        results: Dict[str, SeedCountResponse] = {}
        if not tasks:
            return results

        def _do(bid: str, panel: Path, cls: Dict[str, Any]) -> Tuple[str, SeedCountResponse]:
            block = self.blocks[bid]
            try:
                resp = ask_seed_count(
                    self.client,
                    panel,
                    self.config.semantic,
                    max_seeds=self.config.max_seeds,
                    fallback=2,
                    model=self.config.seed_model,
                    covered_parts=cls.get("covered_parts"),
                    semantic_parts=cls.get("semantic_parts"),
                    extra_parts=cls.get("extra_parts"),
                    block_points=block.size,
                    media_type="pointcloud",
                )
            except Exception as exc:  # noqa: BLE001
                from geometry.seed_split import SeedCountResponse as SCR
                resp = SCR(seed_count=2, reason=f"ask_seed_count_failed: {exc}")
            return bid, resp

        workers = max(1, int(self.config.parallel_vlm_workers))
        if workers > 1 and len(tasks) > 1:
            n = min(workers, len(tasks))
            LOGGER.info("parallel ask_seed_count: %d blocks, %d workers", len(tasks), n)
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = {
                    ex.submit(_do, bid, panel, cls): bid
                    for bid, panel, cls in tasks
                }
                for fut in as_completed(futs):
                    bid = futs[fut]
                    try:
                        _, resp = fut.result()
                        results[bid] = resp
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("parallel ask_seed_count failed %s: %s", bid, exc)
        else:
            for bid, panel, cls in tasks:
                try:
                    _, resp = _do(bid, panel, cls)
                    results[bid] = resp
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("serial ask_seed_count failed %s: %s", bid, exc)
        return results

    @staticmethod
    def _force_binary_label(cls: Dict[str, Any]) -> str:
        """强制终止时只允许 full / none：partial 按 committed_if_forced / dominant_side 二选一。"""
        label = cls.get("label") or LABEL_NONE
        if label == LABEL_FULL:
            return LABEL_FULL
        if label == LABEL_NONE:
            return LABEL_NONE
        committed = cls.get("committed_if_forced") or ""
        if committed == LABEL_FULL:
            return LABEL_FULL
        if committed == LABEL_NONE:
            return LABEL_NONE
        if cls.get("dominant_side") == "semantic":
            return LABEL_FULL
        return LABEL_NONE

    def _apply_force_terminate(
        self,
        block: SemanticBlock,
        cls: Dict[str, Any],
        responses_dir: Path,
        responses_log: List[Dict[str, Any]],
        *,
        iteration: int,
        reason_tag: str,
    ) -> None:
        """用已有 VLM 分类结果强制落为 full/none 后终止（不再 split）。"""
        forced = self._force_binary_label(cls)
        cls = {**cls, "label": forced, "forced_terminate": True, "original_label": cls.get("label")}
        block.classification = cls
        block.status = BlockStatus.FULL if forced == LABEL_FULL else BlockStatus.NONE
        block.notes = (block.notes + f" | {reason_tag}_force→{forced}").strip(" |")
        responses_log.append({"block_id": block.block_id, "iteration": iteration, **cls})
        self._dump_response(responses_dir, block.block_id, cls)
        LOGGER.info(
            "block %s %s → force %s (orig=%s)",
            block.block_id, reason_tag, forced, cls.get("original_label"),
        )

    def _force_terminate_block(
        self,
        block: SemanticBlock,
        panel_dir: Path,
        responses_dir: Path,
        responses_log: List[Dict[str, Any]],
        *,
        iteration: int,
        reason_tag: str,
        cls: Optional[Dict[str, Any]] = None,
    ) -> None:
        """块过小或无法继续切：渲染 + VLM 三分类（或复用缓存 cls），强制 full/none 后终止。"""
        if cls is None:
            try:
                panel_path = self._render_block_panel(block, panel_dir)
                block.panel_path = str(panel_path)
                cls = self._classify(block, panel_path, force_terminate=True)
            except (OSError, ImportError, ValueError) as exc:
                LOGGER.warning("block %s %s 渲染/分类失败，强制 none: %s", block.block_id, reason_tag, exc)
                cls = {
                    "label": LABEL_NONE,
                    "confidence": 0.0,
                    "reason": f"{reason_tag} classify/render failed: {exc}",
                    "forced_terminate": True,
                }
        self._apply_force_terminate(
            block, cls, responses_dir, responses_log,
            iteration=iteration, reason_tag=reason_tag,
        )

    # ---------- 视角选择 ----------
    def _select_view_for_display(self, out_dir: Path) -> Optional[BestViewSelection]:
        """渲染 12 缩略图 → VLM 挑最佳展示视角 → 渲一张透明背景 overview PNG。

        - 关掉视角选择 (select_best_view=False) 时退回默认 overview 视角；
        - 任何阶段出错都 fallback 到默认视角，主流程不被阻断；
        - 只影响展示图（report/timeline），不影响分类。
        """
        from rendering.heatmap_renderer import VIEW_GRID

        def _fallback(reason: str) -> BestViewSelection:
            target = (float(self.config.display_elev), float(self.config.display_azim))
            best_idx, best_d = 0, float("inf")
            for i, (el, az) in enumerate(VIEW_GRID):
                d = (el - target[0]) ** 2 + (az - target[1]) ** 2
                if d < best_d:
                    best_d, best_idx = d, i
            el, az = VIEW_GRID[best_idx]
            return BestViewSelection(
                index=best_idx, elev=float(el), azim=float(az),
                reason=reason, fallback=True,
            )

        if not self.config.select_best_view:
            bv = _fallback("select_best_view=False，跳过 VLM，直接用默认 overview 视角")
        else:
            thumb_path = out_dir / "view_thumbnails.png"
            try:
                render_pointcloud_thumbnails_panel(
                    self.points,
                    thumb_path,
                    title=f"候选视角（语义：{self.config.semantic}）",
                    point_size=self.config.point_size,
                )
            except (OSError, ImportError, ValueError) as exc:
                LOGGER.warning("缩略图渲染失败: %s", exc)
                thumb_path = None
            if thumb_path is None:
                bv = _fallback("缩略图渲染失败，退回默认视角")
            else:
                LOGGER.info("调用 VLM 选择最佳展示视角 ...")
                bv = select_best_view(
                    self.client, thumb_path, self.config.semantic,
                    model=self.config.view_select_model,
                )
                bv.thumbnails_panel = thumb_path

        # 用 chosen view 渲一张透明背景 overview PNG（报告镇页图）
        ov_path = out_dir / "best_view_overview.png"
        try:
            render_pointcloud_overview(
                self.points, ov_path,
                elev=bv.elev, azim=bv.azim,
                image_size=self.config.single_view_image_size,
                background="transparent",
                point_size=self.config.point_size,
            )
            bv.overview_png = ov_path
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("best view overview 渲染失败: %s", exc)
            bv.overview_png = None

        try:
            (out_dir / "best_view.json").write_text(
                json.dumps(
                    {
                        "index_0based": bv.index,
                        "index_1based": bv.index + 1,
                        "elev": bv.elev,
                        "azim": bv.azim,
                        "fallback": bv.fallback,
                        "confidence": bv.confidence,
                        "reason": bv.reason,
                        "provider": bv.provider,
                        "model": bv.model,
                        "elapsed_seconds": bv.elapsed_seconds,
                        "error": bv.error,
                        "thumbnails_panel": str(bv.thumbnails_panel) if bv.thumbnails_panel else None,
                        "overview_png": str(bv.overview_png) if bv.overview_png else None,
                    },
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.warning("写 best_view.json 失败: %s", exc)

        LOGGER.info(
            "最佳展示视角: #%d (elev=%.1f, azim=%.1f, fallback=%s) — %s",
            bv.index + 1, bv.elev, bv.azim, bv.fallback, (bv.reason or "")[:80],
        )
        return bv

    # ---------- 主循环 ----------
    def run(self, points: np.ndarray, output_dir: str | Path) -> PointCloudRefinementResult:
        t0 = time.perf_counter()
        self.points = np.asarray(points, dtype=np.float64)[:, :3]
        n_points = len(self.points)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_dir = out_dir / "panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        responses_dir = out_dir / "responses"
        if self.config.save_responses:
            responses_dir.mkdir(parents=True, exist_ok=True)

        result = PointCloudRefinementResult(
            ok=False, output_dir=out_dir, semantic=self.config.semantic
        )
        if n_points < self.config.min_block_size:
            result.message = f"点数过少 ({n_points})"
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        # ====== NEO 全局特征空间（可选，整朵点云一次性前馈）======
        # 成功后所有子块分裂走 lbo_spectral 切片复用，避免每块重算 scipy eigsh。
        self._neo_eigvecs = None
        if self.config.use_neo:
            try:
                self._compute_neo_eigenspace()
            except Exception as exc:  # noqa: BLE001 — 缺环境/权重/推理失败都回退
                LOGGER.warning(
                    "NEO 特征空间失败，回退到 %s（scipy 现算）: %s",
                    self.config.split_method, exc,
                )
                self._neo_eigvecs = None
            result.extra["neo_used"] = self._neo_eigvecs is not None

        # ====== 视角选择（最先做）：VLM 看 12 缩略图挑展示视角，失败自动 fallback ======
        try:
            self._best_view = self._select_view_for_display(out_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("视角选择失败，使用默认视角: %s", exc)
            self._best_view = None

        # ====== 没有初始分块：整朵点云作为根，force_split 直接切第一刀 ======
        root = SemanticBlock(
            block_id="C0",
            face_ids=list(range(n_points)),
            depth=0,
            parent_id=None,
            status=BlockStatus.PENDING,
            notes="root | initial_force_split",
            force_split=True,
        )
        self._register_block(root)

        responses_log: List[Dict[str, Any]] = []
        queue: List[str] = ["C0"]
        iteration = 0
        terminated = False

        # 初始 live_report：全 pending 底图
        try:
            self._update_live_report(
                out_dir, iteration=0, queue_after=queue, finished=False,
                extra_note="初始化中：整朵点云作为根，尚未开始分裂",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("初始 live report 失败: %s", exc)

        while queue:
            iteration += 1
            LOGGER.info("== iteration %d, queue=%d ==", iteration, len(queue))
            next_queue: List[str] = []

            # ----- Pass A（串行，CPU）：渲染 + force_split 立即几何分块（不等 VLM）-----
            iter_panel: Dict[str, Path] = {}
            force_terminate_ids: Set[str] = set()

            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue

                if block.size < self.config.min_block_size:
                    try:
                        panel_path = self._render_block_panel(block, panel_dir)
                        block.panel_path = str(panel_path)
                        iter_panel[block_id] = panel_path
                        force_terminate_ids.add(block_id)
                    except (OSError, ImportError, ValueError) as exc:
                        LOGGER.warning("block %s too_small 渲染失败: %s", block_id, exc)
                        block.status = BlockStatus.ABANDONED
                        block.notes = f"render_failed: {exc}"
                    continue

                at_max_depth = block.depth >= self.config.max_depth

                if block.force_split and not at_max_depth:
                    seed_n = self.config.root_seed_count
                    LOGGER.info("block %s force_split → 立即几何分块 seeds=%d", block_id, seed_n)
                    children = self._split_block(block, seed_n)
                    if len(children) >= 2:
                        block.status = BlockStatus.SPLIT
                        block.seed_count = seed_n
                        block.seed_decision = {"seed_count": seed_n, "reason": "root_force_split"}
                        next_queue.extend(c.block_id for c in children)
                    else:
                        block.force_split = False
                        next_queue.append(block_id)
                    continue

                try:
                    panel_path = self._render_block_panel(block, panel_dir)
                    block.panel_path = str(panel_path)
                    iter_panel[block_id] = panel_path
                except (OSError, ImportError, ValueError) as exc:
                    LOGGER.error("渲染 block %s 失败: %s", block_id, exc)
                    block.status = BlockStatus.ABANDONED
                    block.notes = f"render_failed: {exc}"

            # ----- Pass B（并行 VLM）：批量三分类（含 too_small 块）-----
            classify_tasks = list(iter_panel.items())
            cls_cache = self._batch_classify(
                classify_tasks,
                force_terminate_ids=force_terminate_ids,
            )

            # ----- Pass C（串行决策）：读 cache，partial 再批量 ask_seed_count + 几何分块 -----
            partial_seed_tasks: List[Tuple[str, Path, Dict[str, Any]]] = []
            partial_block_ids: List[str] = []

            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue

                if block_id in force_terminate_ids:
                    panel_path = iter_panel.get(block_id)
                    cls = cls_cache.get(block_id)
                    if cls is None and panel_path is not None:
                        try:
                            cls = self._classify(
                                block, panel_path, force_terminate=True,
                            )
                        except (RuntimeError, ValueError, OSError) as exc:
                            LOGGER.warning("too_small classify fallback failed %s: %s", block_id, exc)
                            cls = {"label": LABEL_NONE, "reason": str(exc)}
                    if cls is not None:
                        self._apply_force_terminate(
                            block, cls, responses_dir, responses_log,
                            iteration=iteration, reason_tag="too_small",
                        )
                    continue

                if block.status != BlockStatus.PENDING:
                    continue

                panel_path = iter_panel.get(block_id)
                if panel_path is None:
                    continue

                at_max_depth = block.depth >= self.config.max_depth
                cls = cls_cache.get(block_id)
                if cls is None:
                    try:
                        cls = self._classify(block, panel_path)
                    except (RuntimeError, ValueError, OSError) as exc:
                        LOGGER.error("classify fallback failed %s: %s", block_id, exc)
                        block.status = BlockStatus.ABANDONED
                        block.notes = f"classify_failed: {exc}"
                        continue

                block.classification = cls
                label = cls.get("label") or LABEL_NONE
                responses_log.append({"block_id": block_id, "iteration": iteration, **cls})
                self._dump_response(responses_dir, block_id, cls)
                LOGGER.info("block %s classify → %s", block_id, label)

                if label == LABEL_FULL:
                    block.status = BlockStatus.FULL
                    continue
                if label == LABEL_NONE:
                    block.status = BlockStatus.NONE
                    continue

                if at_max_depth:
                    if (
                        self.config.accept_partial_at_max_depth
                        and (cls.get("committed_if_forced") or "none") == LABEL_FULL
                        and block.size >= self.config.accept_partial_min_points
                    ):
                        block.status = BlockStatus.FULL
                        block.notes = (block.notes + " | accepted_partial_at_max_depth").strip(" |")
                    else:
                        self._apply_force_terminate(
                            block, cls, responses_dir, responses_log,
                            iteration=iteration, reason_tag="partial_at_max_depth",
                        )
                    continue

                partial_seed_tasks.append((block_id, panel_path, cls))
                partial_block_ids.append(block_id)

            seed_cache = self._batch_ask_seed_count(partial_seed_tasks)

            for block_id in partial_block_ids:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                cls = block.classification or {}
                panel_path = iter_panel[block_id]
                seed_resp = seed_cache.get(block_id)
                if seed_resp is None:
                    try:
                        seed_resp = ask_seed_count(
                            self.client, panel_path, self.config.semantic,
                            max_seeds=self.config.max_seeds, fallback=2,
                            model=self.config.seed_model,
                            covered_parts=cls.get("covered_parts"),
                            semantic_parts=cls.get("semantic_parts"),
                            extra_parts=cls.get("extra_parts"),
                            block_faces=block.size,
                        )
                    except Exception as exc:  # noqa: BLE001
                        from geometry.seed_split import SeedCountResponse
                        seed_resp = SeedCountResponse(seed_count=2, reason=str(exc))
                seed_n = seed_resp.seed_count
                seed_reason = seed_resp.reason

                children = self._split_block(block, seed_n)
                if len(children) >= 2:
                    block.status = BlockStatus.SPLIT
                    block.seed_count = seed_n
                    block.seed_decision = {"seed_count": seed_n, "reason": seed_reason}
                    next_queue.extend(c.block_id for c in children)
                    LOGGER.info("block %s split → %d children", block_id, len(children))
                else:
                    self._apply_force_terminate(
                        block, cls, responses_dir, responses_log,
                        iteration=iteration, reason_tag="split_failed",
                    )

            # 每轮末尾：进度图 + live_report
            try:
                self._save_iteration_progress(out_dir, iteration=iteration)
                self._update_live_report(
                    out_dir, iteration=iteration, queue_after=next_queue, finished=False,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("iter %d 进度刷新失败: %s", iteration, exc)

            # 第一刀切完后导出 initial_segmentation.png（按当前叶子块各一色）
            if iteration == 1:
                self._render_initial_segmentation(out_dir)

            queue = next_queue

        # ====== 收尾 ======
        full_ids: List[int] = []
        none_ids: List[int] = []
        pending_ids: List[int] = []
        for block in self.blocks.values():
            if block.status == BlockStatus.SPLIT:
                continue
            if block.status == BlockStatus.FULL:
                full_ids.extend(block.face_ids)
            elif block.status in (BlockStatus.NONE, BlockStatus.ABANDONED):
                none_ids.extend(block.face_ids)
            else:
                pending_ids.extend(block.face_ids)

        result.full_point_ids = sorted(set(full_ids))
        result.none_point_ids = sorted(set(none_ids))
        result.pending_point_ids = sorted(set(pending_ids))
        result.num_iterations = iteration
        result.num_final_blocks = sum(
            1 for b in self.blocks.values() if b.status != BlockStatus.SPLIT
        )

        # 最终掩码图
        mask_png = out_dir / "semantic_mask.png"
        try:
            render_pointcloud_highlight_panel(
                self.points,
                result.full_point_ids,
                mask_png,
                title=f"语义掩码「{self.config.semantic}」full={len(result.full_point_ids)}",
                image_size=(self.config.image_size, self.config.image_size),
                cols=self.config.panel_cols,
                point_size=self.config.point_size,
            )
            result.semantic_panel = mask_png
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("最终掩码渲染失败: %s", exc)

        # block_tree.json（schema 与 mesh 版一致，可喂 report.generate_report）
        tree_json = out_dir / "block_tree.json"
        export_tree_json(
            self.blocks,
            tree_json,
            extra={
                "semantic": self.config.semantic,
                "num_iterations": iteration,
                "num_initial_blocks": 1,
                "num_final_blocks": result.num_final_blocks,
                "max_depth": self.config.max_depth,
                "terminated_by_vlm": terminated,
                "unit": "point",
                "num_points": n_points,
            },
        )
        result.tree_json = tree_json

        # semantic_mask.json
        mask_json = out_dir / "semantic_mask.json"
        mask_json.write_text(
            json.dumps(
                {
                    "semantic": self.config.semantic,
                    "unit": "point",
                    "num_points": n_points,
                    "full_point_ids": result.full_point_ids,
                    "none_point_ids": result.none_point_ids,
                    "pending_point_ids": result.pending_point_ids,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result.mask_json = mask_json

        (out_dir / "responses_log.json").write_text(
            json.dumps(responses_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 最后一次 live_report：标记完成（去掉 meta refresh）
        try:
            self._update_live_report(
                out_dir, iteration=iteration, queue_after=[], finished=True,
                extra_note=f"已完成：full={len(result.full_point_ids)} 点 · "
                           f"详细 block 卡片见 report.html。",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("收尾 live report 失败: %s", exc)

        # 最终 HTML 报告（复用 mesh 版 report.generate_report）
        try:
            from pipeline.report import generate_report

            result.report_html = generate_report(out_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("report.html 生成失败: %s", exc)

        # 视角选择产物落到 result
        if self._best_view is not None:
            result.best_view = self._best_view
            result.best_view_overview = self._best_view.overview_png
            result.best_view_thumbnails = self._best_view.thumbnails_panel
            bv_json = out_dir / "best_view.json"
            if bv_json.is_file():
                result.best_view_json = bv_json

        result.ok = True
        result.elapsed_seconds = time.perf_counter() - t0
        result.message = (
            f"完成：{iteration} 轮迭代，full={len(result.full_point_ids)} 点 / "
            f"{result.num_final_blocks} 叶块"
        )
        return result

    def _render_initial_segmentation(self, out_dir: Path) -> None:
        """第一刀切完后，把当前叶子块按块号着色，导出 initial_segmentation.png。"""
        n_points = len(self.points)
        labels = np.full(n_points, -1, dtype=np.int64)
        leaf_idx = 0
        for block in self.blocks.values():
            if block.status == BlockStatus.SPLIT or block.parent_id is None:
                continue
            for pid in block.face_ids:
                if 0 <= pid < n_points:
                    labels[pid] = leaf_idx
            leaf_idx += 1
        if leaf_idx == 0:
            return
        try:
            render_pointcloud_segmentation_panel(
                self.points,
                labels.tolist(),
                out_dir / "initial_segmentation.png",
                title=f"初始分块（第一刀）blocks={leaf_idx}",
                image_size=(self.config.image_size, self.config.image_size),
                cols=self.config.panel_cols,
                point_size=self.config.point_size,
            )
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("初始分块图渲染失败: %s", exc)
