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
from geometry.pointcloud_partition import (
    VALID_POINT_STRATEGIES,
    ask_point_split_strategy,
    default_point_split_params,
    split_points_by_strategy,
)
from pipeline.geometry_postprocess import (
    analyze_selected_components,
    apply_ops as _postprocess_apply_ops,
    ask_component_split_decision,
    ask_postprocess_ops,
    build_neighbor_index,
    compute_selection_metrics,
    op_keep_components,
)
from geometry.seed_split import SplitStrategyResponse, ask_seed_count
from rendering.heatmap_renderer import DEFAULT_OVERVIEW_VIEW, VIEW_GRID
from rendering.pointcloud_render import (
    STATUS_COLOR_MAP,
    component_color_legend,
    compute_point_normals,
    render_pointcloud_components_panel,
    render_pointcloud_highlight_panel,
    render_pointcloud_overview,
    render_pointcloud_segmentation_panel,
    render_pointcloud_status_view,
    render_pointcloud_thumbnails_panel,
)
from vlm.classifier import ask_revive_pick, classify_component
from vlm.client import VLMClient
from vlm.prompts import LABEL_FULL, LABEL_NONE, LABEL_PARTIAL
from vlm.view_selector import (
    BestViewSelection,
    select_best_view,
    select_diverse_view_indices,
)

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
    # ---- VLM 选分块策略（与 mesh pipeline 对齐）----
    # 每个 partial 块分裂前，让 VLM 看高亮图，从 6 种点云策略（graph_cut / spectral /
    # cvt / sdf / skeleton / geodesic_fps）里挑一种 + 给参数（曲率指标 curvature_weight、
    # 凸包/凹凸指标 concavity_weight 等）。NEO 仅在所选策略为 spectral 时介入加速。
    ask_split_strategy: bool = True
    # 策略提示 / 强制：auto=每块由 VLM 决定；指定 6 种之一则强制该策略（不调 VLM 选策略）。
    split_strategy: str = "auto"
    split_strategy_model: Optional[str] = None
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
    # 分类视角子集：选完最佳展示视角后，让 VLM 同时决定后续「逐块三分类高亮拼图」用几张视角。
    # 视角越少越精，能看清语义即可——视角太多会让 VLM 在拼图里反复找红色、产生幻觉、虚报部位。
    # select_classify_views=True 时由 VLM 决定数量与具体视角（在 select_best_view 那一步一起出）；
    # classify_view_count>0 时强制固定张数（以 best view 为中心按角度互补挑），不问 VLM；
    # 关闭 select_best_view 或 VLM 选择失败时，退回 default_classify_view_count 张（角度互补）。
    select_classify_views: bool = True
    classify_view_count: int = 0          # >0 强制固定张数；0=自动（VLM 决定 / 兜底默认）
    default_classify_view_count: int = 4  # 兜底默认张数
    min_classify_views: int = 2           # VLM 可选的最少张数
    max_classify_views: int = 6           # VLM 可选的最多张数（上限 = 全部 12 视角）
    # VLM 三分类 / ask_seed_count 并发数。>1 时一批块并行调 VLM（HTTP IO），
    # 等待期间主线程继续处理 force_split 等几何分块（见 run() 三轮 pass）。
    parallel_vlm_workers: int = 4
    # ---- 几何后处理（最终掩码清洗）----
    # 主循环结束后，对最终 full 选区算几何指标（凸包/solidity、边缘平滑度、曲率、碎片、
    # 空洞、离群），喂给 VLM 让它挑形态学清洗算子（去小岛/去离群/平滑边界/填洞/保留最大块），
    # 在 kNN 图上执行后得到更干净的最终分块。失败安全回退原始 mask。
    postprocess: bool = True
    postprocess_max_rounds: int = 2       # 体检→VLM→清洗 最多迭代几轮（VLM 判干净即停）
    postprocess_model: Optional[str] = None
    # 选区分裂成「两个都很大」的连通域时，结合语义+分色渲染请 VLM 判断该保留哪些块、
    # 丢弃溢出到别的部件的那块。一个连通域占比 ≥ 该阈值才算「大」（避免对小岛触发）。
    postprocess_large_component_fraction: float = 0.15
    # ---- VLM 跨迭代记忆 ----
    # 每轮三分类前，把此前各块已落定的判定（full / none / partial→split）汇总成文字摘要，
    # 注入到 reason 阶段的 prompt（不进 observe 阶段，避免确认偏差），让 VLM 保持跨迭代记忆、
    # 判定彼此一致（不重复认领、不与已定父块矛盾）。
    vlm_memory: bool = True
    vlm_memory_max_entries: int = 40         # 摘要里最多保留多少条最近判定，控制 token
    # ---- 兜底复活 ----
    # 整轮跑完队列清空后，若全模型 0 个 full 块（语义完全没找到），强制让 VLM 从所有候选叶块里
    # 挑最可能是该语义的一个，复活成 full，避免返回空掩码。
    revive_when_empty: bool = True
    revive_max_candidates: int = 8           # 喂给 VLM 选择的候选叶块上限（含高亮图）


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
        # VLM（或兜底）选定的「分类视角子集」(elev, azim)，逐块高亮拼图只渲染这些视角。
        # None / 空 = 用全部 VIEW_GRID（12 视角，旧行为）。
        self._classify_views: Optional[List[Tuple[float, float]]] = None
        # NEO 全局特征空间（整朵点云一次性前馈得到，子块分裂复用）。None=未用 NEO。
        self._neo_eigvecs: Optional[np.ndarray] = None
        # 整朵点云每点 PCA 法向 (N, 3)，一次性算好供所有渲染做逐点法向 shader。
        self._point_normals: Optional[np.ndarray] = None

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

    # ---------- 分类视角子集 ----------
    def _n_classify_views(self) -> int:
        """当前逐块分类拼图实际用的视角数（喂给 prompt，让它如实知道有几张图）。"""
        if self._classify_views:
            return len(self._classify_views)
        return len(VIEW_GRID)

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
            views=self._classify_views,
            normals=self._point_normals,
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
    def _default_strategy(self) -> str:
        """无 VLM 选策略时的默认 / 强制策略。"""
        s = str(self.config.split_strategy or "auto").strip().lower()
        if s in VALID_POINT_STRATEGIES:
            return s
        # auto 但未调 VLM，或 legacy split_method：feature_spectral/lbo_spectral → spectral
        method = str(self.config.split_method or "").strip().lower()
        if method == "geodesic_fps":
            return "geodesic_fps"
        return "spectral"

    def _resolve_strat_resp(
        self, strat_resp: Optional[SplitStrategyResponse]
    ) -> SplitStrategyResponse:
        if strat_resp is not None and strat_resp.strategy:
            return strat_resp
        strat = self._default_strategy()
        return SplitStrategyResponse(
            strategy=strat,
            reason="默认策略（未调用 VLM 选策略）",
            params=default_point_split_params(strat),
        )

    def _split_block(
        self,
        block: SemanticBlock,
        seed_count: int,
        strat_resp: Optional[SplitStrategyResponse] = None,
    ) -> List[SemanticBlock]:
        # 由 VLM（或默认）选定的策略 + 参数做几何分裂；NEO 仅在 spectral 时介入加速。
        resolved = self._resolve_strat_resp(strat_resp)
        strategy = resolved.strategy
        params = resolved.params or {}
        precomputed = self._neo_eigvecs if strategy == "spectral" else None

        sub = split_points_by_strategy(
            self.points,
            block.face_ids,
            seed_count,
            strategy,
            lambda_smooth=float(params.get("lambda_smooth", 1.0)),
            lambda_concave=float(params.get("lambda_concave", 0.85)),
            scalar_weight=float(params.get("scalar_weight", 0.35)),
            curvature_weight=params.get("curvature_weight", self.config.split_curvature_weight),
            concavity_weight=params.get("concavity_weight", self.config.split_convexity_weight),
            normal_weight=float(params.get("normal_weight", self.config.split_normal_weight)),
            lloyd_iter=int(params.get("lloyd_iter", self.config.split_lloyd_iter)),
            num_rays=int(params.get("num_rays", 16)),
            cone_angle_deg=float(params.get("cone_angle_deg", 60.0)),
            contraction_iters=int(params.get("contraction_iters", 5)),
            n_neighbors=self.config.split_n_neighbors,
            enforce_connectivity=self.config.split_enforce_connectivity,
            precomputed_eigvecs=precomputed,
            random_state=0,
        )
        neo_tag = "+neo" if precomputed is not None else ""
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
                notes=f"split({strategy}{neo_tag}:{sub.debug.get('strategy','')})",
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
                normals=self._point_normals,
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
                normals=self._point_normals,
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
            cv_html = ""
            if self._classify_views:
                cv_nums = "、".join(
                    f"#{i + 1}" for i in (self._best_view.classify_view_indices or [])
                ) or f"{len(self._classify_views)} 张"
                cv_html = (
                    f'<div class="view-meta"><b>分类视角子集</b> '
                    f'{esc(cv_nums)}（共 {len(self._classify_views)} 张，少而精降幻觉）</div>'
                )
            view_card_html = (
                '<section class="view-card">'
                f'<div class="view-meta"><b>展示视角</b> {tag}</div>'
                f'{cv_html}'
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

    # ---------- 跨迭代记忆 ----------
    def _build_memory_digest(self) -> str:
        """把此前各块已落定的判定汇总成文字摘要，注入下一轮 reason 阶段 prompt。

        只收集非 pending 块（full / none / abandoned / split），每条形如：
          - block C0.1 (pts=820, depth=2): full (counted as target); covered=[head, ears]
        最多保留最近 vlm_memory_max_entries 条，控制 token。
        """
        if not self.config.vlm_memory:
            return ""
        entries: List[str] = []
        for b in self.blocks.values():
            if b.status == BlockStatus.PENDING:
                continue
            cls = b.classification or {}
            if b.status == BlockStatus.SPLIT:
                verdict = f"partial -> split into {len(b.children_ids)} children"
            elif b.status == BlockStatus.FULL:
                verdict = "full (counted as target)"
            elif b.status == BlockStatus.ABANDONED:
                verdict = "none (abandoned)"
            else:
                verdict = "none"
            covered = cls.get("covered_parts") or []
            cov = ", ".join(str(c) for c in covered[:4])
            reason = str(cls.get("reason") or "").strip()
            if len(reason) > 140:
                reason = reason[:140] + "..."
            line = f"- block {b.block_id} (pts={b.size}, depth={b.depth}): {verdict}; covered=[{cov}]"
            if reason:
                line += f"; reason={reason}"
            entries.append(line)
        if not entries:
            return ""
        max_n = max(1, int(self.config.vlm_memory_max_entries))
        if len(entries) > max_n:
            entries = entries[-max_n:]
        return "\n".join(entries)

    # ---------- 兜底复活 ----------
    def _revive_block(
        self,
        responses_dir: Path,
        responses_log: List[Dict[str, Any]],
        iteration: int,
    ) -> Optional[str]:
        """全模型 0 个 full 时，让 VLM 从候选叶块里挑最可能的一个，复活成 full。

        候选 = 非 split / 非 full 的叶块（none / abandoned）。按已存判定启发式排序后取前
        revive_max_candidates 个喂给 VLM（带各自高亮图，图标题印有 block id）。VLM 选择失败
        时回退到启发式最优块。返回被复活的 block_id（无候选则 None）。
        """
        leaves = [
            b for b in self.blocks.values()
            if b.status in (BlockStatus.NONE, BlockStatus.ABANDONED)
            and b.block_id != "C0"
        ]
        if not leaves:
            LOGGER.warning("强制复活：无候选叶块（none/abandoned）")
            return None

        sized = [b for b in leaves if b.size >= self.config.min_block_size]
        candidates = sized or leaves

        def _score(b: SemanticBlock) -> Tuple[float, float, float, int]:
            cls = b.classification or {}
            committed = 1.0 if (cls.get("committed_if_forced") == LABEL_FULL) else 0.0
            ratio = float(cls.get("semantic_area_ratio") or 0.0)
            conf = float(cls.get("confidence") or 0.0)
            return (committed, ratio, conf, int(b.size))

        candidates.sort(key=_score, reverse=True)
        top = candidates[: max(1, int(self.config.revive_max_candidates))]

        payload: List[Dict[str, Any]] = []
        images: List[Path] = []
        for b in top:
            cls = b.classification or {}
            payload.append({
                "block_id": b.block_id,
                "points": b.size,
                "covered_parts": cls.get("covered_parts") or [],
                "dominant_region": cls.get("dominant_region") or "",
                "extra_parts": cls.get("extra_parts") or [],
                "semantic_area_ratio": cls.get("semantic_area_ratio"),
                "prior_reason": cls.get("reason") or "",
            })
            if b.panel_path and Path(b.panel_path).is_file():
                images.append(Path(b.panel_path))

        chosen_id: Optional[str] = None
        try:
            chosen_id = ask_revive_pick(
                self.client,
                self.config.semantic,
                payload,
                image_paths=images,
                model=self.config.classification_model,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ask_revive_pick 异常，回退启发式: %s", exc)
            chosen_id = None

        block: Optional[SemanticBlock] = None
        if chosen_id and chosen_id in self.blocks:
            cand = self.blocks[chosen_id]
            if cand.status in (BlockStatus.NONE, BlockStatus.ABANDONED):
                block = cand
        if block is None:
            block = top[0]
            chosen_id = block.block_id
            LOGGER.info("强制复活：VLM 未给出有效选择，回退启发式最优块 %s", chosen_id)

        prev = block.classification or {}
        cls = {
            **prev,
            "label": LABEL_FULL,
            "revived": True,
            "original_label": prev.get("label"),
        }
        block.classification = cls
        block.status = BlockStatus.FULL
        block.notes = (block.notes + " | revived_no_full").strip(" |")
        responses_log.append({
            "block_id": block.block_id, "iteration": iteration, "revived": True, **cls,
        })
        self._dump_response(responses_dir, block.block_id, cls)
        LOGGER.info(
            "强制复活块 %s 为 full（全模型原本 0 个 full，%d 候选叶块）",
            chosen_id, len(candidates),
        )
        return chosen_id

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
        memory: str = "",
    ) -> Dict[str, Any]:
        cls = classify_component(
            self.client,
            panel_path,
            self.config.semantic,
            model=self.config.classification_model,
            media_type="pointcloud",
            force_terminate=force_terminate,
            n_views=self._n_classify_views(),
            memory=memory,
        )
        return self._cls_dict_from_component(cls)

    def _batch_classify(
        self,
        tasks: Sequence[Tuple[str, Path]],
        *,
        force_terminate_ids: Optional[Set[str]] = None,
        memory: str = "",
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
                memory=memory,
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
                    n_views=self._n_classify_views(),
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

    def _batch_ask_split_strategy(
        self,
        tasks: Sequence[Tuple[str, Path, Dict[str, Any], int]],
    ) -> Dict[str, SplitStrategyResponse]:
        """并行批量让 VLM 选分块策略 + 参数（partial 块）。

        非 auto（强制策略）或关闭 ask_split_strategy 时，直接返回默认策略响应，不调 VLM。
        """
        results: Dict[str, SplitStrategyResponse] = {}
        if not tasks:
            return results

        forced = str(self.config.split_strategy or "auto").strip().lower()
        if not self.config.ask_split_strategy or forced in VALID_POINT_STRATEGIES:
            strat = self._default_strategy()
            for bid, _panel, _cls, _seed in tasks:
                results[bid] = SplitStrategyResponse(
                    strategy=strat,
                    reason=(
                        "CLI 强制策略" if forced in VALID_POINT_STRATEGIES
                        else "未启用 VLM 选策略，用默认"
                    ),
                    params=default_point_split_params(strat),
                )
            return results

        def _do(bid: str, panel: Path, cls: Dict[str, Any], seed_n: int) -> Tuple[str, SplitStrategyResponse]:
            block = self.blocks[bid]
            try:
                resp = ask_point_split_strategy(
                    self.client,
                    panel,
                    semantic_context=self.config.semantic,
                    seed_count=seed_n,
                    covered_parts=cls.get("covered_parts"),
                    semantic_parts=cls.get("semantic_parts"),
                    extra_parts=cls.get("extra_parts"),
                    matched_parts=cls.get("semantic_parts"),
                    block_points=block.size,
                    split_stage="main",
                    strategy_hint=str(self.config.split_strategy or "auto"),
                    fallback="spectral",
                    model=self.config.split_strategy_model or self.config.seed_model,
                )
            except Exception as exc:  # noqa: BLE001
                resp = SplitStrategyResponse(strategy="spectral", error=str(exc))
            return bid, resp

        workers = max(1, int(self.config.parallel_vlm_workers))
        if workers > 1 and len(tasks) > 1:
            n = min(workers, len(tasks))
            LOGGER.info("parallel ask_split_strategy: %d blocks, %d workers", len(tasks), n)
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = {
                    ex.submit(_do, bid, panel, cls, seed_n): bid
                    for bid, panel, cls, seed_n in tasks
                }
                for fut in as_completed(futs):
                    bid = futs[fut]
                    try:
                        _, resp = fut.result()
                        results[bid] = resp
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("parallel ask_split_strategy failed %s: %s", bid, exc)
        else:
            for bid, panel, cls, seed_n in tasks:
                try:
                    _, resp = _do(bid, panel, cls, seed_n)
                    results[bid] = resp
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("serial ask_split_strategy failed %s: %s", bid, exc)
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
                    normals=self._point_normals,
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
                    min_classify_views=self.config.min_classify_views,
                    max_classify_views=self.config.max_classify_views,
                    default_classify_views=(
                        self.config.classify_view_count
                        if self.config.classify_view_count > 0
                        else self.config.default_classify_view_count
                    ),
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
                normals=self._point_normals,
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
                        "classify_view_indices_1based": [
                            i + 1 for i in (bv.classify_view_indices or [])
                        ],
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

    def _resolve_classify_views(self) -> None:
        """根据 best_view + 配置，确定逐块分类拼图实际使用的视角子集。

        优先级：
          1. classify_view_count>0 → 以 best view 为中心按角度互补强制固定张数；
          2. select_classify_views 且 VLM 给了 classify_view_indices → 用 VLM 的选择；
          3. 否则（关闭或缺失）→ 用 default_classify_view_count 张角度互补视角。
        关闭分类视角裁剪（select_classify_views=False 且 classify_view_count<=0）→ None（全 12 视角）。
        """
        n_grid = len(VIEW_GRID)
        bv = self._best_view
        best_idx = int(bv.index) if bv is not None else 0

        forced = int(self.config.classify_view_count or 0)
        if forced <= 0 and not self.config.select_classify_views:
            self._classify_views = None
            LOGGER.info("分类视角裁剪关闭：逐块仍用全部 %d 视角", n_grid)
            return

        if forced > 0:
            idxs = select_diverse_view_indices(best_idx, min(forced, n_grid))
        elif bv is not None and bv.classify_view_indices:
            idxs = [i for i in bv.classify_view_indices if 0 <= i < n_grid]
            if not idxs:
                idxs = select_diverse_view_indices(
                    best_idx, self.config.default_classify_view_count
                )
        else:
            idxs = select_diverse_view_indices(
                best_idx, self.config.default_classify_view_count
            )

        self._classify_views = [VIEW_GRID[i] for i in idxs]
        if bv is not None:
            bv.classify_view_indices = idxs
        LOGGER.info(
            "逐块分类视角子集: %s（共 %d 张，全网格 %d；视角越少越降幻觉）",
            [i + 1 for i in idxs], len(idxs), n_grid,
        )

    # ---------- 主循环 ----------
    def run(self, points: np.ndarray, output_dir: str | Path) -> PointCloudRefinementResult:
        t0 = time.perf_counter()
        self.points = np.asarray(points, dtype=np.float64)[:, :3]
        n_points = len(self.points)

        # 逐点 PCA 法向：一次性算好，所有渲染（高亮拼图/状态图/概览）用它对 base color
        # 做明暗 shader（扁平点，不画球体）。算不出时退化为无 shader。
        try:
            self._point_normals = compute_point_normals(self.points)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("逐点法向计算失败，渲染退回无 shader: %s", exc)
            self._point_normals = None

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
        try:
            self._resolve_classify_views()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("分类视角子集解析失败，退回全部视角: %s", exc)
            self._classify_views = None

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
            # 跨迭代记忆：把此前各块已落定的判定汇总成文字摘要，注入 reason 阶段 prompt。
            memory = self._build_memory_digest()
            classify_tasks = list(iter_panel.items())
            cls_cache = self._batch_classify(
                classify_tasks,
                force_terminate_ids=force_terminate_ids,
                memory=memory,
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
                                block, panel_path, force_terminate=True, memory=memory,
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
                        cls = self._classify(block, panel_path, memory=memory)
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

            # 先确定每个 partial 块的 seed_count，再批量让 VLM 选分块策略 + 参数。
            strat_tasks: List[Tuple[str, Path, Dict[str, Any], int]] = []
            seed_resolved: Dict[str, Any] = {}
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
                            block_points=block.size,
                            media_type="pointcloud",
                            n_views=self._n_classify_views(),
                        )
                    except Exception as exc:  # noqa: BLE001
                        from geometry.seed_split import SeedCountResponse
                        seed_resp = SeedCountResponse(seed_count=2, reason=str(exc))
                seed_resolved[block_id] = seed_resp
                strat_tasks.append((block_id, panel_path, cls, seed_resp.seed_count))

            strat_cache = self._batch_ask_split_strategy(strat_tasks)

            for block_id in partial_block_ids:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                cls = block.classification or {}
                seed_resp = seed_resolved.get(block_id)
                if seed_resp is None:
                    continue
                seed_n = seed_resp.seed_count
                seed_reason = seed_resp.reason
                strat_resp = strat_cache.get(block_id) or self._resolve_strat_resp(None)

                children = self._split_block(block, seed_n, strat_resp)
                if len(children) >= 2:
                    block.status = BlockStatus.SPLIT
                    block.seed_count = seed_n
                    block.seed_decision = {
                        "seed_count": seed_n,
                        "reason": seed_reason,
                        "strategy": strat_resp.strategy,
                        "strategy_reason": strat_resp.reason,
                        "strategy_params": strat_resp.params,
                        "strategy_error": strat_resp.error,
                    }
                    next_queue.extend(c.block_id for c in children)
                    LOGGER.info(
                        "block %s split → %d children (strategy=%s)",
                        block_id, len(children), strat_resp.strategy,
                    )
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

        # ====== 兜底复活：整轮跑完后若全模型 0 个 full 块，强制让 VLM 挑一个最可能的复活 ======
        if self.config.revive_when_empty and not any(
            b.status == BlockStatus.FULL for b in self.blocks.values()
        ):
            LOGGER.info("全模型 0 个 full 块，触发强制复活")
            try:
                self._revive_block(responses_dir, responses_log, iteration)
            except Exception as exc:  # noqa: BLE001 — 复活失败不影响收尾
                LOGGER.warning("强制复活块失败: %s", exc)

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

        # ====== 几何后处理：对最终 full 选区算指标 + VLM 清洗（凸包/边缘/曲率）======
        if self.config.postprocess:
            try:
                self._postprocess_final_mask(out_dir, result)
            except Exception as exc:  # noqa: BLE001 — 后处理失败不影响主结果
                LOGGER.warning("几何后处理失败，沿用原始最终掩码: %s", exc)

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
                normals=self._point_normals,
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

    # ---------- 几何后处理 ----------
    def _split_large_components(
        self,
        pp_dir: Path,
        mask: np.ndarray,
        nbr_idx: np.ndarray,
        metrics_fn,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """选区分裂成「两个都很大」的连通域时，请 VLM 判断哪些块属于该语义。

        - 仅当存在 ≥2 个占比 ≥ ``postprocess_large_component_fraction`` 的连通域时触发；
        - 把各大块分色渲染成多视角拼图 + 各块占比 + 语义喂给 VLM；
        - VLM 返回应保留的块号，丢弃其余（溢出到别的部件的）块。
        返回 (新 mask, 决策记录 dict 或 None)。任何失败/判定全保留 → 原样返回。
        """
        n_neighbors = self.config.split_n_neighbors
        comps = analyze_selected_components(self.points, mask, n_neighbors)
        large_thresh = float(self.config.postprocess_large_component_fraction)
        large = [c for c in comps if c.fraction >= large_thresh]
        if len(large) < 2:
            return mask, None

        comp_render = [c.indices.tolist() for c in large]
        panel_path = pp_dir / "components.png"
        try:
            render_pointcloud_components_panel(
                self.points,
                comp_render,
                panel_path,
                title=f"[postprocess] large components: {component_color_legend(len(large))}",
                image_size=(self.config.image_size, self.config.image_size),
                cols=self.config.panel_cols,
                point_size=self.config.point_size,
                views=self._classify_views,
                normals=self._point_normals,
            )
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("postprocess 连通域分色渲染失败: %s", exc)
            panel_path = None  # type: ignore[assignment]

        decision = ask_component_split_decision(
            self.client,
            panel_path,
            self.config.semantic,
            large,
            metrics_fn(mask),
            media_type="pointcloud",
            n_views=self._n_classify_views(),
            model=self.config.postprocess_model or self.config.classification_model,
        )
        rec: Dict[str, Any] = {
            "num_large_components": len(large),
            "components": [
                {"id": c.rank, "size": c.size, "fraction": round(c.fraction, 4)} for c in large
            ],
            "keep_ranks": decision.keep_ranks,
            "split": decision.split,
            "reason": decision.reason,
            "error": decision.error,
        }
        if not decision.split:
            LOGGER.info("postprocess 连通域判定：全部保留（%d 个大块）", len(large))
            return mask, rec

        before = int(mask.sum())
        new_mask = op_keep_components(self.points, mask, n_neighbors, decision.keep_ranks)
        after = int(new_mask.sum())
        rec["before"] = before
        rec["after"] = after
        rec["delta"] = after - before
        if after == 0 or np.array_equal(new_mask, mask):
            LOGGER.info("postprocess 连通域拆分无变化/会清空，保留原 mask")
            return mask, rec
        LOGGER.info(
            "postprocess 连通域拆分：保留块 %s，%d -> %d (%+d)",
            decision.keep_ranks, before, after, after - before,
        )
        return new_mask, rec

    def _postprocess_final_mask(
        self, out_dir: Path, result: PointCloudRefinementResult
    ) -> None:
        """对最终 full 选区做「几何指标体检 + VLM 决策 + 形态学清洗」。

        指标（凸包/solidity、边缘平滑度、曲率、碎片、空洞、离群）由 VLM 据此挑清洗算子；
        在整朵点云的 kNN 图上执行。把优化后的 mask 写回 result 并落产物到 postprocess/。
        """
        n_points = len(self.points)
        original_full = sorted(set(result.full_point_ids))
        if n_points == 0 or len(original_full) < 8:
            return

        pp_dir = out_dir / "postprocess"
        pp_dir.mkdir(parents=True, exist_ok=True)

        from geometry.pointcloud_cluster import compute_point_normals_and_curvature

        nbr_idx = build_neighbor_index(self.points, n_neighbors=self.config.split_n_neighbors)
        _normals, curvature = compute_point_normals_and_curvature(
            self.points, n_neighbors=self.config.split_n_neighbors
        )
        mask = np.zeros(n_points, dtype=bool)
        mask[np.asarray(original_full, dtype=np.int64)] = True

        def _metrics(m: np.ndarray) -> Dict[str, Any]:
            return compute_selection_metrics(
                self.points, m,
                n_neighbors=self.config.split_n_neighbors,
                nbr_idx=nbr_idx, curvature=curvature,
            ).to_dict()

        rounds: List[Dict[str, Any]] = []
        max_rounds = max(1, int(self.config.postprocess_max_rounds))
        metrics_before = _metrics(mask)

        # 先处理「两个都很大的连通域」：分色渲染 + 语义 → VLM 判断是否拆分（丢弃溢出块），
        # 在形态学清洗（会无脑保留最大块）之前做，避免误删另一块合法部件。
        mask, split_rec = self._split_large_components(pp_dir, mask, nbr_idx, _metrics)

        for r in range(1, max_rounds + 1):
            metrics = _metrics(mask)

            panel_path = pp_dir / f"round_{r:02d}.png"
            try:
                render_pointcloud_highlight_panel(
                    self.points,
                    np.flatnonzero(mask).tolist(),
                    panel_path,
                    title=f"[postprocess r{r}] full={int(mask.sum())}",
                    image_size=(self.config.image_size, self.config.image_size),
                    cols=self.config.panel_cols,
                    point_size=self.config.point_size,
                    views=self._classify_views,
                    normals=self._point_normals,
                )
            except (OSError, ImportError, ValueError) as exc:
                LOGGER.warning("postprocess 渲染失败: %s", exc)
                panel_path = None  # type: ignore[assignment]

            decision = ask_postprocess_ops(
                self.client,
                panel_path,
                self.config.semantic,
                metrics,
                media_type="pointcloud",
                n_views=self._n_classify_views(),
                model=self.config.postprocess_model or self.config.classification_model,
            )
            round_rec: Dict[str, Any] = {
                "round": r,
                "metrics": metrics,
                "ops_requested": decision.ops,
                "reason": decision.reason,
                "error": decision.error,
            }
            if not decision.ops:
                rounds.append(round_rec)
                LOGGER.info("postprocess round %d: VLM 判定无需清洗，停止", r)
                break

            new_mask, op_log = _postprocess_apply_ops(
                self.points, mask, decision.ops,
                n_neighbors=self.config.split_n_neighbors, nbr_idx=nbr_idx,
            )
            round_rec["ops_applied"] = op_log
            rounds.append(round_rec)

            if int(new_mask.sum()) == 0:
                LOGGER.warning("postprocess round %d 清空选区，回退本轮", r)
                break
            if np.array_equal(new_mask, mask):
                LOGGER.info("postprocess round %d: mask 无变化，停止", r)
                break
            mask = new_mask

        new_full = set(int(i) for i in np.flatnonzero(mask).tolist())
        old_full = set(original_full)
        removed = old_full - new_full
        added = new_full - old_full

        none_set = (set(result.none_point_ids) | removed) - added
        pending_set = set(result.pending_point_ids) - added
        result.full_point_ids = sorted(new_full)
        result.none_point_ids = sorted(none_set)
        result.pending_point_ids = sorted(pending_set)

        metrics_after = _metrics(mask)

        summary = {
            "semantic": self.config.semantic,
            "num_points": n_points,
            "num_full_before": len(old_full),
            "num_full_after": len(new_full),
            "num_removed": len(removed),
            "num_added": len(added),
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "component_split": split_rec,
            "rounds": rounds,
        }
        try:
            (out_dir / "postprocess.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            LOGGER.warning("写 postprocess.json 失败: %s", exc)
        result.extra["postprocess"] = {
            "num_full_before": len(old_full),
            "num_full_after": len(new_full),
            "num_removed": len(removed),
            "num_added": len(added),
            "rounds": len(rounds),
            "json": str(out_dir / "postprocess.json"),
        }
        LOGGER.info(
            "几何后处理完成: full %d -> %d (removed=%d, added=%d, rounds=%d)",
            len(old_full), len(new_full), len(removed), len(added), len(rounds),
        )

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
                normals=self._point_normals,
            )
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("初始分块图渲染失败: %s", exc)
