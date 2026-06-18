"""
语义细化流水线：

Step 1：对每个分块渲染 12 视角（该块高亮、其余淡色）→ VLM 三分类。
Step 2：把"完全符合"标为接受、"完全不符合"标为拒绝；
       "局部符合"的分块由 VLM 决定种子数，做谱聚类分裂，
       新分块继续走 Step 1 / Step 2，直到没有"局部符合"或达到深度上限。
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from geometry.blocks import (
    BlockStatus,
    SemanticBlock,
    build_initial_blocks,
    export_tree_json,
    gather_face_status,
)
from geometry.seed_split import (
    _extract_submesh,
    ask_seed_count,
    split_block_by_connected_components,
    split_block_by_geodesic_fps,
)
from geometry.spectral_cluster import vertex_labels_to_face_labels
from rendering.render_component import (
    ACCEPTED_RGB,
    DIM_RGB,
    HIGHLIGHT_RGB,
    REJECTED_RGB,
    render_highlighted_component_panel,
    render_initial_segmentation_panel,
    render_progress_panel,
    render_semantic_mask_panel,
    render_single_view_overview,
    render_single_view_status,
)
from rendering.view_thumbnails import render_thumbnails_panel
from vlm.classifier import classify_component
from vlm.client import VLMClient
from vlm.prompts import LABEL_FULL, LABEL_NONE, LABEL_PARTIAL
from vlm.review import ProgressReview, review_progress
from vlm.view_selector import BestViewSelection, select_best_view

LOGGER = logging.getLogger("pipeline.pipeline")


@dataclass
class SemanticRefinementResult:
    ok: bool
    output_dir: Path
    semantic: str
    full_face_ids: List[int] = field(default_factory=list)
    none_face_ids: List[int] = field(default_factory=list)
    pending_face_ids: List[int] = field(default_factory=list)
    num_initial_blocks: int = 0
    num_final_blocks: int = 0
    num_iterations: int = 0
    tree_json: Optional[Path] = None
    mask_json: Optional[Path] = None
    semantic_panel: Optional[Path] = None
    initial_panel: Optional[Path] = None
    elapsed_seconds: float = 0.0
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    # 视角选择产物（VLM 选最佳视角 + 透明背景 overview + 每轮 iter PNG）
    best_view: Optional[BestViewSelection] = None
    best_view_json: Optional[Path] = None
    best_view_overview: Optional[Path] = None
    best_view_thumbnails: Optional[Path] = None
    iter_progress_pngs: List[Path] = field(default_factory=list)


@dataclass
class SemanticRefinementConfig:
    semantic: str
    max_depth: int = 6
    max_seeds: int = 6
    min_block_size: int = 20
    image_size: int = 512
    panel_cols: int = 4
    save_intermediate_panels: bool = True
    save_responses: bool = True
    classification_model: Optional[str] = None
    seed_model: Optional[str] = None
    # 在 max_depth 处遇到 partial 时是否当作"最佳估计 full"接受。
    # 适用于几何上无法被谱聚类干净切分的细分语义（例如发型 vs 头颅）。
    accept_partial_at_max_depth: bool = False
    # accept 时所需的最小面数下限——小块容易是 VLM 幻觉，不值得无条件信任。
    accept_partial_min_faces: int = 500
    # 每轮迭代结束后让 VLM 看一眼「整模型染色进度图」并自评 + 决定是否终止。
    # max_depth 仍是硬上限。
    vlm_self_review: bool = False
    # 分裂算法与 full_coverage 保持一致：ask_seed_count 后统一走
    # connected components → segmat → geodesic FPS/Voronoi/Lloyd。
    # 并行 VLM 请求开关：每轮迭代里把 N 个块的 classify（observe+reason）放进
    # ThreadPoolExecutor 同时跑。1 = 完全串行（默认，与旧行为一致）。
    # 渲染 (pyvista/VTK) 仍串行（线程不安全）；ask_seed_count 也仍串行
    # （只在 CC+segmat 都失败时偶发，量小）。
    parallel_vlm_workers: int = 1
    # 实时 HTML 进度报告：每轮迭代结束后重新渲染一张「整模型染色进度图」
    # （none=白 / partial 待分裂=浅黄 / full=深红 / abandoned=极浅灰）并写到
    # output_dir/live_report.html，HTML 自带 meta refresh，浏览器直接打开就能
    # 看到实时进度。默认开启；CLI 可用 --no-live-report 关掉。
    live_report: bool = True
    live_report_refresh_seconds: int = 10
    # 视角选择：进入主循环之前让 VLM 看 12 张缩略图，挑出最能展示语义的视角，
    # 用作 live_report.html / report.html / 每轮 iter PNG 的统一展示视角。
    # 失败时退回到 heatmap_renderer.DEFAULT_OVERVIEW_VIEW。
    select_best_view: bool = True
    view_select_model: Optional[str] = None
    # 单视角渲染分辨率（像素）。VLM 不看这张图，只是 HTML 展示用，可以开高一点。
    single_view_image_size: int = 720


class SemanticRefinementPipeline:
    """
    一个完整的"VLM 三分类 + 谱聚类分裂"循环。
    用法：
        pipeline = SemanticRefinementPipeline(client, config)
        result = pipeline.run(vertices, faces, initial_face_labels, output_dir)
    """

    def __init__(self, client: VLMClient, config: SemanticRefinementConfig):
        self.client = client
        self.config = config
        self.blocks: Dict[str, SemanticBlock] = {}
        self._counter_per_parent: Dict[str, int] = {}
        # 视角选择 + 迭代时间线（在 run() 里被初始化 / 填充）
        self._best_view: Optional[BestViewSelection] = None
        self._iter_progress_pngs: List[Path] = []

    # ---------- 树工具 ----------
    def _next_child_id(self, parent_id: str) -> str:
        n = self._counter_per_parent.get(parent_id, 0)
        self._counter_per_parent[parent_id] = n + 1
        return f"{parent_id}.{n}"

    def _register_block(self, block: SemanticBlock) -> None:
        self.blocks[block.block_id] = block

    # ---------- 面级状态（用于全局进度图 / 自评） ----------
    def _compute_face_status(self, n_faces: int) -> List[str]:
        """
        返回长度为 n_faces 的状态字符串数组：
            "full"      已确认属于语义
            "pending"   仍在处理 / 等待分裂
            "none"      已确认不属于语义
            "abandoned" 触底放弃（max_depth / too_small / split_failed）
        """
        status = ["none"] * n_faces
        for block in self.blocks.values():
            if block.status == BlockStatus.SPLIT:
                # SPLIT 不直接拥有面（已交给 children）
                continue
            if block.status == BlockStatus.FULL:
                tag = "full"
            elif block.status == BlockStatus.PENDING:
                tag = "pending"
            elif block.status == BlockStatus.ABANDONED:
                tag = "abandoned"
            else:  # NONE
                tag = "none"
            for fid in block.face_ids:
                if 0 <= fid < n_faces:
                    status[fid] = tag
        return status

    def _drain_pending_to_terminate(
        self, queue: Sequence[str], reason: str
    ) -> int:
        """
        VLM 决定终止时调用：把还在 queue 里、尚未分类的 PENDING 块按
        accept_partial_at_max_depth 规则尽量接受，否则标 ABANDONED。
        返回被接受为 full 的块数。
        """
        accepted = 0
        for bid in queue:
            blk = self.blocks.get(bid)
            if blk is None or blk.status != BlockStatus.PENDING:
                continue
            blk.status = BlockStatus.ABANDONED
            blk.notes = (blk.notes + f" | terminated_by_vlm({reason})").strip(" |")
        return accepted

    # ---------- 几何 segmat 预分割 ----------
    def _segmat_split(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_ids: Sequence[int],
        vertex_scalars: Optional[np.ndarray],
        *,
        min_clusters: int = 3,
        max_clusters: int = 8,
    ) -> Dict[str, Any]:
        """单连通块的第一次几何切。

        统一走 **测地 FPS + Voronoi + Lloyd 平衡**：
          ① 子 mesh 顶点图（边权 = 欧氏边长）
          ② 测地最远点采样选 K = max_clusters 个种子顶点
          ③ Lloyd 迭代：Voronoi 划分 + 簇内 3D 重心更新种子，让 K 片切分大小相近
          ④ 顶点 label → 面 label，再做一次连通性清洗
        K 取 max_clusters；Lloyd 后实际簇数可能因 mask 退化而少于 K。
        不再做谱聚类（user 明确反馈 spectral 退化为轴向切片，不可用）。
        """
        face_ids_list = list(face_ids)
        if len(face_ids_list) <= 1:
            return {"groups": [], "error": "block too small"}

        try:
            sub_v, sub_f, _ = _extract_submesh(vertices, faces, face_ids_list)
            n_v = len(sub_v)
            if n_v <= max(min_clusters, 4) or len(sub_f) <= 1:
                return {"groups": [], "error": "submesh too small"}

            from geometry.seed_split import (
                _build_submesh_vertex_graph,
                _geodesic_fps_seeds,
                _lloyd_relax_geodesic,
                enforce_per_label_face_connectivity,
            )

            graph = _build_submesh_vertex_graph(sub_v, sub_f)
            k_target = int(np.clip(max_clusters, min_clusters, max(2, n_v - 1)))
            fps_seeds = _geodesic_fps_seeds(graph, sub_v, k_target)
            if len(fps_seeds) < 2:
                return {"groups": [], "error": "fps 不足 2 个种子"}
            seeds_final, v_labels = _lloyd_relax_geodesic(
                graph, sub_v, list(fps_seeds), n_iter=8
            )

            sub_face_labels = vertex_labels_to_face_labels(v_labels, sub_f)
            arr = np.asarray(sub_face_labels, dtype=np.int64)
            arr = enforce_per_label_face_connectivity(sub_f, arr)
            mode = "geodesic_fps_voronoi_lloyd"
            sp_dbg = {
                "method": mode,
                "k_requested": k_target,
                "fps_seeds": list(fps_seeds),
                "lloyd_seeds": list(seeds_final),
            }

            used = sorted({int(x) for x in arr})
            groups: List[List[int]] = []
            for k in used:
                idx = np.flatnonzero(arr == k)
                if idx.size == 0:
                    continue
                groups.append([face_ids_list[i] for i in idx.tolist()])
            k_actual = len(groups)
            return {
                "groups": groups,
                "k_actual": int(k_actual),
                "k_eigengap": int(k_actual),
                "sizes": [len(g) for g in groups],
                "mode": mode,
                "spectral_debug": sp_dbg if isinstance(sp_dbg, dict) else None,
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("segmat split 失败: %s", exc)
            return {"groups": [], "error": str(exc)}

    def _split_block_by_seeds(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        block: SemanticBlock,
        seed_count: int,
        *,
        responses_dir: Path,
    ) -> Optional[List[List[int]]]:
        """对单个 block 走 CC → segmat → FPS 级联，返回子块 face_id 分组。

        这与 ``FullCoveragePipeline._split_block_by_seeds`` 保持一致：
        - 未做过 segmat：先按拓扑连通分量拆；若得到多个 CC，直接采用。
        - 仍是单连通且未做过 segmat：跑一次 segmat 预切。
        - 已做过 segmat 或 segmat 失败：只走 geodesic FPS + Voronoi + Lloyd。
        """
        # Step A: 连通分量（仅未做过 segmat 的块）
        if not block.segmat_done:
            cc_groups = split_block_by_connected_components(
                vertices,
                faces,
                block.face_ids,
                min_component_faces=max(1, self.config.min_block_size // 2),
            )
            if cc_groups is not None and len(cc_groups) >= 2:
                block.seed_decision = {
                    "mode": "connected_components",
                    "num_components": len(cc_groups),
                    "component_sizes": [len(g) for g in cc_groups],
                    "seed_count_target": seed_count,
                }
                self._dump_response(
                    responses_dir, f"{block.block_id}_cc", block.seed_decision
                )
                return cc_groups

        # Step B: segmat 第一次切（仅未做过的）
        if not block.segmat_done:
            segmat = self._segmat_split(
                vertices,
                faces,
                block.face_ids,
                vertex_scalars=None,
                min_clusters=max(2, min(3, seed_count)),
                max_clusters=max(seed_count, 4),
            )
            block.seed_decision = {
                "mode": "segmat",
                "k_actual": segmat.get("k_actual", 0),
                "k_eigengap": segmat.get("k_eigengap", 0),
                "component_sizes": segmat.get("sizes", []),
                "seed_count_target": seed_count,
                "error": segmat.get("error"),
            }
            self._dump_response(
                responses_dir, f"{block.block_id}_segmat", block.seed_decision
            )
            sub_groups = segmat.get("groups", [])
            if len(sub_groups) >= 2:
                block.segmat_done = True
                return sub_groups
            LOGGER.info(
                "block %s segmat 失败 (%s) → 改走 FPS-Voronoi K=%d",
                block.block_id,
                segmat.get("error", "unknown"),
                seed_count,
            )
            block.segmat_done = True

        # Step C: FPS-Voronoi-Lloyd
        try:
            split = split_block_by_geodesic_fps(
                vertices,
                faces,
                block.face_ids,
                seed_count=seed_count,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("block %s geodesic_fps 失败: %s", block.block_id, exc)
            block.seed_decision = {
                "mode": "geodesic_fps",
                "seed_count_target": seed_count,
                "error": str(exc),
            }
            self._dump_response(
                responses_dir, f"{block.block_id}_seed", block.seed_decision
            )
            return None

        block.seed_decision = {
            "mode": "geodesic_fps",
            "seed_count_target": seed_count,
            "num_clusters": split.num_clusters,
            "debug": split.debug,
        }
        self._dump_response(
            responses_dir, f"{block.block_id}_seed", block.seed_decision
        )
        if split.num_clusters <= 1:
            return None
        return split.sub_face_groups

    # ---------- 渲染 ----------
    def _render_block_panel(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        block: SemanticBlock,
        panel_dir: Path,
    ) -> Path:
        # NOTE: 不要把 semantic 写进图片标题——这会被 VLM 当成提示从而引发确认偏差。
        title = f"[block {block.block_id}] depth={block.depth}  faces={block.size}"
        panel_path = panel_dir / f"block_{block.block_id}.png"
        return render_highlighted_component_panel(
            vertices,
            faces,
            block.face_ids,
            panel_path,
            title=title,
            image_size=self.config.image_size,
            cols=self.config.panel_cols,
            highlight_rgb=HIGHLIGHT_RGB,
            dim_rgb=DIM_RGB,
        )

    # ---------- 实时 HTML 进度报告 ----------
    # 用户配色：none=白 / partial(pending)=浅黄 / full=深红 / abandoned=极浅灰。
    _LIVE_COLOR_MAP = {
        "full": (140, 25, 35),       # 深红高亮
        "pending": (255, 215, 130),  # 浅黄（partial 待分裂）
        "none": (255, 255, 255),     # 纯白
        "abandoned": (235, 235, 235),  # 极浅灰
    }

    def _save_iteration_progress(
        self,
        out_dir: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        iteration: int,
    ) -> Optional[Path]:
        """每轮迭代末尾把「当前面状态」按 chosen 视角 + 透明背景渲染一张 PNG，
        放到 output_dir/iterations/iter_<NN>.png。供 live_report / 最终 report
        作时间线展示。失败时返回 None。
        """
        if self._best_view is None:
            return None
        n_faces = int(faces.shape[0])
        face_status = self._compute_face_status(n_faces)
        it_dir = out_dir / "iterations"
        it_dir.mkdir(parents=True, exist_ok=True)
        png = it_dir / f"iter_{iteration:02d}.png"
        try:
            render_single_view_status(
                vertices,
                faces,
                face_status,
                png,
                elev=self._best_view.elev,
                azim=self._best_view.azim,
                image_size=self.config.single_view_image_size,
                color_map=self._LIVE_COLOR_MAP,
                background="transparent",
                title=None,
            )
        except (OSError, ValueError, ImportError) as exc:
            LOGGER.warning("iter %d 单视角 PNG 渲染失败: %s", iteration, exc)
            return None
        # 同步记录到 result 列表（pipeline 末尾会把这个列表带到 report）
        self._iter_progress_pngs.append(png)
        return png

    def _update_live_report(
        self,
        out_dir: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        iteration: int,
        queue_after: Sequence[str],
        finished: bool = False,
        extra_note: str = "",
    ) -> None:
        """重新渲染 live_progress.png 并写 live_report.html。

        在每轮迭代末尾、终止 / 结束时调用。HTML 自带 <meta refresh> 不需要
        前端 JS，浏览器打开后定时自动刷新整页（含图片）。
        """
        if not self.config.live_report:
            return

        n_faces = int(faces.shape[0])
        face_status = self._compute_face_status(n_faces)

        live_png = out_dir / "live_progress.png"
        rendered = False
        # 优先用 chosen view 单视角 + 透明背景（贴合用户最新要求）。
        # 视角选择失败 / fallback 时也照样能用——BestViewSelection 始终有 elev/azim。
        try:
            if self._best_view is not None:
                render_single_view_status(
                    vertices,
                    faces,
                    face_status,
                    live_png,
                    elev=self._best_view.elev,
                    azim=self._best_view.azim,
                    image_size=self.config.single_view_image_size,
                    color_map=self._LIVE_COLOR_MAP,
                    background="transparent",
                    title=None,
                )
            else:
                # 兜底：未选视角时维持 12 视角拼图
                render_progress_panel(
                    vertices,
                    faces,
                    face_status,
                    live_png,
                    title=(
                        f"iter {iteration}  "
                        f"({'已完成' if finished else '进行中'})  "
                        f"semantic={self.config.semantic}"
                    ),
                    image_size=self.config.image_size,
                    cols=self.config.panel_cols,
                    color_map=self._LIVE_COLOR_MAP,
                )
            rendered = True
        except (OSError, ValueError, ImportError) as exc:
            LOGGER.warning("live progress 渲染失败: %s", exc)

        # 按 status 统计 block / face
        from geometry.blocks import BlockStatus as _BS

        block_counts = {st.value: 0 for st in _BS}
        for b in self.blocks.values():
            block_counts[b.status.value] = block_counts.get(b.status.value, 0) + 1
        full_faces = sum(1 for s in face_status if s == "full")
        pending_faces = sum(1 for s in face_status if s == "pending")
        abandoned_faces = sum(1 for s in face_status if s == "abandoned")
        none_faces = n_faces - full_faces - pending_faces - abandoned_faces

        import html as _html

        def esc(x: Any) -> str:
            return _html.escape(str(x))

        refresh_meta = (
            ""
            if finished
            else (
                f'<meta http-equiv="refresh" '
                f'content="{int(self.config.live_report_refresh_seconds)}">'
            )
        )
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        status_tag = (
            '<span class="tag tag-done">已完成</span>'
            if finished
            else '<span class="tag tag-live">运行中</span>'
        )
        img_html = (
            f'<img class="progress" src="live_progress.png?t={int(time.time())}" '
            f'alt="live progress" />'
            if rendered
            else '<div class="muted">(尚未生成 live_progress.png)</div>'
        )
        extra_html = (
            f'<div class="extra">{esc(extra_note)}</div>' if extra_note else ""
        )

        # 视角选择卡片
        view_card_html = ""
        if self._best_view is not None:
            bv = self._best_view
            tag = (
                '<span class="badge badge-fallback">默认 fallback</span>'
                if bv.fallback
                else '<span class="badge badge-vlm">VLM 选定</span>'
            )
            ov_html = ""
            if bv.overview_png is not None:
                try:
                    rel = bv.overview_png.relative_to(out_dir).as_posix()
                except ValueError:
                    rel = bv.overview_png.name
                ov_html = (
                    f'<img class="overview" src="{esc(rel)}" alt="best view overview" />'
                )
            view_card_html = f"""
<section class="view-card">
  <div class="view-meta">
    <b>展示视角</b> {tag}
    <span class="muted"> · #{bv.index + 1}（VIEW_GRID）</span>
    <span class="muted"> · elev={bv.elev:.1f}° azim={bv.azim:.1f}°</span>
  </div>
  <div class="view-reason">{esc(bv.reason or '—')}</div>
  {ov_html}
</section>
"""

        # 迭代时间线（已经生成的 iter_*.png 全部按顺序排开）
        timeline_html = ""
        if self._iter_progress_pngs:
            items: List[str] = []
            for png in self._iter_progress_pngs:
                try:
                    rel = png.relative_to(out_dir).as_posix()
                except ValueError:
                    rel = f"iterations/{png.name}"
                m = png.stem.split("_")[-1]  # iter_03 -> "03"
                items.append(
                    f'<figure class="timeline-item">'
                    f'<img src="{esc(rel)}?t={int(time.time())}" '
                    f'alt="iter {esc(m)}" />'
                    f'<figcaption>iter {esc(m)}</figcaption>'
                    f"</figure>"
                )
            timeline_html = (
                f'<section class="timeline">'
                f'<h2>迭代时间线（共 {len(items)} 帧）</h2>'
                f'<div class="timeline-row">{"".join(items)}</div>'
                f"</section>"
            )

        css = """
            body { font-family: -apple-system, "Segoe UI", "Microsoft Yahei",
                   sans-serif; background:#f5f5f7; color:#1d1d1f; margin:0;
                   padding:24px; }
            h1 { margin:0 0 8px 0; font-size:22px; }
            h2 { font-size:16px; margin:20px 0 10px 0; }
            .row { display:flex; gap:16px; flex-wrap:wrap; margin:14px 0; }
            .card { background:#fff; border:1px solid #d2d2d7; border-radius:10px;
                    padding:14px 18px; min-width:200px; }
            .card b { display:block; color:#6e6e73; font-size:12px;
                      font-weight:500; margin-bottom:4px; }
            .card span.v { font-size:20px; font-weight:600; }
            .legend span { display:inline-flex; align-items:center; gap:6px;
                           margin-right:14px; font-size:13px; }
            .legend i { display:inline-block; width:14px; height:14px;
                        border:1px solid #999; border-radius:3px; }
            .tag { padding:3px 10px; border-radius:999px; font-size:12px;
                   font-weight:600; vertical-align:middle; margin-left:8px; }
            .tag-live { background:#fff7d9; color:#a16400;
                        border:1px solid #f1d27a; }
            .tag-done { background:#d6f4e0; color:#0a6c2c;
                        border:1px solid #7dc99a; }
            .badge { display:inline-block; padding:2px 10px; border-radius:999px;
                     font-size:12px; font-weight:600; margin-left:6px; }
            .badge-vlm { background:#e5f0ff; color:#1a55b8;
                         border:1px solid #b6d3ff; }
            .badge-fallback { background:#ffe9d9; color:#a14a00;
                              border:1px solid #ffc69b; }
            .muted { color:#6e6e73; }
            .ts { color:#86868b; font-size:12px; margin-bottom:10px; }
            /* 透明背景 PNG，靠 checkerboard 衬出来更直观 */
            .progress, .overview, .timeline-item img {
              max-width:100%; border:1px solid #d2d2d7; border-radius:8px;
              background-color:#fff;
              background-image: linear-gradient(45deg, #f0f0f5 25%, transparent 25%),
                                linear-gradient(-45deg, #f0f0f5 25%, transparent 25%),
                                linear-gradient(45deg, transparent 75%, #f0f0f5 75%),
                                linear-gradient(-45deg, transparent 75%, #f0f0f5 75%);
              background-size: 18px 18px;
              background-position: 0 0, 0 9px, 9px -9px, -9px 0;
            }
            .progress { max-width:720px; display:block; margin:0 auto; }
            .overview { max-width:380px; margin-top:8px; }
            .view-card { background:#fff; border:1px solid #d2d2d7;
                         border-radius:10px; padding:12px 16px; margin:14px 0; }
            .view-meta { font-size:14px; }
            .view-reason { color:#3a3a3c; font-size:13px; margin-top:6px;
                           padding:6px 10px; border-left:3px solid #0a84ff;
                           background:#f5f5f7; border-radius:4px; }
            .timeline { margin-top:24px; }
            .timeline-row { display:flex; gap:10px; overflow-x:auto;
                            padding:8px 0; }
            .timeline-item { margin:0; flex:0 0 auto; text-align:center; }
            .timeline-item img { width:180px; height:auto; display:block; }
            .timeline-item figcaption { color:#6e6e73; font-size:12px;
                                        margin-top:4px; }
            .extra { background:#fff7d9; border:1px solid #f1d27a;
                     padding:10px 14px; border-radius:8px; margin:10px 0; }
        """

        body = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
{refresh_meta}
<title>VLM live · {esc(self.config.semantic)} · iter {iteration}</title>
<style>{css}</style>
</head>
<body>
<h1>VLM 语义细化 · 实时进度 {status_tag}</h1>
<div class="ts">更新时间: {esc(ts)} · semantic: <b>{esc(self.config.semantic)}</b>
 · iteration: <b>{iteration}</b> · 下轮队列: <b>{len(queue_after)}</b></div>

<div class="legend">
  <span><i style="background:#fff"></i>none（白）</span>
  <span><i style="background:#ffd782"></i>partial 待分裂（浅黄）</span>
  <span><i style="background:#8c1923"></i>full（深红高亮）</span>
  <span><i style="background:#ebebeb"></i>abandoned（极浅灰）</span>
</div>

<div class="row">
  <div class="card"><b>full faces</b><span class="v">{full_faces}</span></div>
  <div class="card"><b>partial(pending) faces</b><span class="v">{pending_faces}</span></div>
  <div class="card"><b>none faces</b><span class="v">{none_faces}</span></div>
  <div class="card"><b>abandoned faces</b><span class="v">{abandoned_faces}</span></div>
  <div class="card"><b>total faces</b><span class="v">{n_faces}</span></div>
</div>

<div class="row">
  <div class="card"><b>full blocks</b><span class="v">{block_counts.get('full', 0)}</span></div>
  <div class="card"><b>pending blocks</b><span class="v">{block_counts.get('pending', 0)}</span></div>
  <div class="card"><b>split blocks</b><span class="v">{block_counts.get('split', 0)}</span></div>
  <div class="card"><b>none blocks</b><span class="v">{block_counts.get('none', 0)}</span></div>
  <div class="card"><b>abandoned blocks</b><span class="v">{block_counts.get('abandoned', 0)}</span></div>
</div>

{extra_html}

{view_card_html}

<h2>当前进度图（单视角 · 透明背景）</h2>
{img_html}

{timeline_html}
</body>
</html>
"""
        try:
            (out_dir / "live_report.html").write_text(body, encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("live report 写文件失败: %s", exc)

    # ---------- 并行 classify ----------
    @staticmethod
    def _make_forced_partial_classification():
        """force_split=True 时的占位分类，等价于"伪造一个 partial"让分裂路径接管。"""
        from vlm.classifier import ComponentClassification

        return ComponentClassification(
            label=LABEL_PARTIAL,
            confidence=0.0,
            reason="force_split_by_review",
            covered_parts=[],
            dominant_region="",
            semantic_parts=[],
            extra_parts=[],
            missing_overlap="",
            overridden_from="full",
        )

    def _batch_classify(
        self,
        tasks: Sequence[tuple[str, Path]],
    ) -> Dict[str, Any]:
        """对一批 (block_id, panel_path) 调 classify_component。

        当 parallel_vlm_workers > 1 时用 ThreadPoolExecutor 并发。VLM 调用主要是
        HTTP IO，GIL 不阻塞，线程池足够；不用 asyncio 是为了避免侵入式改造
        client.py。返回 block_id -> ComponentClassification（含失败时的 error 字段）。
        """
        results: Dict[str, Any] = {}
        if not tasks:
            return results

        def _do(bid: str, panel: Path):
            return bid, classify_component(
                self.client,
                panel,
                self.config.semantic,
                model=self.config.classification_model,
            )

        workers = max(1, int(self.config.parallel_vlm_workers))
        if workers > 1 and len(tasks) > 1:
            n = min(workers, len(tasks))
            LOGGER.info(
                "parallel classify: %d tasks across %d workers", len(tasks), n
            )
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = {ex.submit(_do, bid, panel): bid for bid, panel in tasks}
                for fut in as_completed(futs):
                    bid = futs[fut]
                    try:
                        _, cls = fut.result()
                        results[bid] = cls
                    except (RuntimeError, ValueError, OSError) as exc:
                        LOGGER.warning(
                            "parallel classify failed for %s: %s", bid, exc
                        )
                        # 留空让主循环 fallback 到串行 classify
        else:
            for bid, panel in tasks:
                try:
                    _, cls = _do(bid, panel)
                    results[bid] = cls
                except (RuntimeError, ValueError, OSError) as exc:
                    LOGGER.warning("serial classify failed for %s: %s", bid, exc)
        return results

    # ---------- 视角选择 ----------
    def _select_view_for_display(
        self,
        out_dir: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Optional[BestViewSelection]:
        """渲染 12 缩略图 → VLM 挑最佳 → 渲一张透明背景 overview PNG。

        - 关掉视角选择 (select_best_view=False) 时直接退回 DEFAULT_OVERVIEW_VIEW。
        - 任何阶段出错都 fallback 到 DEFAULT_OVERVIEW_VIEW，主流程不被阻断。
        """
        if not self.config.select_best_view:
            from rendering.heatmap_renderer import DEFAULT_OVERVIEW_VIEW, VIEW_GRID as _VG

            target = np.asarray(DEFAULT_OVERVIEW_VIEW, dtype=np.float64)
            best_idx = 0
            best_dist = float("inf")
            for i, (elev, azim) in enumerate(_VG):
                d = (elev - target[0]) ** 2 + (azim - target[1]) ** 2
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            elev, azim = _VG[best_idx]
            bv = BestViewSelection(
                index=best_idx, elev=float(elev), azim=float(azim),
                reason="select_best_view=False，跳过 VLM，直接用默认 overview 视角",
                fallback=True,
            )
        else:
            thumb_path = out_dir / "view_thumbnails.png"
            try:
                render_thumbnails_panel(
                    vertices, faces, thumb_path,
                    title=f"候选视角（语义：{self.config.semantic}）",
                )
            except (OSError, ImportError, ValueError) as exc:
                LOGGER.warning("缩略图渲染失败: %s", exc)
                thumb_path = None
            if thumb_path is None:
                bv = BestViewSelection(
                    index=0, elev=float(VIEW_GRID[0][0]),
                    azim=float(VIEW_GRID[0][1]),
                    reason="缩略图渲染失败，退回 VIEW_GRID[0]",
                    fallback=True,
                )
            else:
                LOGGER.info("调用 VLM 选择最佳视角 ...")
                bv = select_best_view(
                    self.client, thumb_path,
                    self.config.semantic,
                    model=self.config.view_select_model,
                )
                bv.thumbnails_panel = thumb_path

        # 用 chosen view 渲一张透明背景 overview PNG（用作 HTML 报告镇页图）
        ov_path = out_dir / "best_view_overview.png"
        try:
            render_single_view_overview(
                vertices, faces, ov_path,
                elev=bv.elev, azim=bv.azim,
                image_size=self.config.single_view_image_size,
                background="transparent",
            )
            bv.overview_png = ov_path
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("best view overview 渲染失败: %s", exc)
            bv.overview_png = None

        # 落盘 best_view.json，方便 report.py / 用户后期查看
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
                        "thumbnails_panel": (
                            str(bv.thumbnails_panel)
                            if bv.thumbnails_panel else None
                        ),
                        "overview_png": (
                            str(bv.overview_png) if bv.overview_png else None
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.warning("写 best_view.json 失败: %s", exc)

        LOGGER.info(
            "最佳视角: #%d (elev=%.1f, azim=%.1f, fallback=%s) — %s",
            bv.index + 1, bv.elev, bv.azim, bv.fallback,
            (bv.reason or "")[:80],
        )
        return bv

    # ---------- 主流程 ----------
    def run(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        initial_face_labels: Sequence[int],
        output_dir: str | Path,
        *,
        vertex_scalars: Optional[np.ndarray] = None,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> SemanticRefinementResult:
        t0 = time.perf_counter()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_dir = out_dir / "panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        responses_dir = out_dir / "responses"
        if self.config.save_responses:
            responses_dir.mkdir(parents=True, exist_ok=True)

        # ---- 视角选择（最先做）----
        # 让 VLM 看 12 张缩略图挑出最能展示语义的视角。失败时自动 fallback 到
        # DEFAULT_OVERVIEW_VIEW，主流程不会被阻断。
        self._iter_progress_pngs = []
        self._best_view = self._select_view_for_display(out_dir, vertices, faces)

        n_faces = int(faces.shape[0])
        if len(initial_face_labels) != n_faces:
            raise ValueError(
                f"initial_face_labels 长度 ({len(initial_face_labels)}) "
                f"与 faces 数量 ({n_faces}) 不一致"
            )

        raw_initial_blocks = build_initial_blocks(
            initial_face_labels, min_block_size=self.config.min_block_size
        )

        # 连通域分块优先于 segmat：把每个初始 block 按拓扑连通分量展开，
        # 再交给 VLM。这样开局就是「每片独立连通几何 → 一次 12 视角分类」，
        # 比让 spectral 把多个不相连的部件归到同一标签靠谱得多。
        # 初始展开不合并小 CC（保留拓扑纯度），太小的会在主循环里被
        # `too_small` 跳过。
        initial_blocks: List[SemanticBlock] = []
        cc_min_faces = 1
        for src in raw_initial_blocks:
            cc_groups = split_block_by_connected_components(
                vertices,
                faces,
                src.face_ids,
                min_component_faces=cc_min_faces,
            )
            if cc_groups is None or len(cc_groups) <= 1:
                initial_blocks.append(src)
                continue
            base_id = src.block_id
            base_note = src.notes
            for k, group_face_ids in enumerate(cc_groups):
                cc_block = SemanticBlock(
                    block_id=f"{base_id}.{k}",
                    face_ids=list(group_face_ids),
                    depth=0,
                    parent_id=None,
                    status=BlockStatus.PENDING,
                    notes=(
                        f"{base_note} | cc_init({k+1}/{len(cc_groups)})"
                    ).strip(" |"),
                )
                initial_blocks.append(cc_block)

        # 重新编号成连续 C0/C1/... 让后续 child id (Cx.y) 不冲突。
        # 初始块一律 force_split=True：跳过首轮 classify，直接走级联
        # （CC → segmat → VLM-K spectral）。VLM 在整 mesh 一片红/全模型上做
        # 三分类高度不可靠（容易判 full 或 none 提前结束），先切再判更稳。
        renumbered: List[SemanticBlock] = []
        for new_idx, b in enumerate(initial_blocks):
            renumbered.append(
                SemanticBlock(
                    block_id=f"C{new_idx}",
                    face_ids=b.face_ids,
                    depth=0,
                    parent_id=None,
                    status=BlockStatus.PENDING,
                    notes=(b.notes + " | initial_force_split").strip(" |"),
                    force_split=True,
                )
            )
        initial_blocks = renumbered

        LOGGER.info(
            "初始 segmat=%d 块 → CC 展开后=%d 块",
            len(raw_initial_blocks), len(initial_blocks),
        )

        for b in initial_blocks:
            self._register_block(b)

        result = SemanticRefinementResult(
            ok=False,
            output_dir=out_dir,
            semantic=self.config.semantic,
            num_initial_blocks=len(initial_blocks),
        )

        if not initial_blocks:
            result.message = "初始分块为空（min_block_size 是否过大？）"
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        # 用展开后的 CC 标签渲染初始分块图（每个 CC 一种颜色）
        cc_face_labels = np.full(n_faces, -1, dtype=np.int64)
        for new_idx, b in enumerate(initial_blocks):
            cc_face_labels[np.asarray(b.face_ids, dtype=np.int64)] = new_idx

        initial_panel = out_dir / "initial_segmentation.png"
        try:
            render_initial_segmentation_panel(
                vertices,
                faces,
                cc_face_labels.tolist(),
                initial_panel,
                title=(
                    f"初始分块（CC优先）  segmat={len(raw_initial_blocks)} "
                    f"→ cc={len(initial_blocks)}"
                ),
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
            )
            result.initial_panel = initial_panel
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("初始分块拼图渲染失败: %s", exc)

        queue: List[str] = [b.block_id for b in initial_blocks]
        iteration = 0
        responses_log: List[Dict[str, Any]] = []
        reviews_log: List[Dict[str, Any]] = []
        reviews_dir = out_dir / "reviews"
        if self.config.vlm_self_review:
            reviews_dir.mkdir(parents=True, exist_ok=True)
        terminated_by_vlm = False

        # 初始 live_report：让用户在 VLM 还没开始干活时就看到「全部 pending」的底图
        try:
            self._update_live_report(
                out_dir,
                vertices,
                faces,
                iteration=0,
                queue_after=queue,
                finished=False,
                extra_note="初始化中：尚未开始 VLM classify",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("初始 live report 失败: %s", exc)

        while queue:
            iteration += 1
            LOGGER.info("== iteration %d, queue=%d ==", iteration, len(queue))
            next_queue: List[str] = []

            # ===== Pre-pass A（串行）：pre-check + render =====
            # pyvista/VTK 不是线程安全的，所以渲染留串行；同时把 too_small / 渲染失败
            # 的块在这里就处理掉，避免污染并行批次。
            iter_panel: Dict[str, Path] = {}  # block_id -> panel_path（仅渲染成功）
            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                if block.size < self.config.min_block_size:
                    block.status = BlockStatus.ABANDONED
                    block.notes = (block.notes + " | too_small").strip(" |")
                    self._dump_response(responses_dir, block_id, {"reason": "too_small"})
                    if progress_cb:
                        progress_cb("abandoned", {"block_id": block_id, "reason": "too_small"})
                    continue
                try:
                    panel_path = self._render_block_panel(vertices, faces, block, panel_dir)
                    block.panel_path = str(panel_path)
                    iter_panel[block_id] = panel_path
                except (OSError, ImportError, ValueError) as exc:
                    LOGGER.error("渲染 block %s 失败: %s", block_id, exc)
                    block.status = BlockStatus.ABANDONED
                    block.notes = f"render_failed: {exc}"

            # ===== Pre-pass B（可并行）：classify 非 force_split 的块 =====
            classify_tasks: List[tuple[str, Path]] = []
            for block_id, panel_path in iter_panel.items():
                if self.blocks[block_id].force_split:
                    continue
                classify_tasks.append((block_id, panel_path))
            iter_cls_cache = self._batch_classify(classify_tasks)

            # ===== 决策主循环（串行；只读 cache） =====
            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                panel_path = iter_panel.get(block_id)
                if panel_path is None:
                    # 已在 pre-pass A 被 too_small / render_failed 处理
                    continue

                # Step 1：三分类（force_split 时跳过，伪造一个 partial 让分裂路径接管）
                if block.force_split:
                    LOGGER.info(
                        "block %s force_split=True (review 降级)，跳过 classify 直接分裂",
                        block_id,
                    )
                    cls = self._make_forced_partial_classification()
                else:
                    cls = iter_cls_cache.get(block_id)
                    if cls is None:
                        # 并行 batch 失败兜底：串行重试一次
                        cls = classify_component(
                            self.client,
                            panel_path,
                            self.config.semantic,
                            model=self.config.classification_model,
                        )
                block.classification = {
                    "label": cls.label,
                    "confidence": cls.confidence,
                    "reason": cls.reason,
                    "covered_parts": cls.covered_parts,
                    "dominant_region": cls.dominant_region,
                    "semantic_parts": cls.semantic_parts,
                    "extra_parts": cls.extra_parts,
                    "missing_overlap": cls.missing_overlap,
                    "overridden_from": cls.overridden_from,
                    "semantic_area_ratio": cls.semantic_area_ratio,
                    "dominant_side": cls.dominant_side,
                    "committed_if_forced": cls.committed_if_forced,
                    "provider": cls.provider,
                    "model": cls.model,
                    "elapsed_seconds": cls.elapsed_seconds,
                    "error": cls.error,
                }
                self._dump_response(
                    responses_dir, f"{block_id}_classify", block.classification
                )
                responses_log.append(
                    {
                        "block_id": block_id,
                        "kind": "classify",
                        "label": cls.label,
                        "confidence": cls.confidence,
                        "reason": cls.reason,
                    }
                )
                if progress_cb:
                    progress_cb("classified", {"block_id": block_id, "label": cls.label})

                if cls.label == LABEL_FULL:
                    block.status = BlockStatus.FULL
                    continue
                if cls.label == LABEL_NONE:
                    block.status = BlockStatus.NONE
                    continue

                # PARTIAL：深度已经到顶
                if block.depth >= self.config.max_depth:
                    n_sem = len(cls.semantic_parts)
                    n_extra = len(cls.extra_parts)
                    block_faces = block.size
                    committed = cls.committed_if_forced or ""
                    dom = cls.dominant_side or ""
                    sar = cls.semantic_area_ratio
                    # 新版接受规则（更保守，要求 VLM 自己「强制承诺 full」）：
                    # 必要条件：
                    #   ① accept_partial_at_max_depth 开启；
                    #   ② 含有语义部位（n_sem ≥ 1）；
                    #   ③ block 面数 ≥ 阈值（小块多半是 VLM 幻觉边角）；
                    #   ④ VLM committed_if_forced == "full"（表示语义占主导值得保留）；
                    #   ⑤ dominant_side ∈ {"semantic", ""}（balanced/extra 一律不接受）。
                    # 这样 (头发+整张脸) 这种主导边是 "extra" 的块会被收为 abandoned，
                    # 不会污染最终掩码。
                    accept = (
                        self.config.accept_partial_at_max_depth
                        and n_sem >= 1
                        and cls.missing_overlap is not True
                        and block_faces >= self.config.accept_partial_min_faces
                        and committed == LABEL_FULL
                        and dom in ("semantic", "")
                    )
                    if accept:
                        block.status = BlockStatus.FULL
                        block.notes = (
                            block.notes
                            + f" | accepted_partial_at_max_depth"
                            f"(faces={block_faces},sem={n_sem},extra={n_extra},"
                            f"dom={dom or 'na'},ratio="
                            f"{f'{sar:.2f}' if sar is not None else 'na'},"
                            f"commit={committed})"
                        ).strip(" |")
                        if progress_cb:
                            progress_cb(
                                "classified",
                                {
                                    "block_id": block_id,
                                    "label": "full*",  # 带 * 表示 best-effort
                                },
                            )
                        continue
                    block.status = BlockStatus.ABANDONED
                    if self.config.accept_partial_at_max_depth:
                        note_extra = (
                            f"max_depth_reached(faces={block_faces},"
                            f"sem={n_sem},extra={n_extra},"
                            f"dom={dom or 'na'},"
                            f"ratio={f'{sar:.2f}' if sar is not None else 'na'},"
                            f"commit={committed or 'na'})"
                        )
                    else:
                        note_extra = "max_depth_reached"
                    block.notes = (block.notes + " | " + note_extra).strip(" |")
                    if progress_cb:
                        progress_cb(
                            "abandoned",
                            {"block_id": block_id, "reason": "max_depth"},
                        )
                    continue

                # ---- ask_seed_count + 与 full_coverage 一致的几何级联 ----
                seed_resp = ask_seed_count(
                    self.client,
                    panel_path,
                    self.config.semantic,
                    max_seeds=self.config.max_seeds,
                    fallback=max(2, min(self.config.max_seeds, len(cls.covered_parts) + 1)),
                    model=self.config.seed_model,
                    covered_parts=cls.covered_parts,
                    semantic_parts=cls.semantic_parts,
                    extra_parts=cls.extra_parts,
                    block_faces=block.size,
                )
                block.seed_decision = {
                    "seed_count": seed_resp.seed_count,
                    "reason": seed_resp.reason,
                    "error": seed_resp.error,
                    "strategy": "connected_components→segmat→geodesic_fps",
                }
                block.seed_count = seed_resp.seed_count
                self._dump_response(
                    responses_dir, f"{block_id}_seed", block.seed_decision
                )

                sub_groups = self._split_block_by_seeds(
                    vertices,
                    faces,
                    block,
                    seed_count=seed_resp.seed_count,
                    responses_dir=responses_dir,
                )
                if not sub_groups or len(sub_groups) < 2:
                    block.status = BlockStatus.ABANDONED
                    block.notes = (block.notes + " | split_failed").strip(" |")
                    if progress_cb:
                        progress_cb(
                            "abandoned",
                            {"block_id": block_id, "reason": "split_failed"},
                        )
                    continue

                block.status = BlockStatus.SPLIT
                for group_face_ids in sub_groups:
                    child_id = self._next_child_id(block_id)
                    child = SemanticBlock(
                        block_id=child_id,
                        face_ids=list(group_face_ids),
                        depth=block.depth + 1,
                        parent_id=block_id,
                        status=BlockStatus.PENDING,
                        notes=(
                            f"split from {block_id} "
                            f"(k={len(sub_groups)},mode={block.seed_decision.get('mode')})"
                        ),
                        segmat_done=block.segmat_done,
                    )
                    self._register_block(child)
                    block.children_ids.append(child_id)
                    next_queue.append(child_id)

                if progress_cb:
                    progress_cb(
                        "split",
                        {
                            "block_id": block_id,
                            "seed_count": seed_resp.seed_count,
                            "children": list(block.children_ids),
                            "mode": block.seed_decision.get("mode"),
                        },
                    )

            # ---------- VLM 全局自评 ----------
            if self.config.vlm_self_review:
                face_status = self._compute_face_status(n_faces)
                full_cnt = sum(1 for s in face_status if s == "full")
                pending_cnt = sum(1 for s in face_status if s == "pending")
                abandoned_cnt = sum(1 for s in face_status if s == "abandoned")

                progress_path = reviews_dir / f"iter_{iteration:02d}_progress.png"
                try:
                    render_progress_panel(
                        vertices,
                        faces,
                        face_status,
                        progress_path,
                        title=(
                            f"iter {iteration} progress  "
                            f"full={full_cnt}  pending={pending_cnt}  "
                            f"abandoned={abandoned_cnt}"
                        ),
                        image_size=self.config.image_size,
                        cols=self.config.panel_cols,
                    )
                except (OSError, ImportError, ValueError) as exc:
                    LOGGER.warning("iter %d 进度面板渲染失败: %s", iteration, exc)
                    progress_path = None

                review: Optional[ProgressReview] = None
                if progress_path is not None:
                    review = review_progress(
                        self.client,
                        progress_path,
                        self.config.semantic,
                        iter_done=iteration,
                        max_depth=self.config.max_depth,
                        full_faces=full_cnt,
                        pending_faces=pending_cnt,
                        abandoned_faces=abandoned_cnt,
                        total_faces=n_faces,
                        model=self.config.classification_model,
                    )

                review_entry = {
                    "iteration": iteration,
                    "progress_panel": (
                        str(progress_path) if progress_path else None
                    ),
                    "full_faces": full_cnt,
                    "pending_faces": pending_cnt,
                    "abandoned_faces": abandoned_cnt,
                    "queue_after_iter": list(next_queue),
                    "review": (
                        {
                            "should_terminate": review.should_terminate,
                            "red_assessment": review.red_assessment,
                            "yellow_assessment": review.yellow_assessment,
                            "red_covered_parts": review.red_covered_parts,
                            "missing_regions": review.missing_regions,
                            "extra_regions": review.extra_regions,
                            "confidence": review.confidence,
                            "reason": review.reason,
                            "elapsed_seconds": review.elapsed_seconds,
                            "error": review.error,
                        }
                        if review is not None
                        else None
                    ),
                }
                reviews_log.append(review_entry)
                self._dump_response(
                    reviews_dir, f"iter_{iteration:02d}_review", review_entry
                )
                if progress_cb:
                    progress_cb(
                        "review",
                        {
                            "iteration": iteration,
                            "should_terminate": (
                                review.should_terminate if review else False
                            ),
                            "red_assessment": (
                                review.red_assessment if review else ""
                            ),
                            "reason": review.reason if review else "",
                        },
                    )
                # over_claimed：review 觉得「红色覆盖明显超出语义」，把最大的几个
                # FULL 块降级回 PENDING + force_split，让下一轮跳过 classify 直接分裂。
                if (
                    review is not None
                    and not review.should_terminate
                    and (review.red_assessment or "").lower() == "over_claimed"
                ):
                    full_blocks = [
                        b for b in self.blocks.values()
                        if b.status == BlockStatus.FULL
                        and b.depth < self.config.max_depth
                        and b.size >= max(self.config.min_block_size, 1)
                    ]
                    full_blocks.sort(key=lambda b: -b.size)
                    demoted = 0
                    for b in full_blocks:
                        b.status = BlockStatus.PENDING
                        b.force_split = True
                        b.classification = None
                        b.notes = (
                            b.notes + f" | demoted_by_review(over_claimed,iter={iteration})"
                        ).strip(" |")
                        next_queue.append(b.block_id)
                        demoted += 1
                        if demoted >= 3:  # 一次最多降级 3 个，防止抖动
                            break
                    if demoted:
                        LOGGER.info(
                            "review over_claimed: demoted %d FULL blocks → force_split",
                            demoted,
                        )

                if review is not None and review.should_terminate and next_queue:
                    self._drain_pending_to_terminate(
                        next_queue, reason=review.red_assessment or "vlm_review"
                    )
                    terminated_by_vlm = True
                    next_queue = []

            # 每轮迭代末尾：先把这一轮的「单视角进度」存进 iterations/ 时间线，
            # 再刷新 live_report.html（包含 progress png + 时间线缩略图）
            try:
                self._save_iteration_progress(
                    out_dir, vertices, faces, iteration=iteration,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("iter %d 时间线 PNG 失败: %s", iteration, exc)
            try:
                self._update_live_report(
                    out_dir,
                    vertices,
                    faces,
                    iteration=iteration,
                    queue_after=next_queue,
                    finished=False,
                    extra_note=(
                        "VLM 自评提前终止" if terminated_by_vlm else ""
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("live report 更新失败: %s", exc)

            queue = next_queue
            if terminated_by_vlm:
                break

        result.num_iterations = iteration
        result.num_final_blocks = sum(
            1 for b in self.blocks.values() if b.status != BlockStatus.SPLIT
        )

        status_groups = gather_face_status(self.blocks, n_faces)
        result.full_face_ids = status_groups["full_face_ids"]
        result.none_face_ids = status_groups["none_face_ids"]
        result.pending_face_ids = status_groups["pending_face_ids"]

        # 导出 mask + tree
        mask_path = out_dir / "semantic_mask.json"
        with open(mask_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "semantic": self.config.semantic,
                    "num_faces": n_faces,
                    "num_full": len(result.full_face_ids),
                    "num_none": len(result.none_face_ids),
                    "num_pending": len(result.pending_face_ids),
                    "full_face_ids": result.full_face_ids,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        result.mask_json = mask_path

        tree_path = export_tree_json(
            self.blocks,
            out_dir / "block_tree.json",
            extra={
                "semantic": self.config.semantic,
                "num_iterations": iteration,
                "num_initial_blocks": result.num_initial_blocks,
                "num_final_blocks": result.num_final_blocks,
                "max_depth": self.config.max_depth,
                "max_seeds": self.config.max_seeds,
                "min_block_size": self.config.min_block_size,
                "vlm_self_review": self.config.vlm_self_review,
                "terminated_by_vlm": terminated_by_vlm,
                "accept_partial_at_max_depth": (
                    self.config.accept_partial_at_max_depth
                ),
                "accept_partial_min_faces": self.config.accept_partial_min_faces,
            },
        )
        result.tree_json = tree_path

        if responses_log:
            with open(out_dir / "responses_log.json", "w", encoding="utf-8") as f:
                json.dump(responses_log, f, indent=2, ensure_ascii=False)
        if reviews_log:
            with open(out_dir / "reviews_log.json", "w", encoding="utf-8") as f:
                json.dump(reviews_log, f, indent=2, ensure_ascii=False)
            result.extra["reviews_log"] = str(out_dir / "reviews_log.json")
        if terminated_by_vlm:
            result.extra["terminated_by_vlm"] = True

        # 最终语义掩码渲染
        try:
            semantic_panel = render_semantic_mask_panel(
                vertices,
                faces,
                result.full_face_ids,
                out_dir / "semantic_mask.png",
                title=(
                    f"VLM 语义 = 「{self.config.semantic}」  "
                    f"full={len(result.full_face_ids)}  "
                    f"none={len(result.none_face_ids)}  "
                    f"pending={len(result.pending_face_ids)}"
                ),
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
                semantic_rgb=ACCEPTED_RGB,
                rest_rgb=REJECTED_RGB,
            )
            result.semantic_panel = semantic_panel
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("语义掩码渲染失败: %s", exc)

        # 所有叶分块随机色总览图（每个叶 block 一个随机色，便于核对最终切分粒度）
        try:
            leaf_blocks = [
                b for b in self.blocks.values()
                if b.status != BlockStatus.SPLIT
            ]
            leaf_blocks.sort(key=lambda b: b.block_id)
            leaf_face_labels = np.full(n_faces, -1, dtype=np.int64)
            for idx, b in enumerate(leaf_blocks):
                leaf_face_labels[np.asarray(b.face_ids, dtype=np.int64)] = idx
            leaf_panel = out_dir / "leaf_blocks_overview.png"
            render_initial_segmentation_panel(
                vertices,
                faces,
                leaf_face_labels.tolist(),
                leaf_panel,
                title=(
                    f"叶分块总览  blocks={len(leaf_blocks)}  "
                    f"(每块一个独立随机色，便于核对最终切分粒度)"
                ),
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
            )
            result.extra["leaf_blocks_overview"] = str(leaf_panel)
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("叶分块总览渲染失败: %s", exc)

        # 最后一次 live_report：标记为「已完成」（去掉 meta refresh）
        try:
            self._update_live_report(
                out_dir,
                vertices,
                faces,
                iteration=iteration,
                queue_after=[],
                finished=True,
                extra_note=(
                    f"全部完成，共 {iteration} 轮迭代。"
                    f"详细 block 卡片见同目录 report.html。"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("最终 live report 失败: %s", exc)

        # 视角选择产物 / 迭代时间线落到 result，让 CLI / report 能直接消费
        if self._best_view is not None:
            result.best_view = self._best_view
            result.best_view_overview = self._best_view.overview_png
            result.best_view_thumbnails = self._best_view.thumbnails_panel
            bv_json = out_dir / "best_view.json"
            if bv_json.is_file():
                result.best_view_json = bv_json
        result.iter_progress_pngs = list(self._iter_progress_pngs)

        result.ok = True
        result.elapsed_seconds = time.perf_counter() - t0
        result.message = (
            f"完成 {iteration} 轮，初始 {result.num_initial_blocks} 块 → "
            f"最终 {result.num_final_blocks} 叶块；"
            f"full={len(result.full_face_ids)} faces, "
            f"none={len(result.none_face_ids)} faces"
        )
        return result

    # ---------- IO ----------
    def _dump_response(
        self, responses_dir: Path, name: str, payload: Dict[str, Any]
    ) -> None:
        if not self.config.save_responses:
            return
        try:
            with open(responses_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            LOGGER.warning("写响应日志失败 %s: %s", name, exc)


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------


def run_semantic_refinement(
    client: VLMClient,
    vertices: np.ndarray,
    faces: np.ndarray,
    initial_face_labels: Sequence[int],
    output_dir: str | Path,
    semantic: str,
    *,
    max_depth: int = 6,
    max_seeds: int = 6,
    min_block_size: int = 20,
    image_size: int = 512,
    panel_cols: int = 4,
    scalar_weight: float = 0.35,
    vertex_scalars: Optional[np.ndarray] = None,
    classification_model: Optional[str] = None,
    seed_model: Optional[str] = None,
    accept_partial_at_max_depth: bool = False,
    vlm_self_review: bool = False,
    vlm_grouped_split: bool = False,
    grouped_split_n_candidates: int = 8,
    grouped_split_max_groups: int = 4,
    split_strategy: str = "geodesic_fps",
    parallel_vlm_workers: int = 1,
    live_report: bool = True,
    live_report_refresh_seconds: int = 10,
    select_best_view: bool = True,
    view_select_model: Optional[str] = None,
    single_view_image_size: int = 720,
    progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> SemanticRefinementResult:
    """函数式封装。"""
    cfg = SemanticRefinementConfig(
        semantic=semantic,
        max_depth=max_depth,
        max_seeds=max_seeds,
        min_block_size=min_block_size,
        image_size=image_size,
        panel_cols=panel_cols,
        classification_model=classification_model,
        seed_model=seed_model,
        accept_partial_at_max_depth=accept_partial_at_max_depth,
        vlm_self_review=vlm_self_review,
        parallel_vlm_workers=parallel_vlm_workers,
        live_report=live_report,
        live_report_refresh_seconds=live_report_refresh_seconds,
        select_best_view=select_best_view,
        view_select_model=view_select_model,
        single_view_image_size=single_view_image_size,
    )
    pipeline = SemanticRefinementPipeline(client, cfg)
    return pipeline.run(
        vertices,
        faces,
        initial_face_labels,
        output_dir,
        vertex_scalars=vertex_scalars,
        progress_cb=progress_cb,
    )
