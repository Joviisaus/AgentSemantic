"""
点云全覆盖（Full-coverage）语义分割流水线（mesh 版 ``pipeline.full_coverage`` 的点云孪生版）。

与 mesh 版 ``FullCoveragePipeline`` 同构，区别仅在于：
    - 标签单元是 **point_id** 而不是 face_id；
    - **没有初始 face_labels**——整朵点云作为唯一的根 block（force_split 切第一刀），
      靠几何谱聚类逐层切分（迁移自 mesh spectral_cluster 的点云版 ``split_points_by_strategy``）；
    - **默认可选 NEO**：整朵点云一次性前馈全局 LBO 特征空间，子块 spectral 分裂切片复用；
    - 去掉 mesh 专属的凸包边界优化 / handle-loop / 基于面邻接的拓扑修复（点云无面邻接），
      收尾用 kNN 邻居多数表决把未标注点传播到已定语义。

核心逻辑与 mesh 版完全一致：
  先让 VLM 观察整点云并枚举一份**全包围、互不重叠**的语义清单（如人形 → 头/躯干/四肢），
  随后对每个分块 multi-label 判断它命中了清单里的哪些部位：
    - 1 个   → 直接归到该部位（叶子）
    - ≥2 个  → 分裂成 min(max_seeds, len(matched)) 个种子，继续
    - 0 个   → 强制归入父块/兜底语义
  直到每个叶分块只命中一个语义，或到 max_depth。不允许 background；枚举不区分左右。

产物（默认 output/<模型>/pointcloud_vlm_full/）：
    enumeration.json               VLM 枚举到的部位清单
    semantics_overview.png         枚举所用的整点云 12 视角图
    panels/  responses/            每个 block 的高亮 12 视角图 / VLM 原始响应
    point_semantic_labels.json     每个点最终的语义索引（全覆盖，无 background）
    full_coverage_mask.png         全覆盖最终结果（按部位上色）
    block_tree.json                完整 block 树（含 multi-label 决策）
    live_report.html               实时进度（每轮迭代刷新）
    iterations/iter_NN.png         单视角时间线
"""

from __future__ import annotations

import html as _html
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.blocks import BlockStatus, SemanticBlock, export_tree_json
from geometry.pointcloud_cluster import (
    DEFAULT_CURVATURE_WEIGHT,
    DEFAULT_KNN,
    DEFAULT_NORMAL_WEIGHT,
)
from geometry.pointcloud_partition import (
    VALID_POINT_STRATEGIES,
    ask_point_split_strategy,
    default_point_split_params,
    split_points_by_strategy,
)
from geometry.seed_split import SplitStrategyResponse
from rendering.heatmap_renderer import DEFAULT_OVERVIEW_VIEW, VIEW_GRID
from rendering.pointcloud_render import (
    compute_point_normals,
    render_pointcloud_highlight_panel,
    render_pointcloud_overview,
    render_pointcloud_segmentation_panel,
    render_pointcloud_status_view,
    render_pointcloud_thumbnails_panel,
)
from rendering.render_component import (
    FULL_COVERAGE_BG_RGB,
    FULL_COVERAGE_PALETTE,
    FULL_COVERAGE_PENDING_RGB,
)
from vlm.client import VLMClient
from vlm.full_coverage import (
    MultiLabelClassification,
    SemanticEnumeration,
    classify_multi_label,
    enumerate_semantics,
)
from vlm.view_selector import BestViewSelection, select_best_view

LOGGER = logging.getLogger("pipeline.pointcloud_full_coverage")


def _safe_palette_rgb(
    idx: int, palette: Sequence[Tuple[int, int, int]] = FULL_COVERAGE_PALETTE
) -> Tuple[int, int, int]:
    if not palette:
        return (180, 180, 180)
    return tuple(palette[idx % len(palette)])


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class PointCloudFullCoverageConfig:
    """点云全覆盖语义分割配置（与 mesh FullCoverageConfig 同名字段对齐）。"""

    # 用户提示，传给 VLM 枚举阶段（宜短；粗粒度短名全覆盖，不区分左右）
    enumerate_hint: str = ""
    # 显式提供候选语义集（跳过 VLM 枚举）。两者都给时优先用这个
    semantic_parts: Optional[List[str]] = None
    enumerate_min_parts: int = 2
    enumerate_max_parts: int = 12

    max_depth: int = 6
    min_block_size: int = 60           # 小于此点数的块不再切，直接归属
    # 无初始分块：整朵点云作为根，第一刀切多少块
    root_seed_count: int = 6
    # 单次分裂的最多种子数（VLM 即使返回更多 matched，也不超过这个数）
    max_seeds_per_split: int = 4

    image_size: int = 512
    panel_cols: int = 4
    point_size: Optional[float] = None
    classification_model: Optional[str] = None
    enumerate_model: Optional[str] = None

    # ---- 几何分块（迁移自 mesh spectral_cluster / 点云 split_points_by_strategy）----
    # 每个 partial 块分裂前，让 VLM 看高亮图，从点云策略里挑一种 + 给参数；auto=每块 VLM 决定。
    ask_split_strategy: bool = True
    split_strategy: str = "auto"        # auto | spectral | cvt | sdf | skeleton | geodesic_fps
    split_strategy_model: Optional[str] = None
    split_n_neighbors: int = DEFAULT_KNN
    split_enforce_connectivity: bool = True
    split_curvature_weight: float = DEFAULT_CURVATURE_WEIGHT
    split_convexity_weight: float = 0.0
    split_normal_weight: float = DEFAULT_NORMAL_WEIGHT

    # ---- NEO 后端：整朵点云一次性前馈全局 LBO 特征空间，spectral 子块分裂切片复用 ----
    use_neo: bool = False
    neo_ckpt: Optional[str] = None
    neo_python: Optional[str] = None
    neo_k: int = 0
    neo_device: str = "cpu"
    neo_no_mass: bool = False

    parallel_vlm_workers: int = 4
    save_responses: bool = True
    live_report: bool = True
    live_report_refresh_seconds: int = 10
    # 视角选择：进入主循环前让 VLM 选最佳视角，用作 live_report / 时间线的单视角展示
    select_best_view: bool = True
    view_select_model: Optional[str] = None
    view_select_semantic_hint: str = ""
    single_view_image_size: int = 720
    display_elev: float = DEFAULT_OVERVIEW_VIEW[0]
    display_azim: float = DEFAULT_OVERVIEW_VIEW[1]
    # 主流程结束后是否生成最终 report.html（复用 mesh full_coverage_report）
    generate_report: bool = True


@dataclass
class PointCloudFullCoverageResult:
    """点云全覆盖语义分割最终产物（point 单元）。"""

    ok: bool
    output_dir: Path
    semantic_parts: List[str] = field(default_factory=list)
    enumeration: Optional[SemanticEnumeration] = None
    # 长度 = n_points，每个点对应的部位索引（0..K-1；-2 = 迭代中 pending）
    point_semantic_index: List[int] = field(default_factory=list)
    # 每个部位 → 包含的 point_id 列表
    semantic_point_groups: Dict[str, List[int]] = field(default_factory=dict)
    pending_point_ids: List[int] = field(default_factory=list)
    num_points: int = 0
    num_initial_blocks: int = 0
    num_final_blocks: int = 0
    num_iterations: int = 0
    tree_json: Optional[Path] = None
    labels_json: Optional[Path] = None
    coverage_panel: Optional[Path] = None
    overview_panel: Optional[Path] = None
    initial_panel: Optional[Path] = None
    elapsed_seconds: float = 0.0
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
    best_view: Optional[BestViewSelection] = None
    best_view_overview: Optional[Path] = None
    iter_progress_pngs: List[Path] = field(default_factory=list)
    report_html: Optional[Path] = None


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class PointCloudFullCoveragePipeline:
    """
    点云全覆盖语义分割流水线：
        pipeline = PointCloudFullCoveragePipeline(client, config)
        result = pipeline.run(points, output_dir)

    与 ``PointCloudRefinementPipeline`` 的关键差异（同 mesh full_coverage vs pipeline）：
      - 不做 full/partial/none 三分类，而是 multi-label 命中数；
      - 命中数决定每个 block 要切的种子数（K = min(max_seeds, len(matched)))；
      - 命中数 = 1 的块自动归到对应语义；空匹配强制归入一项（无 background）。
    """

    def __init__(
        self, client: VLMClient, config: PointCloudFullCoverageConfig
    ) -> None:
        self.client = client
        self.config = config
        self.points: np.ndarray = np.zeros((0, 3))
        self.blocks: Dict[str, SemanticBlock] = {}
        self._counter_per_parent: Dict[str, int] = {}
        self.semantic_parts: List[str] = []
        self._part_index: Dict[str, int] = {}
        self._best_view: Optional[BestViewSelection] = None
        self._iter_progress_pngs: List[Path] = []
        # block_id -> assigned semantic name (set when matched == 1 / forced)
        self._assigned_semantic: Dict[str, str] = {}
        # NEO 全局特征空间（整朵点云一次性前馈得到，spectral 子块分裂复用）
        self._neo_eigvecs: Optional[np.ndarray] = None
        # 整朵点云每点 PCA 法向 (N, 3)，供所有渲染做逐点法向 shader
        self._point_normals: Optional[np.ndarray] = None

    def _resolve_vlm_model(self, explicit: Optional[str]) -> str:
        from vlm.config import resolve_model_name

        return resolve_model_name(explicit, self.client.config)

    # ---------- 展示视角 ----------
    def _display_view(self) -> Tuple[float, float]:
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

    def _inherit_parent_semantic(self, block: SemanticBlock) -> Optional[str]:
        pid = block.parent_id
        if not pid:
            return None
        sem = self._assigned_semantic.get(pid)
        if sem and sem in self._part_index:
            return sem
        return None

    def _semantic_from_classification(
        self, cls: Optional[MultiLabelClassification]
    ) -> Optional[str]:
        if cls is None:
            return None
        for raw in list(cls.matched_parts) + ([cls.dominant] if cls.dominant else []):
            if raw in self._part_index:
                return raw
        return None

    @staticmethod
    def _matched_parts_from_cls_dict(
        cls_dict: Optional[Dict[str, Any]],
    ) -> List[str]:
        if not cls_dict:
            return []
        out: List[str] = []
        seen: Dict[str, int] = {}
        for raw in cls_dict.get("matched_parts") or []:
            s = str(raw).strip()
            if s and s not in seen:
                seen[s] = 1
                out.append(s)
        return out

    def _block_matched_parts(self, block: SemanticBlock) -> List[str]:
        return self._matched_parts_from_cls_dict(block.classification)

    def _parent_matched_for_block(
        self, block: Optional[SemanticBlock]
    ) -> Optional[List[str]]:
        if block is None or not block.parent_id:
            return None
        parent = self.blocks.get(block.parent_id)
        if parent is None:
            return None
        parts = self._block_matched_parts(parent)
        return list(parts) if parts else None

    def _fallback_semantic(
        self,
        cls: Optional[MultiLabelClassification] = None,
        *,
        block: Optional[SemanticBlock] = None,
    ) -> str:
        inherited = self._inherit_parent_semantic(block) if block else None
        if inherited:
            return inherited
        from_cls = self._semantic_from_classification(cls)
        if from_cls:
            return from_cls
        return self.semantic_parts[0]

    def _force_assign_block(
        self, block: SemanticBlock, sem: str, note: str
    ) -> None:
        if sem not in self._part_index:
            sem = self._fallback_semantic(block=block)
        block.status = BlockStatus.FULL
        self._assigned_semantic[block.block_id] = sem
        block.notes = (block.notes + f" | {note}").strip(" |")

    # ---------- NEO 全局特征空间 ----------
    def _compute_neo_eigenspace(self) -> None:
        from geometry.neo_backend import neo_eigenspace

        LOGGER.info(
            "调用 NEO 前馈全局特征空间（整朵点云 %d 点，一次性）...", len(self.points)
        )
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
            "NEO 全局特征空间就绪: eigvecs=%s，耗时 %.1fs",
            self._neo_eigvecs.shape, time.perf_counter() - t,
        )

    # ---------- 渲染 ----------
    def _render_block_panel(
        self, block: SemanticBlock, panel_dir: Path
    ) -> Path:
        # 标题不写 semantic，避免给 VLM 暗示（确认偏差）
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
            normals=self._point_normals,
        )

    def _dump_response(
        self, responses_dir: Path, block_id: str, payload: Dict[str, Any]
    ) -> None:
        if not self.config.save_responses:
            return
        try:
            (responses_dir / f"block_{block_id}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            LOGGER.warning("写 response 失败 %s: %s", block_id, exc)

    # ---------- 分裂策略 ----------
    def _default_strategy(self) -> str:
        s = str(self.config.split_strategy or "auto").strip().lower()
        if s in VALID_POINT_STRATEGIES:
            return s
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

    def _batch_ask_split_strategy(
        self,
        tasks: Sequence[Tuple[str, Path, Dict[str, Any], int]],
    ) -> Dict[str, SplitStrategyResponse]:
        """并行批量让 VLM 选分块策略 + 参数；强制策略/关闭时直接返回默认。"""
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

        def _do(
            bid: str, panel: Path, cls: Dict[str, Any], seed_n: int
        ) -> Tuple[str, SplitStrategyResponse]:
            block = self.blocks[bid]
            matched = self._matched_parts_from_cls_dict(cls)
            try:
                resp = ask_point_split_strategy(
                    self.client,
                    panel,
                    semantic_context="、".join(matched) or "全覆盖语义细分",
                    seed_count=seed_n,
                    covered_parts=matched,
                    semantic_parts=self.semantic_parts,
                    matched_parts=matched,
                    block_points=block.size,
                    split_stage="main",
                    strategy_hint=str(self.config.split_strategy or "auto"),
                    fallback="spectral",
                    model=self._resolve_vlm_model(
                        self.config.split_strategy_model
                        or self.config.classification_model
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                resp = SplitStrategyResponse(strategy="spectral", error=str(exc))
            return bid, resp

        workers = max(1, int(self.config.parallel_vlm_workers))
        if workers > 1 and len(tasks) > 1:
            n = min(workers, len(tasks))
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
                        LOGGER.warning("ask_split_strategy failed %s: %s", bid, exc)
        else:
            for bid, panel, cls, seed_n in tasks:
                try:
                    _, resp = _do(bid, panel, cls, seed_n)
                    results[bid] = resp
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("ask_split_strategy failed %s: %s", bid, exc)
        return results

    def _split_block(
        self,
        block: SemanticBlock,
        seed_count: int,
        strat_resp: Optional[SplitStrategyResponse] = None,
    ) -> List[List[int]]:
        """按策略几何分裂当前 block，返回子块 point_id 组。NEO 仅在 spectral 时介入。"""
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
            curvature_weight=params.get(
                "curvature_weight", self.config.split_curvature_weight
            ),
            concavity_weight=params.get(
                "concavity_weight", self.config.split_convexity_weight
            ),
            normal_weight=float(
                params.get("normal_weight", self.config.split_normal_weight)
            ),
            lloyd_iter=int(params.get("lloyd_iter", 0)),
            num_rays=int(params.get("num_rays", 16)),
            cone_angle_deg=float(params.get("cone_angle_deg", 60.0)),
            contraction_iters=int(params.get("contraction_iters", 5)),
            n_neighbors=self.config.split_n_neighbors,
            enforce_connectivity=self.config.split_enforce_connectivity,
            precomputed_eigvecs=precomputed,
            random_state=0,
        )
        neo_tag = "+neo" if precomputed is not None else ""
        groups = [list(g) for g in sub.sub_point_groups if g]
        if groups:
            block.seed_decision = {
                "seed_count": seed_count,
                "strategy": strategy,
                "strategy_tag": f"{strategy}{neo_tag}",
                "strategy_reason": resolved.reason,
                "strategy_params": resolved.params,
                "strategy_error": resolved.error,
                "debug": sub.debug.get("strategy", "") if sub.debug else "",
            }
        return groups

    # ---------- 语义枚举 ----------
    def _resolve_semantic_parts(
        self, out_dir: Path
    ) -> Tuple[List[str], Optional[SemanticEnumeration], Optional[Path]]:
        overview_path: Optional[Path] = out_dir / "semantics_overview.png"
        try:
            render_pointcloud_overview(
                self.points,
                overview_path,
                elev=self.config.display_elev,
                azim=self.config.display_azim,
                image_size=self.config.single_view_image_size,
                background="white",
                point_size=self.config.point_size,
                normals=self._point_normals,
            )
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("整点云 overview 渲染失败: %s", exc)
            overview_path = None

        # 枚举阶段用 12 视角拼图（信息更全）；overview 单视角仅作备用镇页图。
        enum_panel: Optional[Path] = out_dir / "semantics_overview_panel.png"
        try:
            render_pointcloud_segmentation_panel(
                self.points,
                [0] * len(self.points),
                enum_panel,
                title="整点云 12 视角（用于全覆盖语义枚举）",
                image_size=(self.config.image_size, self.config.image_size),
                cols=self.config.panel_cols,
                point_size=self.config.point_size,
                normals=self._point_normals,
            )
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("枚举拼图渲染失败: %s", exc)
            enum_panel = None

        if self.config.semantic_parts:
            seen: Dict[str, int] = {}
            unique: List[str] = []
            for x in self.config.semantic_parts:
                p = str(x).strip()
                if p and p not in seen:
                    seen[p] = 1
                    unique.append(p)
            LOGGER.info("使用用户提供的 semantic_parts: %s", unique)
            return unique, None, overview_path

        panel_for_enum = enum_panel or overview_path
        if panel_for_enum is None:
            raise RuntimeError(
                "无法渲染 overview，且未提供 semantic_parts，无法继续。"
            )

        LOGGER.info("调用 VLM 枚举全覆盖语义清单 ...")
        enumeration = enumerate_semantics(
            self.client,
            panel_for_enum,
            hint=self.config.enumerate_hint,
            model=self._resolve_vlm_model(self.config.enumerate_model),
            min_parts=self.config.enumerate_min_parts,
            max_parts=self.config.enumerate_max_parts,
        )
        try:
            (out_dir / "enumeration.json").write_text(
                json.dumps(enumeration.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.warning("写 enumeration.json 失败: %s", exc)

        if not enumeration.ok:
            raise RuntimeError(
                f"VLM 枚举语义清单失败: {enumeration.error or '未知错误'}"
            )
        LOGGER.info(
            "VLM 枚举到 %d 项语义部位: %s", len(enumeration.parts), enumeration.parts
        )
        return list(enumeration.parts), enumeration, overview_path

    # ---------- 视角选择 ----------
    def _select_view_for_display(self, out_dir: Path) -> Optional[BestViewSelection]:
        hint = (
            self.config.view_select_semantic_hint.strip()
            if self.config.view_select_semantic_hint
            else (
                "整体结构"
                + (
                    f"（{', '.join(self.semantic_parts[:8])}）"
                    if self.semantic_parts
                    else ""
                )
            )
        )

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
            bv = _fallback("select_best_view=False，直接用默认 overview 视角")
        else:
            thumb_path = out_dir / "view_thumbnails.png"
            try:
                render_pointcloud_thumbnails_panel(
                    self.points,
                    thumb_path,
                    title=f"候选视角（全覆盖：{hint}）",
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
                    self.client, thumb_path, hint,
                    model=self._resolve_vlm_model(self.config.view_select_model),
                )
                bv.thumbnails_panel = thumb_path

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
                        "fallback": bv.fallback,
                        "confidence": bv.confidence,
                        "reason": bv.reason,
                        "provider": bv.provider,
                        "model": bv.model,
                        "elapsed_seconds": bv.elapsed_seconds,
                        "error": bv.error,
                        "thumbnails_panel": (
                            str(bv.thumbnails_panel) if bv.thumbnails_panel else None
                        ),
                        "overview_png": (
                            str(bv.overview_png) if bv.overview_png else None
                        ),
                        "semantic_hint": hint,
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

    # ---------- 并行 multi-label ----------
    def _batch_multi_label(
        self, tasks: Sequence[Tuple[str, Path]]
    ) -> Dict[str, MultiLabelClassification]:
        results: Dict[str, MultiLabelClassification] = {}
        if not tasks:
            return results

        def _do(bid: str, panel: Path) -> Tuple[str, MultiLabelClassification]:
            block = self.blocks.get(bid)
            parent_matched = self._parent_matched_for_block(block)
            depth = int(block.depth) if block is not None else 0
            at_max = depth >= int(self.config.max_depth)
            return bid, classify_multi_label(
                self.client,
                panel,
                self.semantic_parts,
                block_id=bid,
                parent_id=(block.parent_id if block else "") or "",
                parent_matched=parent_matched,
                at_max_depth=at_max,
                current_depth=depth,
                max_depth=int(self.config.max_depth),
                model=self._resolve_vlm_model(self.config.classification_model),
            )

        workers = max(1, int(self.config.parallel_vlm_workers))
        if workers > 1 and len(tasks) > 1:
            n = min(workers, len(tasks))
            LOGGER.info("parallel multi-label: %d tasks, %d workers", len(tasks), n)
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = {ex.submit(_do, bid, panel): bid for bid, panel in tasks}
                for fut in as_completed(futs):
                    bid = futs[fut]
                    try:
                        _, cls = fut.result()
                        results[bid] = cls
                    except (RuntimeError, ValueError, OSError) as exc:
                        LOGGER.warning("multi-label failed %s: %s", bid, exc)
        else:
            for bid, panel in tasks:
                try:
                    _, cls = _do(bid, panel)
                    results[bid] = cls
                except (RuntimeError, ValueError, OSError) as exc:
                    LOGGER.warning("multi-label failed %s: %s", bid, exc)
        return results

    def _write_block_classification(
        self,
        block: SemanticBlock,
        cls: MultiLabelClassification,
        responses_dir: Path,
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        block.classification = {
            "matched_parts": list(cls.matched_parts),
            "is_background": bool(cls.is_background),
            "dominant": cls.dominant,
            "confidences": dict(cls.confidences),
            "notes": cls.notes,
            "provider": cls.provider,
            "model": cls.model,
            "elapsed_seconds": cls.elapsed_seconds,
            "error": cls.error,
        }
        if block.parent_id:
            parent_matched = self._parent_matched_for_block(block)
            if parent_matched:
                block.classification["parent_id"] = block.parent_id
                block.classification["parent_matched"] = list(parent_matched)
        self._dump_response(
            responses_dir, f"{block.block_id}_classify", block.classification
        )
        if progress_cb:
            progress_cb(
                "classified",
                {
                    "block_id": block.block_id,
                    "matched": list(cls.matched_parts),
                    "is_background": bool(cls.is_background),
                },
            )

    # ---------- 点语义索引 / 传播 ----------
    def _compute_point_semantic_index(self, n_points: int) -> np.ndarray:
        idx = np.full(n_points, -2, dtype=np.int64)
        for block in self.blocks.values():
            if block.status == BlockStatus.SPLIT:
                continue
            sem = self._assigned_semantic.get(block.block_id)
            val = self._part_index[sem] if (sem in self._part_index) else -2
            arr = np.asarray(block.face_ids, dtype=np.int64)
            if arr.size:
                mask = (arr >= 0) & (arr < n_points)
                idx[arr[mask]] = val
        return idx

    def _propagate_unlabeled_point_indices(self, idx: np.ndarray) -> np.ndarray:
        """把仍为 -1/-2 的点按 kNN 邻居多数表决传播到已标注语义；仍孤立则归入 0。"""
        out = idx.copy()
        if not np.any(out < 0):
            return out
        try:
            from pipeline.geometry_postprocess import build_neighbor_index

            nbr = build_neighbor_index(
                self.points, n_neighbors=self.config.split_n_neighbors
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("kNN 传播建图失败，未标注点直接归入 0: %s", exc)
            out[out < 0] = 0
            return out

        n_sem = len(self.semantic_parts)
        for _ in range(50):
            unlabeled = np.flatnonzero(out < 0)
            if unlabeled.size == 0:
                break
            changed = False
            for pid in unlabeled:
                nbrs = out[nbr[pid]]
                nbrs = nbrs[(nbrs >= 0) & (nbrs < n_sem)]
                if nbrs.size == 0:
                    continue
                vals, counts = np.unique(nbrs, return_counts=True)
                out[pid] = int(vals[int(np.argmax(counts))])
                changed = True
            if not changed:
                break
        out[(out < 0) | (out >= n_sem)] = 0
        return out

    def _finalize_block_assignments(self) -> None:
        for block in self.blocks.values():
            if block.status == BlockStatus.SPLIT:
                continue
            if block.block_id in self._assigned_semantic:
                if block.status != BlockStatus.FULL:
                    block.status = BlockStatus.FULL
                continue
            cls_dict = block.classification or {}
            dom = str(cls_dict.get("dominant") or "").strip()
            matched = self._matched_parts_from_cls_dict(cls_dict)
            sem: Optional[str] = None
            if dom in self._part_index:
                sem = dom
            else:
                for m in matched:
                    if m in self._part_index:
                        sem = m
                        break
            if sem is None:
                sem = self._fallback_semantic(block=block)
            self._force_assign_block(
                block, sem, f"finalize_assign({block.status.value})"
            )

    # ---------- 状态图 / live 报告 ----------
    def _color_map_for_status(self) -> Dict[str, Tuple[int, int, int]]:
        cmap: Dict[str, Tuple[int, int, int]] = {
            "bg": FULL_COVERAGE_BG_RGB,
            "pending": FULL_COVERAGE_PENDING_RGB,
            "none": FULL_COVERAGE_BG_RGB,
        }
        for k, _name in enumerate(self.semantic_parts):
            cmap[f"sem_{k}"] = _safe_palette_rgb(k)
        return cmap

    def _point_index_to_status_strings(self, idx: np.ndarray) -> List[str]:
        out: List[str] = []
        for v in idx.tolist():
            if v == -2:
                out.append("pending")
            elif v < 0:
                out.append("bg")
            else:
                out.append(f"sem_{int(v)}")
        return out

    def _save_iteration_progress(
        self, out_dir: Path, *, iteration: int
    ) -> Optional[Path]:
        n_points = len(self.points)
        idx = self._compute_point_semantic_index(n_points)
        status = self._point_index_to_status_strings(idx)
        it_dir = out_dir / "iterations"
        it_dir.mkdir(parents=True, exist_ok=True)
        png = it_dir / f"iter_{iteration:02d}.png"
        elev, azim = self._display_view()
        try:
            render_pointcloud_status_view(
                self.points, status, png,
                elev=elev, azim=azim,
                image_size=self.config.single_view_image_size,
                color_map=self._color_map_for_status(),
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
        idx = self._compute_point_semantic_index(n_points)
        status = self._point_index_to_status_strings(idx)
        live_png = out_dir / "live_progress.png"
        rendered = False
        elev, azim = self._display_view()
        try:
            render_pointcloud_status_view(
                self.points, status, live_png,
                elev=elev, azim=azim,
                image_size=self.config.single_view_image_size,
                color_map=self._color_map_for_status(),
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
        assigned_pts = int(np.sum(idx >= 0))
        pending_pts = int(np.sum(idx == -2))

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

        legend_items: List[str] = []
        for k, name in enumerate(self.semantic_parts):
            r, g, b = _safe_palette_rgb(k)
            legend_items.append(
                f'<span><i style="background:rgb({r},{g},{b})"></i>{esc(name)}</span>'
            )
        legend_items.append(
            f'<span><i style="background:rgb{tuple(FULL_COVERAGE_PENDING_RGB)}"></i>pending（待细分）</span>'
        )
        legend_html = "".join(legend_items)

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
                    padding:14px 18px; min-width:160px; }
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
        """

        body = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">{refresh_meta}
<title>点云全覆盖 live · iter {iteration}</title>
<style>{css}</style></head><body>
<h1>点云全覆盖语义分割 · 实时进度 {status_tag}</h1>
<div class="ts">更新时间: {esc(ts)} · 部位数: <b>{len(self.semantic_parts)}</b>
 · iteration: <b>{iteration}</b> · 下轮队列: <b>{len(queue_after)}</b></div>
<div class="legend">{legend_html}</div>
<div class="row">
  <div class="card"><b>assigned points</b><span class="v">{assigned_pts}</span></div>
  <div class="card"><b>pending points</b><span class="v">{pending_pts}</span></div>
  <div class="card"><b>total points</b><span class="v">{n_points}</span></div>
</div>
<div class="row">
  <div class="card"><b>full/assigned blocks</b><span class="v">{block_counts.get('full', 0)}</span></div>
  <div class="card"><b>pending blocks</b><span class="v">{block_counts.get('pending', 0)}</span></div>
  <div class="card"><b>split blocks</b><span class="v">{block_counts.get('split', 0)}</span></div>
</div>
{extra_html}
<h2>当前进度图（单视角 · 透明背景 · 按部位上色）</h2>
{img_html}
{timeline_html}
</body></html>
"""
        try:
            (out_dir / "live_report.html").write_text(body, encoding="utf-8")
        except OSError as exc:
            LOGGER.warning("live report 写文件失败: %s", exc)

    # ---------- 主循环 ----------
    def run(
        self,
        points: np.ndarray,
        output_dir: str | Path,
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> PointCloudFullCoverageResult:
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

        result = PointCloudFullCoverageResult(
            ok=False, output_dir=out_dir, num_points=n_points
        )
        if n_points < max(self.config.min_block_size, 4):
            result.message = f"点数过少 ({n_points})"
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        # 逐点 PCA 法向（供所有渲染做明暗 shader）
        try:
            self._point_normals = compute_point_normals(self.points)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("逐点法向计算失败，渲染退回无 shader: %s", exc)
            self._point_normals = None

        # ---- NEO 全局特征空间（可选）----
        self._neo_eigvecs = None
        if self.config.use_neo:
            try:
                self._compute_neo_eigenspace()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("NEO 特征空间失败，回退 scipy 现算: %s", exc)
                self._neo_eigvecs = None
            result.extra["neo_used"] = self._neo_eigvecs is not None

        # ---- Step 1: 枚举语义部位 ----
        try:
            parts, enumeration, overview_path = self._resolve_semantic_parts(out_dir)
        except RuntimeError as exc:
            result.message = f"语义枚举失败: {exc}"
            result.elapsed_seconds = time.perf_counter() - t0
            return result
        self.semantic_parts = parts
        self._part_index = {n: k for k, n in enumerate(parts)}
        result.semantic_parts = list(parts)
        result.enumeration = enumeration
        result.overview_panel = overview_path

        # ---- Step 2: 视角选择 ----
        self._iter_progress_pngs = []
        try:
            self._best_view = self._select_view_for_display(out_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("视角选择失败，使用默认视角: %s", exc)
            self._best_view = None
        if self._best_view is not None:
            result.best_view = self._best_view
            result.best_view_overview = self._best_view.overview_png

        # ---- Step 3: 初始分块（整朵点云作为根，force_split 切第一刀）----
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
        result.num_initial_blocks = 1

        queue: List[str] = ["C0"]
        iteration = 0

        try:
            self._update_live_report(
                out_dir, iteration=0, queue_after=queue, finished=False,
                extra_note="初始化中：整朵点云作为根，尚未开始 VLM multi-label",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("初始 live report 失败: %s", exc)

        # ---- Step 4: 主迭代 ----
        while queue:
            iteration += 1
            LOGGER.info("== iteration %d, queue=%d ==", iteration, len(queue))
            next_queue: List[str] = []

            # --- Pass A（串行 CPU）：too_small / 渲染 / 根 force_split 立即切 ---
            iter_panel: Dict[str, Path] = {}
            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue

                at_max_depth = block.depth >= self.config.max_depth

                # 根块 force_split：不调 VLM，直接几何切第一刀
                if block.force_split and not at_max_depth:
                    seed_n = self.config.root_seed_count
                    LOGGER.info("block %s force_split → 立即几何分块 seeds=%d", block_id, seed_n)
                    groups = self._split_block(block, seed_n)
                    if len(groups) >= 2:
                        block.status = BlockStatus.SPLIT
                        block.seed_count = seed_n
                        for group in groups:
                            child_id = self._next_child_id(block_id)
                            child = SemanticBlock(
                                block_id=child_id,
                                face_ids=list(group),
                                depth=block.depth + 1,
                                parent_id=block_id,
                                status=BlockStatus.PENDING,
                                notes=f"root_force_split ({len(groups)} children)",
                                segmat_done=True,
                            )
                            self._register_block(child)
                            block.children_ids.append(child_id)
                            next_queue.append(child_id)
                        if progress_cb:
                            progress_cb("split", {
                                "block_id": block_id, "seed_count": seed_n,
                                "children": list(block.children_ids),
                            })
                        continue
                    block.force_split = False

                if block.size < self.config.min_block_size:
                    try:
                        panel_path = self._render_block_panel(block, panel_dir)
                        block.panel_path = str(panel_path)
                        iter_panel[block_id] = panel_path
                    except (OSError, ImportError, ValueError) as exc:
                        LOGGER.warning("block %s too_small 渲染失败: %s", block_id, exc)
                        sem = self._fallback_semantic(block=block)
                        self._force_assign_block(block, sem, f"too_small_render_failed→{sem}")
                    continue

                try:
                    panel_path = self._render_block_panel(block, panel_dir)
                    block.panel_path = str(panel_path)
                    iter_panel[block_id] = panel_path
                except (OSError, ImportError, ValueError) as exc:
                    LOGGER.error("渲染 block %s 失败: %s", block_id, exc)
                    sem = self._fallback_semantic(block=block)
                    self._force_assign_block(block, sem, f"render_failed→{sem}: {exc}")

            # --- Pass B（并行 VLM）：批量 multi-label ---
            cls_cache = self._batch_multi_label(list(iter_panel.items()))
            for block_id, panel_path in iter_panel.items():
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                cls = cls_cache.get(block_id)
                if cls is None:
                    try:
                        parent_matched = self._parent_matched_for_block(block)
                        at_max = block.depth >= int(self.config.max_depth)
                        cls = classify_multi_label(
                            self.client, panel_path, self.semantic_parts,
                            block_id=block_id,
                            parent_id=block.parent_id or "",
                            parent_matched=parent_matched,
                            at_max_depth=at_max,
                            current_depth=int(block.depth),
                            max_depth=int(self.config.max_depth),
                            model=self._resolve_vlm_model(self.config.classification_model),
                        )
                    except (RuntimeError, ValueError) as exc:
                        LOGGER.warning("fallback multi-label 失败 (%s): %s", block_id, exc)
                        cls = MultiLabelClassification(error=str(exc))
                self._write_block_classification(
                    block, cls, responses_dir, progress_cb=progress_cb
                )

            # --- Pass C（串行决策）：读 classification，收集 partial 待切块 ---
            partial_ids: List[str] = []
            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                if block_id not in iter_panel:
                    continue

                cls_dict = block.classification or {}
                matched_parts = self._matched_parts_from_cls_dict(cls_dict)
                dominant = str(cls_dict.get("dominant") or "").strip()
                if not matched_parts and dominant:
                    matched_parts = [dominant]
                n_matched = len(matched_parts)

                if n_matched == 0:
                    sem = self._fallback_semantic(block=block)
                    self._force_assign_block(block, sem, f"forced_assign(empty→{sem})")
                    continue

                if n_matched == 1:
                    sem = matched_parts[0]
                    if sem not in self._part_index:
                        sem = self._fallback_semantic(block=block)
                        self._force_assign_block(block, sem, f"forced_assign(unknown→{sem})")
                        continue
                    block.status = BlockStatus.FULL
                    self._assigned_semantic[block_id] = sem
                    block.notes = (block.notes + f" | assigned={sem}").strip(" |")
                    continue

                # matched ≥ 2：需要继续切
                if block.depth >= self.config.max_depth:
                    dom = dominant if dominant in self._part_index else matched_parts[0]
                    if dom not in self._part_index:
                        dom = self._fallback_semantic(block=block)
                    self._force_assign_block(
                        block, dom, f"max_depth→{dom} (matched={matched_parts})"
                    )
                    if progress_cb:
                        progress_cb("max_depth", {"block_id": block_id, "dominant": dom})
                    continue

                if block.size < self.config.min_block_size:
                    dom = dominant if dominant in self._part_index else matched_parts[0]
                    self._force_assign_block(block, dom, f"too_small_multi→{dom}")
                    continue

                partial_ids.append(block_id)

            # --- Pass D：批量选策略 + 几何分裂 ---
            strat_tasks: List[Tuple[str, Path, Dict[str, Any], int]] = []
            seed_by_id: Dict[str, int] = {}
            for block_id in partial_ids:
                block = self.blocks[block_id]
                cls_dict = block.classification or {}
                matched_parts = self._matched_parts_from_cls_dict(cls_dict)
                seed_count = max(
                    2, min(int(self.config.max_seeds_per_split), len(matched_parts))
                )
                seed_by_id[block_id] = seed_count
                strat_tasks.append((block_id, iter_panel[block_id], cls_dict, seed_count))
            strat_cache = self._batch_ask_split_strategy(strat_tasks)

            for block_id in partial_ids:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                cls_dict = block.classification or {}
                matched_parts = self._matched_parts_from_cls_dict(cls_dict)
                dominant = str(cls_dict.get("dominant") or "").strip()
                seed_count = seed_by_id[block_id]
                strat_resp = strat_cache.get(block_id) or self._resolve_strat_resp(None)

                if progress_cb:
                    progress_cb("split_start", {
                        "block_id": block_id, "faces": block.size,
                        "matched": list(matched_parts), "seed_count": seed_count,
                    })
                groups = self._split_block(block, seed_count, strat_resp)
                if len(groups) < 2:
                    dom = dominant if dominant in self._part_index else matched_parts[0]
                    if dom in self._part_index:
                        block.status = BlockStatus.FULL
                        self._assigned_semantic[block_id] = dom
                        block.notes = (
                            block.notes + f" | split_failed → assigned={dom}"
                        ).strip(" |")
                    else:
                        dom = self._fallback_semantic(block=block)
                        self._force_assign_block(block, dom, f"split_failed→{dom}")
                    if progress_cb:
                        progress_cb("split_failed", {"block_id": block_id, "dominant": dom})
                    continue

                block.status = BlockStatus.SPLIT
                block.force_split = False
                block.seed_count = seed_count
                for group in groups:
                    child_id = self._next_child_id(block_id)
                    child = SemanticBlock(
                        block_id=child_id,
                        face_ids=list(group),
                        depth=block.depth + 1,
                        parent_id=block_id,
                        status=BlockStatus.PENDING,
                        notes=f"split from {block_id} (k={len(groups)}, matched={matched_parts})",
                        segmat_done=True,
                    )
                    self._register_block(child)
                    block.children_ids.append(child_id)
                    next_queue.append(child_id)
                if progress_cb:
                    progress_cb("split", {
                        "block_id": block_id, "seed_count": seed_count,
                        "matched": list(matched_parts),
                        "children": list(block.children_ids),
                        "strategy": strat_resp.strategy,
                    })

            # 每轮末尾：时间线 + live report
            try:
                self._save_iteration_progress(out_dir, iteration=iteration)
                self._update_live_report(
                    out_dir, iteration=iteration, queue_after=next_queue, finished=False,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("iter %d 进度刷新失败: %s", iteration, exc)

            if progress_cb:
                progress_cb("iteration_end", {
                    "iteration": iteration, "queue_after": len(next_queue),
                })
            queue = next_queue

        LOGGER.info("主循环结束（%d 轮），进入收尾 ...", iteration)
        if progress_cb:
            progress_cb("main_loop_done", {
                "iterations": iteration, "num_blocks": len(self.blocks),
            })
        self._finalize_block_assignments()

        # ---- Step 5: 产出 ----
        result.num_iterations = iteration
        result.num_final_blocks = sum(
            1 for b in self.blocks.values() if b.status != BlockStatus.SPLIT
        )

        point_idx = self._compute_point_semantic_index(n_points)
        point_idx = self._propagate_unlabeled_point_indices(point_idx)
        result.point_semantic_index = point_idx.tolist()

        sem_groups: Dict[str, List[int]] = {n: [] for n in self.semantic_parts}
        pending_ids: List[int] = []
        for pid, v in enumerate(point_idx.tolist()):
            if v == -2:
                pending_ids.append(pid)
            elif 0 <= v < len(self.semantic_parts):
                sem_groups[self.semantic_parts[v]].append(pid)
        result.semantic_point_groups = sem_groups
        result.pending_point_ids = pending_ids

        labels_json = out_dir / "point_semantic_labels.json"
        try:
            labels_json.write_text(
                json.dumps(
                    {
                        "unit": "point",
                        "num_points": n_points,
                        "semantic_parts": list(self.semantic_parts),
                        "point_semantic_index": result.point_semantic_index,
                        "num_background": 0,
                        "num_pending": len(pending_ids),
                        "semantic_point_counts": {
                            n: len(v) for n, v in sem_groups.items()
                        },
                    },
                    ensure_ascii=False, indent=2,
                ),
                encoding="utf-8",
            )
            result.labels_json = labels_json
        except OSError as exc:
            LOGGER.warning("写 point_semantic_labels.json 失败: %s", exc)

        # 全覆盖结果图（按部位上色）
        coverage_panel = out_dir / "full_coverage_mask.png"
        try:
            render_pointcloud_segmentation_panel(
                self.points,
                result.point_semantic_index,
                coverage_panel,
                title="VLM 点云全覆盖语义分割（按部位上色）",
                image_size=(self.config.image_size, self.config.image_size),
                cols=self.config.panel_cols,
                point_size=self.config.point_size,
                normals=self._point_normals,
            )
            result.coverage_panel = coverage_panel
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("全覆盖结果图渲染失败: %s", exc)

        # 叶分块总览
        try:
            leaf_blocks = [
                b for b in self.blocks.values() if b.status != BlockStatus.SPLIT
            ]
            leaf_blocks.sort(key=lambda b: b.block_id)
            leaf_labels = np.full(n_points, -1, dtype=np.int64)
            for k, b in enumerate(leaf_blocks):
                arr = np.asarray(b.face_ids, dtype=np.int64)
                if arr.size:
                    mask = (arr >= 0) & (arr < n_points)
                    leaf_labels[arr[mask]] = k
            leaf_panel = out_dir / "leaf_blocks_overview.png"
            render_pointcloud_segmentation_panel(
                self.points,
                leaf_labels.tolist(),
                leaf_panel,
                title=f"叶分块总览 blocks={len(leaf_blocks)}（每块一个随机色）",
                image_size=(self.config.image_size, self.config.image_size),
                cols=self.config.panel_cols,
                point_size=self.config.point_size,
                normals=self._point_normals,
            )
            result.extra["leaf_blocks_overview"] = str(leaf_panel)
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("叶分块总览渲染失败: %s", exc)

        # block_tree.json
        tree_extra: Dict[str, Any] = {
            "mode": "full_coverage",
            "unit": "point",
            "num_points": n_points,
            "semantic_parts": list(self.semantic_parts),
            "num_iterations": iteration,
            "num_initial_blocks": result.num_initial_blocks,
            "num_final_blocks": result.num_final_blocks,
            "max_depth": self.config.max_depth,
            "min_block_size": self.config.min_block_size,
            "max_seeds_per_split": self.config.max_seeds_per_split,
            "assigned_semantic": dict(self._assigned_semantic),
            "pending_point_count": len(pending_ids),
            "semantic_point_counts": {n: len(v) for n, v in sem_groups.items()},
        }
        if enumeration is not None:
            tree_extra["enumeration"] = enumeration.to_dict()
        tree_path = export_tree_json(
            self.blocks, out_dir / "block_tree.json", extra=tree_extra,
        )
        result.tree_json = tree_path

        # 最终 live_report：标记完成
        try:
            self._update_live_report(
                out_dir, iteration=iteration, queue_after=[], finished=True,
                extra_note=(
                    f"全部完成，共 {iteration} 轮迭代。"
                    f"最终配色见 full_coverage_mask.png，详情见 block_tree.json。"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("最终 live report 失败: %s", exc)

        result.iter_progress_pngs = list(self._iter_progress_pngs)

        # 最终 HTML 复盘报告（复用 mesh full_coverage_report）
        if self.config.generate_report:
            try:
                from pipeline.full_coverage_report import generate_full_coverage_report

                result.report_html = generate_full_coverage_report(out_dir)
                LOGGER.info("report.html 已生成: %s", result.report_html)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("report.html 生成失败: %s", exc)

        result.ok = True
        result.elapsed_seconds = time.perf_counter() - t0
        sem_counts_str = ", ".join(
            f"{n}={len(v)}" for n, v in sem_groups.items()
        )
        result.message = (
            f"完成 {iteration} 轮，初始 1 根块 → 最终 {result.num_final_blocks} 叶块；"
            f"{sem_counts_str}；pending={len(pending_ids)}"
        )
        return result


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------


def run_pointcloud_full_coverage(
    client: VLMClient,
    points: np.ndarray,
    output_dir: str | Path,
    *,
    enumerate_hint: str = "",
    semantic_parts: Optional[List[str]] = None,
    enumerate_min_parts: int = 2,
    enumerate_max_parts: int = 12,
    max_depth: int = 6,
    min_block_size: int = 60,
    root_seed_count: int = 6,
    max_seeds_per_split: int = 4,
    image_size: int = 512,
    panel_cols: int = 4,
    point_size: Optional[float] = None,
    classification_model: Optional[str] = None,
    enumerate_model: Optional[str] = None,
    ask_split_strategy: bool = True,
    split_strategy: str = "auto",
    split_strategy_model: Optional[str] = None,
    split_n_neighbors: int = DEFAULT_KNN,
    split_enforce_connectivity: bool = True,
    split_curvature_weight: float = DEFAULT_CURVATURE_WEIGHT,
    split_convexity_weight: float = 0.0,
    split_normal_weight: float = DEFAULT_NORMAL_WEIGHT,
    use_neo: bool = False,
    neo_ckpt: Optional[str] = None,
    neo_python: Optional[str] = None,
    neo_k: int = 0,
    neo_device: str = "cpu",
    neo_no_mass: bool = False,
    parallel_vlm_workers: int = 4,
    live_report: bool = True,
    live_report_refresh_seconds: int = 10,
    select_best_view: bool = True,
    view_select_model: Optional[str] = None,
    view_select_semantic_hint: str = "",
    single_view_image_size: int = 720,
    generate_report: bool = True,
    progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> PointCloudFullCoverageResult:
    """点云全覆盖语义分割便捷入口（mesh ``run_full_coverage`` 的点云孪生版）。"""
    config = PointCloudFullCoverageConfig(
        enumerate_hint=enumerate_hint,
        semantic_parts=semantic_parts,
        enumerate_min_parts=enumerate_min_parts,
        enumerate_max_parts=enumerate_max_parts,
        max_depth=max_depth,
        min_block_size=min_block_size,
        root_seed_count=root_seed_count,
        max_seeds_per_split=max_seeds_per_split,
        image_size=image_size,
        panel_cols=panel_cols,
        point_size=point_size,
        classification_model=classification_model,
        enumerate_model=enumerate_model,
        ask_split_strategy=ask_split_strategy,
        split_strategy=split_strategy,
        split_strategy_model=split_strategy_model,
        split_n_neighbors=split_n_neighbors,
        split_enforce_connectivity=split_enforce_connectivity,
        split_curvature_weight=split_curvature_weight,
        split_convexity_weight=split_convexity_weight,
        split_normal_weight=split_normal_weight,
        use_neo=use_neo,
        neo_ckpt=neo_ckpt,
        neo_python=neo_python,
        neo_k=neo_k,
        neo_device=neo_device,
        neo_no_mass=neo_no_mass,
        parallel_vlm_workers=parallel_vlm_workers,
        live_report=live_report,
        live_report_refresh_seconds=live_report_refresh_seconds,
        select_best_view=select_best_view,
        view_select_model=view_select_model,
        view_select_semantic_hint=view_select_semantic_hint,
        single_view_image_size=single_view_image_size,
        generate_report=generate_report,
    )
    pipeline = PointCloudFullCoveragePipeline(client, config)
    return pipeline.run(points, output_dir, progress_cb=progress_cb)
