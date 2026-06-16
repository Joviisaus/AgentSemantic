"""
全覆盖（Full-coverage）语义分割流水线：

与 ``pipeline.pipeline`` 的「单语义 full/partial/none」不同，本流水线先让 VLM
观察整模型并枚举出一份**全包围、互不重叠**的语义清单（如人形角色 → 头/躯干/四肢），
随后对每个分块 multi-label 判断它命中了清单里的哪些部位：

  - 1 个 → 直接归到该部位
  - ≥2 个 → 分裂成 len(matched) 个种子（CC → segmat → FPS-spectral 级联），继续

直到每个叶分块只命中一个语义为止。不允许 background；枚举不区分左右。
收尾阶段：0 face 的枚举语义由 VLM 决定 split（从已有块 carve）或 delete；
随后按枚举给出的拓扑约束审计：块数多了→relabel 到正确语义；
块数少了→geometric_split 几何切分（VLM 定块数与每块语义）。

输出：
  output/<模型>/vlm_full/
      enumeration.json              VLM 枚举到的部位清单
      semantics_overview.png        枚举所用的整模型 12 视角图
      panels/                       每个 block 的高亮 12 视角图
      responses/                    每次 VLM multi-label 调用的原始 JSON
      face_semantic_labels.json     每张面最终的语义索引（全覆盖，无 background）
      full_coverage_mask.png        全覆盖最终结果的 12 视角图（按部位上色）
      block_tree.json               完整 block 树（含 multi-label 决策）
      live_report.html              实时进度（每轮迭代刷新）
      convex_optimize_report.html   凸包边界优化报告（语义部位，可选）
      convex_optimize_before.png / convex_optimize_after.png
      convex_optimize_log.json
      iterations/iter_NN.png        单视角时间线
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

from geometry.blocks import (
    BlockStatus,
    SemanticBlock,
    build_initial_blocks,
    export_tree_json,
)
from geometry.partition.common import SubBlockSplit
from geometry.partition.strategy import (
    split_block_by_strategy,
    split_block_by_strategy_timed,
    split_needs_strategy_retry,
)
from geometry.seed_split import (
    SplitStrategyResponse,
    _extract_submesh,
    ask_split_strategy,
    ask_split_strategy_retry,
    infer_split_strategy_fallback,
    infer_split_strategy_heuristic,
    infer_split_strategy_retry,
    split_block_by_connected_components,
)
from geometry.spectral_cluster import vertex_labels_to_face_labels
from rendering.render_component import (
    DIM_RGB,
    FULL_COVERAGE_BG_RGB,
    FULL_COVERAGE_PALETTE,
    FULL_COVERAGE_PENDING_RGB,
    HIGHLIGHT_RGB,
    render_full_coverage_panel,
    render_highlighted_component_panel,
    render_initial_segmentation_panel,
    render_overview_only_panel,
    render_single_view_overview,
    render_single_view_status,
)
from rendering.view_thumbnails import render_thumbnails_panel
from vlm.client import VLMClient
from vlm.full_coverage import (
    MultiLabelClassification,
    SemanticEnumeration,
    TopologyChildAssignment,
    TopologyConstraint,
    TopologyRepair,
    assign_geometric_split_children,
    audit_topology,
    classify_multi_label,
    classify_multi_label_coverage_repair,
    enumerate_semantics,
    infer_topology_constraints,
    resolve_empty_semantic_repairs,
    resolve_topology_repairs,
)
from vlm.view_selector import BestViewSelection, select_best_view

LOGGER = logging.getLogger("pipeline.full_coverage")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class FullCoverageConfig:
    """全覆盖语义分割配置。"""

    # 用户提示，传给 VLM 枚举阶段（宜短；粗粒度短名全覆盖，不区分左右）
    enumerate_hint: str = ""
    # 显式提供候选语义集（跳过 VLM 枚举）。两者都给时优先用这个
    semantic_parts: Optional[List[str]] = None
    enumerate_min_parts: int = 2
    enumerate_max_parts: int = 12

    max_depth: int = 6
    min_block_size: int = 20
    image_size: int = 512
    panel_cols: int = 4
    classification_model: Optional[str] = None
    enumerate_model: Optional[str] = None
    empty_semantic_repair_model: Optional[str] = None
    empty_semantic_repair_max_rounds: int = 5
    # 拓扑约束审计与修复（枚举阶段给出约束，收尾由 VLM 修复违规）
    topology_repair: bool = True
    topology_repair_model: Optional[str] = None
    topology_repair_max_rounds: int = 3
    # 单次分裂的最多种子数（VLM 即使返回更多 matched，分裂时也不会超过这个数）
    max_seeds_per_split: int = 4
    # 分裂策略提示：auto=全权交给 VLM；非 auto 仅作倾向提示
    split_strategy: str = "spectral"
    split_strategy_model: Optional[str] = None
    skip_segmat: bool = True
    # False（默认）= 按面数启发式选策略（小块 spectral，大块 geodesic_fps），不调 VLM
    vlm_split_strategy: bool = False
    # 单次几何分裂超时（秒）；≤0 表示不限时
    split_timeout_seconds: float = 60.0
    # 策略失败/超时后最多再试几次（可触发 VLM 换策略）
    split_strategy_max_retries: int = 2
    # 重试时是否调 VLM 换策略（False 则只用启发式轮换）
    split_retry_with_vlm: bool = True
    parallel_vlm_workers: int = 1
    save_responses: bool = True
    live_report: bool = True
    live_report_refresh_seconds: int = 10
    # 视角选择：进入主循环前让 VLM 选最佳视角，用作 live_report / 时间线
    select_best_view: bool = True
    view_select_model: Optional[str] = None
    view_select_semantic_hint: str = ""
    single_view_image_size: int = 720
    # 主流程结束后是否生成最终 report.html（区别于 live_report.html）
    generate_report: bool = True
    # VLM 完成后是否对语义部位做凸包边界松弛（不改变 VLM 切分逻辑）
    convex_boundary_opt: bool = True
    # 与 convex_boundary_opt 独立：仅沿 handle_loop / 凹缝拉直交界（可与 --no-convex-opt 联用）
    boundary_handle_loop: bool = False
    convex_opt_max_swaps: int = 800
    # 语义部位至少保留的面数（低于 min_block_size，便于边界在部位间移动）
    convex_opt_min_faces: int = 8
    # 凸包优化前：同语义不连通区域拆成独立连通块（与 semantic_rig 同口径）
    convex_opt_min_component_faces: int = 20


@dataclass
class FullCoverageResult:
    """全覆盖语义分割最终产物。"""

    ok: bool
    output_dir: Path
    semantic_parts: List[str] = field(default_factory=list)
    enumeration: Optional[SemanticEnumeration] = None
    # 长度 = n_faces，每个面对应的部位索引（0..K-1；-2 = 迭代中 pending）
    face_semantic_index: List[int] = field(default_factory=list)
    # 每个部位 → 包含的 face_id 列表
    semantic_face_groups: Dict[str, List[int]] = field(default_factory=dict)
    background_face_ids: List[int] = field(default_factory=list)
    pending_face_ids: List[int] = field(default_factory=list)
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
    # 最终单文件 HTML 复盘报告（与 live_report.html 不同：包含每个 block 的
    # 完整 VLM multi-label 决策、语义图例、枚举/视角/时间线/最终结果等）
    report_html: Optional[Path] = None


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _segmat_split_groups(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    strat_resp: SplitStrategyResponse,
    *,
    min_clusters: int = 3,
    max_clusters: int = 8,
) -> Dict[str, Any]:
    """单连通块的「auto-K」几何预切（VLM 选定策略）。"""
    face_ids_list = list(face_ids)
    if len(face_ids_list) <= 1:
        return {"groups": [], "error": "block too small"}
    try:
        sub_v, sub_f, _ = _extract_submesh(vertices, faces, face_ids_list)
        n_v = len(sub_v)
        if n_v <= max(min_clusters, 4) or len(sub_f) <= 1:
            return {"groups": [], "error": "submesh too small"}

        k_target = int(np.clip(max_clusters, min_clusters, max(2, n_v - 1)))
        params = strat_resp.params or {}
        split = split_block_by_strategy(
            vertices,
            faces,
            face_ids_list,
            k_target,
            strat_resp.strategy,
            lambda_smooth=float(params.get("lambda_smooth", 1.0)),
            lambda_concave=float(params.get("lambda_concave", 0.85)),
            scalar_weight=float(params.get("scalar_weight", 0.35)),
            geodesic_weight=params.get("geodesic_weight"),
            curvature_weight=params.get("curvature_weight"),
            concavity_weight=params.get("concavity_weight"),
            lloyd_iter=int(params.get("lloyd_iter", 0)),
            num_rays=int(params.get("num_rays", 16)),
            cone_angle_deg=float(params.get("cone_angle_deg", 60.0)),
            contraction_iters=int(params.get("contraction_iters", 5)),
            graph_vertex_budget=params.get("graph_vertex_budget"),
        )
        if split.num_clusters <= 1:
            return {"groups": [], "error": "segmat 退化为 1 簇"}

        groups = split.sub_face_groups
        return {
            "groups": groups,
            "k_actual": len(groups),
            "sizes": [len(g) for g in groups],
            "mode": strat_resp.strategy,
            "strategy_reason": strat_resp.reason,
            "strategy_params": strat_resp.params,
            "spectral_debug": split.debug,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("segmat split 失败: %s", exc)
        return {"groups": [], "error": str(exc)}


def _safe_palette_rgb(
    idx: int, palette: Sequence[Tuple[int, int, int]] = FULL_COVERAGE_PALETTE
) -> Tuple[int, int, int]:
    if not palette:
        return (180, 180, 180)
    return tuple(palette[idx % len(palette)])


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------


class FullCoveragePipeline:
    """
    全覆盖语义分割流水线：
        pipeline = FullCoveragePipeline(client, config)
        result = pipeline.run(vertices, faces, output_dir)

    与 ``SemanticRefinementPipeline`` 的关键差异：
      - 不在每轮做 full/partial/none 三分类，而是 multi-label 命中数；
      - 命中数决定每个 PARTIAL 块要切的种子数（K = len(matched_parts)）；
      - 命中数 = 1 的块自动归到对应语义；空匹配强制归入一项（无 background）。
    """

    def __init__(self, client: VLMClient, config: FullCoverageConfig):
        self.client = client
        self.config = config
        self.blocks: Dict[str, SemanticBlock] = {}
        self._counter_per_parent: Dict[str, int] = {}
        self.semantic_parts: List[str] = []
        self._part_index: Dict[str, int] = {}
        self._best_view: Optional[BestViewSelection] = None
        self._iter_progress_pngs: List[Path] = []
        # block_id -> assigned semantic name (set when matched == 1)
        self._assigned_semantic: Dict[str, str] = {}
        self._empty_semantic_repairs: List[Dict[str, Any]] = []
        self._topology_constraints: List[TopologyConstraint] = []
        self._topology_repairs: List[Dict[str, Any]] = []
        self._topology_face_overrides: Dict[int, int] = {}
        self._split_coverage_repairs: List[Dict[str, Any]] = []
        self._coverage_vlm_rounds: Dict[str, int] = {}
        self._coverage_geometric_resplit_done: set[str] = set()
        self._convex_opt_extra_html: str = ""
        self._convex_optimized_face_index: Optional[np.ndarray] = None

    def _resolve_vlm_model(self, explicit: Optional[str]) -> str:
        """解析 model 参数；避免把 provider 名（如 qwen）误传给 API。"""
        from vlm.config import resolve_model_name

        return resolve_model_name(explicit, self.client.config)

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
        """父块上一轮 multi-label 标注（几何分裂前的 matched_parts）。"""
        if block is None or not block.parent_id:
            return None
        parent = self.blocks.get(block.parent_id)
        if parent is None:
            return None
        parts = self._block_matched_parts(parent)
        return list(parts) if parts else None

    def _semantic_claimed_globally(self, semantic: str) -> bool:
        """整棵 block 树中是否已有块命中或认领该语义。"""
        sem = str(semantic).strip()
        if not sem:
            return False
        for block in self.blocks.values():
            if block.status == BlockStatus.SPLIT:
                continue
            if self._assigned_semantic.get(block.block_id) == sem:
                return True
            if sem in self._block_matched_parts(block):
                return True
        return False

    def _filter_globally_covered_missing(
        self,
        missing: Sequence[str],
        *,
        parent_id: str,
    ) -> List[str]:
        """去掉已在分区其他块中出现的「缺失」语义（避免误触发覆盖修复）。"""
        out: List[str] = []
        for sem in missing:
            s = str(sem).strip()
            if not s:
                continue
            if self._semantic_claimed_globally(s):
                LOGGER.info(
                    "覆盖审计 parent=%s：语义 '%s' 已在其他块命中/认领，"
                    "不视为缺失",
                    parent_id,
                    s,
                )
                continue
            out.append(s)
        return out

    def _parent_coverage_ok(self, parent_id: str) -> bool:
        """父块子树语义并集（含全局认领）是否已覆盖 parent matched。"""
        parent = self.blocks.get(parent_id)
        if parent is None or parent.status != BlockStatus.SPLIT:
            return True
        parent_set = set(self._block_matched_parts(parent))
        if not parent_set:
            return True
        children = [
            b for b in self.blocks.values() if b.parent_id == parent_id
        ]
        if len(children) < 2:
            return False
        covered: set[str] = set()
        for ch in children:
            covered.update(self._subtree_matched_parts(ch))
        missing = self._filter_globally_covered_missing(
            sorted(parent_set - covered), parent_id=parent_id
        )
        return not missing

    def _subtree_fully_resolved(self, block_id: str) -> bool:
        """该块及其子孙是否均已决策完毕（无 PENDING、无未收敛的 SPLIT）。"""
        block = self.blocks.get(block_id)
        if block is None:
            return True
        if block.status == BlockStatus.PENDING:
            return False
        if block.status == BlockStatus.SPLIT:
            children = [
                b for b in self.blocks.values() if b.parent_id == block_id
            ]
            if len(children) < 2:
                return False
            return all(
                self._subtree_fully_resolved(ch.block_id) for ch in children
            )
        return True

    def _subtree_matched_parts(self, block: SemanticBlock) -> set[str]:
        """收集块及其子孙的 matched 并集；SPLIT 中间节点向下递归到叶块。"""
        if block.status == BlockStatus.SPLIT:
            covered: set[str] = set()
            for ch in self.blocks.values():
                if ch.parent_id == block.block_id:
                    covered.update(self._subtree_matched_parts(ch))
            return covered
        assigned = self._assigned_semantic.get(block.block_id)
        if assigned:
            return {assigned}
        return set(self._block_matched_parts(block))

    def _split_children_ready(self, parent_id: str) -> bool:
        children = [
            b for b in self.blocks.values() if b.parent_id == parent_id
        ]
        if len(children) < 2:
            return False
        return all(self._subtree_fully_resolved(ch.block_id) for ch in children)

    def _apply_child_after_coverage_reclassify(
        self,
        child: SemanticBlock,
        cls: MultiLabelClassification,
        *,
        parent_id: str,
        missing_parts: Sequence[str],
        was_full: bool,
    ) -> bool:
        """写入 VLM 覆盖修复分类；必要时从 FULL 改回 PENDING。返回是否 requeue。"""
        cls_dict = {
            "matched_parts": list(cls.matched_parts),
            "is_background": bool(cls.is_background),
            "dominant": cls.dominant,
            "confidences": dict(cls.confidences),
            "notes": cls.notes,
            "provider": cls.provider,
            "model": cls.model,
            "elapsed_seconds": cls.elapsed_seconds,
            "error": cls.error,
            "coverage_vlm_repair": True,
            "coverage_parent_id": parent_id,
            "coverage_missing": list(missing_parts),
        }
        child.classification = cls_dict
        n_matched = len(cls.matched_parts)
        reopened = False
        if was_full and (
            n_matched != 1
            or (
                n_matched == 1
                and self._assigned_semantic.get(child.block_id)
                != cls.matched_parts[0]
            )
        ):
            self._assigned_semantic.pop(child.block_id, None)
            child.status = BlockStatus.PENDING
            reopened = True
        elif child.status == BlockStatus.FULL and n_matched >= 2:
            self._assigned_semantic.pop(child.block_id, None)
            child.status = BlockStatus.PENDING
            reopened = True
        child.notes = (
            child.notes
            + f" | coverage_vlm_reclassify(missing={','.join(missing_parts)})"
        ).strip(" |")
        return reopened

    def _batch_coverage_repair_reclassify(
        self,
        *,
        parent_id: str,
        parent_matched: Sequence[str],
        sibling_matched: Dict[str, List[str]],
        missing_parts: Sequence[str],
        tasks: Sequence[Tuple[str, Path]],
    ) -> Dict[str, MultiLabelClassification]:
        results: Dict[str, MultiLabelClassification] = {}
        if not tasks:
            return results

        def _do(bid: str, panel: Path):
            return bid, classify_multi_label_coverage_repair(
                self.client,
                panel,
                self.semantic_parts,
                parent_id=parent_id,
                block_id=bid,
                parent_matched=parent_matched,
                sibling_matched=sibling_matched,
                missing_parts=missing_parts,
                model=self._resolve_vlm_model(
                    self.config.classification_model
                ),
            )

        workers = max(1, int(self.config.parallel_vlm_workers))
        if workers > 1 and len(tasks) > 1:
            n = min(workers, len(tasks))
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = {
                    ex.submit(_do, bid, panel): bid for bid, panel in tasks
                }
                for fut in as_completed(futs):
                    bid = futs[fut]
                    try:
                        _, cls = fut.result()
                        results[bid] = cls
                    except (RuntimeError, ValueError, OSError) as exc:
                        LOGGER.warning(
                            "parallel coverage repair failed for %s: %s",
                            bid, exc,
                        )
        else:
            for bid, panel in tasks:
                try:
                    _, cls = _do(bid, panel)
                    results[bid] = cls
                except (RuntimeError, ValueError, OSError) as exc:
                    LOGGER.warning(
                        "coverage repair reclassify failed for %s: %s",
                        bid, exc,
                    )
        return results

    def _reset_parent_for_geometric_resplit(
        self,
        parent_id: str,
        parent_matched: Sequence[str],
        missing_parts: Sequence[str],
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> List[str]:
        """VLM 覆盖修复不收敛时：撤销子块，按父块 matched 数重新几何切分。"""
        parent = self.blocks.get(parent_id)
        if parent is None:
            return []
        if parent_id in self._coverage_geometric_resplit_done:
            return []

        self._coverage_geometric_resplit_done.add(parent_id)
        target_k = max(2, len(parent_matched))
        LOGGER.warning(
            "覆盖修复不收敛 parent=%s missing=%s → 撤销子块并按 k=%d 重新几何切分",
            parent_id,
            list(missing_parts),
            target_k,
        )

        for cid in list(parent.children_ids):
            self.blocks.pop(cid, None)
            self._assigned_semantic.pop(cid, None)
        parent.children_ids = []
        parent.status = BlockStatus.PENDING
        parent.segmat_done = False
        parent.force_split = True
        parent.seed_count = target_k
        if not parent.classification:
            dom = parent_matched[0] if parent_matched else ""
            parent.classification = {
                "matched_parts": list(parent_matched),
                "dominant": dom,
            }
        parent.notes = (
            parent.notes
            + f" | coverage_geometric_resplit(k={target_k}, missing={','.join(missing_parts)})"
        ).strip(" |")

        if progress_cb:
            progress_cb(
                "coverage_geometric_resplit",
                {
                    "parent_id": parent_id,
                    "seed_count": target_k,
                    "missing": list(missing_parts),
                },
            )
        return [parent_id]

    def _repair_split_coverage_gaps(
        self,
        parent_id: str,
        children: Sequence[SemanticBlock],
        panel_by_id: Dict[str, Path],
        responses_dir: Path,
        next_queue: Optional[List[str]] = None,
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> List[str]:
        """子块 matched 并集须覆盖父块分裂前 matched；缺失则让 VLM 重新分类各子块。"""
        parent = self.blocks.get(parent_id)
        if parent is None or parent.status != BlockStatus.SPLIT:
            return []
        if len(children) < 2:
            return []

        parent_matched = self._block_matched_parts(parent)
        parent_set = set(parent_matched)
        if not parent_set:
            return []

        covered: set[str] = set()
        for ch in children:
            covered.update(self._subtree_matched_parts(ch))

        missing = sorted(parent_set - covered)
        missing = self._filter_globally_covered_missing(
            missing, parent_id=parent_id
        )
        if not missing:
            return []

        LOGGER.warning(
            "分裂语义覆盖缺失 parent=%s parent_matched=%s children_cover=%s "
            "missing=%s → VLM 重新分类",
            parent_id,
            sorted(parent_set),
            sorted(covered),
            missing,
        )

        sibling_matched = {
            ch.block_id: list(self._subtree_matched_parts(ch)) for ch in children
        }
        tasks: List[Tuple[str, Path]] = []
        for ch in children:
            if ch.status == BlockStatus.SPLIT:
                continue
            panel = panel_by_id.get(ch.block_id)
            if panel is None and ch.panel_path:
                panel = Path(ch.panel_path)
            if panel is not None and panel.is_file():
                tasks.append((ch.block_id, panel))
        if not tasks:
            LOGGER.warning(
                "覆盖修复跳过 parent=%s：无可用子块 panel", parent_id
            )
            return []

        vlm_round = int(self._coverage_vlm_rounds.get(parent_id, 0))
        if vlm_round >= 1:
            LOGGER.info(
                "parent=%s 已做过 VLM 覆盖修复，跳过重复 VLM 调用", parent_id
            )
            cls_cache: Dict[str, MultiLabelClassification] = {}
        else:
            cls_cache = self._batch_coverage_repair_reclassify(
                parent_id=parent_id,
                parent_matched=parent_matched,
                sibling_matched=sibling_matched,
                missing_parts=missing,
                tasks=tasks,
            )
            self._coverage_vlm_rounds[parent_id] = vlm_round + 1

        requeued: List[str] = []
        if cls_cache:
            for child in children:
                cls = cls_cache.get(child.block_id)
                if cls is None:
                    continue
                was_full = child.status == BlockStatus.FULL
                before = self._block_matched_parts(child)
                reopened = self._apply_child_after_coverage_reclassify(
                    child,
                    cls,
                    parent_id=parent_id,
                    missing_parts=missing,
                    was_full=was_full,
                )
                after = self._block_matched_parts(child)
                self._dump_response(
                    responses_dir,
                    f"{child.block_id}_coverage_repair",
                    {
                        "parent_id": parent_id,
                        "missing": missing,
                        "before": before,
                        "after": after,
                        "classification": child.classification,
                    },
                )
                if (
                    reopened
                    and next_queue is not None
                    and child.block_id not in next_queue
                ):
                    next_queue.append(child.block_id)
                    requeued.append(child.block_id)
                self._split_coverage_repairs.append(
                    {
                        "parent_id": parent_id,
                        "child_id": child.block_id,
                        "method": "vlm_reclassify",
                        "missing": list(missing),
                        "before": before,
                        "after": after,
                        "was_full": was_full,
                        "parent_matched": sorted(parent_set),
                        "children_cover_before": sorted(covered),
                    }
                )
                if progress_cb:
                    progress_cb(
                        "split_coverage_repair",
                        {
                            "parent_id": parent_id,
                            "child_id": child.block_id,
                            "missing": missing,
                            "before": before,
                            "after": after,
                            "requeued": bool(
                                requeued and child.block_id in requeued
                            ),
                        },
                    )

        covered_after: set[str] = set()
        for ch in children:
            if ch.block_id in self.blocks:
                covered_after.update(self._subtree_matched_parts(ch))
        still_missing = sorted(parent_set - covered_after)
        still_missing = self._filter_globally_covered_missing(
            still_missing, parent_id=parent_id
        )
        if still_missing:
            LOGGER.warning(
                "VLM 覆盖修复后仍缺失 parent=%s still_missing=%s",
                parent_id,
                still_missing,
            )
            geo = self._reset_parent_for_geometric_resplit(
                parent_id,
                parent_matched,
                still_missing,
                progress_cb=progress_cb,
            )
            if geo and next_queue is not None:
                for pid in geo:
                    if pid not in next_queue:
                        next_queue.append(pid)
            return geo or requeued

        return requeued

    def _audit_split_coverage_before_decisions(
        self,
        queue: List[str],
        panel_by_id: Dict[str, Path],
        responses_dir: Path,
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """本轮所有子块分类写入后、分裂决策前：校验 SPLIT 父块语义覆盖。"""
        queue_set = set(queue)
        parent_ids: set[str] = set()
        for bid in queue:
            block = self.blocks.get(bid)
            if block is None or not block.parent_id:
                continue
            parent = self.blocks.get(block.parent_id)
            if parent is not None and parent.status == BlockStatus.SPLIT:
                parent_ids.add(block.parent_id)

        for pid in sorted(parent_ids):
            children = [
                b
                for b in self.blocks.values()
                if b.parent_id == pid
                and b.block_id in queue_set
                and b.status == BlockStatus.PENDING
                and b.classification
            ]
            if len(children) < 2:
                continue
            # 本轮 queue 须包含该父块全部分裂子块（同一批分类）
            all_siblings = [
                b
                for b in self.blocks.values()
                if b.parent_id == pid and b.status != BlockStatus.SPLIT
            ]
            if len(children) != len(all_siblings):
                continue
            requeued = self._repair_split_coverage_gaps(
                pid,
                children,
                panel_by_id,
                responses_dir,
                next_queue=queue,
                progress_cb=progress_cb,
            )
            if requeued:
                dead = [
                    bid for bid in queue
                    if bid not in self.blocks and bid not in requeued
                ]
                if dead:
                    queue[:] = [bid for bid in queue if bid not in dead]

    def _audit_and_repair_split_coverage(
        self,
        parent_id: str,
        panel_by_id: Dict[str, Path],
        responses_dir: Path,
        next_queue: List[str],
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> List[str]:
        """迭代末尾兜底：父块子块均已决策完毕后再查一次。"""
        parent = self.blocks.get(parent_id)
        if parent is None or parent.status != BlockStatus.SPLIT:
            return []
        if not self._split_children_ready(parent_id):
            return []

        children = [
            b for b in self.blocks.values() if b.parent_id == parent_id
        ]
        return self._repair_split_coverage_gaps(
            parent_id,
            children,
            panel_by_id,
            responses_dir,
            next_queue,
            progress_cb=progress_cb,
        )

    def _audit_all_split_coverage(
        self,
        panel_by_id: Dict[str, Path],
        responses_dir: Path,
        next_queue: List[str],
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """每轮迭代末尾：检查所有已分类完毕的 SPLIT 父块。"""
        parents = sorted(
            {
                b.block_id
                for b in self.blocks.values()
                if b.status == BlockStatus.SPLIT
            }
        )
        if not parents:
            return
        todo = [pid for pid in parents if not self._parent_coverage_ok(pid)]
        if progress_cb:
            progress_cb(
                "coverage_audit",
                {
                    "total_parents": len(parents),
                    "audit_parents": len(todo),
                    "skipped_ok": len(parents) - len(todo),
                },
            )
        for i, pid in enumerate(todo):
            if progress_cb:
                progress_cb(
                    "coverage_audit",
                    {
                        "parent_id": pid,
                        "index": i + 1,
                        "total": len(todo),
                    },
                )
            self._audit_and_repair_split_coverage(
                pid,
                panel_by_id,
                responses_dir,
                next_queue,
                progress_cb=progress_cb,
            )

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
            responses_dir,
            f"{block.block_id}_classify",
            block.classification,
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
        self,
        block: SemanticBlock,
        sem: str,
        note: str,
    ) -> None:
        if sem not in self._part_index:
            sem = self._fallback_semantic(block=block)
        block.status = BlockStatus.FULL
        self._assigned_semantic[block.block_id] = sem
        block.notes = (block.notes + f" | {note}").strip(" |")

    @staticmethod
    def _build_face_adjacency(faces: np.ndarray) -> List[List[int]]:
        n_faces = int(faces.shape[0])
        adj: List[List[int]] = [[] for _ in range(n_faces)]
        edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
        for fi in range(n_faces):
            tri = faces[fi]
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                e = (min(int(a), int(b)), max(int(a), int(b)))
                edge_to_faces.setdefault(e, []).append(fi)
        for face_list in edge_to_faces.values():
            if len(face_list) < 2:
                continue
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    f1, f2 = face_list[i], face_list[j]
                    adj[f1].append(f2)
                    adj[f2].append(f1)
        return adj

    def _propagate_unlabeled_face_indices(
        self, idx: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """把仍为 -1 / -2 的面按邻接传播到已标注语义；仍孤立则归入索引 0。"""
        out = idx.copy()
        adj = self._build_face_adjacency(faces)
        changed = True
        while changed:
            changed = False
            for fid in range(out.shape[0]):
                if out[fid] >= 0:
                    continue
                nbrs = [int(out[n]) for n in adj[fid] if out[n] >= 0]
                if not nbrs:
                    continue
                out[fid] = int(max(set(nbrs), key=nbrs.count))
                changed = True
        unlabeled = (out < 0) | (out >= len(self.semantic_parts))
        if np.any(unlabeled):
            out[unlabeled] = 0
        return out

    def _finalize_block_assignments(self) -> None:
        """结束前：所有非 SPLIT 叶块都必须有语义归属。"""
        for block in self.blocks.values():
            if block.status == BlockStatus.SPLIT:
                continue
            if block.block_id in self._assigned_semantic:
                if block.status != BlockStatus.FULL:
                    block.status = BlockStatus.FULL
                continue
            cls_dict = block.classification or {}
            dom = str(cls_dict.get("dominant") or "").strip()
            matched = [
                str(x).strip()
                for x in (cls_dict.get("matched_parts") or [])
                if str(x).strip()
            ]
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

    def _face_semantic_counts(self, n_faces: int, faces: np.ndarray) -> Dict[str, int]:
        face_idx = self._compute_face_semantic_index(n_faces)
        face_idx = self._propagate_unlabeled_face_indices(face_idx, faces)
        counts: Dict[str, int] = {n: 0 for n in self.semantic_parts}
        for v in face_idx.tolist():
            if 0 <= v < len(self.semantic_parts):
                counts[self.semantic_parts[v]] += 1
        return counts

    def _empty_semantic_parts(self, counts: Dict[str, int]) -> List[str]:
        return [n for n in self.semantic_parts if counts.get(n, 0) == 0]

    def _delete_semantic_part(self, name: str, *, reason: str) -> None:
        if name not in self.semantic_parts:
            return
        deleted_idx = self._part_index.get(name)
        self.semantic_parts = [p for p in self.semantic_parts if p != name]
        self._part_index = {n: i for i, n in enumerate(self.semantic_parts)}
        self._topology_constraints = [
            c for c in self._topology_constraints if c.part != name
        ]
        if deleted_idx is not None and self._topology_face_overrides:
            remapped: Dict[int, int] = {}
            for fid, old_idx in self._topology_face_overrides.items():
                if old_idx == deleted_idx:
                    continue
                if old_idx > deleted_idx:
                    remapped[fid] = old_idx - 1
                else:
                    remapped[fid] = old_idx
            self._topology_face_overrides = remapped
        self._empty_semantic_repairs.append(
            {"empty_semantic": name, "action": "delete", "reason": reason}
        )
        LOGGER.info("0-face 语义 '%s' 已从清单删除（%s）", name, reason)

    def _pick_child_for_empty_semantic(
        self,
        child_ids: Sequence[str],
        cls_by_id: Dict[str, MultiLabelClassification],
        empty_sem: str,
        donor_sem: str,
    ) -> Optional[str]:
        for cid in child_ids:
            cls = cls_by_id.get(cid)
            if cls is None:
                continue
            if empty_sem in cls.matched_parts and donor_sem not in cls.matched_parts:
                return cid
        for cid in child_ids:
            cls = cls_by_id.get(cid)
            if cls and cls.dominant == empty_sem:
                return cid
        for cid in child_ids:
            cls = cls_by_id.get(cid)
            if cls and empty_sem in cls.matched_parts:
                return cid
        if child_ids:
            smallest = min(child_ids, key=lambda c: self.blocks[c].size)
            return smallest
        return None

    def _apply_split_repair(
        self,
        empty_sem: str,
        donor_block_id: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        panel_dir: Path,
        responses_dir: Path,
        *,
        reason: str,
    ) -> bool:
        block = self.blocks.get(donor_block_id)
        if block is None or block.status != BlockStatus.FULL:
            LOGGER.warning(
                "split repair '%s': donor %s 不可用",
                empty_sem,
                donor_block_id,
            )
            return False
        donor_sem = self._assigned_semantic.get(donor_block_id)
        if not donor_sem or donor_sem == empty_sem:
            return False

        repair_panel = panel_dir / f"block_{donor_block_id}.png"
        sub_groups = self._split_block_by_seeds(
            vertices,
            faces,
            block,
            seed_count=2,
            responses_dir=responses_dir,
            panel_path=repair_panel if repair_panel.is_file() else None,
            matched_parts=[empty_sem, donor_sem],
        )
        if not sub_groups or len(sub_groups) < 2:
            LOGGER.warning(
                "split repair '%s': donor %s 几何分裂失败",
                empty_sem,
                donor_block_id,
            )
            return False

        self._assigned_semantic.pop(donor_block_id, None)
        block.status = BlockStatus.SPLIT
        child_ids: List[str] = []
        cls_by_id: Dict[str, MultiLabelClassification] = {}
        binary_candidates = [empty_sem, donor_sem]

        for group_face_ids in sub_groups:
            child_id = self._next_child_id(donor_block_id)
            child = SemanticBlock(
                block_id=child_id,
                face_ids=list(group_face_ids),
                depth=block.depth + 1,
                parent_id=donor_block_id,
                status=BlockStatus.PENDING,
                notes=(
                    f"empty_repair split from {donor_block_id} "
                    f"for '{empty_sem}'"
                ),
                segmat_done=block.segmat_done,
            )
            self._register_block(child)
            block.children_ids.append(child_id)
            child_ids.append(child_id)

            try:
                panel_path = self._render_block_panel(
                    vertices, faces, child, panel_dir
                )
                cls = classify_multi_label(
                    self.client,
                    panel_path,
                    binary_candidates,
                    block_id=child_id,
                    parent_id=donor_block_id,
                    parent_matched=[empty_sem, donor_sem],
                    model=self._resolve_vlm_model(
                        self.config.classification_model
                    ),
                )
            except (RuntimeError, ValueError, OSError) as exc:
                LOGGER.warning(
                    "split repair classify 失败 (%s): %s", child_id, exc
                )
                cls = MultiLabelClassification(error=str(exc))
            cls_by_id[child_id] = cls
            self._dump_response(
                responses_dir,
                f"{child_id}_empty_repair_classify",
                cls.to_dict(),
            )

        target = self._pick_child_for_empty_semantic(
            child_ids, cls_by_id, empty_sem, donor_sem
        )
        if target is None:
            return False

        for cid in child_ids:
            if cid == target:
                self._force_assign_block(
                    self.blocks[cid],
                    empty_sem,
                    f"empty_repair_split→{empty_sem}",
                )
            else:
                self._force_assign_block(
                    self.blocks[cid],
                    donor_sem,
                    f"empty_repair_split_remain→{donor_sem}",
                )

        self._empty_semantic_repairs.append(
            {
                "empty_semantic": empty_sem,
                "action": "split",
                "donor_block_id": donor_block_id,
                "target_child_id": target,
                "reason": reason,
            }
        )
        LOGGER.info(
            "0-face 语义 '%s' 已从 %s carve → child %s",
            empty_sem,
            donor_block_id,
            target,
        )
        return True

    def _repair_empty_semantics(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        out_dir: Path,
        panel_dir: Path,
        responses_dir: Path,
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """收尾：0 face 语义必须由 VLM 决定 split 或 delete，直到清单无空项。"""
        n_faces = int(faces.shape[0])
        repair_model = self._resolve_vlm_model(
            self.config.empty_semantic_repair_model
            or self.config.classification_model
        )
        max_rounds = max(1, int(self.config.empty_semantic_repair_max_rounds))

        for round_idx in range(1, max_rounds + 1):
            counts = self._face_semantic_counts(n_faces, faces)
            empty = self._empty_semantic_parts(counts)
            if not empty:
                break

            LOGGER.info(
                "0-face 语义补救 round %d: %s", round_idx, empty
            )
            if progress_cb:
                progress_cb(
                    "empty_repair",
                    {
                        "round": round_idx,
                        "empty_semantic": ",".join(empty),
                        "action": "audit",
                    },
                )
            repair_png = out_dir / f"empty_semantic_repair_r{round_idx:02d}.png"
            face_idx = self._compute_face_semantic_index(n_faces)
            face_idx = self._propagate_unlabeled_face_indices(face_idx, faces)
            try:
                render_full_coverage_panel(
                    vertices,
                    faces,
                    face_idx.tolist(),
                    self.semantic_parts,
                    repair_png,
                    title=f"0-face 补救前（round {round_idx}）",
                    image_size=self.config.image_size,
                    cols=self.config.panel_cols,
                )
            except (OSError, ImportError, ValueError) as exc:
                LOGGER.warning("0-face 补救图渲染失败: %s", exc)
                repair_png = out_dir / "full_coverage_mask.png"
                if not repair_png.is_file():
                    break

            all_donors: List[Dict[str, Any]] = []
            for block in self.blocks.values():
                if block.status != BlockStatus.FULL:
                    continue
                sem = self._assigned_semantic.get(block.block_id)
                if not sem:
                    continue
                min_split = max(
                    self.config.min_block_size * 2,
                    self.config.min_block_size + 1,
                )
                if block.size < min_split:
                    continue
                all_donors.append(
                    {
                        "block_id": block.block_id,
                        "semantic": sem,
                        "faces": block.size,
                    }
                )
            all_donors.sort(key=lambda x: int(x["faces"]), reverse=True)

            plan = resolve_empty_semantic_repairs(
                self.client,
                repair_png,
                semantic_parts=self.semantic_parts,
                face_counts=counts,
                empty_parts=empty,
                donor_blocks=all_donors,
                model=repair_model,
            )
            self._dump_response(
                responses_dir,
                f"empty_semantic_repair_r{round_idx:02d}",
                plan.to_dict(),
            )
            if not plan.ok:
                LOGGER.warning(
                    "0-face VLM 补救失败 (%s)，剩余项 fallback delete",
                    plan.error,
                )
                for name in empty:
                    self._delete_semantic_part(name, reason="VLM repair failed")
                continue

            progress_made = False
            for repair in plan.repairs:
                if repair.empty_semantic not in self.semantic_parts:
                    continue
                if progress_cb:
                    progress_cb(
                        "empty_repair",
                        {
                            "round": round_idx,
                            "empty_semantic": repair.empty_semantic,
                            "action": repair.action,
                            "donor_block_id": repair.donor_block_id,
                        },
                    )
                if repair.action == "delete":
                    self._delete_semantic_part(
                        repair.empty_semantic,
                        reason=repair.reason or "VLM delete",
                    )
                    progress_made = True
                elif repair.action == "split":
                    ok = self._apply_split_repair(
                        repair.empty_semantic,
                        repair.donor_block_id,
                        vertices,
                        faces,
                        panel_dir,
                        responses_dir,
                        reason=repair.reason or "VLM split",
                    )
                    if ok:
                        progress_made = True
                    else:
                        self._delete_semantic_part(
                            repair.empty_semantic,
                            reason=f"split failed: {repair.reason}",
                        )
                        progress_made = True

            if not progress_made:
                for name in self._empty_semantic_parts(counts):
                    if name in self.semantic_parts:
                        self._delete_semantic_part(
                            name, reason="repair round made no progress"
                        )
                break

        remaining = self._empty_semantic_parts(
            self._face_semantic_counts(n_faces, faces)
        )
        if remaining:
            LOGGER.warning(
                "仍有 0-face 语义 %s，强制 delete", remaining
            )
            for name in remaining:
                self._delete_semantic_part(name, reason="final fallback delete")

    def _resolve_topology_constraints(
        self,
        parts: List[str],
        enumeration: Optional[SemanticEnumeration],
        overview_path: Optional[Path],
    ) -> List[TopologyConstraint]:
        """从枚举结果或 VLM 推断获取拓扑约束。"""
        if enumeration and enumeration.topology_constraints:
            return list(enumeration.topology_constraints)
        if overview_path is not None and overview_path.is_file():
            LOGGER.info("调用 VLM 推断拓扑约束 ...")
            topo = infer_topology_constraints(
                self.client,
                overview_path,
                parts,
                model=self._resolve_vlm_model(
                    self.config.enumerate_model
                    or self.config.classification_model
                ),
            )
            if topo:
                return topo
        return [TopologyConstraint(part=p, count=1) for p in parts]

    def _get_topology_component_faces(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        n_faces: int,
        semantic: str,
        component_index: int,
    ) -> List[int]:
        """返回某语义第 component_index（1-based）个连通块的面 id 列表。"""
        face_idx = self._compute_face_semantic_index(n_faces)
        face_idx = self._propagate_unlabeled_face_indices(face_idx, faces)
        audit = audit_topology(
            vertices,
            faces,
            face_idx.tolist(),
            self.semantic_parts,
            self._topology_constraints,
            min_component_faces=self.config.convex_opt_min_component_faces,
        )
        for comp in audit.components:
            if comp.semantic == semantic and comp.index == component_index:
                return list(comp.face_ids)
        return []

    def _reassign_component_faces(
        self,
        face_ids: Sequence[int],
        to_semantic: str,
        *,
        reason: str,
    ) -> bool:
        """把一组面改归目标语义（face 级 override）。"""
        if to_semantic not in self._part_index or not face_ids:
            return False
        target_idx = self._part_index[to_semantic]
        for fid in face_ids:
            self._topology_face_overrides[int(fid)] = target_idx
        self._topology_repairs.append(
            {
                "action": "reassign_component",
                "to_semantic": to_semantic,
                "num_faces": len(face_ids),
                "reason": reason,
            }
        )
        return True

    def _sort_split_groups_by_centroid(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        sub_groups: Sequence[Sequence[int]],
    ) -> List[List[int]]:
        """按重心空间顺序排序子块（左→右、前→后），与 VLM 编号约定对齐。"""
        face_centers = vertices[faces].mean(axis=1)

        def _key(group: Sequence[int]) -> Tuple[float, float, float]:
            if not group:
                return (0.0, 0.0, 0.0)
            idx = np.asarray(list(group), dtype=np.int64)
            c = face_centers[idx].mean(axis=0)
            return (-float(c[0]), -float(c[2]), -float(c[1]))

        return sorted(
            [list(g) for g in sub_groups],
            key=_key,
        )

    def _apply_geometric_split_repair(
        self,
        donor_block_id: str,
        seed_count: int,
        child_assignments: Sequence[TopologyChildAssignment],
        vertices: np.ndarray,
        faces: np.ndarray,
        panel_dir: Path,
        responses_dir: Path,
        *,
        reason: str,
    ) -> bool:
        """几何切分 donor 块为 K 块，由 VLM 为每个编号子块指定语义。"""
        block = self.blocks.get(donor_block_id)
        if block is None or block.status != BlockStatus.FULL:
            LOGGER.warning(
                "geometric_split: donor %s 不可用", donor_block_id
            )
            return False
        donor_sem = self._assigned_semantic.get(donor_block_id) or ""

        k = max(2, min(int(seed_count), self.config.max_seeds_per_split))
        matched_parts = list(
            dict.fromkeys(
                [a.semantic for a in child_assignments]
                + ([donor_sem] if donor_sem else [])
            )
        )
        repair_panel = panel_dir / f"block_{donor_block_id}.png"
        LOGGER.info(
            "拓扑 geometric_split: donor=%s faces=%d k=%d strategy=skeleton",
            donor_block_id,
            block.size,
            k,
        )
        sub_groups = self._split_block_by_seeds(
            vertices,
            faces,
            block,
            seed_count=k,
            responses_dir=responses_dir,
            panel_path=repair_panel if repair_panel.is_file() else None,
            matched_parts=matched_parts,
            strategy_hint_override="skeleton",
        )
        if not sub_groups or len(sub_groups) < 2:
            LOGGER.warning(
                "geometric_split: donor %s 几何分裂失败", donor_block_id
            )
            return False

        ordered_groups = self._sort_split_groups_by_centroid(
            vertices, faces, sub_groups
        )
        actual_k = len(ordered_groups)

        n_faces = int(faces.shape[0])
        cand = np.full(n_faces, -1, dtype=np.int64)
        for i, group_face_ids in enumerate(ordered_groups):
            for fid in group_face_ids:
                if 0 <= int(fid) < n_faces:
                    cand[int(fid)] = i

        split_panel = panel_dir / f"split_{donor_block_id}.png"
        try:
            from rendering.render_component import render_candidate_panel

            render_candidate_panel(
                vertices,
                faces,
                cand.tolist(),
                split_panel,
                title=f"几何切分 {donor_block_id} → {actual_k} 块",
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
            )
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("geometric_split 拼图渲染失败: %s", exc)
            split_panel = repair_panel if repair_panel.is_file() else None
            if split_panel is None:
                return False

        planned = list(child_assignments)[:actual_k]
        if len(planned) < actual_k:
            for i in range(len(planned) + 1, actual_k + 1):
                sem = (
                    planned[i - 2].semantic
                    if i - 2 < len(planned)
                    else (donor_sem or matched_parts[0])
                )
                planned.append(
                    TopologyChildAssignment(index=i, semantic=sem)
                )

        LOGGER.info(
            "拓扑 geometric_split: donor=%s 几何切完成 k=%d，VLM 指派子块语义…",
            donor_block_id,
            actual_k,
        )
        assign_result = assign_geometric_split_children(
            self.client,
            split_panel,
            donor_block_id=donor_block_id,
            donor_semantic=donor_sem,
            child_count=actual_k,
            candidate_semantics=matched_parts,
            planned_assignments=planned,
            model=self._resolve_vlm_model(self.config.classification_model),
        )
        self._dump_response(
            responses_dir,
            f"split_assign_{donor_block_id}",
            assign_result.to_dict(),
        )
        if assign_result.error and not assign_result.assignments:
            LOGGER.warning(
                "geometric_split 子块指派失败: %s", assign_result.error
            )
            return False

        self._assigned_semantic.pop(donor_block_id, None)
        block.status = BlockStatus.SPLIT
        assigned_labels: List[str] = []
        for i, group_face_ids in enumerate(ordered_groups):
            child_id = self._next_child_id(donor_block_id)
            sem = assign_result.assignments.get(
                i + 1,
                planned[i].semantic if i < len(planned) else donor_sem,
            )
            if sem not in self._part_index:
                sem = self._fallback_semantic()
            child = SemanticBlock(
                block_id=child_id,
                face_ids=list(group_face_ids),
                depth=block.depth + 1,
                parent_id=donor_block_id,
                status=BlockStatus.PENDING,
                notes=(
                    f"topology geometric_split from {donor_block_id} "
                    f"#{i + 1}→{sem}"
                ),
                segmat_done=block.segmat_done,
            )
            self._register_block(child)
            block.children_ids.append(child_id)
            self._force_assign_block(
                child,
                sem,
                f"topology_geometric_split#{i + 1}→{sem}",
            )
            assigned_labels.append(sem)

        self._topology_repairs.append(
            {
                "action": "geometric_split",
                "donor_block_id": donor_block_id,
                "seed_count": actual_k,
                "child_assignments": assigned_labels,
                "reason": reason,
            }
        )
        LOGGER.info(
            "拓扑 geometric_split %s → %d 子块: %s",
            donor_block_id,
            actual_k,
            assigned_labels,
        )
        return True

    def _apply_geometric_topology_cut(
        self,
        repair: TopologyRepair,
        vertices: np.ndarray,
        faces: np.ndarray,
        n_faces: int,
        *,
        reason: str,
    ) -> bool:
        """几何拓扑手术：谱切割 / handle loop / 边界图割精修。"""
        from geometry.topology_repair import (
            TopologyRepairResult,
            refine_surgery_boundary,
            repair_illegal_adjacency,
            repair_pseudo_bridge,
        )

        part_a = repair.from_semantic.strip()
        part_b = repair.to_semantic.strip()
        if (
            part_a not in self._part_index
            or part_b not in self._part_index
        ):
            return False

        face_idx = self._compute_face_semantic_index(n_faces)
        face_idx = self._propagate_unlabeled_face_indices(face_idx, faces)

        if repair.action == "cut_adjacency":
            result = repair_illegal_adjacency(
                vertices,
                faces,
                face_idx.tolist(),
                self.semantic_parts,
                part_a,
                part_b,
            )
        elif repair.action == "cut_bridge":
            result = repair_pseudo_bridge(
                vertices,
                faces,
                face_idx.tolist(),
                self.semantic_parts,
                part_a,
                part_b,
            )
        elif repair.action == "boundary_refine":
            from geometry.topology_repair import semantic_boundary_faces

            ia = self._part_index[part_a]
            ib = self._part_index[part_b]
            boundary = semantic_boundary_faces(faces, face_idx, ia, ib)
            overrides, dbg = refine_surgery_boundary(
                vertices,
                faces,
                face_idx.tolist(),
                self.semantic_parts,
                boundary,
            )
            result = TopologyRepairResult(
                face_overrides=overrides,
                strategy="boundary_refine",
                ok=bool(overrides),
                debug=dbg,
            )
        else:
            return False

        if not result.ok or not result.face_overrides:
            LOGGER.warning(
                "几何拓扑修复无进展: %s %s↔%s debug=%s",
                repair.action,
                part_a,
                part_b,
                getattr(result, "debug", {}),
            )
            return False

        for fid, sem_idx in result.face_overrides.items():
            if 0 <= int(fid) < n_faces and 0 <= int(sem_idx) < len(
                self.semantic_parts
            ):
                self._topology_face_overrides[int(fid)] = int(sem_idx)

        self._topology_repairs.append(
            {
                "action": repair.action,
                "from_semantic": part_a,
                "to_semantic": part_b,
                "num_faces": len(result.face_overrides),
                "strategy": result.strategy,
                "reason": reason,
                "debug": result.debug,
            }
        )
        return True

    def _apply_topology_repair(
        self,
        repair: TopologyRepair,
        vertices: np.ndarray,
        faces: np.ndarray,
        n_faces: int,
        panel_dir: Path,
        responses_dir: Path,
    ) -> bool:
        """执行单条拓扑修复。"""
        if repair.action in ("cut_adjacency", "cut_bridge", "boundary_refine"):
            return self._apply_geometric_topology_cut(
                repair,
                vertices,
                faces,
                n_faces,
                reason=repair.reason or f"topology {repair.action}",
            )

        if repair.action == "geometric_split":
            if not repair.donor_block_id or not repair.child_assignments:
                return False
            return self._apply_geometric_split_repair(
                repair.donor_block_id,
                repair.seed_count,
                repair.child_assignments,
                vertices,
                faces,
                panel_dir,
                responses_dir,
                reason=repair.reason or "topology geometric_split",
            )

        if repair.action == "reassign_component":
            from_sem = repair.from_semantic.strip()
            to_sem = repair.to_semantic.strip()
            if (
                from_sem not in self._part_index
                or to_sem not in self._part_index
                or repair.component_index < 1
            ):
                return False
            fids = self._get_topology_component_faces(
                vertices,
                faces,
                n_faces,
                from_sem,
                repair.component_index,
            )
            if not fids:
                return False
            ok = self._reassign_component_faces(
                fids,
                to_sem,
                reason=repair.reason or "topology reassign",
            )
            if ok:
                self._topology_repairs.append(
                    {
                        "action": "reassign_component",
                        "from_semantic": from_sem,
                        "to_semantic": to_sem,
                        "component_index": repair.component_index,
                        "num_faces": len(fids),
                        "reason": repair.reason,
                    }
                )
            return ok

        return False

    def _repair_topology(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        out_dir: Path,
        panel_dir: Path,
        responses_dir: Path,
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        """收尾：按拓扑约束审计分割结果，VLM 修复 count/attach 违规。"""
        if not self.config.topology_repair or not self._topology_constraints:
            return

        n_faces = int(faces.shape[0])
        repair_model = self._resolve_vlm_model(
            self.config.topology_repair_model
            or self.config.empty_semantic_repair_model
            or self.config.classification_model
        )
        max_rounds = max(1, int(self.config.topology_repair_max_rounds))

        for round_idx in range(1, max_rounds + 1):
            face_idx = self._compute_face_semantic_index(n_faces)
            face_idx = self._propagate_unlabeled_face_indices(face_idx, faces)
            audit = audit_topology(
                vertices,
                faces,
                face_idx.tolist(),
                self.semantic_parts,
                self._topology_constraints,
                min_component_faces=self.config.convex_opt_min_component_faces,
            )
            self._dump_response(
                responses_dir,
                f"topology_audit_r{round_idx:02d}",
                audit.to_dict(),
            )
            if audit.ok:
                LOGGER.info("拓扑审计通过（round %d）", round_idx)
                break

            LOGGER.info(
                "拓扑违规 round %d: %s",
                round_idx,
                [v.get("detail") for v in audit.violations],
            )

            # 先尝试几何工具集自动修复（非法邻接 / 伪桥梁）
            geo_violations = [
                v
                for v in audit.violations
                if str(v.get("kind", "")).strip().lower()
                in (
                    "illegal_adjacency",
                    "forbidden_adjacency",
                    "pseudo_bridge",
                    "handle_loop",
                )
            ]
            if geo_violations:
                from geometry.topology_repair import apply_geometric_repairs

                if progress_cb:
                    progress_cb(
                        "topology_repair",
                        {
                            "round": round_idx,
                            "action": "auto_geometric_repair",
                            "violations": len(geo_violations),
                        },
                    )
                geo_result = apply_geometric_repairs(
                    vertices,
                    faces,
                    face_idx.tolist(),
                    self.semantic_parts,
                    geo_violations,
                    topology_constraints=self._topology_constraints,
                    auto_detect_forbidden=False,
                    refine_boundary=False,
                    max_repairs=4,
                )
                self._dump_response(
                    responses_dir,
                    f"topology_geo_repair_r{round_idx:02d}",
                    {
                        "strategy": geo_result.strategy,
                        "ok": geo_result.ok,
                        "overrides": len(geo_result.face_overrides),
                        "debug": geo_result.debug,
                    },
                )
                if progress_cb:
                    progress_cb(
                        "topology_repair",
                        {
                            "round": round_idx,
                            "action": "auto_geometric_repair_done",
                            "overrides": len(geo_result.face_overrides),
                            "ok": geo_result.ok,
                        },
                    )
                if geo_result.ok:
                    for fid, sem_idx in geo_result.face_overrides.items():
                        if 0 <= int(fid) < n_faces and 0 <= int(sem_idx) < len(
                            self.semantic_parts
                        ):
                            self._topology_face_overrides[int(fid)] = int(
                                sem_idx
                            )
                    self._topology_repairs.append(
                        {
                            "action": "auto_geometric_repair",
                            "num_faces": len(geo_result.face_overrides),
                            "strategy": geo_result.strategy,
                            "reason": "auto illegal_adjacency / pseudo_bridge",
                        }
                    )
                    continue

            if progress_cb:
                progress_cb(
                    "topology_repair",
                    {
                        "round": round_idx,
                        "action": "vlm_audit",
                        "violations": len(audit.violations),
                    },
                )
            repair_png = out_dir / f"topology_repair_r{round_idx:02d}.png"
            try:
                render_full_coverage_panel(
                    vertices,
                    faces,
                    face_idx.tolist(),
                    self.semantic_parts,
                    repair_png,
                    title=f"拓扑修复前（round {round_idx}）",
                    image_size=self.config.image_size,
                    cols=self.config.panel_cols,
                )
            except (OSError, ImportError, ValueError) as exc:
                LOGGER.warning("拓扑修复图渲染失败: %s", exc)
                repair_png = out_dir / "full_coverage_mask.png"
                if not repair_png.is_file():
                    break

            all_donors: List[Dict[str, Any]] = []
            for block in self.blocks.values():
                if block.status != BlockStatus.FULL:
                    continue
                sem = self._assigned_semantic.get(block.block_id)
                if not sem:
                    continue
                min_split = max(
                    self.config.min_block_size * 2,
                    self.config.min_block_size + 1,
                )
                if block.size < min_split:
                    continue
                all_donors.append(
                    {
                        "block_id": block.block_id,
                        "semantic": sem,
                        "faces": block.size,
                    }
                )
            all_donors.sort(key=lambda x: int(x["faces"]), reverse=True)

            plan = resolve_topology_repairs(
                self.client,
                repair_png,
                semantic_parts=self.semantic_parts,
                topology_constraints=self._topology_constraints,
                audit=audit,
                donor_blocks=all_donors,
                model=repair_model,
                max_split=self.config.max_seeds_per_split,
            )
            self._dump_response(
                responses_dir,
                f"topology_repair_r{round_idx:02d}",
                plan.to_dict(),
            )
            if plan.topology_ok and not plan.repairs:
                LOGGER.info("VLM 判定拓扑合规（round %d）", round_idx)
                break
            if not plan.ok:
                LOGGER.warning(
                    "拓扑 VLM 修复失败 (%s)，停止拓扑修复",
                    plan.error,
                )
                break
            if not plan.repairs:
                break

            progress_made = False
            for repair in plan.repairs:
                if progress_cb:
                    progress_cb(
                        "topology_repair",
                        {
                            "round": round_idx,
                            "action": repair.action,
                            "violation": repair.violation,
                            "from_semantic": repair.from_semantic,
                            "to_semantic": repair.to_semantic,
                            "donor_block_id": repair.donor_block_id,
                            "seed_count": repair.seed_count,
                        },
                    )
                if self._apply_topology_repair(
                    repair,
                    vertices,
                    faces,
                    n_faces,
                    panel_dir,
                    responses_dir,
                ):
                    progress_made = True

            if not progress_made:
                LOGGER.warning("拓扑修复 round %d 无进展，停止", round_idx)
                break

    # ---------- 渲染 ----------
    def _render_block_panel(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        block: SemanticBlock,
        panel_dir: Path,
    ) -> Path:
        # 不写语义到 title 以避免 VLM 确认偏差
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

    # ---------- face 状态 ----------
    def _compute_face_semantic_index(self, n_faces: int) -> np.ndarray:
        """根据当前 blocks 状态计算每个面的语义索引。

        - SPLIT 块不直接拥有面
        - FULL 块且 _assigned_semantic 命中 → semantic_parts 索引
        - 已归属语义的块（含原 NONE/ABANDONED 经强制归属）→ 语义索引
        - 未归属 / PENDING → -2（收尾阶段会传播消除）
        """
        idx = np.full(n_faces, -2, dtype=np.int64)
        for block in self.blocks.values():
            if block.status == BlockStatus.SPLIT:
                continue
            sem = self._assigned_semantic.get(block.block_id)
            if sem is not None and sem in self._part_index:
                val = self._part_index[sem]
            elif block.status == BlockStatus.PENDING:
                val = -2
            else:
                val = -2
            arr = np.asarray(block.face_ids, dtype=np.int64)
            if arr.size:
                mask = (arr >= 0) & (arr < n_faces)
                idx[arr[mask]] = val
        for fid, sem_idx in self._topology_face_overrides.items():
            if 0 <= int(fid) < n_faces and 0 <= int(sem_idx) < len(
                self.semantic_parts
            ):
                idx[int(fid)] = int(sem_idx)
        return idx

    def _apply_convex_optimize_final_blocks(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        out_dir: Path,
        n_faces: int,
        result: FullCoverageResult,
        *,
        boundary_method: str = "planar",
    ) -> None:
        """对 VLM 完成后的语义部位做边界优化（平面+handle_loop 或仅 handle_loop）。"""
        handle_loop_only = str(boundary_method).strip().lower() in (
            "handle_loop", "handle-loop", "handleloop",
        )
        method_label = (
            "handle_loop" if handle_loop_only else "planar_boundary"
        )
        LOGGER.info(
            "边界优化开始 (method=%s, max_pair_passes=%d) …",
            method_label,
            max(1, min(3, self.config.convex_opt_max_swaps // 200 + 1)),
        )
        face_idx = self._compute_face_semantic_index(n_faces)
        face_idx = self._propagate_unlabeled_face_indices(face_idx, faces)

        n_sem = len(self.semantic_parts)
        assigned = (face_idx >= 0) & (face_idx < n_sem)
        active_parts = sorted({int(x) for x in face_idx[assigned]})
        if len(active_parts) < 2:
            return

        try:
            from geometry.convex_partition_optimize import (
                ConvexPartitionOptConfig,
                map_component_labels_to_semantic,
                optimize_face_partition_convex_hull,
                split_semantic_into_connected_components,
                write_convex_optimize_report,
            )
        except ImportError as exc:
            LOGGER.warning("凸包优化模块不可用: %s", exc)
            return

        split = split_semantic_into_connected_components(
            vertices,
            faces,
            face_idx,
            self.semantic_parts,
            min_component_faces=self.config.convex_opt_min_component_faces,
        )
        comp_labels = split.comp_labels.copy()
        valid_comp = comp_labels >= 0
        if int(valid_comp.sum()) == 0 or split.num_components < 2:
            LOGGER.info(
                "凸包优化跳过：连通块不足 2（components=%d）",
                split.num_components,
            )
            return

        comp_labels_for_opt = comp_labels.copy()
        comp_labels_for_opt[~valid_comp] = -1

        convex_cfg = ConvexPartitionOptConfig(
            enabled=True,
            max_swaps=self.config.convex_opt_max_swaps,
            min_block_faces=self.config.convex_opt_min_faces,
            split_connected_components=True,
            min_component_faces=self.config.convex_opt_min_component_faces,
            enforce_connectivity=False,
            image_size=self.config.image_size,
            panel_cols=self.config.panel_cols,
            preserve_label_ids=False,
            stage_title="连通块",
            verbose=True,
            boundary_method=boundary_method,
            max_pair_passes=max(1, min(5, self.config.convex_opt_max_swaps // 200 + 1)),
        )
        split_meta = {
            "num_components": split.num_components,
            "component_names": list(split.component_names),
            "comp_part": list(split.comp_part),
            "min_component_faces": self.config.convex_opt_min_component_faces,
        }
        try:
            opt = optimize_face_partition_convex_hull(
                vertices, faces, comp_labels_for_opt, convex_cfg,
            )
            face_idx_opt = map_component_labels_to_semantic(
                np.asarray(opt.face_labels, dtype=np.int64),
                split.comp_part,
                face_idx,
            )
            opt = write_convex_optimize_report(
                out_dir,
                vertices,
                faces,
                comp_labels_for_opt,
                opt,
                convex_cfg,
                semantic_parts=self.semantic_parts,
                display_labels_before=face_idx,
                display_labels_after=face_idx_opt,
                component_names=split.component_names,
                comp_labels_split=comp_labels,
                split_meta=split_meta,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("语义部位凸包优化失败（沿用 VLM 分块）: %s", exc)
            return

        self._convex_optimized_face_index = face_idx_opt

        compare_mask = face_idx >= 0
        changed_ratio = (
            float(np.mean(face_idx_opt[compare_mask] != face_idx[compare_mask]))
            if compare_mask.any()
            else opt.changed_face_ratio
        )
        num_faces_changed = int(np.sum(face_idx_opt != face_idx))

        convex_meta = {
            "enabled": True,
            "method": (
                "boundary_handle_loop"
                if handle_loop_only
                else "planar_boundary_handle_loop"
            ),
            "stage": "connected_components",
            "num_semantic_parts": len(active_parts),
            "num_components": split.num_components,
            "component_names": list(split.component_names),
            "component_split": split_meta,
            "num_swaps": opt.num_swaps,
            "num_faces_changed": num_faces_changed,
            "num_connectivity_relabels": opt.num_connectivity_relabels,
            "energy_before": opt.metrics_before.active_total_energy,
            "energy_after": opt.metrics_after.active_total_energy,
            "energy_delta": opt.energy_delta,
            "changed_face_ratio": changed_ratio,
            "parts_before": opt.metrics_before.per_part_list(
                self.semantic_parts,
            ),
            "parts_after": opt.metrics_after.per_part_list(
                self.semantic_parts,
            ),
            "elapsed_seconds": opt.elapsed_seconds,
            "report_html": (
                str(opt.report_html) if opt.report_html else None
            ),
            "log_json": str(opt.log_json) if opt.log_json else None,
            "before_png": str(opt.before_png) if opt.before_png else None,
            "after_png": str(opt.after_png) if opt.after_png else None,
        }
        result.extra["convex_optimize"] = convex_meta

        before_rel = opt.before_png.name if opt.before_png else ""
        after_rel = opt.after_png.name if opt.after_png else ""
        report_rel = (
            opt.report_html.name
            if opt.report_html is not None
            else "convex_optimize_report.html"
        )
        img_pair = ""
        if before_rel and after_rel:
            img_pair = (
                f'<div style="display:flex;gap:12px;flex-wrap:wrap;'
                f'margin-top:10px">'
                f'<figure style="margin:0">'
                f'<img src="{_html.escape(before_rel)}" '
                f'style="max-width:320px;border:1px solid #d2d2d7;'
                f'border-radius:8px" alt="convex before"/>'
                f'<figcaption style="font-size:12px;color:#6e6e73">'
                f'优化前</figcaption></figure>'
                f'<figure style="margin:0">'
                f'<img src="{_html.escape(after_rel)}" '
                f'style="max-width:320px;border:1px solid #d2d2d7;'
                f'border-radius:8px" alt="convex after"/>'
                f'<figcaption style="font-size:12px;color:#6e6e73">'
                f'优化后</figcaption></figure></div>'
            )
        self._convex_opt_extra_html = (
            f'<section class="legend-panel">'
            f'<h2>凸包边界优化（语义部位）</h2>'
            f'<p class="muted">VLM 完成后先拆 {split.num_components} 个连通块'
            f'（{len(active_parts)} 个语义部位内不连通区域独立优化），'
            f'再松弛边界（不合并删块）；'
            f'贪婪交换 <b>{opt.num_swaps}</b> 次，'
            f'实际变更 <b>{num_faces_changed}</b> 面；'
            f'活跃部位凹度能量 '
            f'<b>{opt.metrics_before.active_total_energy:.4f}</b> → '
            f'<b>{opt.metrics_after.active_total_energy:.4f}</b> '
            f'(Δ={opt.energy_delta:+.4f})；'
            f'变更比例 <b>{100 * changed_ratio:.2f}%</b>。'
            f' <a href="{_html.escape(report_rel)}">完整报告</a></p>'
            f'{img_pair}</section>'
        )
        LOGGER.info(
            "凸包优化: sem_parts=%d components=%d swaps=%d changed_faces=%d "
            "E_active %.4f→%.4f (Δ=%.4f) ratio=%.2f%%",
            len(active_parts),
            split.num_components,
            opt.num_swaps,
            num_faces_changed,
            opt.metrics_before.active_total_energy,
            opt.metrics_after.active_total_energy,
            opt.energy_delta,
            100 * changed_ratio,
        )

    def _color_map_for_status(self) -> Dict[str, Tuple[int, int, int]]:
        cmap: Dict[str, Tuple[int, int, int]] = {
            "bg": FULL_COVERAGE_BG_RGB,
            "pending": FULL_COVERAGE_PENDING_RGB,
        }
        for k, _name in enumerate(self.semantic_parts):
            cmap[f"sem_{k}"] = _safe_palette_rgb(k)
        return cmap

    def _face_index_to_status_strings(
        self, face_idx: np.ndarray
    ) -> List[str]:
        out: List[str] = []
        for v in face_idx.tolist():
            if v == -1:
                out.append("bg")
            elif v == -2:
                out.append("pending")
            else:
                out.append(f"sem_{int(v)}")
        return out

    # ---------- 视角选择 ----------
    def _select_view_for_display(
        self,
        out_dir: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Optional[BestViewSelection]:
        """渲染 12 缩略图 → VLM 挑最佳 → 渲一张透明背景 overview PNG。"""
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
                    title=f"候选视角（全覆盖：{hint}）",
                )
            except (OSError, ImportError, ValueError) as exc:
                LOGGER.warning("缩略图渲染失败: %s", exc)
                thumb_path = None
            if thumb_path is None:
                from rendering.heatmap_renderer import VIEW_GRID as _VG
                bv = BestViewSelection(
                    index=0, elev=float(_VG[0][0]),
                    azim=float(_VG[0][1]),
                    reason="缩略图渲染失败，退回 VIEW_GRID[0]",
                    fallback=True,
                )
            else:
                LOGGER.info("调用 VLM 选择最佳视角 ...")
                bv = select_best_view(
                    self.client, thumb_path,
                    hint,
                    model=self._resolve_vlm_model(
                        self.config.view_select_model
                    ),
                )
                bv.thumbnails_panel = thumb_path

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
                        "semantic_hint": hint,
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

    # ---------- 并行 multi-label ----------
    def _batch_multi_label(
        self,
        tasks: Sequence[Tuple[str, Path]],
    ) -> Dict[str, MultiLabelClassification]:
        results: Dict[str, MultiLabelClassification] = {}
        if not tasks:
            return results

        def _do(bid: str, panel: Path):
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
                model=self._resolve_vlm_model(
                    self.config.classification_model
                ),
            )

        workers = max(1, int(self.config.parallel_vlm_workers))
        if workers > 1 and len(tasks) > 1:
            n = min(workers, len(tasks))
            LOGGER.info(
                "parallel multi-label: %d tasks across %d workers", len(tasks), n
            )
            with ThreadPoolExecutor(max_workers=n) as ex:
                futs = {
                    ex.submit(_do, bid, panel): bid for bid, panel in tasks
                }
                for fut in as_completed(futs):
                    bid = futs[fut]
                    try:
                        _, cls = fut.result()
                        results[bid] = cls
                    except (RuntimeError, ValueError, OSError) as exc:
                        LOGGER.warning(
                            "parallel multi-label failed for %s: %s",
                            bid, exc,
                        )
        else:
            for bid, panel in tasks:
                try:
                    _, cls = _do(bid, panel)
                    results[bid] = cls
                except (RuntimeError, ValueError, OSError) as exc:
                    LOGGER.warning(
                        "serial multi-label failed for %s: %s", bid, exc
                    )
        return results

    # ---------- 时间线 / live 报告 ----------
    def _save_iteration_progress(
        self,
        out_dir: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        iteration: int,
    ) -> Optional[Path]:
        if self._best_view is None:
            return None
        n_faces = int(faces.shape[0])
        face_idx = self._compute_face_semantic_index(n_faces)
        status_strings = self._face_index_to_status_strings(face_idx)
        it_dir = out_dir / "iterations"
        it_dir.mkdir(parents=True, exist_ok=True)
        png = it_dir / f"iter_{iteration:02d}.png"
        try:
            render_single_view_status(
                vertices,
                faces,
                status_strings,
                png,
                elev=self._best_view.elev,
                azim=self._best_view.azim,
                image_size=self.config.single_view_image_size,
                color_map=self._color_map_for_status(),
                background="transparent",
                title=None,
            )
        except (OSError, ValueError, ImportError) as exc:
            LOGGER.warning("iter %d 单视角 PNG 渲染失败: %s", iteration, exc)
            return None
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
        if not self.config.live_report:
            return

        n_faces = int(faces.shape[0])
        face_idx = self._compute_face_semantic_index(n_faces)
        status_strings = self._face_index_to_status_strings(face_idx)
        cmap = self._color_map_for_status()

        live_png = out_dir / "live_progress.png"
        rendered = False
        try:
            if self._best_view is not None:
                render_single_view_status(
                    vertices,
                    faces,
                    status_strings,
                    live_png,
                    elev=self._best_view.elev,
                    azim=self._best_view.azim,
                    image_size=self.config.single_view_image_size,
                    color_map=cmap,
                    background="transparent",
                    title=None,
                )
            else:
                render_full_coverage_panel(
                    vertices,
                    faces,
                    face_idx.tolist(),
                    self.semantic_parts,
                    live_png,
                    title=(
                        f"iter {iteration}  "
                        f"({'已完成' if finished else '进行中'})"
                    ),
                    image_size=self.config.image_size,
                    cols=self.config.panel_cols,
                )
            rendered = True
        except (OSError, ValueError, ImportError) as exc:
            LOGGER.warning("live progress 渲染失败: %s", exc)

        # 统计
        sem_face_counts: Dict[str, int] = {n: 0 for n in self.semantic_parts}
        pending_faces = 0
        for v in face_idx.tolist():
            if v == -2:
                pending_faces += 1
            else:
                if 0 <= v < len(self.semantic_parts):
                    sem_face_counts[self.semantic_parts[v]] += 1

        # block 状态计数
        block_counts = {st.value: 0 for st in BlockStatus}
        for b in self.blocks.values():
            block_counts[b.status.value] = block_counts.get(b.status.value, 0) + 1

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
        if self._convex_opt_extra_html:
            extra_html += self._convex_opt_extra_html

        # 语义图例 + 计数
        legend_rows: List[str] = []
        for k, name in enumerate(self.semantic_parts):
            rgb = _safe_palette_rgb(k)
            hex_color = _rgb_to_hex(rgb)
            cnt = sem_face_counts.get(name, 0)
            legend_rows.append(
                f'<div class="legend-row">'
                f'<span class="swatch" style="background:{hex_color}"></span>'
                f'<span class="legend-name">{esc(name)}</span>'
                f'<span class="legend-count">{cnt}</span>'
                f"</div>"
            )
        legend_rows.append(
            f'<div class="legend-row">'
            f'<span class="swatch" style="background:'
            f'{_rgb_to_hex(FULL_COVERAGE_PENDING_RGB)}"></span>'
            f'<span class="legend-name">pending</span>'
            f'<span class="legend-count">{pending_faces}</span>'
            f"</div>"
        )
        legend_html = (
            f'<section class="legend-panel"><h2>语义图例</h2>'
            f'<div class="legend-list">{"".join(legend_rows)}</div></section>'
        )

        # 视角卡片
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

        # 时间线
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
            .legend-panel { background:#fff; border:1px solid #d2d2d7;
                            border-radius:10px; padding:12px 16px; margin:14px 0; }
            .legend-list { display:grid;
                           grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                           gap:6px 14px; }
            .legend-row { display:flex; align-items:center; gap:8px;
                          font-size:13px; }
            .swatch { display:inline-block; width:14px; height:14px;
                      border:1px solid #999; border-radius:3px; flex-shrink:0; }
            .legend-name { flex:1; color:#1d1d1f; }
            .legend-count { color:#6e6e73; font-variant-numeric: tabular-nums; }
        """

        body = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
{refresh_meta}
<title>VLM 全覆盖语义 · iter {iteration}</title>
<style>{css}</style>
</head>
<body>
<h1>VLM 全覆盖语义分割 · 实时进度 {status_tag}</h1>
<div class="ts">更新时间: {esc(ts)}
 · semantic_parts: <b>{esc(len(self.semantic_parts))}</b> 项
 · iteration: <b>{iteration}</b>
 · 下轮队列: <b>{len(queue_after)}</b></div>

<div class="row">
  <div class="card"><b>已分配 face</b><span class="v">{sum(sem_face_counts.values())}</span></div>
  <div class="card"><b>pending face</b><span class="v">{pending_faces}</span></div>
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

{legend_html}

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

    # ---------- 枚举阶段 ----------
    def _resolve_semantic_parts(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        out_dir: Path,
    ) -> Tuple[List[str], Optional[SemanticEnumeration], Optional[Path]]:
        """渲染整模型 overview 并向 VLM 询问语义清单，或直接采用用户给的清单。

        返回 (semantic_parts, enumeration_or_none, overview_panel_path_or_none)
        """
        overview_path = out_dir / "semantics_overview.png"
        try:
            render_overview_only_panel(
                vertices,
                faces,
                overview_path,
                title="整模型 12 视角（用于全覆盖语义枚举）",
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
            )
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("整模型 overview 渲染失败: %s", exc)
            overview_path = None

        if self.config.semantic_parts:
            parts = [
                str(x).strip() for x in self.config.semantic_parts
                if str(x).strip()
            ]
            # 去重保序
            seen: Dict[str, int] = {}
            unique: List[str] = []
            for p in parts:
                if p in seen:
                    continue
                seen[p] = 1
                unique.append(p)
            LOGGER.info("使用用户提供的 semantic_parts: %s", unique)
            return unique, None, overview_path

        if overview_path is None:
            raise RuntimeError(
                "无法渲染 overview 面板，且未提供 semantic_parts，无法继续。"
            )

        LOGGER.info("调用 VLM 枚举全覆盖语义清单 ...")
        enumeration = enumerate_semantics(
            self.client,
            overview_path,
            hint=self.config.enumerate_hint,
            model=self._resolve_vlm_model(self.config.enumerate_model),
            min_parts=self.config.enumerate_min_parts,
            max_parts=self.config.enumerate_max_parts,
        )
        # 落盘 enumeration.json
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
            "VLM 枚举到 %d 项语义部位: %s",
            len(enumeration.parts), enumeration.parts,
        )
        return list(enumeration.parts), enumeration, overview_path

    def _resolve_split_strategy(
        self,
        panel_path: Optional[Path],
        *,
        seed_count: int,
        matched_parts: Optional[List[str]] = None,
        block_faces: int = 0,
        split_stage: str = "main",
        strategy_hint_override: Optional[str] = None,
    ) -> SplitStrategyResponse:
        forced = str(
            strategy_hint_override or self.config.split_strategy or "spectral"
        ).strip().lower()
        if not self.config.vlm_split_strategy:
            return infer_split_strategy_heuristic(
                block_faces=block_faces,
                strategy_hint=forced,
                seed_count=seed_count,
            )
        semantic_ctx = "、".join(matched_parts or []) or "全覆盖语义细分"
        return ask_split_strategy(
            self.client,
            panel_path,
            semantic_context=semantic_ctx,
            seed_count=seed_count,
            matched_parts=matched_parts,
            block_faces=block_faces,
            split_stage=split_stage,
            strategy_hint=forced,
            fallback="spectral",
            model=self._resolve_vlm_model(
                self.config.split_strategy_model
                or self.config.classification_model
            ),
        )

    def _resolve_split_strategy_after_failure(
        self,
        panel_path: Optional[Path],
        *,
        seed_count: int,
        matched_parts: Optional[List[str]],
        block_faces: int,
        split_stage: str,
        previous_strategy: str,
        failure_reason: str,
        tried_strategies: Sequence[str],
    ) -> SplitStrategyResponse:
        forced = str(self.config.split_strategy or "spectral").strip().lower()
        semantic_ctx = "、".join(matched_parts or []) or "全覆盖语义细分"
        model = self._resolve_vlm_model(
            self.config.split_strategy_model or self.config.classification_model
        )
        if self.config.split_retry_with_vlm:
            return ask_split_strategy_retry(
                self.client,
                panel_path,
                semantic_context=semantic_ctx,
                seed_count=seed_count,
                matched_parts=matched_parts,
                block_faces=block_faces,
                split_stage=split_stage,
                strategy_hint=forced,
                previous_strategy=previous_strategy,
                failure_reason=failure_reason,
                tried_strategies=list(tried_strategies),
                fallback="geodesic_fps",
                model=model,
            )
        return infer_split_strategy_retry(
            block_faces=block_faces,
            tried_strategies=list(tried_strategies),
            failure_reason=failure_reason,
            seed_count=seed_count,
        )

    def _run_timed_strategy_split(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        face_ids: Sequence[int],
        seed_count: int,
        strat_resp: SplitStrategyResponse,
    ) -> Tuple[Optional[SubBlockSplit], Optional[str]]:
        params = strat_resp.params or {}
        return split_block_by_strategy_timed(
            self.config.split_timeout_seconds,
            vertices=vertices,
            faces=faces,
            face_ids=face_ids,
            seed_count=seed_count,
            strategy=strat_resp.strategy,
            lambda_smooth=float(params.get("lambda_smooth", 1.0)),
            lambda_concave=float(params.get("lambda_concave", 0.85)),
            scalar_weight=float(params.get("scalar_weight", 0.35)),
            geodesic_weight=params.get("geodesic_weight"),
            curvature_weight=params.get("curvature_weight"),
            concavity_weight=params.get("concavity_weight"),
            lloyd_iter=int(params.get("lloyd_iter", 0)),
            num_rays=int(params.get("num_rays", 16)),
            cone_angle_deg=float(params.get("cone_angle_deg", 60.0)),
            contraction_iters=int(params.get("contraction_iters", 5)),
            graph_vertex_budget=params.get("graph_vertex_budget"),
        )

    # ---------- 单块决策 ----------
    def _split_block_by_seeds(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        block: SemanticBlock,
        seed_count: int,
        *,
        responses_dir: Path,
        panel_path: Optional[Path] = None,
        matched_parts: Optional[List[str]] = None,
        strategy_hint_override: Optional[str] = None,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Optional[List[List[int]]]:
        """对单个 block 走 CC → segmat → VLM 策略几何切，返回子块 face_id 列表。

        - 未做过 segmat：先 CC（多个 CC 则直接采用），否则 VLM 选策略 + segmat。
        - segmat 之后：VLM 选策略 + K=seed_count 几何切。
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

        # Step B: segmat 预切（默认跳过）
        if not block.segmat_done and self.config.skip_segmat:
            block.segmat_done = True
        if not block.segmat_done:
            strat_resp = self._resolve_split_strategy(
                panel_path,
                seed_count=seed_count,
                matched_parts=matched_parts,
                block_faces=block.size,
                split_stage="segmat",
            )
            segmat = _segmat_split_groups(
                vertices,
                faces,
                block.face_ids,
                strat_resp,
                min_clusters=max(2, min(3, seed_count)),
                max_clusters=max(seed_count, 4),
            )
            block.seed_decision = {
                "mode": "segmat",
                "strategy": strat_resp.strategy,
                "strategy_reason": strat_resp.reason,
                "strategy_params": strat_resp.params,
                "k_actual": segmat.get("k_actual", 0),
                "component_sizes": segmat.get("sizes", []),
                "seed_count_target": seed_count,
                "error": segmat.get("error"),
            }
            self._dump_response(
                responses_dir, f"{block.block_id}_segmat", block.seed_decision
            )
            sub_groups = segmat.get("groups", [])
            if len(sub_groups) >= 2:
                # 标记一下：分裂出的子块都视为已做 segmat
                block.segmat_done = True  # 本块以下都跳过 CC/segmat
                return sub_groups
            LOGGER.info(
                "block %s segmat 失败 (%s) → 改走 FPS-Voronoi K=%d",
                block.block_id,
                segmat.get("error", "unknown"),
                seed_count,
            )
            block.segmat_done = True

        # Step C: 选策略 + 几何切（超时/未收敛则换策略重试）
        max_retries = max(0, int(self.config.split_strategy_max_retries))
        tried: List[str] = []
        strat_hint = strategy_hint_override
        if strat_hint is None and block.force_split:
            if "coverage_geometric_resplit" in (block.notes or ""):
                strat_hint = "skeleton"
        strat_resp = self._resolve_split_strategy(
            panel_path,
            seed_count=seed_count,
            matched_parts=matched_parts,
            block_faces=block.size,
            split_stage="main",
            strategy_hint_override=strat_hint,
        )
        split = None
        last_failure = ""
        attempts_log: List[Dict[str, Any]] = []

        for attempt in range(max_retries + 1):
            if attempt > 0:
                strat_resp = self._resolve_split_strategy_after_failure(
                    panel_path,
                    seed_count=seed_count,
                    matched_parts=matched_parts,
                    block_faces=block.size,
                    split_stage="main",
                    previous_strategy=tried[-1] if tried else "",
                    failure_reason=last_failure,
                    tried_strategies=tried,
                )
                LOGGER.warning(
                    "block %s 分裂重试 %d/%d: %s → strategy=%s (%s)",
                    block.block_id,
                    attempt,
                    max_retries,
                    last_failure,
                    strat_resp.strategy,
                    strat_resp.reason,
                )
                if progress_cb:
                    progress_cb(
                        "split_strategy_retry",
                        {
                            "block_id": block.block_id,
                            "attempt": attempt,
                            "max_retries": max_retries,
                            "previous_strategy": tried[-1] if tried else "",
                            "failure": last_failure,
                            "new_strategy": strat_resp.strategy,
                            "reason": strat_resp.reason,
                        },
                    )

            LOGGER.info(
                "block %s 几何切: strategy=%s faces=%d k=%d (%s)",
                block.block_id,
                strat_resp.strategy,
                block.size,
                seed_count,
                strat_resp.reason,
            )
            if progress_cb:
                progress_cb(
                    "split_geometry",
                    {
                        "block_id": block.block_id,
                        "strategy": strat_resp.strategy,
                        "faces": block.size,
                        "seed_count": seed_count,
                        "attempt": attempt,
                        "reason": strat_resp.reason,
                    },
                )
            split, run_error = self._run_timed_strategy_split(
                vertices,
                faces,
                block.face_ids,
                seed_count,
                strat_resp,
            )
            needs_retry, retry_reason = split_needs_strategy_retry(
                split, error=run_error
            )
            attempts_log.append(
                {
                    "attempt": attempt,
                    "strategy": strat_resp.strategy,
                    "error": run_error,
                    "retry_reason": retry_reason,
                    "num_clusters": int(split.num_clusters) if split else 0,
                }
            )
            if not needs_retry:
                break
            last_failure = retry_reason or run_error or "unknown"
            tried.append(strat_resp.strategy)
            if attempt >= max_retries:
                LOGGER.warning(
                    "block %s 分裂策略重试用尽 (%s)，最后 strategy=%s",
                    block.block_id,
                    last_failure,
                    strat_resp.strategy,
                )
                break

        if split is None:
            block.seed_decision = {
                "mode": strat_resp.strategy,
                "seed_count_target": seed_count,
                "strategy_reason": strat_resp.reason,
                "error": last_failure,
                "attempts": attempts_log,
            }
            self._dump_response(
                responses_dir, f"{block.block_id}_seed", block.seed_decision
            )
            return None

        block.seed_decision = {
            "mode": strat_resp.strategy,
            "seed_count_target": seed_count,
            "strategy_reason": strat_resp.reason,
            "strategy_params": strat_resp.params,
            "num_clusters": split.num_clusters,
            "debug": split.debug,
            "attempts": attempts_log,
        }
        self._dump_response(
            responses_dir, f"{block.block_id}_seed", block.seed_decision
        )
        if split.num_clusters <= 1:
            return None
        return split.sub_face_groups

    # ---------- 主流程 ----------
    def run(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_dir: str | Path,
        *,
        initial_face_labels: Optional[Sequence[int]] = None,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> FullCoverageResult:
        t0 = time.perf_counter()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_dir = out_dir / "panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        responses_dir = out_dir / "responses"
        if self.config.save_responses:
            responses_dir.mkdir(parents=True, exist_ok=True)

        n_faces = int(faces.shape[0])

        result = FullCoverageResult(
            ok=False,
            output_dir=out_dir,
        )

        # ---- Step 1: 枚举语义部位 ----
        try:
            parts, enumeration, overview_path = self._resolve_semantic_parts(
                vertices, faces, out_dir
            )
        except RuntimeError as exc:
            result.message = f"语义枚举失败: {exc}"
            result.elapsed_seconds = time.perf_counter() - t0
            return result
        self.semantic_parts = parts
        self._part_index = {n: k for k, n in enumerate(parts)}
        result.semantic_parts = list(parts)
        result.enumeration = enumeration
        result.overview_panel = overview_path

        self._topology_constraints = self._resolve_topology_constraints(
            parts, enumeration, overview_path
        )
        try:
            topo_path = out_dir / "topology_constraints.json"
            topo_path.write_text(
                json.dumps(
                    {
                        "semantic_parts": list(parts),
                        "topology_constraints": [
                            c.to_dict() for c in self._topology_constraints
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            result.extra["topology_constraints"] = str(topo_path)
        except OSError as exc:
            LOGGER.warning("写 topology_constraints.json 失败: %s", exc)

        # ---- Step 2: 视角选择 ----
        self._iter_progress_pngs = []
        self._best_view = self._select_view_for_display(out_dir, vertices, faces)
        if self._best_view is not None:
            result.best_view = self._best_view
            result.best_view_overview = self._best_view.overview_png

        # ---- Step 3: 初始分块 ----
        if initial_face_labels is None:
            labels_in = [0] * n_faces
        else:
            if len(initial_face_labels) != n_faces:
                raise ValueError(
                    f"initial_face_labels 长度 ({len(initial_face_labels)}) "
                    f"≠ faces 数量 ({n_faces})"
                )
            labels_in = [int(x) for x in initial_face_labels]

        raw_initial_blocks = build_initial_blocks(
            labels_in, min_block_size=self.config.min_block_size
        )

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

        renumbered: List[SemanticBlock] = []
        for new_idx, b in enumerate(initial_blocks):
            renumbered.append(
                SemanticBlock(
                    block_id=f"C{new_idx}",
                    face_ids=b.face_ids,
                    depth=0,
                    parent_id=None,
                    status=BlockStatus.PENDING,
                    notes=b.notes,
                )
            )
        initial_blocks = renumbered

        LOGGER.info(
            "初始 segmat=%d 块 → CC 展开后=%d 块",
            len(raw_initial_blocks), len(initial_blocks),
        )

        for b in initial_blocks:
            self._register_block(b)

        result.num_initial_blocks = len(initial_blocks)

        if not initial_blocks:
            result.message = "初始分块为空（min_block_size 是否过大？）"
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        # 渲染初始分块
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

        # 初始 live_report
        try:
            self._update_live_report(
                out_dir,
                vertices,
                faces,
                iteration=0,
                queue_after=queue,
                finished=False,
                extra_note="初始化中：尚未开始 VLM multi-label",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("初始 live report 失败: %s", exc)

        # ---- Step 4: 主迭代 ----
        while queue:
            iteration += 1
            LOGGER.info("== iteration %d, queue=%d ==", iteration, len(queue))
            next_queue: List[str] = []

            # --- Pre-pass A: too_small / 渲染 ---
            iter_panel: Dict[str, Path] = {}
            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                if block.size < self.config.min_block_size:
                    sem = self._fallback_semantic(block=block)
                    self._force_assign_block(
                        block, sem, f"too_small→{sem}"
                    )
                    self._dump_response(
                        responses_dir, block_id, {"reason": "too_small"}
                    )
                    if progress_cb:
                        progress_cb(
                            "abandoned",
                            {
                                "block_id": block_id,
                                "reason": "too_small",
                                "assigned": sem,
                            },
                        )
                    continue
                try:
                    panel_path = self._render_block_panel(
                        vertices, faces, block, panel_dir
                    )
                    block.panel_path = str(panel_path)
                    iter_panel[block_id] = panel_path
                except (OSError, ImportError, ValueError) as exc:
                    LOGGER.error("渲染 block %s 失败: %s", block_id, exc)
                    sem = self._fallback_semantic(block=block)
                    self._force_assign_block(
                        block, sem, f"render_failed→{sem}: {exc}"
                    )

            # --- Pre-pass B: 并行 multi-label ---
            cls_tasks: List[Tuple[str, Path]] = []
            for bid, panel in iter_panel.items():
                blk = self.blocks.get(bid)
                if (
                    blk is not None
                    and blk.force_split
                    and blk.classification
                    and len(self._matched_parts_from_cls_dict(blk.classification)) >= 2
                ):
                    continue
                cls_tasks.append((bid, panel))
            cls_cache = self._batch_multi_label(cls_tasks)

            # --- Pre-pass C: 写入全部分类结果 ---
            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                panel_path = iter_panel.get(block_id)
                if panel_path is None:
                    continue
                cls = cls_cache.get(block_id)
                if cls is None:
                    try:
                        parent_matched = self._parent_matched_for_block(block)
                        at_max = block.depth >= int(self.config.max_depth)
                        cls = classify_multi_label(
                            self.client,
                            panel_path,
                            self.semantic_parts,
                            block_id=block_id,
                            parent_id=block.parent_id or "",
                            parent_matched=parent_matched,
                            at_max_depth=at_max,
                            current_depth=int(block.depth),
                            max_depth=int(self.config.max_depth),
                            model=self._resolve_vlm_model(
                                self.config.classification_model
                            ),
                        )
                    except (RuntimeError, ValueError) as exc:
                        LOGGER.warning(
                            "fallback multi-label 失败 (%s): %s",
                            block_id, exc,
                        )
                        cls = MultiLabelClassification(error=str(exc))
                self._write_block_classification(
                    block, cls, responses_dir, progress_cb=progress_cb
                )

            # --- Pre-pass D: 分裂子块语义覆盖（必须在再切分决策之前）---
            self._audit_split_coverage_before_decisions(
                queue,
                iter_panel,
                responses_dir,
                progress_cb=progress_cb,
            )

            # --- 决策主循环（基于 block.classification，含覆盖修复）---
            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                if block_id not in iter_panel:
                    continue  # 已在 Pre-pass A 处理

                cls_dict = block.classification or {}
                matched_parts = self._matched_parts_from_cls_dict(cls_dict)
                dominant = str(cls_dict.get("dominant") or "").strip()
                if not matched_parts and dominant:
                    matched_parts = [dominant]

                n_matched = len(matched_parts)

                if n_matched == 0:
                    sem = self._fallback_semantic(block=block)
                    self._force_assign_block(
                        block, sem, f"forced_assign(empty→{sem})"
                    )
                    continue

                # matched=1 → 归到该语义
                if n_matched == 1:
                    sem = matched_parts[0]
                    if sem not in self._part_index:
                        LOGGER.warning(
                            "block %s matched=%r 不在 part_index，强制兜底",
                            block_id, sem,
                        )
                        sem = self._fallback_semantic(block=block)
                        self._force_assign_block(
                            block, sem, f"forced_assign(unknown→{sem})"
                        )
                        continue
                    block.status = BlockStatus.FULL
                    self._assigned_semantic[block_id] = sem
                    block.notes = (
                        block.notes + f" | assigned={sem}"
                    ).strip(" |")
                    continue

                # matched ≥ 2：需要继续切
                if block.depth >= self.config.max_depth:
                    dom = dominant or matched_parts[0]
                    if dom not in self._part_index:
                        dom = self._fallback_semantic(block=block)
                    self._force_assign_block(
                        block,
                        dom,
                        f"max_depth→{dom} (matched={matched_parts})",
                    )
                    if progress_cb:
                        progress_cb(
                            "max_depth",
                            {"block_id": block_id, "dominant": dom},
                        )
                    continue

                seed_count = block.seed_count or max(
                    2,
                    min(
                        int(self.config.max_seeds_per_split),
                        n_matched,
                    ),
                )
                if n_matched > seed_count:
                    LOGGER.warning(
                        "block %s matched=%d 但 seed_count=%d（max_seeds_per_split）；"
                        "几何切分后将以覆盖检测补全缺失语义",
                        block_id,
                        n_matched,
                        seed_count,
                    )
                panel_path = iter_panel.get(block_id)
                LOGGER.info(
                    "block %s 开始几何分裂: faces=%d matched=%s seed_count=%d",
                    block_id,
                    block.size,
                    matched_parts,
                    seed_count,
                )
                if progress_cb:
                    progress_cb(
                        "split_start",
                        {
                            "block_id": block_id,
                            "faces": block.size,
                            "matched": list(matched_parts),
                            "seed_count": seed_count,
                        },
                    )
                sub_groups = self._split_block_by_seeds(
                    vertices,
                    faces,
                    block,
                    seed_count=seed_count,
                    responses_dir=responses_dir,
                    panel_path=panel_path,
                    matched_parts=list(matched_parts),
                    progress_cb=progress_cb,
                )
                if not sub_groups or len(sub_groups) < 2:
                    # 分裂失败：保守地把这个块归到 dominant 语义
                    dom = dominant or matched_parts[0]
                    if dom in self._part_index:
                        block.status = BlockStatus.FULL
                        self._assigned_semantic[block_id] = dom
                        block.notes = (
                            block.notes
                            + f" | split_failed → assigned={dom}"
                            f" (matched={matched_parts})"
                        ).strip(" |")
                    else:
                        dom = self._fallback_semantic(block=block)
                        self._force_assign_block(
                            block, dom, f"split_failed→{dom}"
                        )
                    if progress_cb:
                        progress_cb(
                            "split_failed",
                            {"block_id": block_id, "dominant": dom},
                        )
                    continue

                if len(sub_groups) != seed_count:
                    LOGGER.warning(
                        "block %s 分裂组数 %d != seed_count %d（已做连通块校正）",
                        block_id,
                        len(sub_groups),
                        seed_count,
                    )
                # 成功分裂
                block.status = BlockStatus.SPLIT
                block.force_split = False
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
                            f"(k={len(sub_groups)}, "
                            f"matched={matched_parts})"
                        ),
                        # 与父块的 segmat_done 一致：父块已做过几何切分（CC或segmat或FPS），
                        # 子块再切只会走 FPS（K=新一轮 matched 数）
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
                            "seed_count": seed_count,
                            "matched": list(matched_parts),
                            "children": list(block.children_ids),
                            "strategy": (
                                (block.seed_decision or {}).get("strategy")
                                or (block.seed_decision or {}).get("mode")
                            ),
                        },
                    )

            # 每轮末尾：写迭代时间线 + 刷新 live report
            if progress_cb:
                progress_cb(
                    "iteration_end",
                    {
                        "iteration": iteration,
                        "queue_after": len(next_queue),
                        "phase": "save_progress",
                    },
                )
            try:
                self._save_iteration_progress(
                    out_dir, vertices, faces, iteration=iteration,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("iter %d 时间线 PNG 失败: %s", iteration, exc)
            if progress_cb:
                progress_cb(
                    "iteration_end",
                    {
                        "iteration": iteration,
                        "queue_after": len(next_queue),
                        "phase": "live_report",
                    },
                )
            try:
                self._update_live_report(
                    out_dir,
                    vertices,
                    faces,
                    iteration=iteration,
                    queue_after=next_queue,
                    finished=False,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("live report 更新失败: %s", exc)
            if progress_cb:
                progress_cb(
                    "iteration_end",
                    {
                        "iteration": iteration,
                        "queue_after": len(next_queue),
                        "phase": "coverage_audit",
                    },
                )

            end_panel_by_id: Dict[str, Path] = dict(iter_panel)
            for bid, block in self.blocks.items():
                if bid in end_panel_by_id:
                    continue
                if block.panel_path:
                    p = Path(block.panel_path)
                    if p.is_file():
                        end_panel_by_id[bid] = p
            self._audit_all_split_coverage(
                end_panel_by_id,
                responses_dir,
                next_queue,
                progress_cb=progress_cb,
            )
            queue = next_queue

        LOGGER.info("主循环结束（%d 轮），进入收尾阶段 …", iteration)
        if progress_cb:
            progress_cb(
                "main_loop_done",
                {"iterations": iteration, "num_blocks": len(self.blocks)},
            )
        if progress_cb:
            progress_cb("finalize", {"phase": "assignments"})
        self._finalize_block_assignments()
        if progress_cb:
            progress_cb("finalize", {"phase": "empty_semantic_repair"})
        self._repair_empty_semantics(
            vertices,
            faces,
            out_dir,
            panel_dir,
            responses_dir,
            progress_cb=progress_cb,
        )
        if progress_cb:
            progress_cb("finalize", {"phase": "topology_repair"})
        self._repair_topology(
            vertices,
            faces,
            out_dir,
            panel_dir,
            responses_dir,
            progress_cb=progress_cb,
        )

        # ---- Step 4c: 边界优化（语义部位，VLM 完成后）----
        if self.config.convex_boundary_opt:
            if progress_cb:
                progress_cb("finalize", {"phase": "convex_boundary_opt"})
            self._apply_convex_optimize_final_blocks(
                vertices, faces, out_dir, n_faces, result,
                boundary_method="planar",
            )
        elif self.config.boundary_handle_loop:
            if progress_cb:
                progress_cb("finalize", {"phase": "boundary_handle_loop"})
            self._apply_convex_optimize_final_blocks(
                vertices, faces, out_dir, n_faces, result,
                boundary_method="handle_loop",
            )

        # ---- Step 5: 产出 ----
        result.num_iterations = iteration
        result.num_final_blocks = sum(
            1 for b in self.blocks.values() if b.status != BlockStatus.SPLIT
        )

        if self._convex_optimized_face_index is not None:
            face_idx = self._convex_optimized_face_index.copy()
        else:
            face_idx = self._compute_face_semantic_index(n_faces)
            face_idx = self._propagate_unlabeled_face_indices(face_idx, faces)
        result.face_semantic_index = face_idx.tolist()

        sem_groups: Dict[str, List[int]] = {n: [] for n in self.semantic_parts}
        pending_ids: List[int] = []
        for fid, v in enumerate(face_idx.tolist()):
            if v == -2:
                pending_ids.append(fid)
            elif 0 <= v < len(self.semantic_parts):
                sem_groups[self.semantic_parts[v]].append(fid)
        result.semantic_face_groups = sem_groups
        result.background_face_ids = []
        result.pending_face_ids = pending_ids

        labels_json = out_dir / "face_semantic_labels.json"
        try:
            with open(labels_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "num_faces": n_faces,
                        "semantic_parts": list(self.semantic_parts),
                        "face_semantic_index": result.face_semantic_index,
                        "num_background": 0,
                        "num_pending": len(pending_ids),
                        "semantic_face_counts": {
                            n: len(v) for n, v in sem_groups.items()
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            result.labels_json = labels_json
        except OSError as exc:
            LOGGER.warning("写 face_semantic_labels.json 失败: %s", exc)

        # 渲染全覆盖结果图
        coverage_panel = out_dir / "full_coverage_mask.png"
        try:
            render_full_coverage_panel(
                vertices,
                faces,
                result.face_semantic_index,
                self.semantic_parts,
                coverage_panel,
                title=(
                    "VLM 全覆盖语义分割"
                    + (
                        "（凸包优化后）"
                        if self._convex_optimized_face_index is not None
                        else "（按部位上色）"
                    )
                ),
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
            )
            result.coverage_panel = coverage_panel
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("全覆盖结果图渲染失败: %s", exc)

        # 叶分块总览（每块独立随机色，便于核对粒度）
        try:
            leaf_blocks = [
                b for b in self.blocks.values()
                if b.status != BlockStatus.SPLIT
            ]
            leaf_blocks.sort(key=lambda b: b.block_id)
            leaf_face_labels = np.full(n_faces, -1, dtype=np.int64)
            for k, b in enumerate(leaf_blocks):
                leaf_face_labels[np.asarray(b.face_ids, dtype=np.int64)] = k
            leaf_panel = out_dir / "leaf_blocks_overview.png"
            render_initial_segmentation_panel(
                vertices,
                faces,
                leaf_face_labels.tolist(),
                leaf_panel,
                title=(
                    f"叶分块总览  blocks={len(leaf_blocks)}  "
                    f"（每块一个独立随机色）"
                ),
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
            )
            result.extra["leaf_blocks_overview"] = str(leaf_panel)
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("叶分块总览渲染失败: %s", exc)

        # block_tree.json
        tree_extra: Dict[str, Any] = {
            "mode": "full_coverage",
            "semantic_parts": list(self.semantic_parts),
            "num_iterations": iteration,
            "num_initial_blocks": result.num_initial_blocks,
            "num_final_blocks": result.num_final_blocks,
            "max_depth": self.config.max_depth,
            "min_block_size": self.config.min_block_size,
            "max_seeds_per_split": self.config.max_seeds_per_split,
            "assigned_semantic": dict(self._assigned_semantic),
            "background_face_count": 0,
            "pending_face_count": len(pending_ids),
            "semantic_face_counts": {
                n: len(v) for n, v in sem_groups.items()
            },
            "empty_semantic_repairs": list(self._empty_semantic_repairs),
            "topology_constraints": [
                c.to_dict() for c in self._topology_constraints
            ],
            "topology_repairs": list(self._topology_repairs),
            "split_coverage_repairs": list(self._split_coverage_repairs),
        }
        if enumeration is not None:
            tree_extra["enumeration"] = enumeration.to_dict()
        if result.extra.get("convex_optimize"):
            tree_extra["convex_optimize"] = result.extra["convex_optimize"]
        tree_path = export_tree_json(
            self.blocks,
            out_dir / "block_tree.json",
            extra=tree_extra,
        )
        result.tree_json = tree_path

        # 最终 live_report：标记已完成
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
                    f"详细 block 信息见 block_tree.json，"
                    f"最终配色见 full_coverage_mask.png。"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("最终 live report 失败: %s", exc)

        result.iter_progress_pngs = list(self._iter_progress_pngs)

        # 最终单文件 HTML 复盘报告
        if self.config.generate_report:
            if progress_cb:
                progress_cb("finalize", {"phase": "report_html"})
            try:
                # 延迟 import 避免 pipeline 包内循环依赖
                from .full_coverage_report import generate_full_coverage_report

                report_path = generate_full_coverage_report(out_dir)
                result.report_html = report_path
                LOGGER.info("report.html 已生成: %s", report_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("最终 report.html 生成失败: %s", exc)

        result.ok = True
        result.elapsed_seconds = time.perf_counter() - t0
        pen_n = len(pending_ids)
        sem_counts_str = ", ".join(
            f"{n}={len(v)}" for n, v in sem_groups.items()
        )
        result.message = (
            f"完成 {iteration} 轮，初始 {result.num_initial_blocks} 块 → "
            f"最终 {result.num_final_blocks} 叶块；"
            f"{sem_counts_str}；pending={pen_n}"
        )
        return result


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------


def run_full_coverage(
    client: VLMClient,
    vertices: np.ndarray,
    faces: np.ndarray,
    output_dir: str | Path,
    *,
    initial_face_labels: Optional[Sequence[int]] = None,
    enumerate_hint: str = "",
    semantic_parts: Optional[List[str]] = None,
    enumerate_min_parts: int = 2,
    enumerate_max_parts: int = 12,
    max_depth: int = 6,
    min_block_size: int = 20,
    image_size: int = 512,
    panel_cols: int = 4,
    classification_model: Optional[str] = None,
    enumerate_model: Optional[str] = None,
    max_seeds_per_split: int = 4,
    split_strategy: str = "spectral",
    split_strategy_model: Optional[str] = None,
    skip_segmat: bool = True,
    vlm_split_strategy: bool = False,
    parallel_vlm_workers: int = 1,
    save_responses: bool = True,
    live_report: bool = True,
    live_report_refresh_seconds: int = 10,
    select_best_view: bool = True,
    view_select_model: Optional[str] = None,
    view_select_semantic_hint: str = "",
    single_view_image_size: int = 720,
    generate_report: bool = True,
    convex_boundary_opt: bool = True,
    boundary_handle_loop: bool = False,
    convex_opt_max_swaps: int = 800,
    convex_opt_min_faces: int = 8,
    convex_opt_min_component_faces: int = 20,
    topology_repair: bool = True,
    topology_repair_model: Optional[str] = None,
    topology_repair_max_rounds: int = 3,
    split_timeout_seconds: float = 60.0,
    split_strategy_max_retries: int = 2,
    split_retry_with_vlm: bool = True,
    progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> FullCoverageResult:
    """函数式封装，等价于 ``FullCoveragePipeline(client, cfg).run(...)``。"""
    cfg = FullCoverageConfig(
        enumerate_hint=enumerate_hint,
        semantic_parts=semantic_parts,
        enumerate_min_parts=enumerate_min_parts,
        enumerate_max_parts=enumerate_max_parts,
        max_depth=max_depth,
        min_block_size=min_block_size,
        image_size=image_size,
        panel_cols=panel_cols,
        classification_model=classification_model,
        enumerate_model=enumerate_model,
        max_seeds_per_split=max_seeds_per_split,
        split_strategy=split_strategy,
        split_strategy_model=split_strategy_model,
        skip_segmat=skip_segmat,
        vlm_split_strategy=vlm_split_strategy,
        parallel_vlm_workers=parallel_vlm_workers,
        save_responses=save_responses,
        live_report=live_report,
        live_report_refresh_seconds=live_report_refresh_seconds,
        select_best_view=select_best_view,
        view_select_model=view_select_model,
        view_select_semantic_hint=view_select_semantic_hint,
        single_view_image_size=single_view_image_size,
        generate_report=generate_report,
        convex_boundary_opt=convex_boundary_opt,
        boundary_handle_loop=boundary_handle_loop,
        convex_opt_max_swaps=convex_opt_max_swaps,
        convex_opt_min_faces=convex_opt_min_faces,
        convex_opt_min_component_faces=convex_opt_min_component_faces,
        topology_repair=topology_repair,
        topology_repair_model=topology_repair_model,
        topology_repair_max_rounds=topology_repair_max_rounds,
        split_timeout_seconds=split_timeout_seconds,
        split_strategy_max_retries=split_strategy_max_retries,
        split_retry_with_vlm=split_retry_with_vlm,
    )
    pipeline = FullCoveragePipeline(client, cfg)
    return pipeline.run(
        vertices,
        faces,
        output_dir,
        initial_face_labels=initial_face_labels,
        progress_cb=progress_cb,
    )


__all__ = [
    "FullCoverageConfig",
    "FullCoveragePipeline",
    "FullCoverageResult",
    "run_full_coverage",
]
