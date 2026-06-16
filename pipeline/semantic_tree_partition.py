"""
语义树分块流水线：

1. VLM 看整模型渲染图，生成层级语义树（如 整体→上半身/下半身→头/躯干/手臂→…）
2. **第一层全覆盖**：不管语义树多少层，先用「原版」多标签全覆盖循环把第一层全部
   分好——VLM 对每块判断它命中哪些一级语义，命中 1 个就弹出归属，命中多个就选种子
   + 谱/骨架/FPS 聚类策略几何切分后再判断，直到每块只属于一个一级语义。
3. **子分块的语义树**：第一层全部分好后，再对每个已分好的块做同样的多标签全覆盖
   循环，把它细分成下一层子语义；逐层向下，直到叶子或 refine 范围边界。
4. 用户通过 refine_depth / refine_paths / refine_nodes / refine_all 控制细分范围。

输出目录结构与 full_coverage 类似，mode=semantic_tree。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry.blocks import BlockStatus, SemanticBlock, export_tree_json
from geometry.seed_split import split_block_by_connected_components
from rendering.render_component import (
    render_full_coverage_panel,
    render_initial_segmentation_panel,
    render_overview_only_panel,
    render_single_view_overview,
)
from vlm.client import VLMClient
from vlm.full_coverage import MultiLabelClassification, classify_multi_label
from vlm.semantic_tree import (
    SemanticTreeNode,
    SemanticTreeResult,
    generate_semantic_tree,
    load_semantic_tree_file,
    should_refine_node,
)
from vlm.view_selector import BestViewSelection, select_best_view

from .full_coverage import FullCoverageConfig, FullCoveragePipeline

LOGGER = logging.getLogger("pipeline.semantic_tree_partition")


@dataclass
class SemanticTreePartitionConfig:
    """语义树分块配置。"""

    tree_hint: str = ""
    semantic_tree_file: Optional[str] = None
    max_tree_depth: int = 5
    tree_model: Optional[str] = None

    # 细分控制（三选一或组合；refine_all 优先）
    refine_depth: int = 1
    refine_all: bool = False
    refine_paths: Optional[List[str]] = None
    refine_nodes: Optional[List[str]] = None

    min_block_size: int = 20
    max_depth: int = 6
    image_size: int = 512
    panel_cols: int = 4
    classification_model: Optional[str] = None
    max_seeds_per_split: int = 8
    split_strategy: str = "spectral"
    split_strategy_model: Optional[str] = None
    skip_segmat: bool = True
    vlm_split_strategy: bool = False
    split_timeout_seconds: float = 60.0
    split_strategy_max_retries: int = 2
    split_retry_with_vlm: bool = True
    save_responses: bool = True
    select_best_view: bool = True
    view_select_model: Optional[str] = None
    view_select_semantic_hint: str = ""
    single_view_image_size: int = 720
    generate_report: bool = False
    topology_repair: bool = False
    convex_boundary_opt: bool = False
    boundary_handle_loop: bool = False
    convex_opt_max_swaps: int = 800
    convex_opt_min_faces: int = 8
    convex_opt_min_component_faces: int = 20
    live_report: bool = True
    live_report_refresh_seconds: int = 10
    # 「几乎不可见」的小块阈值（占当前细分区域表面积比例）。低于该比例的块不再调 VLM
    # 切分，而是直接按相邻边最多归入已分好的相邻块语义。
    min_visible_area_frac: float = 0.005


@dataclass
class SemanticTreePartitionResult:
    ok: bool
    output_dir: Path
    message: str = ""
    semantic_tree: Optional[SemanticTreeResult] = None
    semantic_parts: List[str] = field(default_factory=list)
    face_semantic_index: Optional[List[int]] = None
    semantic_face_groups: Dict[str, List[int]] = field(default_factory=dict)
    num_tree_levels: int = 0
    num_final_blocks: int = 0
    tree_json: Optional[Path] = None
    labels_json: Optional[Path] = None
    coverage_panel: Optional[Path] = None
    overview_panel: Optional[Path] = None
    best_view: Optional[BestViewSelection] = None
    elapsed_seconds: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _WorkItem:
    block_id: str
    node: SemanticTreeNode


class SemanticTreePartitionPipeline:
    """语义树逐层分块。"""

    def __init__(self, client: VLMClient, config: SemanticTreePartitionConfig):
        self.client = client
        self.config = config
        fc_cfg = FullCoverageConfig(
            min_block_size=config.min_block_size,
            max_depth=config.max_depth,
            image_size=config.image_size,
            panel_cols=config.panel_cols,
            classification_model=config.classification_model,
            max_seeds_per_split=config.max_seeds_per_split,
            split_strategy=config.split_strategy,
            split_strategy_model=config.split_strategy_model,
            skip_segmat=config.skip_segmat,
            vlm_split_strategy=config.vlm_split_strategy,
            split_timeout_seconds=config.split_timeout_seconds,
            split_strategy_max_retries=config.split_strategy_max_retries,
            split_retry_with_vlm=config.split_retry_with_vlm,
            save_responses=config.save_responses,
            select_best_view=False,
            generate_report=False,
            topology_repair=config.topology_repair,
            convex_boundary_opt=config.convex_boundary_opt,
            boundary_handle_loop=config.boundary_handle_loop,
            convex_opt_max_swaps=config.convex_opt_max_swaps,
            convex_opt_min_faces=config.convex_opt_min_faces,
            convex_opt_min_component_faces=config.convex_opt_min_component_faces,
            live_report=config.live_report,
            live_report_refresh_seconds=config.live_report_refresh_seconds,
        )
        self._fc = FullCoveragePipeline(client, fc_cfg)
        self._semantic_tree: Optional[SemanticTreeResult] = None
        self._node_by_block: Dict[str, SemanticTreeNode] = {}
        self._global_iter: int = 0
        # 几何缓存（首次用到时构建）：面邻接对 + 单面面积
        self._face_adj_pairs: Optional[np.ndarray] = None
        self._face_area: Optional[np.ndarray] = None

    def _resolve_vlm_model(self, explicit: Optional[str]) -> str:
        return self._fc._resolve_vlm_model(explicit)

    def _dump_response(
        self, responses_dir: Path, name: str, payload: Dict[str, Any]
    ) -> None:
        self._fc._dump_response(responses_dir, name, payload)

    def _resolve_semantic_tree(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        out_dir: Path,
    ) -> Tuple[SemanticTreeResult, Path]:
        overview_path = out_dir / "semantics_overview.png"
        render_overview_only_panel(
            vertices,
            faces,
            overview_path,
            title="整模型 12 视角（语义树生成）",
            image_size=self.config.image_size,
            cols=self.config.panel_cols,
        )

        if self.config.semantic_tree_file:
            tree = load_semantic_tree_file(self.config.semantic_tree_file)
            LOGGER.info("使用用户提供的 semantic_tree: %s", self.config.semantic_tree_file)
        else:
            LOGGER.info("调用 VLM 生成层级语义树 ...")
            tree = generate_semantic_tree(
                self.client,
                overview_path,
                hint=self.config.tree_hint,
                model=self._resolve_vlm_model(
                    self.config.tree_model or self.config.classification_model
                ),
                max_depth=self.config.max_tree_depth,
            )

        tree_path = out_dir / "semantic_tree.json"
        tree_path.write_text(
            json.dumps(tree.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if not tree.ok:
            raise RuntimeError(
                f"语义树生成/加载失败: {tree.error or '未知错误'}"
            )
        return tree, overview_path

    def _assign_block_semantic(
        self,
        block_id: str,
        semantic: str,
        *,
        notes: str = "",
        mark_full: bool = True,
    ) -> None:
        block = self._fc.blocks[block_id]
        if semantic not in self._fc._part_index:
            self._fc.semantic_parts.append(semantic)
            self._fc._part_index[semantic] = len(self._fc.semantic_parts) - 1
        if mark_full:
            block.status = BlockStatus.FULL
        self._fc._assigned_semantic[block_id] = semantic
        if notes:
            block.notes = (block.notes + " | " + notes).strip(" |")

    def _refresh_live_report(
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
        try:
            self._fc._update_live_report(
                out_dir,
                vertices,
                faces,
                iteration=iteration,
                queue_after=queue_after,
                finished=finished,
                extra_note=extra_note,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("live report 更新失败: %s", exc)

    def _save_iter_progress(
        self,
        out_dir: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        iteration: int,
    ) -> None:
        try:
            self._fc._save_iteration_progress(
                out_dir, vertices, faces, iteration=iteration,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("iter %d 时间线 PNG 失败: %s", iteration, exc)

    def _should_refine(self, node: SemanticTreeNode) -> bool:
        return should_refine_node(
            node,
            refine_depth=self.config.refine_depth,
            refine_all=self.config.refine_all,
            refine_paths=self.config.refine_paths,
            refine_nodes=self.config.refine_nodes,
        )

    def _resolve_block_to_child(
        self,
        block_id: str,
        child_node: SemanticTreeNode,
        refine_next: List[_WorkItem],
        *,
        notes: str = "",
    ) -> None:
        """把块「弹出」归属到某个子语义；若该子语义还有下一层且需细分，记入下一阶段。"""
        self._node_by_block[block_id] = child_node
        self._assign_block_semantic(
            block_id, child_node.name, notes=notes, mark_full=True
        )
        if not child_node.is_leaf and self._should_refine(child_node):
            refine_next.append(_WorkItem(block_id=block_id, node=child_node))

    def _assign_invisible(
        self,
        block_id: str,
        child_node: SemanticTreeNode,
        *,
        notes: str,
    ) -> None:
        """不可见小块直接归入某子语义（不再向下细分）。"""
        self._node_by_block[block_id] = child_node
        self._assign_block_semantic(
            block_id, child_node.name, notes=notes, mark_full=True
        )

    # ---------- 几何缓存 / 可见性 / 相邻归并 ----------
    def _ensure_geometry_cache(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> None:
        if self._face_adj_pairs is not None and self._face_area is not None:
            return
        try:
            import trimesh

            mesh = trimesh.Trimesh(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                process=False,
            )
            self._face_adj_pairs = np.asarray(mesh.face_adjacency, dtype=np.int64)
            self._face_area = np.asarray(mesh.area_faces, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("几何缓存构建失败（相邻归并将退化）: %s", exc)
            self._face_adj_pairs = np.zeros((0, 2), dtype=np.int64)
            self._face_area = np.ones(int(faces.shape[0]), dtype=np.float64)

    def _block_area(self, block: SemanticBlock) -> float:
        if self._face_area is None:
            return float(block.size)
        fb = np.asarray(block.face_ids, dtype=np.int64)
        fb = fb[(fb >= 0) & (fb < self._face_area.shape[0])]
        if fb.size == 0:
            return 0.0
        return float(self._face_area[fb].sum())

    def _is_invisible(self, block: SemanticBlock, region_area: float) -> bool:
        """块是否「几乎不可见」：面数过小，或占当前区域面积比例过低。"""
        if block.size < self.config.min_block_size:
            return True
        if region_area > 0.0:
            frac = self._block_area(block) / region_area
            if frac < self.config.min_visible_area_frac:
                return True
        return False

    def _neighbor_child_semantic(
        self,
        block: SemanticBlock,
        faces: np.ndarray,
        candidate_names: Sequence[str],
        *,
        face_idx: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """返回与该块共享边界边最多、且已归属候选语义的相邻块语义。

        没有任何已分好的候选相邻块时返回 None（交由上层延后处理）。
        """
        pairs = self._face_adj_pairs
        if pairs is None or pairs.shape[0] == 0:
            return None
        n_faces = int(faces.shape[0])
        in_block = np.zeros(n_faces, dtype=bool)
        fb = np.asarray(block.face_ids, dtype=np.int64)
        fb = fb[(fb >= 0) & (fb < n_faces)]
        if fb.size == 0:
            return None
        in_block[fb] = True

        f0 = pairs[:, 0]
        f1 = pairs[:, 1]
        b0 = in_block[f0]
        b1 = in_block[f1]
        boundary = b0 ^ b1  # 一面在块内、一面在块外
        if not boundary.any():
            return None
        outside = np.where(b0[boundary], f1[boundary], f0[boundary])

        if face_idx is None:
            face_idx = self._fc._compute_face_semantic_index(n_faces)
        parts = self._fc.semantic_parts
        cand_set = set(candidate_names)
        # 每条边界边按相邻面所属语义累加（边界边数 = 相邻边权重）
        sem_of_outside = face_idx[outside]
        counts: Dict[str, int] = {}
        for v in sem_of_outside:
            iv = int(v)
            if iv < 0 or iv >= len(parts):
                continue
            name = parts[iv]
            if name not in cand_set:
                continue
            counts[name] = counts.get(name, 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda k: counts[k])

    def _resolve_deferred(
        self,
        deferred: List[str],
        faces: np.ndarray,
        candidate_parts: Sequence[str],
        child_node_by_name: Dict[str, SemanticTreeNode],
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> None:
        """收尾处理延后的不可见块：此时相邻块多已分好，按相邻边最多归并；
        若始终没有任何已确定语义的相邻块，则兜底归入主候选语义。"""
        n_faces = int(faces.shape[0])
        remaining = list(dict.fromkeys(deferred))
        for _pass in range(6):
            if not remaining:
                break
            face_idx = self._fc._compute_face_semantic_index(n_faces)
            still: List[str] = []
            progressed = False
            for bid in remaining:
                blk = self._fc.blocks.get(bid)
                if blk is None or blk.status != BlockStatus.PENDING:
                    continue
                name = self._neighbor_child_semantic(
                    blk, faces, candidate_parts, face_idx=face_idx
                )
                if name is not None:
                    self._assign_invisible(
                        bid, child_node_by_name[name],
                        notes=f"invisible→邻接归并:{name}",
                    )
                    progressed = True
                    if progress_cb:
                        progress_cb(
                            "abandoned",
                            {"block_id": bid, "reason": "invisible_merge", "assigned": name},
                        )
                else:
                    still.append(bid)
            remaining = still
            if not progressed:
                break

        # 仍无任何已分好的相邻块：兜底
        for bid in remaining:
            blk = self._fc.blocks.get(bid)
            if blk is None or blk.status != BlockStatus.PENDING:
                continue
            dom = self._dominant_child(blk, candidate_parts)
            self._assign_invisible(
                bid, child_node_by_name[dom],
                notes="invisible→无已分相邻块兜底",
            )
            if progress_cb:
                progress_cb(
                    "abandoned",
                    {"block_id": bid, "reason": "invisible_no_neighbor", "assigned": dom},
                )

    def _coverage_partition(
        self,
        parent_block_id: str,
        node: SemanticTreeNode,
        vertices: np.ndarray,
        faces: np.ndarray,
        panel_dir: Path,
        responses_dir: Path,
        out_dir: Path,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]],
    ) -> List[_WorkItem]:
        """对 ``parent_block`` 所在区域跑「原版」多标签全覆盖循环。

        候选语义 = ``node.children``。流程与 full_coverage 一致：
          1. 区域按连通分量拆成初始块；
          2. VLM 对每块多标签判断它命中哪些子语义；
          3. 命中 1 个 → 弹出（归属该语义）；命中 ≥2 个 → 选种子+策略几何切分，
             子块再次进入下一轮判断；命中 0 个 → 兜底归属；
          4. 直到区域内每块只命中一个子语义。

        返回需要继续向下细分的 ``(block_id, child_node)``。
        """
        children_nodes = list(node.children)
        if not children_nodes:
            return []

        candidate_parts = [c.name for c in children_nodes]
        for name in candidate_parts:
            if name not in self._fc._part_index:
                self._fc.semantic_parts.append(name)
                self._fc._part_index[name] = len(self._fc.semantic_parts) - 1
        child_node_by_name: Dict[str, SemanticTreeNode] = {
            c.name: c for c in children_nodes
        }
        max_seeds = max(2, min(self.config.max_seeds_per_split, len(candidate_parts)))

        self._ensure_geometry_cache(vertices, faces)

        # 区域初始块：先把 parent 区域按连通分量拆开
        parent = self._fc.blocks[parent_block_id]
        parent.status = BlockStatus.PENDING
        self._fc._assigned_semantic.pop(parent_block_id, None)
        region_area = self._block_area(parent)

        queue: List[str] = []
        cc_groups = split_block_by_connected_components(
            vertices,
            faces,
            parent.face_ids,
            min_component_faces=max(1, self.config.min_block_size // 2),
        )
        if cc_groups is not None and len(cc_groups) >= 2:
            parent.status = BlockStatus.SPLIT
            for g in cc_groups:
                cid = self._fc._next_child_id(parent_block_id)
                cb = SemanticBlock(
                    block_id=cid,
                    face_ids=list(g),
                    depth=parent.depth + 1,
                    parent_id=parent_block_id,
                    status=BlockStatus.PENDING,
                    notes=f"cc_init under {node.path or node.name}",
                    segmat_done=parent.segmat_done,
                )
                self._fc._register_block(cb)
                parent.children_ids.append(cid)
                queue.append(cid)
        else:
            queue = [parent_block_id]

        # 单一候选：整区域直接归属
        if len(candidate_parts) == 1:
            refine_next: List[_WorkItem] = []
            for bid in queue:
                self._resolve_block_to_child(
                    bid, children_nodes[0], refine_next, notes="single_child"
                )
            return refine_next

        refine_next = []
        # 延后处理的不可见块（相邻块尚未分好时先存起来，收尾再按邻接归并）
        deferred: List[str] = []
        # 安全迭代上限（仅兜底；正常靠「命中=1 弹出 + 不可见块邻接归并」自然收敛停止）
        max_local_iter = max(20, len(candidate_parts) * 12)
        local_iter = 0

        n_faces_total = int(faces.shape[0])
        while queue:
            local_iter += 1
            capped = local_iter >= max_local_iter
            next_queue: List[str] = []
            # 本轮相邻归并用的 face 语义索引（懒计算一次）
            iter_face_idx: Optional[np.ndarray] = None
            for block_id in queue:
                block = self._fc.blocks.get(block_id)
                if block is None or block.status != BlockStatus.PENDING:
                    continue

                # 「几乎不可见」的小块：不再调 VLM 切分，直接按相邻边最多归入
                # 已分好的相邻块；相邻块都还没确定语义则延后到收尾处理。
                if self._is_invisible(block, region_area) or capped:
                    if iter_face_idx is None:
                        iter_face_idx = self._fc._compute_face_semantic_index(
                            n_faces_total
                        )
                    name = self._neighbor_child_semantic(
                        block, faces, candidate_parts, face_idx=iter_face_idx
                    )
                    if name is not None:
                        self._assign_invisible(
                            block_id, child_node_by_name[name],
                            notes=f"invisible→邻接归并:{name}",
                        )
                        if progress_cb:
                            progress_cb(
                                "abandoned",
                                {
                                    "block_id": block_id,
                                    "reason": "invisible_merge",
                                    "assigned": name,
                                },
                            )
                    else:
                        deferred.append(block_id)
                    continue

                try:
                    panel_path = self._fc._render_block_panel(
                        vertices, faces, block, panel_dir
                    )
                    block.panel_path = str(panel_path)
                except (OSError, ImportError, ValueError) as exc:
                    LOGGER.warning("渲染 block %s 失败: %s", block_id, exc)
                    dom = self._dominant_child(block, candidate_parts)
                    self._resolve_block_to_child(
                        block_id, child_node_by_name[dom], refine_next,
                        notes="render_failed",
                    )
                    continue

                parent_matched = self._fc._parent_matched_for_block(block)
                at_max_depth = block.depth >= int(self._fc.config.max_depth)
                try:
                    cls = classify_multi_label(
                        self.client,
                        panel_path,
                        candidate_parts,
                        block_id=block_id,
                        parent_id=block.parent_id or "",
                        parent_matched=parent_matched,
                        at_max_depth=at_max_depth,
                        current_depth=int(block.depth),
                        max_depth=int(self._fc.config.max_depth),
                        model=self._resolve_vlm_model(
                            self.config.classification_model
                        ),
                    )
                except (RuntimeError, ValueError) as exc:
                    LOGGER.warning("multi-label 分类失败 (%s): %s", block_id, exc)
                    cls = MultiLabelClassification(error=str(exc))
                self._fc._write_block_classification(
                    block, cls, responses_dir, progress_cb=progress_cb
                )

                cls_dict = block.classification or {}
                matched = [
                    m
                    for m in self._fc._matched_parts_from_cls_dict(cls_dict)
                    if m in child_node_by_name
                ]
                dominant = str(cls_dict.get("dominant") or "").strip()
                if not matched and dominant in child_node_by_name:
                    matched = [dominant]
                n_matched = len(matched)

                if n_matched == 0:
                    dom = self._dominant_child(block, candidate_parts)
                    self._resolve_block_to_child(
                        block_id, child_node_by_name[dom], refine_next,
                        notes=f"empty→{dom}",
                    )
                    continue

                if n_matched == 1:
                    self._resolve_block_to_child(
                        block_id, child_node_by_name[matched[0]], refine_next,
                        notes="matched=1",
                    )
                    continue

                # 已达 max_depth：不再几何切分，强制归入 dominant
                if at_max_depth:
                    dom = dominant if dominant in child_node_by_name else matched[0]
                    self._resolve_block_to_child(
                        block_id, child_node_by_name[dom], refine_next,
                        notes=f"max_depth→{dom}",
                    )
                    if progress_cb:
                        progress_cb(
                            "max_depth",
                            {
                                "block_id": block_id,
                                "dominant": dom,
                                "matched": list(matched),
                                "depth": int(block.depth),
                            },
                        )
                    continue

                # n_matched >= 2 → 选种子+策略几何切分
                seed_count = max(2, min(max_seeds, n_matched))
                if progress_cb:
                    progress_cb(
                        "split_start",
                        {
                            "block_id": block_id,
                            "faces": block.size,
                            "matched": list(matched),
                            "seed_count": seed_count,
                            "tree_node": node.path or node.name,
                        },
                    )
                sub_groups = self._fc._split_block_by_seeds(
                    vertices,
                    faces,
                    block,
                    seed_count=seed_count,
                    responses_dir=responses_dir,
                    panel_path=Path(block.panel_path) if block.panel_path else None,
                    matched_parts=list(matched),
                    progress_cb=progress_cb,
                )
                if not sub_groups or len(sub_groups) < 2:
                    dom = dominant if dominant in child_node_by_name else matched[0]
                    self._resolve_block_to_child(
                        block_id, child_node_by_name[dom], refine_next,
                        notes=f"split_failed→{dom}",
                    )
                    if progress_cb:
                        progress_cb(
                            "split_failed",
                            {"block_id": block_id, "dominant": dom},
                        )
                    continue

                block.status = BlockStatus.SPLIT
                self._fc._assigned_semantic.pop(block_id, None)
                new_children: List[str] = []
                for g in sub_groups:
                    cid = self._fc._next_child_id(block_id)
                    cb = SemanticBlock(
                        block_id=cid,
                        face_ids=list(g),
                        depth=block.depth + 1,
                        parent_id=block_id,
                        status=BlockStatus.PENDING,
                        notes=(
                            f"split under {node.path or node.name} "
                            f"matched={matched}"
                        ),
                        segmat_done=block.segmat_done,
                    )
                    self._fc._register_block(cb)
                    block.children_ids.append(cid)
                    next_queue.append(cid)
                    new_children.append(cid)
                if progress_cb:
                    seed_dec = block.seed_decision or {}
                    progress_cb(
                        "split",
                        {
                            "block_id": block_id,
                            "seed_count": seed_count,
                            "matched": list(matched),
                            "children": new_children,
                            "tree_node": node.path or node.name,
                            "strategy": (
                                seed_dec.get("strategy")
                                or seed_dec.get("mode")
                            ),
                        },
                    )

            self._global_iter += 1
            self._save_iter_progress(
                out_dir, vertices, faces, iteration=self._global_iter
            )
            self._refresh_live_report(
                out_dir,
                vertices,
                faces,
                iteration=self._global_iter,
                queue_after=next_queue,
                finished=False,
                extra_note=(
                    f"分块 {node.path or node.name} → "
                    f"候选 {candidate_parts}，待处理 {len(next_queue)} 块"
                ),
            )
            queue = next_queue

        # 收尾：把延后的不可见块按相邻边最多归并（相邻块都没语义则兜底）
        if deferred:
            self._resolve_deferred(
                deferred, faces, candidate_parts, child_node_by_name, progress_cb
            )

        return refine_next

    def _dominant_child(
        self, block: SemanticBlock, candidate_parts: Sequence[str]
    ) -> str:
        """从块当前分类结果里取一个最可能的候选子语义（兜底用）。"""
        cls_dict = block.classification or {}
        dominant = str(cls_dict.get("dominant") or "").strip()
        if dominant in candidate_parts:
            return dominant
        for m in self._fc._matched_parts_from_cls_dict(cls_dict):
            if m in candidate_parts:
                return m
        return candidate_parts[0]

    def run(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_dir: str | Path,
        *,
        progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> SemanticTreePartitionResult:
        t0 = time.perf_counter()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        panel_dir = out_dir / "panels"
        panel_dir.mkdir(parents=True, exist_ok=True)
        responses_dir = out_dir / "responses"
        if self.config.save_responses:
            responses_dir.mkdir(parents=True, exist_ok=True)

        n_faces = int(faces.shape[0])
        result = SemanticTreePartitionResult(ok=False, output_dir=out_dir)

        if progress_cb:
            progress_cb("tree_generate", {"phase": "vlm_semantic_tree"})

        try:
            tree, overview_path = self._resolve_semantic_tree(
                vertices, faces, out_dir
            )
        except RuntimeError as exc:
            result.message = str(exc)
            result.elapsed_seconds = time.perf_counter() - t0
            return result

        self._semantic_tree = tree
        result.semantic_tree = tree
        result.overview_panel = overview_path

        # 初始语义表：预注册树中所有节点名
        for node in tree.root.iter_nodes():
            name = node.name.strip()
            if name and name != "整体" and name not in self._fc._part_index:
                self._fc.semantic_parts.append(name)
                self._fc._part_index[name] = len(self._fc.semantic_parts) - 1

        # 整 mesh 单块
        root_block = SemanticBlock(
            block_id="C0",
            face_ids=list(range(n_faces)),
            depth=0,
            parent_id=None,
            status=BlockStatus.PENDING,
            notes="semantic_tree root",
        )
        self._fc._register_block(root_block)
        self._node_by_block["C0"] = tree.root

        if self.config.select_best_view and overview_path and overview_path.is_file():
            try:
                thumb = out_dir / "view_thumbnails.png"
                from rendering.view_thumbnails import render_thumbnails_panel

                render_thumbnails_panel(
                    vertices,
                    faces,
                    thumb,
                    thumb_size=min(256, self.config.image_size),
                )
                hint = self.config.view_select_semantic_hint or "整体结构"
                bv = select_best_view(
                    self.client,
                    thumb,
                    hint,
                    model=self._resolve_vlm_model(self.config.view_select_model),
                )
                bv.thumbnails_panel = thumb
                ov_path = out_dir / "best_view_overview.png"
                try:
                    render_single_view_overview(
                        vertices,
                        faces,
                        ov_path,
                        elev=bv.elev,
                        azim=bv.azim,
                        image_size=self.config.single_view_image_size,
                        background="transparent",
                    )
                    bv.overview_png = ov_path
                except (OSError, ImportError, ValueError) as exc:
                    LOGGER.warning("best view overview 渲染失败: %s", exc)
                result.best_view = bv
                self._fc._best_view = bv
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("视角选择失败: %s", exc)

        self._global_iter = 0
        self._refresh_live_report(
            out_dir,
            vertices,
            faces,
            iteration=0,
            queue_after=["C0"],
            finished=False,
            extra_note="语义树已生成，开始第一层全覆盖切分",
        )

        root_node = tree.root
        root_block = self._fc.blocks["C0"]

        # ---- Phase 1: 第一层全覆盖（原版多标签循环，与 refine 无关）----
        # 不管语义树多少层，先用 VLM 多标签 + 谱/骨架/FPS 聚类把第一层全部分好，
        # 直到每块只属于一个一级语义。
        LOGGER.info("== Phase 1: 第一层全覆盖 ==")
        if progress_cb:
            progress_cb("phase_start", {"phase": "base_coverage"})

        if not root_node.children:
            self._assign_block_semantic(
                "C0", root_node.name or "整体", notes="no_children"
            )
            refine_queue: List[_WorkItem] = []
        else:
            refine_queue = self._coverage_partition(
                "C0",
                root_node,
                vertices,
                faces,
                panel_dir,
                responses_dir,
                out_dir,
                progress_cb,
            )

        if progress_cb:
            progress_cb(
                "phase_end",
                {"phase": "base_coverage", "refine_queue": len(refine_queue)},
            )
        self._refresh_live_report(
            out_dir,
            vertices,
            faces,
            iteration=self._global_iter,
            queue_after=[w.block_id for w in refine_queue],
            finished=False,
            extra_note=(
                f"第一层全覆盖完成，待细分 {len(refine_queue)} 块"
                if refine_queue
                else "第一层全覆盖完成（无需进一步细分）"
            ),
        )

        # ---- Phase 2+: 对已分好的块逐层做子分块的语义树 ----
        level = 1
        queue = refine_queue
        while queue:
            level += 1
            LOGGER.info("== Phase 2: 第 %d 层细分, items=%d ==", level, len(queue))
            if progress_cb:
                progress_cb(
                    "phase_start",
                    {"phase": "refine", "level": level, "items": len(queue)},
                )
            next_queue: List[_WorkItem] = []
            for item in queue:
                block = self._fc.blocks.get(item.block_id)
                if block is None:
                    continue
                next_queue.extend(
                    self._coverage_partition(
                        item.block_id,
                        item.node,
                        vertices,
                        faces,
                        panel_dir,
                        responses_dir,
                        out_dir,
                        progress_cb,
                    )
                )
            if progress_cb:
                progress_cb(
                    "iteration_end",
                    {
                        "iteration": level,
                        "queue_after": len(next_queue),
                        "phase": "semantic_tree_refine",
                    },
                )
            queue = next_queue

        result.num_tree_levels = level

        # 边界优化（可选）
        if self.config.convex_boundary_opt:
            if progress_cb:
                progress_cb("finalize", {"phase": "convex_boundary_opt"})
            self._fc._apply_convex_optimize_final_blocks(
                vertices, faces, out_dir, n_faces, result,
                boundary_method="planar",
            )
        elif self.config.boundary_handle_loop:
            if progress_cb:
                progress_cb("finalize", {"phase": "boundary_handle_loop"})
            self._fc._apply_convex_optimize_final_blocks(
                vertices, faces, out_dir, n_faces, result,
                boundary_method="handle_loop",
            )

        # 产出 face 语义索引
        if self._fc._convex_optimized_face_index is not None:
            face_idx = self._fc._convex_optimized_face_index.copy()
        else:
            face_idx = self._fc._compute_face_semantic_index(n_faces)
            face_idx = self._fc._propagate_unlabeled_face_indices(face_idx, faces)
        result.face_semantic_index = face_idx.tolist()

        sem_groups: Dict[str, List[int]] = {
            n: [] for n in self._fc.semantic_parts
        }
        pending: List[int] = []
        for fid, v in enumerate(face_idx.tolist()):
            if v == -2:
                pending.append(fid)
            elif 0 <= v < len(self._fc.semantic_parts):
                sem_groups[self._fc.semantic_parts[v]].append(fid)

        result.semantic_parts = list(self._fc.semantic_parts)
        result.semantic_face_groups = sem_groups
        result.num_final_blocks = sum(
            1 for b in self._fc.blocks.values() if b.status != BlockStatus.SPLIT
        )

        labels_json = out_dir / "face_semantic_labels.json"
        labels_json.write_text(
            json.dumps(
                {
                    "num_faces": n_faces,
                    "semantic_parts": result.semantic_parts,
                    "face_semantic_index": result.face_semantic_index,
                    "num_pending": len(pending),
                    "semantic_face_counts": {
                        n: len(v) for n, v in sem_groups.items()
                    },
                    "mode": "semantic_tree",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        result.labels_json = labels_json

        coverage_panel = out_dir / "full_coverage_mask.png"
        try:
            render_full_coverage_panel(
                vertices,
                faces,
                result.face_semantic_index,
                result.semantic_parts,
                coverage_panel,
                title="语义树分块结果",
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
            )
            result.coverage_panel = coverage_panel
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("全覆盖图渲染失败: %s", exc)

        # 树层级可视化
        try:
            leaf_blocks = [
                b for b in self._fc.blocks.values()
                if b.status != BlockStatus.SPLIT
            ]
            leaf_blocks.sort(key=lambda b: b.block_id)
            leaf_labels = np.full(n_faces, -1, dtype=np.int64)
            for k, b in enumerate(leaf_blocks):
                leaf_labels[np.asarray(b.face_ids, dtype=np.int64)] = k
            leaf_panel = out_dir / "leaf_blocks_overview.png"
            render_initial_segmentation_panel(
                vertices,
                faces,
                leaf_labels.tolist(),
                leaf_panel,
                title=f"语义树叶分块 blocks={len(leaf_blocks)}",
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
            )
            result.extra["leaf_blocks_overview"] = str(leaf_panel)
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("叶分块图渲染失败: %s", exc)

        tree_extra: Dict[str, Any] = {
            "mode": "semantic_tree",
            "semantic_parts": result.semantic_parts,
            "num_tree_levels": level,
            "num_final_blocks": result.num_final_blocks,
            "refine_depth": self.config.refine_depth,
            "refine_all": self.config.refine_all,
            "refine_paths": list(self.config.refine_paths or []),
            "refine_nodes": list(self.config.refine_nodes or []),
            "semantic_tree": tree.to_dict(),
            "block_semantic_nodes": {
                bid: (self._node_by_block[bid].path or self._node_by_block[bid].name)
                for bid in self._node_by_block
            },
            "assigned_semantic": dict(self._fc._assigned_semantic),
            "semantic_face_counts": {
                n: len(v) for n, v in sem_groups.items()
            },
        }
        for bid, block in self._fc.blocks.items():
            sem = self._fc._assigned_semantic.get(bid)
            if sem:
                block.notes = (block.notes + f" | semantic={sem}").strip(" |")

        tree_path = export_tree_json(
            self._fc.blocks,
            out_dir / "block_tree.json",
            extra=tree_extra,
        )
        result.tree_json = tree_path

        result.ok = True
        result.elapsed_seconds = time.perf_counter() - t0
        counts = ", ".join(
            f"{n}={len(v)}" for n, v in sem_groups.items() if v
        )
        result.message = (
            f"语义树分块完成：第一层全覆盖 + {max(0, level - 1)} 层细分，"
            f"{result.num_final_blocks} 个叶块；{counts}"
        )

        self._refresh_live_report(
            out_dir,
            vertices,
            faces,
            iteration=level,
            queue_after=[],
            finished=True,
            extra_note=result.message,
        )
        return result


def run_semantic_tree_partition(
    client: VLMClient,
    vertices: np.ndarray,
    faces: np.ndarray,
    output_dir: str | Path,
    *,
    tree_hint: str = "",
    semantic_tree_file: Optional[str] = None,
    max_tree_depth: int = 5,
    tree_model: Optional[str] = None,
    refine_depth: int = 1,
    refine_all: bool = False,
    refine_paths: Optional[List[str]] = None,
    refine_nodes: Optional[List[str]] = None,
    min_block_size: int = 20,
    max_depth: int = 6,
    image_size: int = 512,
    panel_cols: int = 4,
    classification_model: Optional[str] = None,
    max_seeds_per_split: int = 8,
    split_strategy: str = "spectral",
    split_strategy_model: Optional[str] = None,
    skip_segmat: bool = True,
    vlm_split_strategy: bool = False,
    split_timeout_seconds: float = 60.0,
    split_strategy_max_retries: int = 2,
    split_retry_with_vlm: bool = True,
    select_best_view: bool = True,
    view_select_model: Optional[str] = None,
    view_select_semantic_hint: str = "",
    single_view_image_size: int = 720,
    topology_repair: bool = False,
    convex_boundary_opt: bool = False,
    boundary_handle_loop: bool = False,
    convex_opt_max_swaps: int = 800,
    convex_opt_min_faces: int = 8,
    convex_opt_min_component_faces: int = 20,
    live_report: bool = True,
    live_report_refresh_seconds: int = 10,
    min_visible_area_frac: float = 0.005,
    progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> SemanticTreePartitionResult:
    cfg = SemanticTreePartitionConfig(
        tree_hint=tree_hint,
        semantic_tree_file=semantic_tree_file,
        max_tree_depth=max_tree_depth,
        tree_model=tree_model,
        refine_depth=refine_depth,
        refine_all=refine_all,
        refine_paths=refine_paths,
        refine_nodes=refine_nodes,
        min_block_size=min_block_size,
        max_depth=max_depth,
        image_size=image_size,
        panel_cols=panel_cols,
        classification_model=classification_model,
        max_seeds_per_split=max_seeds_per_split,
        split_strategy=split_strategy,
        split_strategy_model=split_strategy_model,
        skip_segmat=skip_segmat,
        vlm_split_strategy=vlm_split_strategy,
        split_timeout_seconds=split_timeout_seconds,
        split_strategy_max_retries=split_strategy_max_retries,
        split_retry_with_vlm=split_retry_with_vlm,
        select_best_view=select_best_view,
        view_select_model=view_select_model,
        view_select_semantic_hint=view_select_semantic_hint,
        single_view_image_size=single_view_image_size,
        topology_repair=topology_repair,
        convex_boundary_opt=convex_boundary_opt,
        boundary_handle_loop=boundary_handle_loop,
        convex_opt_max_swaps=convex_opt_max_swaps,
        convex_opt_min_faces=convex_opt_min_faces,
        convex_opt_min_component_faces=convex_opt_min_component_faces,
        live_report=live_report,
        live_report_refresh_seconds=live_report_refresh_seconds,
        min_visible_area_frac=min_visible_area_frac,
    )
    pipeline = SemanticTreePartitionPipeline(client, cfg)
    return pipeline.run(
        vertices, faces, output_dir, progress_cb=progress_cb
    )


__all__ = [
    "SemanticTreePartitionConfig",
    "SemanticTreePartitionPipeline",
    "SemanticTreePartitionResult",
    "run_semantic_tree_partition",
]
