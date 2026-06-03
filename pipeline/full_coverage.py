"""
全覆盖（Full-coverage）语义分割流水线：

与 ``pipeline.pipeline`` 的「单语义 full/partial/none」不同，本流水线先让 VLM
观察整模型并枚举出一份**全包围、互不重叠**的语义清单（如人形角色 → 头/躯干/四肢），
随后对每个分块 multi-label 判断它命中了清单里的哪些部位：

  - 1 个 → 直接归到该部位
  - ≥2 个 → 分裂成 len(matched) 个种子（CC → segmat → FPS-spectral 级联），继续

直到每个叶分块只命中一个语义为止。不允许 background；枚举不区分左右。
收尾阶段：0 face 的枚举语义由 VLM 决定 split（从已有块 carve）或 delete。

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
from geometry.seed_split import (
    _extract_submesh,
    split_block_by_connected_components,
    split_block_by_geodesic_fps,
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
    classify_multi_label,
    enumerate_semantics,
    resolve_empty_semantic_repairs,
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
    # 单次分裂的最多种子数（VLM 即使返回更多 matched，分裂时也不会超过这个数）
    max_seeds_per_split: int = 6
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
    *,
    min_clusters: int = 3,
    max_clusters: int = 8,
) -> Dict[str, Any]:
    """单连通块的「auto-K」几何预切（测地 FPS + Voronoi + Lloyd）。

    返回 dict：
      - groups: List[List[int]] 全局 face_id 分组
      - k_actual: int
      - sizes: List[int]
      - mode: str
      - error: Optional[str]

    与 ``pipeline.pipeline.SemanticRefinementPipeline._segmat_split`` 实现一致，
    只把它独立成模块级函数避免跨类耦合。
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

        used = sorted({int(x) for x in arr})
        groups: List[List[int]] = []
        for k in used:
            idx = np.flatnonzero(arr == k)
            if idx.size == 0:
                continue
            groups.append([face_ids_list[i] for i in idx.tolist()])
        return {
            "groups": groups,
            "k_actual": len(groups),
            "sizes": [len(g) for g in groups],
            "mode": "geodesic_fps_voronoi_lloyd",
            "spectral_debug": {
                "fps_seeds": list(fps_seeds),
                "lloyd_seeds": list(seeds_final),
                "k_requested": k_target,
            },
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
        self.semantic_parts = [p for p in self.semantic_parts if p != name]
        self._part_index = {n: i for i, n in enumerate(self.semantic_parts)}
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

        sub_groups = self._split_block_by_seeds(
            vertices,
            faces,
            block,
            seed_count=2,
            responses_dir=responses_dir,
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
                    model=self.config.classification_model,
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
        repair_model = (
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
        return idx

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
                    model=self.config.view_select_model,
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
            return bid, classify_multi_label(
                self.client,
                panel,
                self.semantic_parts,
                model=self.config.classification_model,
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
            model=self.config.enumerate_model,
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

    # ---------- 单块决策 ----------
    def _split_block_by_seeds(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        block: SemanticBlock,
        seed_count: int,
        *,
        responses_dir: Path,
    ) -> Optional[List[List[int]]]:
        """对单个 block 走 CC → segmat → FPS 级联，返回每个子块的 face_id 列表，
        或 None 表示分裂失败（调用方 → ABANDONED）。

        - 未做过 segmat：先 CC（多个 CC 则直接采用），否则 segmat。
        - segmat 之后：只走 FPS-Voronoi-Lloyd，K = seed_count。
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
            segmat = _segmat_split_groups(
                vertices, faces, block.face_ids,
                min_clusters=max(2, min(3, seed_count)),
                max_clusters=max(seed_count, 4),
            )
            block.seed_decision = {
                "mode": "segmat",
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

        # Step C: FPS-Voronoi-Lloyd
        try:
            split = split_block_by_geodesic_fps(
                vertices,
                faces,
                block.face_ids,
                seed_count=seed_count,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "block %s geodesic_fps 失败: %s", block.block_id, exc
            )
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
            cls_tasks: List[Tuple[str, Path]] = list(iter_panel.items())
            cls_cache = self._batch_multi_label(cls_tasks)

            # --- 决策主循环 ---
            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue
                panel_path = iter_panel.get(block_id)
                if panel_path is None:
                    continue  # 已在 Pre-pass A 处理

                cls = cls_cache.get(block_id)
                if cls is None:
                    # 兜底：串行重试一次
                    try:
                        cls = classify_multi_label(
                            self.client,
                            panel_path,
                            self.semantic_parts,
                            model=self.config.classification_model,
                        )
                    except (RuntimeError, ValueError) as exc:
                        LOGGER.warning(
                            "fallback multi-label 失败 (%s): %s",
                            block_id, exc,
                        )
                        cls = MultiLabelClassification(error=str(exc))

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
                self._dump_response(
                    responses_dir,
                    f"{block_id}_classify",
                    block.classification,
                )
                if progress_cb:
                    progress_cb(
                        "classified",
                        {
                            "block_id": block_id,
                            "matched": list(cls.matched_parts),
                            "is_background": bool(cls.is_background),
                        },
                    )

                n_matched = len(cls.matched_parts)

                if n_matched == 0:
                    sem = self._fallback_semantic(cls, block=block)
                    self._force_assign_block(
                        block, sem, f"forced_assign(empty→{sem})"
                    )
                    continue

                # matched=1 → 归到该语义
                if n_matched == 1:
                    sem = cls.matched_parts[0]
                    if sem not in self._part_index:
                        LOGGER.warning(
                            "block %s matched=%r 不在 part_index，强制兜底",
                            block_id, sem,
                        )
                        sem = self._fallback_semantic(cls, block=block)
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
                    dom = cls.dominant or cls.matched_parts[0]
                    if dom not in self._part_index:
                        dom = self._fallback_semantic(cls, block=block)
                    self._force_assign_block(
                        block,
                        dom,
                        f"max_depth→{dom} (matched={cls.matched_parts})",
                    )
                    if progress_cb:
                        progress_cb(
                            "max_depth",
                            {"block_id": block_id, "dominant": dom},
                        )
                    continue

                seed_count = max(
                    2,
                    min(
                        int(self.config.max_seeds_per_split),
                        n_matched,
                    ),
                )
                sub_groups = self._split_block_by_seeds(
                    vertices,
                    faces,
                    block,
                    seed_count=seed_count,
                    responses_dir=responses_dir,
                )
                if not sub_groups or len(sub_groups) < 2:
                    # 分裂失败：保守地把这个块归到 dominant 语义
                    dom = cls.dominant or cls.matched_parts[0]
                    if dom in self._part_index:
                        block.status = BlockStatus.FULL
                        self._assigned_semantic[block_id] = dom
                        block.notes = (
                            block.notes
                            + f" | split_failed → assigned={dom}"
                            f" (matched={cls.matched_parts})"
                        ).strip(" |")
                    else:
                        dom = self._fallback_semantic(cls, block=block)
                        self._force_assign_block(
                            block, dom, f"split_failed→{dom}"
                        )
                    if progress_cb:
                        progress_cb(
                            "split_failed",
                            {"block_id": block_id, "dominant": dom},
                        )
                    continue

                # 成功分裂
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
                            f"(k={len(sub_groups)}, "
                            f"matched={cls.matched_parts})"
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
                            "matched": list(cls.matched_parts),
                            "children": list(block.children_ids),
                        },
                    )

            # 每轮末尾：写迭代时间线 + 刷新 live report
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
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("live report 更新失败: %s", exc)

            queue = next_queue

        self._finalize_block_assignments()
        self._repair_empty_semantics(
            vertices,
            faces,
            out_dir,
            panel_dir,
            responses_dir,
            progress_cb=progress_cb,
        )

        # ---- Step 5: 产出 ----
        result.num_iterations = iteration
        result.num_final_blocks = sum(
            1 for b in self.blocks.values() if b.status != BlockStatus.SPLIT
        )

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
                title="VLM 全覆盖语义分割（按部位上色）",
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
        }
        if enumeration is not None:
            tree_extra["enumeration"] = enumeration.to_dict()
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
    max_seeds_per_split: int = 6,
    parallel_vlm_workers: int = 1,
    save_responses: bool = True,
    live_report: bool = True,
    live_report_refresh_seconds: int = 10,
    select_best_view: bool = True,
    view_select_model: Optional[str] = None,
    view_select_semantic_hint: str = "",
    single_view_image_size: int = 720,
    generate_report: bool = True,
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
        parallel_vlm_workers=parallel_vlm_workers,
        save_responses=save_responses,
        live_report=live_report,
        live_report_refresh_seconds=live_report_refresh_seconds,
        select_best_view=select_best_view,
        view_select_model=view_select_model,
        view_select_semantic_hint=view_select_semantic_hint,
        single_view_image_size=single_view_image_size,
        generate_report=generate_report,
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
