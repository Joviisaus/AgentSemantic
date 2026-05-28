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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .blocks import (
    BlockStatus,
    SemanticBlock,
    build_initial_blocks,
    export_tree_json,
    gather_face_status,
)
from .classifier import classify_component
from .client import VLMClient
from .prompts import LABEL_FULL, LABEL_NONE, LABEL_PARTIAL
from .render_component import (
    ACCEPTED_RGB,
    DIM_RGB,
    HIGHLIGHT_RGB,
    REJECTED_RGB,
    render_highlighted_component_panel,
    render_initial_segmentation_panel,
    render_semantic_mask_panel,
)
from .seed_split import ask_seed_count, split_block_by_spectral_seeds

LOGGER = logging.getLogger("vlm.pipeline")


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


@dataclass
class SemanticRefinementConfig:
    semantic: str
    max_depth: int = 3
    max_seeds: int = 6
    min_block_size: int = 20
    image_size: int = 512
    panel_cols: int = 4
    scalar_weight: float = 0.35
    save_intermediate_panels: bool = True
    save_responses: bool = True
    classification_model: Optional[str] = None
    seed_model: Optional[str] = None


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

    # ---------- 树工具 ----------
    def _next_child_id(self, parent_id: str) -> str:
        n = self._counter_per_parent.get(parent_id, 0)
        self._counter_per_parent[parent_id] = n + 1
        return f"{parent_id}.{n}"

    def _register_block(self, block: SemanticBlock) -> None:
        self.blocks[block.block_id] = block

    # ---------- 渲染 ----------
    def _render_block_panel(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        block: SemanticBlock,
        panel_dir: Path,
    ) -> Path:
        title = (
            f"[block {block.block_id}] depth={block.depth}  "
            f"faces={block.size}  semantic=「{self.config.semantic}」"
        )
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

        n_faces = int(faces.shape[0])
        if len(initial_face_labels) != n_faces:
            raise ValueError(
                f"initial_face_labels 长度 ({len(initial_face_labels)}) "
                f"与 faces 数量 ({n_faces}) 不一致"
            )

        initial_blocks = build_initial_blocks(
            initial_face_labels, min_block_size=self.config.min_block_size
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

        initial_panel = out_dir / "initial_segmentation.png"
        try:
            render_initial_segmentation_panel(
                vertices,
                faces,
                list(initial_face_labels),
                initial_panel,
                title=f"初始分块  blocks={len(initial_blocks)}",
                image_size=self.config.image_size,
                cols=self.config.panel_cols,
            )
            result.initial_panel = initial_panel
        except (OSError, ImportError, ValueError) as exc:
            LOGGER.warning("初始分块拼图渲染失败: %s", exc)

        queue: List[str] = [b.block_id for b in initial_blocks]
        iteration = 0
        responses_log: List[Dict[str, Any]] = []

        while queue:
            iteration += 1
            LOGGER.info("== iteration %d, queue=%d ==", iteration, len(queue))
            next_queue: List[str] = []

            for block_id in queue:
                block = self.blocks[block_id]
                if block.status != BlockStatus.PENDING:
                    continue

                # 太小或太深直接放弃（避免无限分裂）
                if block.size < self.config.min_block_size:
                    block.status = BlockStatus.ABANDONED
                    block.notes = (block.notes + " | too_small").strip(" |")
                    self._dump_response(responses_dir, block_id, {"reason": "too_small"})
                    if progress_cb:
                        progress_cb("abandoned", {"block_id": block_id, "reason": "too_small"})
                    continue

                # 渲染高亮 panel
                try:
                    panel_path = self._render_block_panel(vertices, faces, block, panel_dir)
                    block.panel_path = str(panel_path)
                except (OSError, ImportError, ValueError) as exc:
                    LOGGER.error("渲染 block %s 失败: %s", block_id, exc)
                    block.status = BlockStatus.ABANDONED
                    block.notes = f"render_failed: {exc}"
                    continue

                # Step 1：三分类
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

                # PARTIAL：深度已经到顶就放弃
                if block.depth >= self.config.max_depth:
                    block.status = BlockStatus.ABANDONED
                    block.notes = (block.notes + " | max_depth_reached").strip(" |")
                    if progress_cb:
                        progress_cb(
                            "abandoned",
                            {"block_id": block_id, "reason": "max_depth"},
                        )
                    continue

                # Step 2a：问 VLM 要 seed_count
                seed_resp = ask_seed_count(
                    self.client,
                    panel_path,
                    self.config.semantic,
                    max_seeds=self.config.max_seeds,
                    fallback=2,
                    model=self.config.seed_model,
                )
                block.seed_decision = {
                    "seed_count": seed_resp.seed_count,
                    "reason": seed_resp.reason,
                    "error": seed_resp.error,
                }
                block.seed_count = seed_resp.seed_count
                self._dump_response(
                    responses_dir, f"{block_id}_seed", block.seed_decision
                )

                # Step 2b：子谱聚类
                split = split_block_by_spectral_seeds(
                    vertices,
                    faces,
                    block.face_ids,
                    seed_count=seed_resp.seed_count,
                    vertex_scalars=vertex_scalars,
                    scalar_weight=self.config.scalar_weight,
                )
                if split.num_clusters <= 1:
                    block.status = BlockStatus.ABANDONED
                    block.notes = (block.notes + " | split_failed").strip(" |")
                    if progress_cb:
                        progress_cb(
                            "abandoned",
                            {"block_id": block_id, "reason": "split_failed"},
                        )
                    continue

                block.status = BlockStatus.SPLIT
                for group_face_ids in split.sub_face_groups:
                    child_id = self._next_child_id(block_id)
                    child = SemanticBlock(
                        block_id=child_id,
                        face_ids=list(group_face_ids),
                        depth=block.depth + 1,
                        parent_id=block_id,
                        status=BlockStatus.PENDING,
                        notes=f"split from {block_id} (k={split.num_clusters})",
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
                        },
                    )

            queue = next_queue

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
            },
        )
        result.tree_json = tree_path

        if responses_log:
            with open(out_dir / "responses_log.json", "w", encoding="utf-8") as f:
                json.dump(responses_log, f, indent=2, ensure_ascii=False)

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
    max_depth: int = 3,
    max_seeds: int = 6,
    min_block_size: int = 20,
    image_size: int = 512,
    panel_cols: int = 4,
    scalar_weight: float = 0.35,
    vertex_scalars: Optional[np.ndarray] = None,
    classification_model: Optional[str] = None,
    seed_model: Optional[str] = None,
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
        scalar_weight=scalar_weight,
        classification_model=classification_model,
        seed_model=seed_model,
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
