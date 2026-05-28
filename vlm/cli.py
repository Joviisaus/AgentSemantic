"""
VLM 语义细化命令行入口。

输入：
    - mesh / .off 文件（渲染姿态网格，例如 output/<模型>/medoid_pose/render_pose.off）
    - face_labels JSON（SEG-MAT 或谱聚类的产物，例如 face_segment_labels.json）
    - 用户语义文本（例如 "面部"）

输出：
    output/<模型>/vlm/<语义>/
        panels/                        每个 block 的高亮 12 视角图
        responses/                     每次 VLM 的原始响应
        initial_segmentation.png       初始分块拼图
        semantic_mask.png              最终绿色=符合 / 淡色=不符合
        semantic_mask.json             face_id 列表
        block_tree.json                完整 block 树
        responses_log.json             调用流水
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import trimesh

from .client import build_vlm_client
from .config import DEFAULT_CONFIG_PATH, get_provider_config, load_vlm_config
from .pipeline import run_semantic_refinement


def _safe_dir_name(name: str) -> str:
    out = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    return out or "semantic"


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    读取任意 trimesh 支持的网格，返回 (V, F) 数组。
    显式传 file_type 以绕开某些 Windows 上的 trimesh 自动探测崩溃。
    """
    ext = path.suffix.lstrip(".").lower() or None
    try:
        mesh = trimesh.load(str(path), file_type=ext)
    except (ValueError, OSError) as exc:
        # 退回不传 file_type，让 trimesh 自己探测
        mesh = trimesh.load(str(path))
        if mesh is None:
            raise ValueError(f"trimesh 无法加载 {path}: {exc}") from exc

    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"{path} 中找不到 trimesh.Trimesh")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"无法把 {path} 解析为 Trimesh，得到 {type(mesh)}")
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def _load_face_labels(path: Path, n_faces: int) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "face_labels" not in data:
        raise ValueError(f"{path} 不含 face_labels 字段")
    labels = [int(x) for x in data["face_labels"]]
    if len(labels) != n_faces:
        raise ValueError(
            f"face_labels 长度 {len(labels)} 与 mesh.faces ({n_faces}) 不一致"
        )
    return labels


def _resolve_output_dir(
    mesh_path: Path,
    semantic: str,
    explicit: Optional[str],
    base: str = "output",
) -> Path:
    if explicit:
        return Path(explicit)
    return (
        Path(base) / mesh_path.parent.parent.name / "vlm" / _safe_dir_name(semantic)
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vlm.cli",
        description="基于 VLM 的语义细化（在 SEG-MAT / 谱聚类基础上）",
    )
    p.add_argument(
        "mesh",
        type=str,
        help="渲染姿态 mesh 路径（推荐 output/<模型>/medoid_pose/render_pose.off）",
    )
    p.add_argument(
        "--labels",
        type=str,
        required=True,
        help="face_labels JSON（face_segment_labels.json 或 face_cluster_labels.json）",
    )
    p.add_argument(
        "--semantic",
        type=str,
        required=True,
        help="用户语义，例如 '面部'、'右手'",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认 output/<模型>/vlm/<语义>/）",
    )
    p.add_argument(
        "--output-base",
        type=str,
        default="output",
        help="输出根目录（与 --output-dir 二选一，默认 output）",
    )

    p.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="VLM 配置 YAML 路径",
    )
    p.add_argument(
        "--provider",
        type=str,
        default=None,
        help="覆盖 vlm_config.yaml 的 active_provider",
    )
    p.add_argument("--model", type=str, default=None, help="覆盖该 provider 的 default_model")
    p.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="额外 .env 文件路径（默认自动加载 .env / .env.local）",
    )

    p.add_argument("--max-depth", type=int, default=3, help="分裂递归最大深度")
    p.add_argument("--max-seeds", type=int, default=6, help="单次分裂最多种子数")
    p.add_argument("--min-block-size", type=int, default=20, help="忽略 face 数低于该值的分块")
    p.add_argument("--image-size", type=int, default=512, help="单视角分辨率")
    p.add_argument("--panel-cols", type=int, default=4, help="拼图列数（默认 4×3=12）")
    p.add_argument("--scalar-weight", type=float, default=0.35, help="子谱聚类畸变特征权重")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只渲染初始分块拼图，不调用 VLM",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    mesh_path = Path(args.mesh)
    labels_path = Path(args.labels)
    if not mesh_path.is_file():
        print(f"[错误] mesh 不存在: {mesh_path}", file=sys.stderr)
        return 2
    if not labels_path.is_file():
        print(f"[错误] labels JSON 不存在: {labels_path}", file=sys.stderr)
        return 2

    print(f"[mesh] {mesh_path}")
    vertices, faces = _load_mesh(mesh_path)
    print(f"[mesh] V={len(vertices)} F={len(faces)}")

    face_labels = _load_face_labels(labels_path, n_faces=len(faces))
    unique_labels = sorted(set(face_labels))
    print(
        f"[labels] {labels_path.name}: {len(unique_labels)} 个分块"
        f"（label 范围: {min(unique_labels)}..{max(unique_labels)}）"
    )

    out_dir = _resolve_output_dir(mesh_path, args.semantic, args.output_dir, args.output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir.resolve()}")

    if args.dry_run:
        from .render_component import render_initial_segmentation_panel

        panel = render_initial_segmentation_panel(
            vertices,
            faces,
            face_labels,
            out_dir / "initial_segmentation.png",
            title=f"初始分块（dry-run）blocks={len(unique_labels)}",
            image_size=args.image_size,
            cols=args.panel_cols,
        )
        print(f"[dry-run] 初始分块拼图: {panel}")
        return 0

    cfg = load_vlm_config(
        args.config,
        load_env=True,
        env_paths=[args.env_file] if args.env_file else None,
    )
    provider_cfg = get_provider_config(cfg, args.provider, model=args.model)

    if not provider_cfg.has_key() and not provider_cfg.is_local():
        print(
            f"[错误] provider '{provider_cfg.name}' 缺少 API key。"
            " 请在 .env 中设置对应环境变量（参考 .env.example）",
            file=sys.stderr,
        )
        return 3

    print(
        f"[vlm] provider={provider_cfg.name} model={provider_cfg.default_model} "
        f"base_url={provider_cfg.base_url or '<n/a>'}"
    )
    client = build_vlm_client(provider_cfg)

    def progress(event: str, payload: Dict[str, Any]) -> None:
        if event == "classified":
            print(f"  [classify] {payload['block_id']} -> {payload['label']}")
        elif event == "split":
            print(
                f"  [split]    {payload['block_id']} seeds={payload['seed_count']} "
                f"children={len(payload.get('children', []))}"
            )
        elif event == "abandoned":
            print(f"  [skip]     {payload['block_id']} ({payload.get('reason')})")

    result = run_semantic_refinement(
        client,
        vertices,
        faces,
        face_labels,
        out_dir,
        semantic=args.semantic,
        max_depth=args.max_depth,
        max_seeds=args.max_seeds,
        min_block_size=args.min_block_size,
        image_size=args.image_size,
        panel_cols=args.panel_cols,
        scalar_weight=args.scalar_weight,
        classification_model=args.model,
        seed_model=args.model,
        progress_cb=progress,
    )

    print()
    print(f"[result] {result.message}")
    print(f"  iterations         : {result.num_iterations}")
    print(f"  initial / final    : {result.num_initial_blocks} / {result.num_final_blocks}")
    print(f"  full faces         : {len(result.full_face_ids)}")
    print(f"  none faces         : {len(result.none_face_ids)}")
    if result.pending_face_ids:
        print(f"  pending faces      : {len(result.pending_face_ids)}")
    if result.semantic_panel:
        print(f"  semantic mask png  : {result.semantic_panel}")
    if result.tree_json:
        print(f"  block_tree.json    : {result.tree_json}")
    if result.mask_json:
        print(f"  semantic_mask.json : {result.mask_json}")
    print(f"  elapsed            : {result.elapsed_seconds:.1f}s")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
