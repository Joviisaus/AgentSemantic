"""
全覆盖语义分割命令行入口。

与 ``pipeline.cli`` 的区别：本 CLI 不需要用户提供单语义文本，而是让 VLM 自己看
整模型枚举出全覆盖语义清单，随后对每个分块做 multi-label 判断、按命中数继续切分。

输入：
    - mesh / .off 文件
    - (可选) face_labels JSON，用于初始分块；省略时用整 mesh 的连通分量
    - (可选) --semantic-parts 直接给定语义清单，跳过 VLM 枚举

输出：
    output/<模型>/vlm_full/
        enumeration.json
        semantics_overview.png
        panels/
        responses/
        initial_segmentation.png
        full_coverage_mask.png
        face_semantic_labels.json
        block_tree.json
        live_report.html
        iterations/iter_NN.png
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

from vlm.client import build_vlm_client
from vlm.config import DEFAULT_CONFIG_PATH, get_provider_config, load_vlm_config

from .full_coverage import run_full_coverage


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取任意 trimesh 支持的网格，返回 (V, F) 数组。"""
    ext = path.suffix.lstrip(".").lower() or None
    try:
        mesh = trimesh.load(str(path), file_type=ext)
    except (ValueError, OSError) as exc:
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


def _default_model_dir_name(mesh_path: Path, base: str = "output") -> str:
    """根据 mesh 路径推断"模型名"子目录：

    1. 如果 mesh 已经位于 ``<base>/<model_dir>/...``（典型 legacy 约定：
       ``output/<model>/medoid_pose/render_pose.off``），用 ``<model_dir>`` —
       这样能把全覆盖产物和已有的 pose / segmat 产物归到同一棵目录。
    2. 否则（例如 ``E:\\data\\Thingi10K\\raw_meshes\\55280.stl``）用 mesh 的
       文件名 stem，避免同数据集多个 mesh 互相覆盖到 ``output/Thingi10K/`` 下。
    """
    base_path = Path(base)
    try:
        rel = mesh_path.resolve().relative_to(base_path.resolve())
        if rel.parts and rel.parts[0] not in ("", "."):
            return rel.parts[0]
    except (ValueError, OSError):
        pass
    stem = mesh_path.stem or "mesh"
    return stem


def _resolve_output_dir(
    mesh_path: Path,
    explicit: Optional[str],
    base: str = "output",
) -> Path:
    if explicit:
        return Path(explicit)
    return Path(base) / _default_model_dir_name(mesh_path, base) / "vlm_full"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.full_coverage_cli",
        description="基于 VLM 的全覆盖语义分割（先枚举再 multi-label 切分）",
    )
    p.add_argument(
        "mesh",
        type=str,
        help="渲染姿态 mesh 路径（推荐 output/<模型>/medoid_pose/render_pose.off）",
    )
    p.add_argument(
        "--labels",
        type=str,
        default=None,
        help=(
            "face_labels JSON（face_segment_labels.json / face_cluster_labels.json）。"
            "省略时直接用整 mesh 的连通分量作为初始分块。"
        ),
    )

    # ---- 枚举阶段 ----
    p.add_argument(
        "--enumerate-hint",
        type=str,
        default="",
        help=(
            "VLM 枚举阶段补充说明（宜短）。引导粗粒度、短名、全覆盖分区，"
            "例如「四足猫，动物骨架」或「类人，6~8 项短名，不区分左右」。"
            "后续切分会再细化，此处不必拆手指/装饰细节。"
        ),
    )
    p.add_argument(
        "--semantic-parts",
        type=str,
        default=None,
        help=(
            "直接给定全覆盖语义清单，用英文逗号或中文「、」分隔，例如 "
            "'头,颈,躯干,手臂,大腿'（不区分左右）。给了这个就不再调 VLM 枚举。"
        ),
    )
    p.add_argument(
        "--semantic-parts-file",
        type=str,
        default=None,
        help=(
            "从 JSON 文件读语义清单。文件可以是 ['头','躯干',...] 或 "
            "{'semantic_parts': ['头',...]}；给了这个就不再调 VLM 枚举。"
        ),
    )
    p.add_argument(
        "--enumerate-min-parts",
        type=int,
        default=2,
        help="枚举最少部位数；简单物体可保持 2（默认 2）",
    )
    p.add_argument(
        "--enumerate-max-parts",
        type=int,
        default=12,
        help="枚举最多部位数；粗粒度初分上限，默认 12",
    )
    p.add_argument(
        "--enumerate-model",
        type=str,
        default=None,
        help="枚举阶段单独指定模型；不传则沿用 --model / provider 默认模型。",
    )

    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=(
            "输出目录。默认 <output-base>/<模型>/vlm_full/，其中 <模型> 的"
            "推断规则：mesh 若已经在 <output-base> 下则取首级子目录"
            "（保留 output/<model>/medoid_pose/render_pose.off 这种 legacy"
            "约定），否则按 mesh 文件名 stem。"
        ),
    )
    p.add_argument(
        "--output-base",
        type=str,
        default="output",
        help="输出根目录（默认 output）",
    )

    # ---- VLM 通用 ----
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
    p.add_argument(
        "--model", type=str, default=None,
        help="覆盖该 provider 的 default_model",
    )
    p.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="额外 .env 文件路径（默认自动加载 .env / .env.local）",
    )

    # ---- 主循环参数 ----
    p.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="分裂递归最大深度（硬上限，默认 6）",
    )
    p.add_argument(
        "--min-block-size",
        type=int,
        default=20,
        help="忽略 face 数低于该值的分块（默认 20）",
    )
    p.add_argument(
        "--max-seeds-per-split",
        type=int,
        default=6,
        help=(
            "单次分裂的最多种子数（matched 超过该值会被截断；默认 6）。"
            "正常情况下 K = len(matched_parts)。"
        ),
    )
    p.add_argument("--image-size", type=int, default=512, help="单视角分辨率")
    p.add_argument(
        "--panel-cols",
        type=int,
        default=4,
        help="拼图列数（默认 4×3=12）",
    )
    p.add_argument(
        "--parallel-vlm",
        type=int,
        default=1,
        metavar="N",
        help=(
            "每轮迭代里把 N 个块的 multi-label 用线程池并发请求 VLM。"
            "1=完全串行（默认）。"
        ),
    )
    p.add_argument(
        "--no-live-report",
        action="store_true",
        help="关闭 live_report.html 实时进度页。",
    )
    p.add_argument(
        "--live-report-refresh",
        type=int,
        default=10,
        metavar="SEC",
        help="live_report.html 浏览器自动刷新间隔（秒，默认 10）",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help=(
            "关闭最终 report.html 复盘报告（默认会在主流程结束时生成，"
            "包含每个 block 的 multi-label 决策卡片、语义图例、枚举/"
            "视角/时间线/最终结果等）。"
        ),
    )

    # ---- 视角选择 ----
    p.add_argument(
        "--no-view-select",
        action="store_true",
        help=(
            "默认会让 VLM 看 12 张缩略图挑出最能展示整体结构的视角，"
            "随后 live_report / 时间线只用这一个视角渲染（透明背景）。"
            "加这个 flag 关掉 VLM 选视角。"
        ),
    )
    p.add_argument(
        "--view-select-model",
        type=str,
        default=None,
        help="视角选择步骤单独指定模型；不传则沿用 --model。",
    )
    p.add_argument(
        "--view-select-hint",
        type=str,
        default="",
        help=(
            "视角选择 VLM 的简短提示。默认「整体结构」并附带前若干枚举短名；"
            "可覆写为如「正面看清头与躯干」。"
        ),
    )
    p.add_argument(
        "--single-view-size",
        type=int,
        default=720,
        metavar="PX",
        help="单视角（live_report / 时间线 / overview）渲染分辨率，默认 720。",
    )

    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "只渲染初始分块拼图 + 整模型 overview，不调用 VLM，"
            "用来快速验证渲染路径。"
        ),
    )
    return p


def _parse_semantic_parts_cli(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    raw = arg.strip()
    if not raw:
        return None
    # 同时支持英文逗号 / 中文「、」/ 中文「，」
    for sep in ("、", "，"):
        raw = raw.replace(sep, ",")
    parts = [x.strip() for x in raw.split(",")]
    return [x for x in parts if x]


def _parse_semantic_parts_file(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    path = Path(arg)
    if not path.is_file():
        raise FileNotFoundError(f"semantic_parts 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "semantic_parts" in data:
        items = data["semantic_parts"]
    else:
        raise ValueError(
            f"{path} 不是 list，也没有 semantic_parts 字段"
        )
    out: List[str] = []
    for x in items:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    mesh_path = Path(args.mesh)
    if not mesh_path.is_file():
        print(f"[错误] mesh 不存在: {mesh_path}", file=sys.stderr)
        return 2

    print(f"[mesh] {mesh_path}")
    vertices, faces = _load_mesh(mesh_path)
    print(f"[mesh] V={len(vertices)} F={len(faces)}")

    if args.labels:
        labels_path = Path(args.labels)
        if not labels_path.is_file():
            print(f"[错误] labels JSON 不存在: {labels_path}", file=sys.stderr)
            return 2
        face_labels: Optional[List[int]] = _load_face_labels(
            labels_path, n_faces=len(faces)
        )
        unique_labels = sorted(set(face_labels))
        print(
            f"[labels] {labels_path.name}: {len(unique_labels)} 个分块"
            f"（label 范围: {min(unique_labels)}..{max(unique_labels)}），"
            f"将由 pipeline 再按 CC 展开"
        )
    else:
        face_labels = None
        print("[labels] 未提供 --labels；使用整 mesh 的连通分量作为初始分块")

    out_dir = _resolve_output_dir(mesh_path, args.output_dir, args.output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir.resolve()}")
    if not args.no_live_report and not args.dry_run:
        print(
            f"[live]   {(out_dir / 'live_report.html').resolve()}"
            f"  (浏览器打开，每 {args.live_report_refresh}s 自动刷新)"
        )

    # 解析显式语义清单
    sem_parts_cli = _parse_semantic_parts_cli(args.semantic_parts)
    sem_parts_file = _parse_semantic_parts_file(args.semantic_parts_file)
    # 文件优先于命令行
    semantic_parts = sem_parts_file or sem_parts_cli
    if semantic_parts:
        print(f"[semantic_parts] 用户提供 {len(semantic_parts)} 项: {semantic_parts}")

    if args.dry_run:
        from rendering.render_component import (
            render_initial_segmentation_panel,
            render_overview_only_panel,
        )

        ov_path = out_dir / "semantics_overview.png"
        render_overview_only_panel(
            vertices,
            faces,
            ov_path,
            title="整模型 12 视角（dry-run）",
            image_size=args.image_size,
            cols=args.panel_cols,
        )
        print(f"[dry-run] overview: {ov_path}")

        # 初始分块（CC 展开）
        if face_labels is None:
            face_labels_for_init = [0] * len(faces)
        else:
            face_labels_for_init = face_labels
        panel = render_initial_segmentation_panel(
            vertices,
            faces,
            face_labels_for_init,
            out_dir / "initial_segmentation.png",
            title="初始分块（dry-run）",
            image_size=args.image_size,
            cols=args.panel_cols,
        )
        print(f"[dry-run] initial segmentation: {panel}")
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
            matched = payload.get("matched", []) or []
            tag = f"matched={matched}"
            print(f"  [classify] {payload['block_id']} -> {tag}")
        elif event == "split":
            print(
                f"  [split]    {payload['block_id']} "
                f"k={payload['seed_count']} "
                f"children={len(payload.get('children', []))} "
                f"matched={payload.get('matched', [])}"
            )
        elif event == "empty_repair":
            print(
                f"  [repair]   round={payload.get('round')} "
                f"{payload.get('empty_semantic')} → "
                f"{payload.get('action')}"
                + (
                    f" from {payload['donor_block_id']}"
                    if payload.get("donor_block_id")
                    else ""
                )
            )
        elif event == "abandoned":
            print(f"  [skip]     {payload['block_id']} "
                  f"({payload.get('reason')})")
        elif event == "max_depth":
            print(
                f"  [maxdep]   {payload['block_id']} "
                f"→ assigned={payload.get('dominant')}"
            )
        elif event == "split_failed":
            print(
                f"  [splfail]  {payload['block_id']} "
                f"→ assigned={payload.get('dominant')}"
            )

    result = run_full_coverage(
        client,
        vertices,
        faces,
        out_dir,
        initial_face_labels=face_labels,
        enumerate_hint=args.enumerate_hint,
        semantic_parts=semantic_parts,
        enumerate_min_parts=args.enumerate_min_parts,
        enumerate_max_parts=args.enumerate_max_parts,
        max_depth=args.max_depth,
        min_block_size=args.min_block_size,
        image_size=args.image_size,
        panel_cols=args.panel_cols,
        classification_model=args.model,
        enumerate_model=args.enumerate_model or args.model,
        max_seeds_per_split=args.max_seeds_per_split,
        parallel_vlm_workers=args.parallel_vlm,
        live_report=not args.no_live_report,
        live_report_refresh_seconds=args.live_report_refresh,
        generate_report=not args.no_report,
        select_best_view=not args.no_view_select,
        view_select_model=args.view_select_model or args.model,
        view_select_semantic_hint=args.view_select_hint,
        single_view_image_size=args.single_view_size,
        progress_cb=progress,
    )

    print()
    print(f"[result] {result.message}")
    print(f"  iterations         : {result.num_iterations}")
    print(
        f"  initial / final    : "
        f"{result.num_initial_blocks} / {result.num_final_blocks}"
    )
    print(f"  semantic parts     : {len(result.semantic_parts)}")
    for name in result.semantic_parts:
        cnt = len(result.semantic_face_groups.get(name, []))
        print(f"    - {name:<20s} {cnt} face")
    if result.pending_face_ids:
        print(f"  pending faces      : {len(result.pending_face_ids)}")
    if result.best_view is not None:
        bv = result.best_view
        tag = "fallback" if bv.fallback else "VLM"
        print(
            f"  best view ({tag:>8}) : #{bv.index + 1}  "
            f"elev={bv.elev:.1f}° azim={bv.azim:.1f}°"
        )
        if bv.reason:
            print(f"                       reason: {bv.reason[:140]}")
    if result.best_view_overview:
        print(f"  best view overview : {result.best_view_overview}")
    if result.iter_progress_pngs:
        print(
            f"  iter timeline      : {len(result.iter_progress_pngs)} 帧 "
            f"({result.iter_progress_pngs[0].parent})"
        )
    if result.coverage_panel:
        print(f"  full coverage png  : {result.coverage_panel}")
    if result.labels_json:
        print(f"  face_semantic_labels.json : {result.labels_json}")
    if result.tree_json:
        print(f"  block_tree.json    : {result.tree_json}")
    if result.report_html:
        print(f"  report.html        : {result.report_html}")
    print(f"  elapsed            : {result.elapsed_seconds:.1f}s")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
