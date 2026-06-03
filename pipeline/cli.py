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

from vlm.client import build_vlm_client
from vlm.config import DEFAULT_CONFIG_PATH, get_provider_config, load_vlm_config

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


def _default_model_dir_name(mesh_path: Path, base: str = "output") -> str:
    """根据 mesh 路径推断"模型名"子目录：

    1. 如果 mesh 已经位于 ``<base>/<model_dir>/...``（典型 legacy 约定：
       ``output/<model>/medoid_pose/render_pose.off``），用 ``<model_dir>`` —
       这样能把 VLM 产物和已有的 pose / segmat 产物归到同一棵目录。
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
    semantic: str,
    explicit: Optional[str],
    base: str = "output",
) -> Path:
    if explicit:
        return Path(explicit)
    return (
        Path(base)
        / _default_model_dir_name(mesh_path, base)
        / "vlm"
        / _safe_dir_name(semantic)
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.cli",
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
        default=None,
        help=(
            "face_labels JSON（face_segment_labels.json 或 face_cluster_labels.json）。"
            "省略时直接用整 mesh 的连通分量作为初始分块（推荐：拓扑纯净，"
            "比 spectral 把多个不相连部件并到同一标签更靠谱）。"
        ),
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
        help=(
            "输出目录。默认 <output-base>/<模型>/vlm/<语义>/，其中 <模型> 的"
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

    p.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help=(
            "分裂递归最大深度（硬上限）。配合 --vlm-self-review 时，"
            "VLM 会更早自主决定终止，把它设大一点更安全。"
        ),
    )
    p.add_argument("--max-seeds", type=int, default=6, help="单次分裂最多种子数")
    p.add_argument("--min-block-size", type=int, default=20, help="忽略 face 数低于该值的分块")
    p.add_argument(
        "--accept-partial-at-max-depth",
        action="store_true",
        help=(
            "在 max-depth 处遇到 partial 时当作'最佳估计 full'接受，"
            "适用于几何上无法被谱聚类干净切分的细分语义（例如发型 vs 头颅）。"
            "默认关闭——partial 在 max-depth 处会被舍弃为 abandoned。"
        ),
    )
    p.add_argument(
        "--vlm-self-review",
        action="store_true",
        help=(
            "每轮迭代结束后让 VLM 看一眼 12 视角的「整模型染色进度图」"
            "（红=已确认符合 / 黄=待分裂 / 灰=不符合）并自评 + 决定是否提前终止。"
            "max-depth 仍是硬上限。默认关闭。"
        ),
    )
    p.add_argument(
        "--split-strategy",
        choices=("geodesic_fps", "spectral"),
        default="geodesic_fps",
        help=(
            "分裂策略："
            "geodesic_fps —— 在子 mesh 上做测地最远点采样选 K 个种子 + 测地 Voronoi 划分（默认，更均衡）；"
            "spectral —— 旧的归一化割谱聚类（兼容 / fallback）。"
        ),
    )
    p.add_argument(
        "--vlm-grouped-split",
        action="store_true",
        help=(
            "[实验性，默认关闭] partial 分块改用「过分割 + VLM 一次性分组」。"
            "实测 12 视角下 VLM 难以稳定指认候选编号，已不推荐主流程使用，保留仅供对比实验。"
        ),
    )
    p.add_argument(
        "--grouped-split-n-candidates",
        type=int,
        default=8,
        help="--vlm-grouped-split 时每个 partial 块过分割出多少个候选区（2~12，默认 8）",
    )
    p.add_argument(
        "--grouped-split-max-groups",
        type=int,
        default=4,
        help="--vlm-grouped-split 时 VLM 最多把候选合并成多少组（默认 4）",
    )
    p.add_argument("--image-size", type=int, default=512, help="单视角分辨率")
    p.add_argument("--panel-cols", type=int, default=4, help="拼图列数（默认 4×3=12）")
    p.add_argument("--scalar-weight", type=float, default=0.35, help="子谱聚类畸变特征权重")
    p.add_argument(
        "--parallel-vlm",
        type=int,
        default=1,
        metavar="N",
        help=(
            "每轮迭代里把 N 个块的 classify (observe+reason) 用线程池并发请求 VLM。"
            "1=完全串行（默认，等同旧行为）；调大可显著缩短网络密集型 provider 的总耗时，"
            "但请注意各家 VLM 的 QPS / RPM 限额。渲染 (pyvista) 仍串行。"
        ),
    )
    p.add_argument(
        "--no-live-report",
        action="store_true",
        help=(
            "默认每轮迭代后会在 output_dir/live_report.html 写一份「整模型染色进度」"
            "（none=白 / partial=浅黄 / full=深红 / abandoned=极浅灰），HTML 自带 "
            "meta refresh，浏览器打开就能看实时进度。加这个 flag 关掉。"
        ),
    )
    p.add_argument(
        "--live-report-refresh",
        type=int,
        default=10,
        metavar="SEC",
        help="live_report.html 浏览器自动刷新间隔（秒，默认 10）",
    )
    p.add_argument(
        "--no-view-select",
        action="store_true",
        help=(
            "默认在进入主循环前会让 VLM 看 12 张缩略图、挑出最能展示该语义的视角，"
            "随后 live_report / report / 每轮迭代帧都只用这一个视角渲染（透明背景）。"
            "加这个 flag 关掉 VLM 选视角，直接用 DEFAULT_OVERVIEW_VIEW。"
        ),
    )
    p.add_argument(
        "--view-select-model",
        type=str,
        default=None,
        help="视角选择步骤单独指定模型；不传则沿用 --model / provider 默认模型。",
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
        "--no-report",
        action="store_true",
        help="跳过 HTML 报告生成（默认会在管道结束后输出 report.html）",
    )
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
        face_labels = _load_face_labels(labels_path, n_faces=len(faces))
        unique_labels = sorted(set(face_labels))
        print(
            f"[labels] {labels_path.name}: {len(unique_labels)} 个分块"
            f"（label 范围: {min(unique_labels)}..{max(unique_labels)}），"
            f"将由 pipeline 再按 CC 展开"
        )
    else:
        # 用整 mesh 的连通分量作为唯一初始 label，pipeline 内部会再按 CC 展开
        # （此处已是 1 label = 整 mesh，CC 展开后正好是逐个连通分量）
        face_labels = [0] * len(faces)
        print("[labels] 未提供 --labels；使用整 mesh 的连通分量作为初始分块")

    out_dir = _resolve_output_dir(mesh_path, args.semantic, args.output_dir, args.output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir.resolve()}")
    if not args.no_live_report and not args.dry_run:
        print(
            f"[live]   {(out_dir / 'live_report.html').resolve()}"
            f"  (浏览器打开，每 {args.live_report_refresh}s 自动刷新)"
        )

    if args.dry_run:
        from rendering.render_component import render_initial_segmentation_panel

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
        elif event == "review":
            term = "TERMINATE" if payload.get("should_terminate") else "continue"
            print(
                f"  [review]   iter={payload['iteration']} -> {term}  "
                f"red={payload.get('red_assessment', '')}  "
                f"reason={payload.get('reason', '')[:80]}"
            )

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
        accept_partial_at_max_depth=args.accept_partial_at_max_depth,
        vlm_self_review=args.vlm_self_review,
        vlm_grouped_split=args.vlm_grouped_split,
        grouped_split_n_candidates=args.grouped_split_n_candidates,
        grouped_split_max_groups=args.grouped_split_max_groups,
        split_strategy=args.split_strategy,
        parallel_vlm_workers=args.parallel_vlm,
        live_report=not args.no_live_report,
        live_report_refresh_seconds=args.live_report_refresh,
        select_best_view=not args.no_view_select,
        view_select_model=args.view_select_model or args.model,
        single_view_image_size=args.single_view_size,
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
    if result.semantic_panel:
        print(f"  semantic mask png  : {result.semantic_panel}")
    if result.tree_json:
        print(f"  block_tree.json    : {result.tree_json}")
    if result.mask_json:
        print(f"  semantic_mask.json : {result.mask_json}")
    print(f"  elapsed            : {result.elapsed_seconds:.1f}s")

    # 生成 HTML 报告
    if not args.no_report and result.tree_json:
        try:
            from .report import generate_report

            html_path = generate_report(out_dir)
            print(f"  html report        : {html_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] HTML 报告生成失败: {exc}", file=sys.stderr)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
