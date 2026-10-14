"""
全覆盖 / 语义树分块命令行入口。

默认 ``full_coverage`` 模式：VLM 枚举全覆盖语义清单 + multi-label 切分。
加 ``--semantic-tree`` 则走层级语义树逐层分块（见 ``semantic_tree_partition``）。

输入：
    - mesh / .off 文件
    - (full_coverage) 可选 face_labels JSON、--semantic-parts
    - (semantic_tree) 可选 --semantic-tree-file、--refine-depth / --refine-paths

输出：
    output/<模型>/vlm_full/
        enumeration.json / semantic_tree.json
        semantics_overview.png
        panels/  responses/
        face_semantic_labels.json
        full_coverage_mask.png
        block_tree.json
        live_report.html          (full_coverage / semantic_tree)
        iterations/iter_NN.png    (full_coverage / semantic_tree)
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
from vlm.config import (
    DEFAULT_CONFIG_PATH,
    get_provider_config,
    load_vlm_config,
    resolve_model_name,
)

from .full_coverage import run_full_coverage
from .semantic_tree_partition import run_semantic_tree_partition


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


def _load_semantic_labels(
    path: Path, n_faces: int,
) -> tuple[List[str], List[int]]:
    """读取 face_semantic_labels.json（semantic_parts + face_semantic_index）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "face_semantic_index" not in data:
        raise ValueError(
            f"{path} 需含 face_semantic_index 字段"
            "（face_semantic_labels.json 格式）"
        )
    if "semantic_parts" not in data:
        raise ValueError(f"{path} 需含 semantic_parts 字段")
    semantic_parts = [str(x) for x in data["semantic_parts"]]
    face_idx = [int(x) for x in data["face_semantic_index"]]
    if len(face_idx) != n_faces:
        raise ValueError(
            f"face_semantic_index 长度 {len(face_idx)} "
            f"与 mesh.faces ({n_faces}) 不一致"
        )
    return semantic_parts, face_idx


def _default_model_dir_name(mesh_path: Path, base: str = "output") -> str:
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


def _discover_semantic_labels(
    mesh_path: Path,
    n_faces: int,
    *,
    out_dir: Path,
    output_base: Path,
) -> Optional[Path]:
    """查找与 mesh 面数匹配的 face_semantic_labels.json。"""
    candidates: List[Tuple[int, float, Path]] = []
    model_dir = _default_model_dir_name(mesh_path, str(output_base))

    def _try(path: Path, priority: int) -> None:
        if not path.is_file():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if int(data.get("num_faces", -1)) != n_faces:
            return
        if "face_semantic_index" not in data or "semantic_parts" not in data:
            return
        candidates.append((priority, path.stat().st_mtime, path))

    _try(out_dir / "face_semantic_labels.json", priority=3)
    if output_base.is_dir():
        for c in output_base.glob("**/face_semantic_labels.json"):
            in_model = bool(model_dir) and model_dir in c.parts
            _try(c, priority=2 if in_model else 1)

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def _resolve_semantic_labels_path(
    args: argparse.Namespace,
    mesh_path: Path,
    n_faces: int,
    out_dir: Path,
) -> Optional[Path]:
    if args.labels:
        return Path(args.labels)
    return _discover_semantic_labels(
        mesh_path,
        n_faces,
        out_dir=out_dir,
        output_base=Path(args.output_base),
    )


def _normalize_cli_args(args: argparse.Namespace, mesh_path: Path, out_dir: Path) -> None:
    """解析后修正模式与边界优化标志（就地修改 args）。"""
    if (
        not args.semantic_tree
        and (
            args.refine_all
            or args.refine_paths
            or args.refine_nodes
            or args.semantic_tree_file
        )
    ):
        args.semantic_tree = True
        print("[mode]   检测到 semantic_tree 专用参数，自动启用 --semantic-tree")

    if args.boundary_handle_loop_only:
        if args.semantic_tree:
            print(
                "[warn] --boundary-handle-loop-only 与分块联跑冲突，"
                "改为分块结束后自动 handle loop 平滑"
            )
            args.boundary_handle_loop = True
            args.boundary_handle_loop_only = False
            return

        labels_path = _resolve_semantic_labels_path(
            args, mesh_path, n_faces=-1, out_dir=out_dir,
        )
        if labels_path is None or not labels_path.is_file():
            print(
                "[warn] 未找到已有 face_semantic_labels.json；"
                "改为分块结束后自动 handle loop 平滑"
                "（联跑分块请去掉 --boundary-handle-loop-only，"
                "用 --boundary-handle-loop 或仅 --no-convex-opt）"
            )
            args.boundary_handle_loop = True
            args.boundary_handle_loop_only = False

    if args.no_convex_opt and not args.boundary_handle_loop:
        args.boundary_handle_loop = True


def _boundary_opt_flags(args: argparse.Namespace) -> tuple[bool, bool]:
    """返回 (convex_boundary_opt, boundary_handle_loop)。"""
    convex = not args.no_convex_opt
    handle_loop = bool(args.boundary_handle_loop) and not convex
    return convex, handle_loop


def _parse_list_arg(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    raw = arg.strip()
    if not raw:
        return None
    for sep in ("、", "，", ";"):
        raw = raw.replace(sep, ",")
    parts = [x.strip().strip("/") for x in raw.split(",")]
    return [x for x in parts if x]


def _parse_semantic_parts_cli(arg: Optional[str]) -> Optional[List[str]]:
    return _parse_list_arg(arg)


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
        raise ValueError(f"{path} 不是 list，也没有 semantic_parts 字段")
    out: List[str] = []
    for x in items:
        if isinstance(x, str) and x.strip():
            out.append(x.strip())
    return out or None


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.full_coverage_cli",
        description=(
            "VLM 语义分块 CLI。默认 full_coverage（枚举+multi-label）；"
            "加 --semantic-tree 走层级语义树逐层切分。"
        ),
    )
    p.add_argument(
        "mesh",
        type=str,
        help="渲染姿态 mesh 路径（推荐 output/<模型>/medoid_pose/render_pose.off）",
    )

    # ---- 模式 ----
    p.add_argument(
        "--semantic-tree",
        action="store_true",
        help="使用层级语义树模式（默认：full_coverage 枚举+multi-label）。",
    )

    # ---- full_coverage：labels / 枚举 ----
    p.add_argument(
        "--labels",
        type=str,
        default=None,
        help=(
            "face_labels JSON（face_segment_labels.json / face_cluster_labels.json）。"
            "省略时直接用整 mesh 的连通分量作为初始分块。（full_coverage）"
        ),
    )
    p.add_argument(
        "--enumerate-hint",
        type=str,
        default="",
        help=(
            "VLM 枚举/语义树生成阶段补充说明（宜短）。"
            "full_coverage：引导粗粒度全覆盖分区；"
            "semantic_tree：等同 --tree-hint。"
            "例如「头部变成两个蘑菇的马匹」。"
        ),
    )
    p.add_argument(
        "--semantic-parts",
        type=str,
        default=None,
        help="直接给定全覆盖语义清单，跳过 VLM 枚举。（full_coverage）",
    )
    p.add_argument(
        "--semantic-parts-file",
        type=str,
        default=None,
        help="从 JSON 文件读语义清单。（full_coverage）",
    )
    p.add_argument(
        "--enumerate-min-parts",
        type=int,
        default=2,
        help="枚举最少部位数（full_coverage，默认 2）",
    )
    p.add_argument(
        "--enumerate-max-parts",
        type=int,
        default=12,
        help="枚举最多部位数（full_coverage，默认 12）",
    )
    p.add_argument(
        "--enumerate-model",
        type=str,
        default=None,
        help="枚举阶段单独指定模型；不传则沿用 --model。",
    )

    # ---- semantic_tree 专用 ----
    p.add_argument(
        "--tree-hint",
        type=str,
        default=None,
        help=(
            "语义树生成补充说明（semantic_tree）。"
            "未指定时沿用 --enumerate-hint。"
        ),
    )
    p.add_argument(
        "--semantic-tree-file",
        type=str,
        default=None,
        help="从 JSON 加载语义树，跳过 VLM 生成。（semantic_tree）",
    )
    p.add_argument(
        "--max-tree-depth",
        type=int,
        default=5,
        help="VLM 生成语义树的最大层级（semantic_tree，默认 5）",
    )
    p.add_argument(
        "--tree-model",
        type=str,
        default=None,
        help="语义树生成 VLM 模型（semantic_tree；默认 --model）",
    )
    p.add_argument(
        "--refine-depth",
        type=int,
        default=1,
        metavar="N",
        help="从根向下自动细分数层（semantic_tree，默认 1）",
    )
    p.add_argument(
        "--refine-all",
        action="store_true",
        help="细分整棵语义树直到叶子（semantic_tree）",
    )
    p.add_argument(
        "--refine-paths",
        type=str,
        default=None,
        help="只细分指定路径，如 '上半身,上半身/手臂'（semantic_tree）",
    )
    p.add_argument(
        "--refine-nodes",
        type=str,
        default=None,
        help="只细分名称匹配的节点，如 '手臂,手'（semantic_tree）",
    )
    p.add_argument(
        "--min-visible-area-frac",
        type=float,
        default=0.005,
        metavar="FRAC",
        help=(
            "（semantic_tree）几乎不可见的小块阈值：占当前细分区域表面积比例低于该值的块，"
            "不再调 VLM 切分，直接按相邻边最多归入已分好的相邻块语义（默认 0.005）。"
        ),
    )
    p.add_argument(
        "--print-tree",
        action="store_true",
        help="打印 output 目录已有 semantic_tree.json 后退出。",
    )

    p.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="输出目录。默认 <output-base>/<模型>/vlm_full/",
    )
    p.add_argument(
        "--output-base", type=str, default="output",
        help="输出根目录（默认 output）",
    )

    # ---- VLM 通用 ----
    p.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="VLM 配置 YAML 路径",
    )
    p.add_argument(
        "--provider", type=str, default=None,
        help="覆盖 vlm_config.yaml 的 active_provider",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="覆盖该 provider 的 default_model",
    )
    p.add_argument(
        "--env-file", type=str, default=None,
        help="额外 .env 文件路径",
    )

    # ---- 主循环 / 几何切分 ----
    p.add_argument(
        "--max-depth", type=int, default=6,
        help=(
            "几何切分递归最大深度（默认 6）。"
            "full_coverage / semantic_tree 均生效；"
            "达到上限时 VLM 必须给出唯一最终归属，不再继续切分。"
        ),
    )
    p.add_argument(
        "--min-block-size", type=int, default=20,
        help="忽略 face 数低于该值的分块（默认 20）",
    )
    p.add_argument(
        "--max-seeds-per-split",
        type=int,
        default=None,
        help=(
            "单次分裂最多种子数。"
            "full_coverage 默认 4；semantic_tree 默认 8。"
        ),
    )
    p.add_argument(
        "--split-strategy",
        choices=("auto", "spectral", "cvt", "sdf", "skeleton", "geodesic_fps"),
        default="spectral",
        help="几何分裂策略（默认 spectral）",
    )
    p.add_argument(
        "--split-strategy-model", type=str, default=None,
        help="分块策略 VLM 模型（默认 --model）",
    )
    p.add_argument(
        "--skip-segmat", action="store_true", default=True,
        help="跳过 segmat 预切（默认开启）",
    )
    p.add_argument(
        "--no-skip-segmat", action="store_false", dest="skip_segmat",
        help="恢复 segmat 预切",
    )
    p.add_argument(
        "--vlm-split-strategy", action="store_true",
        help="由 VLM 看图选择分块策略",
    )
    p.add_argument(
        "--split-timeout", type=float, default=60.0, metavar="SEC",
        help="单次几何分裂超时（秒，0=不限时）",
    )
    p.add_argument(
        "--split-strategy-retries", type=int, default=2, metavar="N",
        help="分裂失败后再试 N 次",
    )
    p.add_argument(
        "--no-split-retry-vlm", action="store_true",
        help="分裂重试时只用启发式轮换策略",
    )
    p.add_argument("--image-size", type=int, default=512, help="单视角分辨率")
    p.add_argument(
        "--panel-cols", type=int, default=4,
        help="拼图列数（默认 4×3=12）",
    )
    p.add_argument(
        "--parallel-vlm", type=int, default=1, metavar="N",
        help="每轮 multi-label 并发 VLM 数（full_coverage，默认 1）",
    )
    p.add_argument(
        "--no-live-report", action="store_true",
        help="关闭 live_report.html",
    )
    p.add_argument(
        "--live-report-refresh", type=int, default=10, metavar="SEC",
        help="live_report 刷新间隔（秒）",
    )
    p.add_argument(
        "--no-report", action="store_true",
        help="关闭最终 report.html（full_coverage）",
    )
    p.add_argument(
        "--no-convex-opt", action="store_true",
        help="跳过凸包/平面边界优化（full_coverage）；可配合 --boundary-handle-loop 仅做 handle loop 平滑",
    )
    p.add_argument(
        "--boundary-handle-loop",
        action="store_true",
        help=(
            "分块结束后仅沿 handle_loop / 凹缝拉直交界，不做平面拟合与凸包能量优化。"
            "可与 --no-convex-opt 联用。"
        ),
    )
    p.add_argument(
        "--boundary-handle-loop-only",
        action="store_true",
        help=(
            "仅对已有 face_semantic_labels.json 做 handle loop 平滑（不调用 VLM）。"
            "若文件不存在则自动改为分块结束后执行。"
        ),
    )
    p.add_argument(
        "--convex-opt-max-swaps", type=int, default=800, metavar="N",
        help="凸包优化最大交换次数（planar 模式下控制 max_pair_passes）",
    )
    p.add_argument(
        "--convex-opt-min-faces", type=int, default=8, metavar="N",
        help="凸包优化每块最少面数",
    )
    p.add_argument(
        "--convex-opt-min-component-faces", type=int, default=20, metavar="N",
        help="凸包优化前连通块去噪阈值",
    )
    p.add_argument(
        "--no-topology-repair", action="store_true",
        help="跳过拓扑约束审计与修复（full_coverage）",
    )
    p.add_argument(
        "--topology-repair-model", type=str, default=None,
        help="拓扑修复 VLM 模型（默认 --model）",
    )
    p.add_argument(
        "--topology-repair-max-rounds", type=int, default=3, metavar="N",
        help="拓扑修复最大轮次",
    )

    # ---- 视角选择 ----
    p.add_argument(
        "--no-view-select", action="store_true",
        help="关闭 VLM 选最佳视角",
    )
    p.add_argument(
        "--view-select-model", type=str, default=None,
        help="视角选择模型（默认 --model）",
    )
    p.add_argument(
        "--view-select-hint", type=str, default="",
        help="视角选择 VLM 简短提示",
    )
    p.add_argument(
        "--single-view-size", type=int, default=720, metavar="PX",
        help="单视角渲染分辨率",
    )

    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--dry-run", action="store_true",
        help="只渲染 overview / 初始分块，不调用 VLM",
    )
    return p


def _run_boundary_handle_loop_only(
    args: argparse.Namespace,
    mesh_path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    out_dir: Path,
) -> int:
    """仅运行 planar_boundary + handle_loop 边界优化（跳过 VLM）。"""
    labels_path = _resolve_semantic_labels_path(
        args, mesh_path, n_faces=len(faces), out_dir=out_dir,
    )
    if labels_path is None:
        print(
            "[错误] 找不到 face_semantic_labels.json。"
            "请先跑完整分块，或用 --labels 指定。",
            file=sys.stderr,
        )
        return 2
    if not labels_path.is_file():
        print(f"[错误] labels JSON 不存在: {labels_path}", file=sys.stderr)
        return 2

    try:
        semantic_parts, face_idx_list = _load_semantic_labels(
            labels_path, n_faces=len(faces),
        )
    except ValueError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    try:
        from geometry.convex_partition_optimize import (
            ConvexPartitionOptConfig,
            map_component_labels_to_semantic,
            optimize_face_partition_convex_hull,
            split_semantic_into_connected_components,
            write_convex_optimize_report,
        )
    except ImportError as exc:
        print(f"[错误] 凸包优化模块不可用: {exc}", file=sys.stderr)
        return 1

    face_idx = np.asarray(face_idx_list, dtype=np.int64)
    n_sem = len(semantic_parts)
    assigned = (face_idx >= 0) & (face_idx < n_sem)
    active_parts = sorted({int(x) for x in face_idx[assigned]})
    if len(active_parts) < 2:
        print(
            f"[错误] 有效语义部位不足 2（active={len(active_parts)}）",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[labels] {labels_path}  ({len(semantic_parts)} 部位)")
    print(
        "[mode]   boundary_handle_loop_only "
        "(handle_loop smoothing, no VLM)"
    )

    split = split_semantic_into_connected_components(
        vertices,
        faces,
        face_idx,
        semantic_parts,
        min_component_faces=args.convex_opt_min_component_faces,
    )
    comp_labels = split.comp_labels.copy()
    valid_comp = comp_labels >= 0
    if int(valid_comp.sum()) == 0 or split.num_components < 2:
        print(
            f"[错误] 连通块不足 2（components={split.num_components}）",
            file=sys.stderr,
        )
        return 1

    comp_labels_for_opt = comp_labels.copy()
    comp_labels_for_opt[~valid_comp] = -1
    max_pair_passes = max(1, min(5, args.convex_opt_max_swaps // 200 + 1))

    convex_cfg = ConvexPartitionOptConfig(
        enabled=True,
        max_swaps=args.convex_opt_max_swaps,
        min_block_faces=args.convex_opt_min_faces,
        split_connected_components=True,
        min_component_faces=args.convex_opt_min_component_faces,
        enforce_connectivity=False,
        image_size=args.image_size,
        panel_cols=args.panel_cols,
        preserve_label_ids=False,
        stage_title="连通块",
        verbose=True,
        boundary_method="handle_loop",
        max_pair_passes=max_pair_passes,
    )
    split_meta = {
        "num_components": split.num_components,
        "component_names": list(split.component_names),
        "comp_part": list(split.comp_part),
        "min_component_faces": args.convex_opt_min_component_faces,
    }

    print(
        f"[opt]    components={split.num_components}  "
        f"max_pair_passes={max_pair_passes}"
    )
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
        semantic_parts=semantic_parts,
        display_labels_before=face_idx,
        display_labels_after=face_idx_opt,
        component_names=split.component_names,
        comp_labels_split=comp_labels,
        split_meta=split_meta,
    )

    num_faces_changed = int(np.sum(face_idx_opt != face_idx))
    sem_groups: Dict[str, List[int]] = {n: [] for n in semantic_parts}
    for fid, v in enumerate(face_idx_opt.tolist()):
        if 0 <= v < len(semantic_parts):
            sem_groups[semantic_parts[v]].append(fid)

    labels_json = out_dir / "face_semantic_labels.json"
    labels_json.write_text(
        json.dumps(
            {
                "num_faces": len(faces),
                "semantic_parts": list(semantic_parts),
                "face_semantic_index": face_idx_opt.tolist(),
                "num_background": 0,
                "num_pending": int(np.sum(face_idx_opt == -2)),
                "semantic_face_counts": {
                    n: len(v) for n, v in sem_groups.items()
                },
                "boundary_handle_loop": {
                    "method": "boundary_handle_loop",
                    "source_labels": str(labels_path.resolve()),
                    "num_components": split.num_components,
                    "num_faces_changed": num_faces_changed,
                    "energy_before": opt.metrics_before.active_total_energy,
                    "energy_after": opt.metrics_after.active_total_energy,
                    "energy_delta": opt.energy_delta,
                    "elapsed_seconds": opt.elapsed_seconds,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("[result] boundary handle loop 完成")
    print(f"  components         : {split.num_components}")
    print(f"  faces changed      : {num_faces_changed}")
    print(
        f"  energy             : "
        f"{opt.metrics_before.active_total_energy:.4f} → "
        f"{opt.metrics_after.active_total_energy:.4f} "
        f"(Δ={opt.energy_delta:+.4f})"
    )
    print(f"  elapsed            : {opt.elapsed_seconds:.1f}s")
    if opt.report_html:
        print(f"  report             : {opt.report_html}")
    if opt.before_png and opt.after_png:
        print(f"  before / after     : {opt.before_png} / {opt.after_png}")
    print(f"  face_semantic_labels.json : {labels_json}")
    return 0


def _print_tree_node(node: Dict[str, Any], indent: int = 0) -> None:
    name = node.get("name", "?")
    path = node.get("path", "")
    prefix = "  " * indent
    tag = f" ({path})" if path else ""
    print(f"{prefix}- {name}{tag}")
    for ch in node.get("children") or []:
        if isinstance(ch, dict):
            _print_tree_node(ch, indent + 1)


def _run_full_coverage_mode(
    args: argparse.Namespace,
    client: Any,
    provider_cfg: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    out_dir: Path,
    face_labels: Optional[List[int]],
    semantic_parts: Optional[List[str]],
) -> int:
    vlm_model = resolve_model_name(args.model, provider_cfg)

    def progress(event: str, payload: Dict[str, Any]) -> None:
        if event == "classified":
            matched = payload.get("matched", []) or []
            print(f"  [classify] {payload['block_id']} -> matched={matched}")
        elif event == "split_geometry":
            reason = payload.get("reason") or ""
            print(
                f"  [strategy] {payload.get('block_id')} "
                f"→ {payload.get('strategy')} "
                f"(faces={payload.get('faces')} k={payload.get('seed_count')})"
                + (f"  ← {reason}" if reason else "")
            )
        elif event == "split_start":
            print(
                f"  [split…]   {payload['block_id']} "
                f"faces={payload.get('faces')} k={payload.get('seed_count')} "
                f"matched={payload.get('matched')}"
            )
        elif event == "split_coverage_repair":
            print(
                f"  [cover]    parent={payload.get('parent_id')} "
                f"child={payload.get('child_id')} "
                f"missing={payload.get('missing')}"
                f"{' →requeue' if payload.get('requeued') else ''}"
            )
        elif event == "split":
            strat = payload.get("strategy") or payload.get("mode") or ""
            strat_tag = f" strategy={strat}" if strat else ""
            print(
                f"  [split]    {payload['block_id']} "
                f"k={payload.get('seed_count')} "
                f"children={len(payload.get('children', []))} "
                f"matched={payload.get('matched', [])}{strat_tag}"
            )
        elif event == "split_strategy_retry":
            print(
                f"  [split↻]   {payload.get('block_id')} "
                f"→ {payload.get('new_strategy')}"
            )
        elif event == "empty_repair":
            action = payload.get("action", "")
            if action == "audit":
                print(
                    f"  [repair]   0-face audit round={payload.get('round')} "
                    f"({payload.get('empty_semantic', '')}) …"
                )
            else:
                print(
                    f"  [repair]   {payload.get('empty_semantic')} → {action}"
                    + (
                        f" from {payload['donor_block_id']}"
                        if payload.get("donor_block_id")
                        else ""
                    )
                )
        elif event == "topology_repair":
            print(f"  [topo]     {payload.get('action', '')}")
        elif event == "abandoned":
            print(f"  [skip]     {payload['block_id']} ({payload.get('reason')})")
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
        elif event == "iteration_end":
            phase = payload.get("phase") or "?"
            print(
                f"  [iter]     round={payload.get('iteration')} "
                f"queue_after={payload.get('queue_after')} "
                f"phase={phase}"
            )
        elif event == "main_loop_done":
            print(
                f"  [done]     main loop finished after "
                f"{payload.get('iterations')} iterations "
                f"({payload.get('num_blocks')} blocks) → finalizing"
            )
        elif event == "coverage_audit":
            if "parent_id" in payload:
                print(
                    f"  [audit]    parent={payload.get('parent_id')} "
                    f"({payload.get('index')}/{payload.get('total')})"
                )
            else:
                print(
                    f"  [audit]    split parents: "
                    f"{payload.get('audit_parents')} to check, "
                    f"{payload.get('skipped_ok')} already ok"
                )
        elif event == "finalize":
            print(f"  [final]    phase={payload.get('phase')}")

    max_seeds = args.max_seeds_per_split if args.max_seeds_per_split is not None else 4
    convex_opt, boundary_handle_loop = _boundary_opt_flags(args)

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
        classification_model=vlm_model,
        enumerate_model=resolve_model_name(
            args.enumerate_model or args.model, provider_cfg
        ),
        max_seeds_per_split=max_seeds,
        split_strategy=args.split_strategy,
        split_strategy_model=resolve_model_name(
            args.split_strategy_model or args.model, provider_cfg
        ),
        skip_segmat=args.skip_segmat,
        vlm_split_strategy=args.vlm_split_strategy,
        split_timeout_seconds=args.split_timeout,
        split_strategy_max_retries=args.split_strategy_retries,
        split_retry_with_vlm=not args.no_split_retry_vlm,
        parallel_vlm_workers=args.parallel_vlm,
        live_report=not args.no_live_report,
        live_report_refresh_seconds=args.live_report_refresh,
        generate_report=not args.no_report,
        convex_boundary_opt=convex_opt,
        boundary_handle_loop=boundary_handle_loop,
        convex_opt_max_swaps=args.convex_opt_max_swaps,
        convex_opt_min_faces=args.convex_opt_min_faces,
        convex_opt_min_component_faces=args.convex_opt_min_component_faces,
        topology_repair=not args.no_topology_repair,
        topology_repair_model=resolve_model_name(
            args.topology_repair_model or args.model, provider_cfg
        ),
        topology_repair_max_rounds=args.topology_repair_max_rounds,
        select_best_view=not args.no_view_select,
        view_select_model=resolve_model_name(
            args.view_select_model or args.model, provider_cfg
        ),
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


def _run_semantic_tree_mode(
    args: argparse.Namespace,
    client: Any,
    provider_cfg: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    out_dir: Path,
) -> int:
    vlm_model = resolve_model_name(args.model, provider_cfg)
    tree_hint = args.tree_hint if args.tree_hint is not None else args.enumerate_hint
    refine_paths = _parse_list_arg(args.refine_paths)
    refine_nodes = _parse_list_arg(args.refine_nodes)
    max_seeds = args.max_seeds_per_split if args.max_seeds_per_split is not None else 8

    if refine_paths:
        print(f"[refine] paths: {refine_paths}")
    if refine_nodes:
        print(f"[refine] nodes: {refine_nodes}")
    if args.refine_all:
        print("[refine] all leaves")
    elif not refine_paths and not refine_nodes:
        print(f"[refine] depth={args.refine_depth}")

    def progress(event: str, payload: Dict[str, Any]) -> None:
        if event == "phase_start":
            phase = payload.get("phase", "")
            if phase == "base_coverage":
                print("  [phase]    底层全覆盖（根节点切分）…")
            elif phase == "refine":
                print(
                    f"  [phase]    逐层细分 level={payload.get('level')} "
                    f"items={payload.get('items')}"
                )
        elif event == "tree_generate":
            print("  [tree]     VLM 生成层级语义树…")
        elif event == "phase_end":
            if payload.get("phase") == "base_coverage":
                print(
                    f"  [base]     全覆盖完成，待细分 {payload.get('refine_queue', 0)} 块"
                )
        elif event == "split_start":
            print(
                f"  [split…]   {payload['block_id']} "
                f"node={payload.get('tree_node', '')} "
                f"k={payload.get('seed_count')} matched={payload.get('matched')}"
            )
        elif event == "split_geometry":
            reason = payload.get("reason") or ""
            print(
                f"  [strategy] {payload.get('block_id')} "
                f"→ {payload.get('strategy')} "
                f"(faces={payload.get('faces')} k={payload.get('seed_count')}"
                + (
                    f" retry#{payload.get('attempt')}"
                    if payload.get("attempt")
                    else ""
                )
                + ")"
                + (f"  ← {reason}" if reason else "")
            )
        elif event == "split_strategy_retry":
            print(
                f"  [strategy↻] {payload.get('block_id')} "
                f"{payload.get('previous_strategy')} → {payload.get('new_strategy')} "
                f"({payload.get('failure')})"
            )
        elif event == "split":
            strat = payload.get("strategy") or ""
            strat_tag = f" strategy={strat}" if strat else ""
            print(
                f"  [split]    {payload['block_id']} "
                f"node={payload.get('tree_node', '')} "
                f"children={len(payload.get('children', []))}{strat_tag}"
            )
        elif event == "iteration_end":
            phase = payload.get("phase", "")
            print(
                f"  [level]    tree_level={payload.get('iteration')} "
                f"queue_after={payload.get('queue_after')} "
                f"phase={phase}"
            )
        elif event == "max_depth":
            print(
                f"  [max_depth] {payload.get('block_id')} "
                f"→ {payload.get('dominant')} "
                f"(depth={payload.get('depth')}, was={payload.get('matched')})"
            )

    convex_opt, boundary_handle_loop = _boundary_opt_flags(args)

    result = run_semantic_tree_partition(
        client,
        vertices,
        faces,
        out_dir,
        tree_hint=tree_hint or "",
        semantic_tree_file=args.semantic_tree_file,
        max_tree_depth=args.max_tree_depth,
        tree_model=resolve_model_name(args.tree_model or args.model, provider_cfg),
        refine_depth=args.refine_depth,
        refine_all=args.refine_all,
        refine_paths=refine_paths,
        refine_nodes=refine_nodes,
        min_block_size=args.min_block_size,
        max_depth=args.max_depth,
        image_size=args.image_size,
        panel_cols=args.panel_cols,
        classification_model=vlm_model,
        max_seeds_per_split=max_seeds,
        split_strategy=args.split_strategy,
        split_strategy_model=resolve_model_name(
            args.split_strategy_model or args.model, provider_cfg
        ),
        skip_segmat=args.skip_segmat,
        vlm_split_strategy=args.vlm_split_strategy,
        split_timeout_seconds=args.split_timeout,
        split_strategy_max_retries=args.split_strategy_retries,
        split_retry_with_vlm=not args.no_split_retry_vlm,
        select_best_view=not args.no_view_select,
        view_select_model=resolve_model_name(
            args.view_select_model or args.model, provider_cfg
        ),
        view_select_semantic_hint=args.view_select_hint,
        single_view_image_size=args.single_view_size,
        topology_repair=not args.no_topology_repair,
        convex_boundary_opt=convex_opt,
        boundary_handle_loop=boundary_handle_loop,
        convex_opt_max_swaps=args.convex_opt_max_swaps,
        convex_opt_min_faces=args.convex_opt_min_faces,
        convex_opt_min_component_faces=args.convex_opt_min_component_faces,
        live_report=not args.no_live_report,
        live_report_refresh_seconds=args.live_report_refresh,
        min_visible_area_frac=args.min_visible_area_frac,
        progress_cb=progress,
    )

    print()
    print(f"[result] {result.message}")
    print(f"  tree levels        : {result.num_tree_levels}")
    print(f"  final leaf blocks  : {result.num_final_blocks}")
    if result.semantic_tree and result.semantic_tree.object_kind:
        print(f"  object_kind        : {result.semantic_tree.object_kind}")
    for name in result.semantic_parts:
        cnt = len(result.semantic_face_groups.get(name, []))
        if cnt:
            print(f"    - {name:<20s} {cnt} face")
    if out_dir.joinpath("semantic_tree.json").is_file():
        print(f"  semantic_tree.json : {out_dir / 'semantic_tree.json'}")
    if result.labels_json:
        print(f"  face_semantic_labels.json : {result.labels_json}")
    if out_dir.joinpath("live_report.html").is_file():
        print(f"  live_report.html   : {out_dir / 'live_report.html'}")
    print(f"  elapsed            : {result.elapsed_seconds:.1f}s")
    return 0 if result.ok else 1


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

    out_dir = _resolve_output_dir(mesh_path, args.output_dir, args.output_base)
    _normalize_cli_args(args, mesh_path, out_dir)

    if args.print_tree:
        tree_path = out_dir / "semantic_tree.json"
        if not tree_path.is_file():
            print(f"[错误] 找不到 {tree_path}", file=sys.stderr)
            return 2
        with open(tree_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _print_tree_node(data.get("tree") or data)
        return 0

    print(f"[mesh] {mesh_path}")
    vertices, faces = _load_mesh(mesh_path)
    print(f"[mesh] V={len(vertices)} F={len(faces)}")

    if args.boundary_handle_loop_only:
        if args.semantic_tree:
            print(
                "[错误] --boundary-handle-loop-only 与 --semantic-tree 互斥",
                file=sys.stderr,
            )
            return 2
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[output] {out_dir.resolve()}")
        return _run_boundary_handle_loop_only(
            args, mesh_path, vertices, faces, out_dir,
        )

    print(f"[mode]   {'semantic_tree' if args.semantic_tree else 'full_coverage'}")

    face_labels: Optional[List[int]] = None
    if not args.semantic_tree:
        if args.labels:
            labels_path = Path(args.labels)
            if not labels_path.is_file():
                print(f"[错误] labels JSON 不存在: {labels_path}", file=sys.stderr)
                return 2
            face_labels = _load_face_labels(labels_path, n_faces=len(faces))
            print(f"[labels] {labels_path.name}")
        else:
            print("[labels] 未提供 --labels；使用整 mesh 连通分量")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {out_dir.resolve()}")
    if not args.no_live_report and not args.dry_run:
        print(
            f"[live]   {(out_dir / 'live_report.html').resolve()} "
            f"(每 {args.live_report_refresh}s 刷新)"
        )

    semantic_parts: Optional[List[str]] = None
    if not args.semantic_tree:
        semantic_parts = (
            _parse_semantic_parts_file(args.semantic_parts_file)
            or _parse_semantic_parts_cli(args.semantic_parts)
        )
        if semantic_parts:
            print(f"[semantic_parts] {len(semantic_parts)} 项: {semantic_parts}")

    if args.dry_run:
        from rendering.render_component import (
            render_initial_segmentation_panel,
            render_overview_only_panel,
        )

        ov_path = out_dir / "semantics_overview.png"
        render_overview_only_panel(
            vertices, faces, ov_path,
            title="整模型 12 视角（dry-run）",
            image_size=args.image_size,
            cols=args.panel_cols,
        )
        print(f"[dry-run] overview: {ov_path}")
        if not args.semantic_tree:
            init_labels = face_labels if face_labels else [0] * len(faces)
            panel = render_initial_segmentation_panel(
                vertices, faces, init_labels,
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
            f"[错误] provider '{provider_cfg.name}' 缺少 API key",
            file=sys.stderr,
        )
        return 3

    vlm_model = resolve_model_name(args.model, provider_cfg)
    print(
        f"[vlm] provider={provider_cfg.name} model={vlm_model} "
        f"base_url={provider_cfg.base_url or '<n/a>'}"
    )
    client = build_vlm_client(provider_cfg)

    if args.semantic_tree:
        return _run_semantic_tree_mode(args, client, provider_cfg, vertices, faces, out_dir)
    return _run_full_coverage_mode(
        args, client, provider_cfg, vertices, faces, out_dir,
        face_labels, semantic_parts,
    )


if __name__ == "__main__":
    raise SystemExit(main())
