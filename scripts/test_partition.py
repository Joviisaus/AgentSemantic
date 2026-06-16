"""测试 geometry.partition 各分块策略。

用法示例::

    python scripts/test_partition.py mesh.stl -s geodesic_fps -k 4
    python scripts/test_partition.py mesh.stl -s spectral -k 6 --out-dir output/partition_test
    python scripts/test_partition.py mesh.stl --benchmark-all -k 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry.partition import VALID_STRATEGIES, split_block_by_strategy
from geometry.partition.connected_components import split_block_by_connected_components
from geometry.partition.strategy import split_block_by_strategy_timed, split_needs_strategy_retry
from pipeline.semantic_rig_cli import _load_mesh

_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (228, 26, 28),
    (55, 126, 184),
    (77, 175, 74),
    (152, 78, 163),
    (255, 127, 0),
    (255, 215, 30),
    (166, 86, 40),
    (247, 129, 191),
    (0, 200, 200),
    (140, 86, 75),
    (23, 190, 207),
    (180, 180, 50),
)


def _resolve_face_block(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    all_faces: bool,
    cc_index: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """选择待分块的面片集合（默认取最大连通块）。"""
    n_faces = int(len(faces))
    if all_faces:
        block = list(range(n_faces))
        meta = {"block_mode": "all_faces", "cc_count": 1, "cc_sizes": [n_faces]}
        return block, meta

    cc_groups = split_block_by_connected_components(
        vertices, faces, list(range(n_faces)), min_component_faces=1
    )
    if not cc_groups:
        block = list(range(n_faces))
        meta = {"block_mode": "single_component", "cc_count": 1, "cc_sizes": [n_faces]}
        return block, meta

    cc_sizes = sorted((len(g) for g in cc_groups), reverse=True)
    cc_groups = sorted(cc_groups, key=len, reverse=True)
    idx = int(cc_index)
    if idx < 0 or idx >= len(cc_groups):
        raise ValueError(
            f"--cc-index={idx} 越界，该网格共 {len(cc_groups)} 个连通块"
        )
    block = list(cc_groups[idx])
    meta = {
        "block_mode": "connected_component",
        "cc_index": idx,
        "cc_count": len(cc_groups),
        "cc_sizes": cc_sizes,
        "block_faces": len(block),
    }
    return block, meta


def _cluster_sizes(groups: Sequence[Sequence[int]]) -> List[int]:
    return sorted((len(g) for g in groups), reverse=True)


def _build_face_colors(
    n_faces: int,
    groups: Sequence[Sequence[int]],
) -> np.ndarray:
    colors = np.full((n_faces, 3), 210, dtype=np.uint8)
    for cluster_id, face_ids in enumerate(groups):
        rgb = np.asarray(_PALETTE[cluster_id % len(_PALETTE)], dtype=np.uint8)
        for fi in face_ids:
            colors[int(fi)] = rgb
    return colors


def _save_colored_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    out_path: Path,
) -> None:
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_colors=face_colors,
        process=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))


def _render_overview_png(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    out_path: Path,
    *,
    image_size: int = 960,
) -> bool:
    try:
        import pyvista as pv
        from rendering.heatmap_renderer import DEFAULT_OVERVIEW_VIEW, _apply_camera, _to_pyvista_faces
    except ImportError as exc:
        print(f"[render] 跳过 PNG（缺少依赖）: {exc}", file=sys.stderr)
        return False

    pv_faces = _to_pyvista_faces(faces)
    pdata = pv.PolyData(vertices, pv_faces)
    pdata.cell_data["RGB"] = face_colors

    elev, azim = DEFAULT_OVERVIEW_VIEW
    plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size))
    plotter.set_background("white")
    plotter.add_mesh(
        pdata,
        scalars="RGB",
        rgb=True,
        show_scalar_bar=False,
        smooth_shading=True,
    )
    _apply_camera(plotter, vertices, elev, azim)
    plotter.enable_anti_aliasing("ssaa")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out_path))
    plotter.close()
    return True


def _run_one_strategy(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_block: Sequence[int],
    *,
    strategy: str,
    seed_count: int,
    timeout: float,
    kwargs: Dict[str, Any],
) -> Tuple[Optional[Any], float, Optional[str]]:
    t0 = time.perf_counter()
    if timeout > 0:
        split, err = split_block_by_strategy_timed(
            timeout,
            vertices=vertices,
            faces=faces,
            face_ids=list(face_block),
            seed_count=seed_count,
            strategy=strategy,
            **kwargs,
        )
    else:
        try:
            split = split_block_by_strategy(
                vertices,
                faces,
                list(face_block),
                seed_count,
                strategy,
                **kwargs,
            )
            err = None
        except Exception as exc:  # noqa: BLE001
            split, err = None, str(exc)
    elapsed = time.perf_counter() - t0
    return split, elapsed, err


def _print_split_summary(
    strategy: str,
    split: Any,
    elapsed: float,
    *,
    error: Optional[str] = None,
) -> None:
    needs_retry, reason = split_needs_strategy_retry(split, error=error)
    status = "FAILED" if needs_retry else "OK"
    clusters = int(split.num_clusters) if split is not None else 0
    sizes = _cluster_sizes(split.sub_face_groups) if split is not None else []
    print(f"  strategy={strategy:14s}  status={status:6s}  time={elapsed:7.2f}s  "
          f"clusters={clusters}")
    if sizes:
        print(f"    cluster face sizes: {sizes}")
    if needs_retry:
        print(f"    reason: {reason}")
    if split is not None and split.debug:
        dbg = {k: v for k, v in split.debug.items() if not isinstance(v, np.ndarray)}
        if dbg:
            print(f"    debug: {dbg}")


def _save_result_json(
    out_path: Path,
    *,
    mesh_path: Path,
    strategy: str,
    seed_count: int,
    block_meta: Dict[str, Any],
    split: Any,
    elapsed: float,
    error: Optional[str],
    params: Dict[str, Any],
) -> None:
    payload: Dict[str, Any] = {
        "mesh": str(mesh_path),
        "strategy": strategy,
        "seed_count": seed_count,
        "elapsed_seconds": round(elapsed, 4),
        "error": error,
        "block": block_meta,
        "params": params,
    }
    if split is not None:
        payload["num_clusters"] = int(split.num_clusters)
        payload["cluster_face_sizes"] = _cluster_sizes(split.sub_face_groups)
        payload["sub_face_groups"] = [list(g) for g in split.sub_face_groups]
        dbg = {}
        for k, v in (split.debug or {}).items():
            if isinstance(v, np.ndarray):
                dbg[k] = v.tolist()
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], np.ndarray):
                dbg[k] = [x.tolist() for x in v]
            else:
                dbg[k] = v
        payload["debug"] = dbg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_strategy_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "lambda_smooth": args.lambda_smooth,
        "lambda_concave": args.lambda_concave,
        "scalar_weight": args.scalar_weight,
        "geodesic_weight": args.geodesic_weight,
        "curvature_weight": args.curvature_weight,
        "concavity_weight": args.concavity_weight,
        "lloyd_iter": args.lloyd_iter,
        "num_rays": args.num_rays,
        "cone_angle_deg": args.cone_angle_deg,
        "contraction_iters": args.contraction_iters,
    }
    if args.graph_vertex_budget is not None:
        kwargs["graph_vertex_budget"] = int(args.graph_vertex_budget)
    return kwargs


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 geometry.partition 分块策略",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mesh", type=Path, help="网格路径（STL/OBJ/PLY 等，trimesh 可读格式）")
    parser.add_argument(
        "-s", "--strategy",
        default="geodesic_fps",
        choices=VALID_STRATEGIES,
        help="分块策略",
    )
    parser.add_argument(
        "-k", "--seed-count",
        type=int,
        default=4,
        help="目标分块数（种子数）",
    )
    parser.add_argument(
        "--all-faces",
        action="store_true",
        help="对整个网格分块，不先取最大连通块",
    )
    parser.add_argument(
        "--cc-index",
        type=int,
        default=0,
        help="使用第 N 个连通块（0=最大），仅在未指定 --all-faces 时生效",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="单策略超时秒数；0 表示不启用进程超时",
    )
    parser.add_argument(
        "--benchmark-all",
        action="store_true",
        help="依次跑全部策略并打印对比",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="列出可用策略后退出",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录：彩色网格、overview PNG、result JSON",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="不生成 PNG 预览",
    )
    parser.add_argument("--lambda-smooth", type=float, default=1.0)
    parser.add_argument("--lambda-concave", type=float, default=0.85)
    parser.add_argument("--scalar-weight", type=float, default=0.35)
    parser.add_argument("--geodesic-weight", type=float, default=0.20)
    parser.add_argument("--curvature-weight", type=float, default=0.15)
    parser.add_argument("--concavity-weight", type=float, default=0.70)
    parser.add_argument("--lloyd-iter", type=int, default=0)
    parser.add_argument("--num-rays", type=int, default=16)
    parser.add_argument("--cone-angle-deg", type=float, default=60.0)
    parser.add_argument("--contraction-iters", type=int, default=5)
    parser.add_argument("--graph-vertex-budget", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list_strategies:
        print("可用策略:")
        for name in VALID_STRATEGIES:
            print(f"  - {name}")
        return 0

    mesh_path = args.mesh.resolve()
    if not mesh_path.is_file():
        print(f"网格不存在: {mesh_path}", file=sys.stderr)
        return 1

    t_load = time.perf_counter()
    vertices, faces = _load_mesh(mesh_path)
    load_s = time.perf_counter() - t_load

    try:
        face_block, block_meta = _resolve_face_block(
            vertices,
            faces,
            all_faces=bool(args.all_faces),
            cc_index=int(args.cc_index),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    strategy_kwargs = _build_strategy_kwargs(args)
    strategies = list(VALID_STRATEGIES) if args.benchmark_all else [str(args.strategy)]

    print(f"mesh: {mesh_path.name}")
    print(f"  load={load_s:.2f}s  V={len(vertices):,}  F={len(faces):,}")
    print(f"  block_faces={len(face_block):,}  mode={block_meta.get('block_mode')}")
    if block_meta.get("cc_count", 1) > 1:
        print(f"  connected_components={block_meta['cc_count']}  sizes(top5)={block_meta['cc_sizes'][:5]}")
    print(f"  seed_count={args.seed_count}")
    if args.benchmark_all:
        print("  mode=benchmark_all")
    else:
        print(f"  strategy={args.strategy}")
    print()

    last_split = None
    last_strategy = strategies[-1]
    last_elapsed = 0.0
    last_error: Optional[str] = None

    for strategy in strategies:
        split, elapsed, err = _run_one_strategy(
            vertices,
            faces,
            face_block,
            strategy=strategy,
            seed_count=int(args.seed_count),
            timeout=float(args.timeout),
            kwargs=strategy_kwargs,
        )
        _print_split_summary(strategy, split, elapsed, error=err)
        last_split, last_strategy, last_elapsed, last_error = split, strategy, elapsed, err

    if args.out_dir is not None and last_split is not None and not split_needs_strategy_retry(last_split, error=last_error)[0]:
        out_dir = args.out_dir.resolve()
        stem = f"{mesh_path.stem}_{last_strategy}_k{args.seed_count}"
        face_colors = _build_face_colors(len(faces), last_split.sub_face_groups)
        _save_colored_mesh(
            vertices,
            faces,
            face_colors,
            out_dir / f"{stem}.ply",
        )
        print(f"\n[save] colored mesh -> {out_dir / f'{stem}.ply'}")
        _save_result_json(
            out_dir / f"{stem}.json",
            mesh_path=mesh_path,
            strategy=last_strategy,
            seed_count=int(args.seed_count),
            block_meta=block_meta,
            split=last_split,
            elapsed=last_elapsed,
            error=last_error,
            params=strategy_kwargs,
        )
        print(f"[save] result json  -> {out_dir / f'{stem}.json'}")
        if not args.no_render:
            png_path = out_dir / f"{stem}_overview.png"
            if _render_overview_png(vertices, faces, face_colors, png_path):
                print(f"[save] overview png -> {png_path}")

    if last_split is None or split_needs_strategy_retry(last_split, error=last_error)[0]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
