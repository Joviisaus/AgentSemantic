"""测试点云分块策略（与 ``scripts/test_partition.py`` 对齐，默认 NEO 辅助 spectral）。

用法示例::

    python scripts/test_pointcloud_partition.py mesh.stl -s geodesic_fps -k 4
    python scripts/test_pointcloud_partition.py points.npy -s spectral -k 6 --out-dir output/pc_partition_test
    python scripts/test_pointcloud_partition.py mesh.stl --benchmark-all -k 4
    python scripts/test_pointcloud_partition.py mesh.stl -s spectral --no-neo -k 4
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

from geometry.pointcloud_cluster import split_points_by_connected_components
from geometry.pointcloud_io import load_pointcloud
from geometry.pointcloud_partition import (
    VALID_POINT_STRATEGIES,
    split_needs_strategy_retry,
    split_points_by_strategy,
)
from rendering.heatmap_renderer import DEFAULT_OVERVIEW_VIEW


def _resolve_point_block(
    points: np.ndarray,
    *,
    all_points: bool,
    cc_index: int,
    n_neighbors: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """选择待分块的点集合（默认取最大 kNN 连通分量）。"""
    n_points = int(len(points))
    if all_points:
        block = list(range(n_points))
        meta = {"block_mode": "all_points", "cc_count": 1, "cc_sizes": [n_points]}
        return block, meta

    cc_groups = split_points_by_connected_components(
        points, list(range(n_points)), n_neighbors=n_neighbors, min_component_points=1
    )
    if not cc_groups:
        block = list(range(n_points))
        meta = {"block_mode": "single_component", "cc_count": 1, "cc_sizes": [n_points]}
        return block, meta

    cc_sizes = sorted((len(g) for g in cc_groups), reverse=True)
    cc_groups = sorted(cc_groups, key=len, reverse=True)
    idx = int(cc_index)
    if idx < 0 or idx >= len(cc_groups):
        raise ValueError(
            f"--cc-index={idx} 越界，该点云共 {len(cc_groups)} 个连通分量"
        )
    block = list(cc_groups[idx])
    meta = {
        "block_mode": "connected_component",
        "cc_index": idx,
        "cc_count": len(cc_groups),
        "cc_sizes": cc_sizes,
        "block_points": len(block),
    }
    return block, meta


def _cluster_sizes(groups: Sequence[Sequence[int]]) -> List[int]:
    return sorted((len(g) for g in groups), reverse=True)


def _build_point_labels(n_points: int, split: Any, point_block: Sequence[int]) -> np.ndarray:
    labels = np.full(n_points, -1, dtype=np.int64)
    block_ids = np.asarray(list(point_block), dtype=np.int64)
    sub_labels = np.asarray(split.sub_labels, dtype=np.int64)
    for i, pid in enumerate(block_ids):
        labels[int(pid)] = int(sub_labels[i])
    return labels


def _save_colored_pointcloud(
    points: np.ndarray,
    point_labels: np.ndarray,
    out_path: Path,
) -> None:
    import trimesh
    from rendering.pointcloud_render import point_labels_to_rgb

    rgb = point_labels_to_rgb(point_labels)
    cloud = trimesh.PointCloud(
        vertices=np.asarray(points, dtype=np.float64)[:, :3],
        colors=rgb,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cloud.export(str(out_path))


def _render_overview_png(
    points: np.ndarray,
    point_labels: np.ndarray,
    out_path: Path,
    *,
    image_size: int = 960,
    point_size: Optional[float] = None,
) -> bool:
    try:
        from PIL import Image
        from rendering.pointcloud_render import point_labels_to_rgb, render_pointcloud_rgb_view
    except ImportError as exc:
        print(f"[render] 跳过 PNG（缺少依赖）: {exc}", file=sys.stderr)
        return False

    rgb = point_labels_to_rgb(point_labels)
    elev, azim = DEFAULT_OVERVIEW_VIEW
    img = render_pointcloud_rgb_view(
        points, rgb, elev, azim,
        image_size=(image_size, image_size),
        point_size=point_size,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(out_path))
    return True


def _compute_neo_eigenspace(
    points: np.ndarray,
    *,
    neo_ckpt: Optional[str],
    neo_python: Optional[str],
    neo_k: int,
    neo_device: str,
    neo_no_mass: bool,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        from geometry.neo_backend import neo_eigenspace
    except ImportError as exc:
        return None, str(exc)

    t0 = time.perf_counter()
    try:
        eigvecs, _eigvals = neo_eigenspace(
            points,
            ckpt=neo_ckpt,
            neo_python=neo_python,
            k=int(neo_k),
            device=neo_device,
            no_mass=neo_no_mass,
        )
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    elapsed = time.perf_counter() - t0
    print(f"  NEO 全局特征空间: eigvecs={eigvecs.shape}  time={elapsed:.2f}s")
    return np.asarray(eigvecs, dtype=np.float64), None


def _run_one_strategy(
    points: np.ndarray,
    point_block: Sequence[int],
    *,
    strategy: str,
    seed_count: int,
    kwargs: Dict[str, Any],
    neo_eigvecs: Optional[np.ndarray],
) -> Tuple[Optional[Any], float, Optional[str]]:
    t0 = time.perf_counter()
    strat_kwargs = dict(kwargs)
    if strategy == "spectral" and neo_eigvecs is not None:
        strat_kwargs["precomputed_eigvecs"] = neo_eigvecs
    try:
        split = split_points_by_strategy(
            points,
            list(point_block),
            seed_count,
            strategy,
            **strat_kwargs,
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
    neo_used: bool = False,
) -> None:
    needs_retry, reason = split_needs_strategy_retry(split, error=error)
    status = "FAILED" if needs_retry else "OK"
    clusters = int(split.num_clusters) if split is not None else 0
    sizes = _cluster_sizes(split.sub_point_groups) if split is not None else []
    neo_tag = " neo" if neo_used and strategy == "spectral" else ""
    print(
        f"  strategy={strategy:14s}{neo_tag:4s}  status={status:6s}  time={elapsed:7.2f}s  "
        f"clusters={clusters}"
    )
    if sizes:
        print(f"    cluster point sizes: {sizes}")
    if needs_retry:
        print(f"    reason: {reason}")
    if split is not None and split.debug:
        dbg = {k: v for k, v in split.debug.items() if not isinstance(v, np.ndarray)}
        if dbg:
            print(f"    debug: {dbg}")


def _save_result_json(
    out_path: Path,
    *,
    input_path: Path,
    source: str,
    strategy: str,
    seed_count: int,
    block_meta: Dict[str, Any],
    split: Any,
    elapsed: float,
    error: Optional[str],
    params: Dict[str, Any],
    neo_used: bool,
) -> None:
    payload: Dict[str, Any] = {
        "input": str(input_path),
        "source": source,
        "strategy": strategy,
        "seed_count": seed_count,
        "elapsed_seconds": round(elapsed, 4),
        "error": error,
        "neo_used": neo_used,
        "block": block_meta,
        "params": params,
    }
    if split is not None:
        payload["num_clusters"] = int(split.num_clusters)
        payload["cluster_point_sizes"] = _cluster_sizes(split.sub_point_groups)
        payload["sub_point_groups"] = [list(g) for g in split.sub_point_groups]
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
        "n_neighbors": args.n_neighbors,
        "random_state": args.seed,
    }
    if args.graph_vertex_budget is not None:
        kwargs["graph_vertex_budget"] = int(args.graph_vertex_budget)
    return kwargs


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试点云分块策略（与 mesh test_partition 对齐；NEO 仅辅助 spectral）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        help="mesh（STL/OBJ/PLY 等）或裸点云（.npy/.pcd/.xyz/.txt）",
    )
    parser.add_argument(
        "-s", "--strategy",
        default="geodesic_fps",
        choices=VALID_POINT_STRATEGIES,
        help="分块策略",
    )
    parser.add_argument(
        "-k", "--seed-count",
        type=int,
        default=4,
        help="目标分块数（种子数）",
    )
    parser.add_argument(
        "--num-sample-points",
        type=int,
        default=15000,
        help="mesh 输入时的表面采样点数",
    )
    parser.add_argument("--seed", type=int, default=0, help="mesh 采样 / KMeans 随机种子")
    parser.add_argument(
        "--all-points",
        action="store_true",
        help="对整个点云分块，不先取最大连通分量",
    )
    parser.add_argument(
        "--cc-index",
        type=int,
        default=0,
        help="使用第 N 个连通分量（0=最大），仅在未指定 --all-points 时生效",
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
        help="输出目录：彩色点云 PLY、overview PNG、result JSON",
    )
    parser.add_argument("--no-render", action="store_true", help="不生成 PNG 预览")
    parser.add_argument("--n-neighbors", type=int, default=16, help="kNN 图邻居数")
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
    parser.add_argument("--point-size", type=float, default=None, help="渲染点大小（默认自动）")

    # NEO：默认启用，但仅在 --strategy spectral 时使用
    parser.add_argument(
        "--no-neo",
        action="store_false",
        dest="use_neo",
        help="关闭 NEO（spectral 策略回退 feature_spectral 现算）",
    )
    parser.set_defaults(use_neo=True)
    parser.add_argument("--neo-ckpt", type=str, default=None, help="NEO 权重 .ckpt")
    parser.add_argument(
        "--neo-python", type=str, default=None,
        help="NEO 环境 python（默认 .conda-neo/bin/python）",
    )
    parser.add_argument("--neo-k", type=int, default=0, help="NEO 模式数；0=模型默认")
    parser.add_argument("--neo-device", type=str, default="cpu", help="NEO 设备 cpu/cuda/mps")
    parser.add_argument("--neo-no-mass", action="store_true", help="NEO 消融 mass-aware attention")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list_strategies:
        print("可用策略:")
        for name in VALID_POINT_STRATEGIES:
            print(f"  - {name}")
        return 0

    in_path = args.input.resolve()
    if not in_path.is_file():
        print(f"输入文件不存在: {in_path}", file=sys.stderr)
        return 1

    t_load = time.perf_counter()
    try:
        points, source = load_pointcloud(
            in_path,
            num_sample_points=int(args.num_sample_points),
            seed=int(args.seed),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    load_s = time.perf_counter() - t_load

    strategy_kwargs = _build_strategy_kwargs(args)
    try:
        point_block, block_meta = _resolve_point_block(
            points,
            all_points=bool(args.all_points),
            cc_index=int(args.cc_index),
            n_neighbors=int(args.n_neighbors),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    neo_eigvecs: Optional[np.ndarray] = None
    neo_ready = False
    if args.use_neo:
        print("NEO: 计算整朵点云全局特征空间（仅 spectral 策略使用）...")
        neo_eigvecs, neo_err = _compute_neo_eigenspace(
            points,
            neo_ckpt=args.neo_ckpt,
            neo_python=args.neo_python,
            neo_k=int(args.neo_k),
            neo_device=str(args.neo_device),
            neo_no_mass=bool(args.neo_no_mass),
        )
        if neo_err:
            print(f"  NEO 不可用: {neo_err}", file=sys.stderr)
        else:
            neo_ready = True

    strategies = list(VALID_POINT_STRATEGIES) if args.benchmark_all else [str(args.strategy)]

    print(f"input: {in_path.name}  source={source}")
    print(f"  load={load_s:.2f}s  N={len(points):,}")
    print(f"  block_points={len(point_block):,}  mode={block_meta.get('block_mode')}")
    if block_meta.get("cc_count", 1) > 1:
        print(
            f"  connected_components={block_meta['cc_count']}  "
            f"sizes(top5)={block_meta['cc_sizes'][:5]}"
        )
    print(f"  seed_count={args.seed_count}  use_neo={args.use_neo}  neo_ready={neo_ready}")
    if args.benchmark_all:
        print("  mode=benchmark_all")
    else:
        print(f"  strategy={args.strategy}")
    print()

    last_split = None
    last_strategy = strategies[-1]
    last_elapsed = 0.0
    last_error: Optional[str] = None
    last_neo_used = False

    for strategy in strategies:
        use_neo = neo_ready and strategy == "spectral"
        split, elapsed, err = _run_one_strategy(
            points,
            point_block,
            strategy=strategy,
            seed_count=int(args.seed_count),
            kwargs=strategy_kwargs,
            neo_eigvecs=neo_eigvecs if use_neo else None,
        )
        _print_split_summary(strategy, split, elapsed, error=err, neo_used=use_neo)
        last_split, last_strategy, last_elapsed, last_error = split, strategy, elapsed, err
        last_neo_used = use_neo

    if (
        args.out_dir is not None
        and last_split is not None
        and not split_needs_strategy_retry(last_split, error=last_error)[0]
    ):
        out_dir = args.out_dir.resolve()
        neo_suffix = "_neo" if last_neo_used else ""
        stem = f"{in_path.stem}_{last_strategy}{neo_suffix}_k{args.seed_count}"
        point_labels = _build_point_labels(len(points), last_split, point_block)
        ply_path = out_dir / f"{stem}.ply"
        _save_colored_pointcloud(points, point_labels, ply_path)
        print(f"\n[save] colored pointcloud -> {ply_path}")
        _save_result_json(
            out_dir / f"{stem}.json",
            input_path=in_path,
            source=source,
            strategy=last_strategy,
            seed_count=int(args.seed_count),
            block_meta=block_meta,
            split=last_split,
            elapsed=last_elapsed,
            error=last_error,
            params={**strategy_kwargs, "use_neo": args.use_neo},
            neo_used=last_neo_used,
        )
        print(f"[save] result json       -> {out_dir / f'{stem}.json'}")
        if not args.no_render:
            png_path = out_dir / f"{stem}_overview.png"
            if _render_overview_png(
                points, point_labels, png_path, point_size=args.point_size
            ):
                print(f"[save] overview png       -> {png_path}")

    if last_split is None or split_needs_strategy_retry(last_split, error=last_error)[0]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
