"""
点云谱聚类分割命令行入口（支持 NEO / scipy 两种特征空间后端）。

输入：
    - 点云来源：mesh 文件（.off/.glb/.obj/.stl... 会表面采样）或裸点云（.npy/.ply/.pcd/.xyz）
    - 簇数（可选，默认按特征值间隙自动估计）

特征空间后端（--backend）：
    - scipy : robust_laplacian 建 (L,M) + scipy eigsh 解广义特征，本机即可跑，无需权重（默认）
    - neo   : 调用 NEO（Neural Eigenspace Operator）前馈预测特征空间，需 .conda-neo 环境 + .ckpt

输出（output/<名称>/pointcloud/）：
    point_cluster_labels.json   每个点的 cluster label + 后端 / 调试信息
    points.npy                  使用的点云坐标（采样/读入后）
    segmentation_panel.png      12 视角分块着色拼图（--no-render 关闭）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.pointcloud_cli",
        description="点云谱聚类分割（NEO / scipy 特征空间后端）",
    )
    p.add_argument(
        "input",
        type=str,
        help="点云来源：mesh（会表面采样）或裸点云 .npy/.ply/.pcd/.xyz",
    )
    p.add_argument(
        "--backend",
        choices=("scipy", "neo"),
        default="scipy",
        help="特征空间后端：scipy（默认，本机直接算）/ neo（NEO 前馈，需环境+权重）",
    )
    p.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="簇数；省略则按特征值间隙自动估计",
    )
    p.add_argument(
        "--num-sample-points",
        type=int,
        default=20000,
        help="当输入是 mesh 时表面采样的点数（裸点云输入时忽略）",
    )
    p.add_argument("--seed", type=int, default=0, help="采样 / KMeans 随机种子")
    p.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="输出目录，默认 output/<输入名>/pointcloud/",
    )
    p.add_argument("--output-base", type=str, default="output", help="输出根目录")

    # NEO 后端
    p.add_argument("--neo-ckpt", type=str, default=None, help="NEO 权重 .ckpt（或环境变量 NEO_CKPT）")
    p.add_argument("--neo-python", type=str, default=None, help="NEO 环境 python（默认 .conda-neo/bin/python）")
    p.add_argument("--neo-k", type=int, default=0, help="NEO 取多少个模式；0=模型默认 targ_dim_model")
    p.add_argument("--neo-device", type=str, default="cpu", help="NEO 推理设备 cpu/cuda/mps")
    p.add_argument("--neo-no-mass", action="store_true", help="NEO 消融 mass-aware attention")

    p.add_argument("--image-size", type=int, default=640, help="单视角分辨率")
    p.add_argument("--point-size", type=float, default=None, help="点渲染半径；默认按点数自适应")
    p.add_argument("--no-render", action="store_true", help="跳过 12 视角渲染")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    from geometry.pointcloud_io import load_pointcloud
    from geometry.pointcloud_cluster import (
        spectral_cluster_points,
        export_point_cluster_labels_json,
    )

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"[错误] 输入不存在: {in_path}", file=sys.stderr)
        return 2

    points, source = load_pointcloud(
        in_path, num_sample_points=args.num_sample_points, seed=args.seed
    )
    print(f"[input] {in_path}  source={source}  points={len(points)}")

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.output_base) / in_path.stem / "pointcloud"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "points.npy", points.astype(np.float32))
    print(f"[output] {out_dir.resolve()}")

    precomputed_evecs = None
    precomputed_evals = None
    if args.backend == "neo":
        from geometry.neo_backend import neo_eigenspace, NEOBackendError

        try:
            print("[neo] 调用 NEO 前馈特征空间 ...")
            precomputed_evecs, precomputed_evals = neo_eigenspace(
                points,
                ckpt=args.neo_ckpt,
                neo_python=args.neo_python,
                k=args.neo_k,
                device=args.neo_device,
                no_mass=args.neo_no_mass,
            )
            print(f"[neo] eigvecs={precomputed_evecs.shape}")
        except NEOBackendError as exc:
            print(f"[错误] NEO 后端失败: {exc}", file=sys.stderr)
            return 3

    labels, n_clusters, debug = spectral_cluster_points(
        points,
        n_clusters=args.clusters,
        precomputed_eigvecs=precomputed_evecs,
        precomputed_eigvals=precomputed_evals,
        random_state=args.seed,
    )
    print(f"[cluster] backend={debug['backend']} clusters={n_clusters} "
          f"sizes={np.bincount(labels).tolist()}")

    labels_path = export_point_cluster_labels_json(
        out_dir / "point_cluster_labels.json",
        labels,
        n_clusters,
        backend=debug["backend"],
        extra={"source": source, "input": str(in_path), "debug": debug},
    )
    print(f"[labels] {labels_path}")

    if not args.no_render:
        try:
            from rendering.pointcloud_render import render_pointcloud_segmentation_panel

            panel = render_pointcloud_segmentation_panel(
                points,
                labels,
                out_dir / "segmentation_panel.png",
                title=f"点云分块 backend={debug['backend']} parts={n_clusters}",
                image_size=(args.image_size, args.image_size),
                point_size=args.point_size,
            )
            print(f"[render] {panel}")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] 渲染失败: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
