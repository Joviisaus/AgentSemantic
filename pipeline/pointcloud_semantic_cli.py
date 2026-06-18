"""
点云语义查找命令行入口（与 mesh 版 pipeline.cli 同构，但无初始分块）。

输入：
    - 点云来源：mesh 文件（会表面采样）或裸点云 .npy/.ply/.pcd/.xyz
    - 用户语义文本（例如「猫猫头」）

流程：整朵点云作为根 → 几何特征谱聚类切分（法向+曲率+可选凹凸，迁移自 mesh
      spectral_cluster）→ 每块 VLM 三分类（full/none/partial）
      → partial 继续细化，直到无 partial 或到 max_depth。

输出（output/<模型名>/pointcloud_vlm/<语义>/）：
    panels/ responses/ iterations/ initial_segmentation.png semantic_mask.png
    live_report.html（实时）  report.html（最终）  block_tree.json  semantic_mask.json

用法：
    python -m pipeline.pointcloud_semantic_cli testmodels/Cat.stl --semantic 猫猫头 --provider qwen
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _safe_dir_name(name: str) -> str:
    out = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    return out or "semantic"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.pointcloud_semantic_cli",
        description="点云语义查找（无初始分块 + VLM 三分类迭代细化）",
    )
    p.add_argument("input", type=str, help="mesh 或裸点云文件")
    p.add_argument("--semantic", "-s", required=True, type=str, help="要查找的语义，如「猫猫头」")
    p.add_argument("--num-sample-points", type=int, default=15000, help="mesh 输入时表面采样点数")
    p.add_argument("--seed", type=int, default=0, help="采样 / KMeans 随机种子")

    p.add_argument("--max-depth", type=int, default=6, help="最大细化深度")
    p.add_argument("--root-seed-count", type=int, default=6, help="第一刀切多少块")
    p.add_argument("--max-seeds", type=int, default=6, help="partial 块每次最多切几刀")
    p.add_argument("--min-block-size", type=int, default=60,
                   help="小于此点数的块不再继续切，但仍渲染并交给 VLM 判 full/none")

    # 几何分块（迁移自 mesh spectral_cluster / seed_split）
    p.add_argument("--split-method", type=str, default="feature_spectral",
                   choices=["feature_spectral", "geodesic_fps", "lbo_spectral", "kmeans"],
                   help="分块算法：feature_spectral=几何特征谱聚类（默认），"
                        "geodesic_fps=kNN 图测地 FPS+Voronoi（子块连通），"
                        "lbo_spectral=内蕴 LBO 谱嵌入，kmeans=坐标 KMeans")
    p.add_argument("--split-n-neighbors", type=int, default=16,
                   help="kNN 图 / PCA 法向曲率的邻居数")
    p.add_argument("--split-lloyd-iter", type=int, default=8,
                   help="geodesic_fps 的 Lloyd 迭代次数（让 Voronoi 切片更均衡）")
    p.add_argument("--no-split-cc-first", action="store_true",
                   help="关闭「分裂前先按连通分量拆」（默认开启，避免远距断岛）")
    p.add_argument("--no-split-enforce-connectivity", action="store_true",
                   help="关闭 KMeans 后逐 label 连通清洗（默认开启，保证每块单连通）")
    p.add_argument("--split-curvature-weight", type=float, default=0.6,
                   help="曲率指标权重（feature_spectral）")
    p.add_argument("--split-convexity-weight", type=float, default=0.0,
                   help="凹凸/凸包指标权重（feature_spectral，>0 启用）")
    p.add_argument("--split-normal-weight", type=float, default=1.0,
                   help="法向权重（feature_spectral）")

    # NEO 后端：整朵点云一次性前馈全局 LBO 特征空间，子块分裂切片复用（大点云更快）
    p.add_argument("--use-neo", action="store_true",
                   help="用 NEO 前馈全局特征空间，分裂走 lbo_spectral 切片复用"
                        "（避免每块重算 scipy eigsh；需 .conda-neo 环境 + 权重）")
    p.add_argument("--neo-ckpt", type=str, default=None,
                   help="NEO 权重 .ckpt（或环境变量 NEO_CKPT）")
    p.add_argument("--neo-python", type=str, default=None,
                   help="NEO 环境 python（默认 .conda-neo/bin/python）")
    p.add_argument("--neo-k", type=int, default=0,
                   help="NEO 取多少个模式；0=模型默认 targ_dim_model")
    p.add_argument("--neo-device", type=str, default="cpu",
                   help="NEO 推理设备 cpu/cuda/mps")
    p.add_argument("--neo-no-mass", action="store_true",
                   help="NEO 消融 mass-aware attention")

    p.add_argument("--accept-partial-at-max-depth", action="store_true",
                   help="到 max_depth 仍 partial 时按 best-effort full 接受")
    p.add_argument("--accept-partial-min-points", type=int, default=400,
                   help="best-effort 接受所需的最小点数（小块易是幻觉）")

    p.add_argument("-o", "--output-dir", type=str, default=None, help="输出目录")
    p.add_argument("--output-base", type=str, default="output", help="输出根目录")

    p.add_argument("--provider", type=str, default=None, help="覆盖 active_provider，如 qwen")
    p.add_argument("--model", type=str, default=None, help="覆盖 default_model")
    p.add_argument("--config", type=str, default=None, help="vlm_config.yaml 路径")

    p.add_argument("--no-view-select", action="store_true",
                   help="关闭 VLM 视角选择，报告/时间线直接用默认 overview 视角")
    p.add_argument("--view-select-model", type=str, default=None,
                   help="视角选择使用的模型（默认与分类同模型）")

    p.add_argument("--image-size", type=int, default=512, help="单视角分辨率（VLM 看的图）")
    p.add_argument("--point-size", type=float, default=None, help="点渲染半径；默认按点数自适应")
    p.add_argument("--no-live-report", action="store_true", help="关闭实时 live_report.html")
    p.add_argument(
        "--parallel-vlm", type=int, default=4, metavar="N",
        help="每轮 VLM 三分类 / ask_seed_count 并发数（默认 4；"
             "force_split 几何分块在 Pass A 立即执行，不等 VLM）",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # 始终打印迭代进度（否则 matplotlib 缓存之后长时间无输出，像卡住）
    logging.getLogger("pipeline.pointcloud_pipeline").setLevel(logging.INFO)

    from geometry.pointcloud_io import load_pointcloud
    from vlm.client import build_vlm_client
    from vlm.config import DEFAULT_CONFIG_PATH, get_provider_config, load_vlm_config
    from pipeline.pointcloud_pipeline import (
        PointCloudRefinementConfig,
        PointCloudRefinementPipeline,
    )

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"[错误] 输入不存在: {in_path}", file=sys.stderr)
        return 2

    # ---- 点云加载 ----
    points, source = load_pointcloud(
        in_path, num_sample_points=args.num_sample_points, seed=args.seed
    )
    print(f"[input] {in_path}  source={source}  points={len(points)}")

    # ---- VLM 客户端 ----
    cfg_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    cfg = load_vlm_config(cfg_path)
    pcfg = get_provider_config(cfg, args.provider, model=args.model)
    if not pcfg.has_key():
        print(f"[错误] provider={pcfg.name} 没有可用 API key", file=sys.stderr)
        return 3
    client = build_vlm_client(pcfg)
    print(f"[vlm] provider={pcfg.name} model={pcfg.default_model}")

    # ---- 输出目录 ----
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.output_base) / in_path.stem / "pointcloud_vlm" / _safe_dir_name(args.semantic)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "points.npy", points.astype(np.float32))
    print(f"[output] {out_dir.resolve()}")
    print(
        "[hint] 首次运行可能先建 matplotlib 字体缓存（约 1–2 分钟）；"
        "之后每轮迭代含 12 视角渲染 + 多次 Qwen 调用，整流程通常 5–15 分钟。"
        "可用 --verbose 看更详细日志。"
    )

    # ---- 跑 pipeline ----
    config = PointCloudRefinementConfig(
        semantic=args.semantic,
        max_depth=args.max_depth,
        root_seed_count=args.root_seed_count,
        max_seeds=args.max_seeds,
        min_block_size=args.min_block_size,
        split_method=args.split_method,
        split_n_neighbors=args.split_n_neighbors,
        split_lloyd_iter=args.split_lloyd_iter,
        split_cc_first=not args.no_split_cc_first,
        split_enforce_connectivity=not args.no_split_enforce_connectivity,
        split_curvature_weight=args.split_curvature_weight,
        split_convexity_weight=args.split_convexity_weight,
        split_normal_weight=args.split_normal_weight,
        use_neo=args.use_neo,
        neo_ckpt=args.neo_ckpt,
        neo_python=args.neo_python,
        neo_k=args.neo_k,
        neo_device=args.neo_device,
        neo_no_mass=args.neo_no_mass,
        image_size=args.image_size,
        point_size=args.point_size,
        classification_model=pcfg.default_model,
        seed_model=pcfg.default_model,
        accept_partial_at_max_depth=args.accept_partial_at_max_depth,
        accept_partial_min_points=args.accept_partial_min_points,
        select_best_view=not args.no_view_select,
        view_select_model=args.view_select_model or pcfg.default_model,
        parallel_vlm_workers=max(1, int(args.parallel_vlm)),
        live_report=not args.no_live_report,
    )
    pipeline = PointCloudRefinementPipeline(client, config)
    result = pipeline.run(points, out_dir)

    print("\n========== 结果 ==========")
    print(f"  {result.message}")
    print(f"  full 点数: {len(result.full_point_ids)} / {len(points)}")
    print(f"  迭代轮数: {result.num_iterations}  叶块: {result.num_final_blocks}")
    if args.use_neo:
        neo_ok = result.extra.get("neo_used", False)
        print(f"  NEO 特征空间: {'已使用（lbo_spectral 切片复用）' if neo_ok else '失败/回退到 scipy'}")
    if result.best_view is not None:
        bv = result.best_view
        print(
            f"  展示视角: #{bv.index + 1} (elev={bv.elev:.0f} azim={bv.azim:.0f}"
            f"{' fallback' if bv.fallback else ''}) {(bv.reason or '')[:60]}"
        )
    if result.semantic_panel:
        print(f"  语义掩码图: {result.semantic_panel}")
    if result.report_html:
        print(f"  最终报告:   {result.report_html}")
    print(f"  实时报告:   {out_dir / 'live_report.html'}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
