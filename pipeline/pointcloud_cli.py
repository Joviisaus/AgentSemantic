"""
点云语义细化命令行入口（mesh 版 pipeline.cli 的点云孪生版）。

与 ``pipeline.cli`` 几乎一致的输入参数，区别仅在于：
    - 输入是 mesh（会表面采样成点云）或裸点云 .npy/.ply/.pcd/.xyz；
    - 无初始 face_labels：整朵点云作为根，靠几何谱聚类逐层切分；
    - **默认启用 NEO**（整朵点云一次性前馈全局 LBO 特征空间，子块分裂 lbo_spectral
      切片复用）；NEO 环境 / 权重缺失时自动回退到 scipy 现算，不阻断流程。

流程：整朵点云 → 几何特征谱聚类切分 → 每块 VLM 三分类（full / none / partial）
      → partial 继续细化，直到无 partial 或到 max_depth。

输出（默认 output/<模型>/pointcloud_vlm/<语义>/）：
    panels/  responses/  iterations/  initial_segmentation.png  semantic_mask.png
    live_report.html（实时）  report.html（最终）  block_tree.json  semantic_mask.json

用法：
    python -m pipeline.pointcloud_cli testmodels/Cat.stl --semantic 猫头 --provider qwen
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from vlm.client import build_vlm_client
from vlm.config import DEFAULT_CONFIG_PATH, get_provider_config, load_vlm_config


def _safe_dir_name(name: str) -> str:
    out = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    return out or "semantic"


def _default_model_dir_name(input_path: Path, base: str = "output") -> str:
    """根据输入路径推断"模型名"子目录（与 mesh 版 cli 同构）：

    1. 若输入已位于 ``<base>/<model_dir>/...`` 则取首级子目录（与已有产物归并）；
    2. 否则用输入文件名 stem。
    """
    base_path = Path(base)
    try:
        rel = input_path.resolve().relative_to(base_path.resolve())
        if rel.parts and rel.parts[0] not in ("", "."):
            return rel.parts[0]
    except (ValueError, OSError):
        pass
    return input_path.stem or "pointcloud"


def _resolve_output_dir(
    input_path: Path,
    semantic: str,
    explicit: Optional[str],
    base: str = "output",
) -> Path:
    if explicit:
        return Path(explicit)
    return (
        Path(base)
        / _default_model_dir_name(input_path, base)
        / "pointcloud_vlm"
        / _safe_dir_name(semantic)
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.pointcloud_cli",
        description="基于 VLM 的点云语义细化（默认启用 NEO 全局特征空间）",
    )
    p.add_argument(
        "input",
        type=str,
        help="点云来源：mesh（.off/.glb/.obj/.stl… 会表面采样）或裸点云 .npy/.ply/.pcd/.xyz",
    )
    p.add_argument(
        "--semantic",
        "-s",
        type=str,
        required=True,
        help="用户语义，例如 '猫头'、'右前腿'",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help=(
            "输出目录。默认 <output-base>/<模型>/pointcloud_vlm/<语义>/，"
            "其中 <模型> 推断规则：输入若已在 <output-base> 下则取首级子目录，"
            "否则按输入文件名 stem。"
        ),
    )
    p.add_argument(
        "--output-base",
        type=str,
        default="output",
        help="输出根目录（默认 output）",
    )

    # ---- VLM provider（与 mesh 版 cli 一致）----
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
        help="覆盖 vlm_config.yaml 的 active_provider，如 qwen",
    )
    p.add_argument("--model", type=str, default=None, help="覆盖该 provider 的 default_model")
    p.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="额外 .env 文件路径（默认自动加载 .env / .env.local）",
    )

    # ---- 迭代细化（与 mesh 版 cli 一致）----
    p.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="分裂递归最大深度（硬上限）。",
    )
    p.add_argument(
        "--max-seeds",
        type=int,
        default=6,
        help="单次分裂最多种子数（默认 6）",
    )
    p.add_argument(
        "--min-block-size",
        type=int,
        default=60,
        help="小于此点数的块不再继续切，但仍渲染并交给 VLM 判 full/none（默认 60）",
    )
    p.add_argument(
        "--accept-partial-at-max-depth",
        action="store_true",
        help=(
            "在 max-depth 处遇到 partial 时当作'最佳估计 full'接受。"
            "默认关闭——partial 在 max-depth 处会被舍弃为 abandoned。"
        ),
    )
    p.add_argument(
        "--accept-partial-min-points",
        type=int,
        default=400,
        help="best-effort 接受所需的最小点数（小块易是幻觉，默认 400）",
    )

    # ---- 点云加载 / 渲染 ----
    p.add_argument(
        "--num-sample-points",
        type=int,
        default=15000,
        help="输入是 mesh 时表面采样的点数（裸点云输入时忽略，默认 15000）",
    )
    p.add_argument("--seed", type=int, default=0, help="采样 / KMeans 随机种子")
    p.add_argument("--image-size", type=int, default=512, help="单视角分辨率（VLM 看的图）")
    p.add_argument("--panel-cols", type=int, default=4, help="拼图列数（默认 4×3=12）")
    p.add_argument("--point-size", type=float, default=None, help="点渲染半径；默认按点数自适应")
    p.add_argument(
        "--root-seed-count",
        type=int,
        default=6,
        help="无初始分块时第一刀切多少块（默认 6）",
    )

    # ---- VLM 选分块策略（与 mesh 版 cli 对齐）----
    p.add_argument(
        "--split-strategy",
        type=str,
        default="auto",
        choices=["auto", "graph_cut", "spectral", "cvt", "sdf", "skeleton", "geodesic_fps"],
        help=(
            "分块策略：auto=每个 partial 块由 VLM 看高亮图自选策略 + 参数（曲率指标、"
            "凸包/凹凸指标等，默认）；也可强制指定 6 种之一。NEO 仅在所选策略为 spectral 时介入。"
        ),
    )
    p.add_argument(
        "--no-ask-split-strategy",
        action="store_false",
        dest="ask_split_strategy",
        help="关闭 VLM 选策略，统一用 --split-strategy（auto 时退化为 spectral）。",
    )
    p.set_defaults(ask_split_strategy=True)
    p.add_argument(
        "--split-strategy-model",
        type=str,
        default=None,
        help="选策略步骤单独指定模型；不传则沿用 seed/默认模型。",
    )

    # ---- 几何分块底层参数（NEO 关闭时生效；NEO 仅辅助 spectral）----
    p.add_argument(
        "--split-method",
        type=str,
        default="feature_spectral",
        choices=["feature_spectral", "geodesic_fps", "lbo_spectral", "kmeans"],
        help=(
            "底层分块算法（legacy；仅当 --split-strategy auto 且关闭 VLM 选策略时作默认映射用）："
            "feature_spectral/lbo_spectral → spectral，geodesic_fps → geodesic_fps。"
        ),
    )
    p.add_argument(
        "--split-n-neighbors", type=int, default=16, help="kNN 图 / PCA 法向曲率的邻居数"
    )
    p.add_argument(
        "--split-lloyd-iter", type=int, default=8, help="geodesic_fps 的 Lloyd 迭代次数"
    )
    p.add_argument(
        "--no-split-cc-first",
        action="store_true",
        help="关闭「分裂前先按连通分量拆」（默认开启，避免远距断岛）",
    )
    p.add_argument(
        "--no-split-enforce-connectivity",
        action="store_true",
        help="关闭 KMeans 后逐 label 连通清洗（默认开启，保证每块单连通）",
    )
    p.add_argument(
        "--split-curvature-weight", type=float, default=0.6, help="曲率指标权重（feature_spectral）"
    )
    p.add_argument(
        "--split-convexity-weight",
        type=float,
        default=0.0,
        help="凹凸/凸包指标权重（feature_spectral，>0 启用）",
    )
    p.add_argument(
        "--split-normal-weight", type=float, default=1.0, help="法向权重（feature_spectral）"
    )

    # ---- NEO 后端：默认启用 ----
    p.add_argument(
        "--no-neo",
        action="store_false",
        dest="use_neo",
        help=(
            "关闭 NEO，spectral 策略改用 scipy 现算谱聚类。"
            "默认启用 NEO：整朵点云一次性前馈全局 LBO 特征空间，**仅 spectral 策略**的子块"
            "分裂直接切片复用该特征空间（需 .conda-neo 环境 + 权重；缺失时自动回退 scipy）。"
        ),
    )
    p.set_defaults(use_neo=True)
    p.add_argument(
        "--neo-ckpt",
        type=str,
        default=None,
        help="NEO 权重 .ckpt（默认：环境变量 NEO_CKPT，或 weights/ 下唯一的 .ckpt）",
    )
    p.add_argument(
        "--neo-python",
        type=str,
        default=None,
        help="NEO 环境 python（默认 .conda-neo/bin/python，或环境变量 NEO_PYTHON）",
    )
    p.add_argument(
        "--neo-k", type=int, default=0, help="NEO 取多少个模式；0=模型默认 targ_dim_model"
    )
    p.add_argument("--neo-device", type=str, default="cpu", help="NEO 推理设备 cpu/cuda/mps")
    p.add_argument("--neo-no-mass", action="store_true", help="NEO 消融 mass-aware attention")

    # ---- 展示视角 / 并发 / 报告（与 mesh 版 cli 一致）----
    p.add_argument(
        "--no-view-select",
        action="store_true",
        help=(
            "默认在进入主循环前让 VLM 看 12 张缩略图、挑出最能展示该语义的视角；"
            "加这个 flag 关掉，直接用默认 overview 视角。"
        ),
    )
    p.add_argument(
        "--view-select-model",
        type=str,
        default=None,
        help="视角选择步骤单独指定模型；不传则沿用 --model / provider 默认模型。",
    )
    # ---- 分类视角子集：VLM 选完最佳视角后，决定逐块三分类用几张视角（少而精降幻觉）----
    p.add_argument(
        "--no-select-classify-views",
        action="store_false",
        dest="select_classify_views",
        help=(
            "关闭「让 VLM 决定逐块分类用几张视角」。默认开启：选最佳视角时一并让 VLM 挑一组"
            "最少但足够的互补视角，逐块高亮拼图只渲染这些（视角太多会让 VLM 幻觉/虚报部位）。"
            "关闭后逐块仍用全部 12 视角（除非用 --classify-view-count 固定）。"
        ),
    )
    p.set_defaults(select_classify_views=True)
    p.add_argument(
        "--classify-view-count",
        type=int,
        default=0,
        metavar="N",
        help=(
            "强制逐块分类固定用 N 张视角（以最佳视角为中心按角度互补挑），不询问 VLM。"
            "0=自动（由 VLM 决定，或兜底默认值）。"
        ),
    )
    p.add_argument(
        "--default-classify-views",
        type=int,
        default=4,
        metavar="N",
        help="VLM 未给/选择失败时，逐块分类兜底用的视角张数（默认 4）。",
    )
    p.add_argument(
        "--min-classify-views",
        type=int,
        default=2,
        metavar="N",
        help="VLM 可挑的最少分类视角张数（默认 2）。",
    )
    p.add_argument(
        "--max-classify-views",
        type=int,
        default=6,
        metavar="N",
        help="VLM 可挑的最多分类视角张数（默认 6，上限 12）。",
    )
    p.add_argument(
        "--single-view-size",
        type=int,
        default=720,
        metavar="PX",
        help="单视角（live_report / 时间线 / overview）渲染分辨率，默认 720。",
    )
    p.add_argument(
        "--parallel-vlm",
        type=int,
        default=4,
        metavar="N",
        help=(
            "每轮迭代里把 N 个块的 VLM 三分类 / ask_seed_count 并发请求。"
            "默认 4；注意各家 VLM 的 QPS / RPM 限额。渲染仍串行。"
        ),
    )
    # ---- 几何后处理（最终掩码清洗）----
    p.add_argument(
        "--no-postprocess",
        action="store_false",
        dest="postprocess",
        help=(
            "关闭最终掩码几何后处理。默认开启：主循环结束后对最终 full 选区算几何指标"
            "（凸包/solidity、边缘平滑度、曲率、碎片、空洞、离群），喂给 VLM 让它挑形态学"
            "清洗算子（去小岛/去离群/平滑边界/填洞/保留最大块），在 kNN 图上执行后得到更"
            "干净的最终分块；产物写到 output_dir/postprocess/ 与 postprocess.json。"
        ),
    )
    p.set_defaults(postprocess=True)
    p.add_argument(
        "--postprocess-rounds",
        type=int,
        default=2,
        metavar="N",
        help="后处理「体检→VLM→清洗」最多迭代轮数（VLM 判干净即提前停，默认 2）。",
    )
    p.add_argument(
        "--postprocess-model",
        type=str,
        default=None,
        help="后处理决策步骤单独指定 VLM 模型；不传则沿用分类模型。",
    )
    p.add_argument(
        "--no-live-report",
        action="store_true",
        help="关闭每轮迭代后写 output_dir/live_report.html 的实时进度图。",
    )
    p.add_argument(
        "--live-report-refresh",
        type=int,
        default=10,
        metavar="SEC",
        help="live_report.html 浏览器自动刷新间隔（秒，默认 10）",
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

    # ---- 输出目录 ----
    out_dir = _resolve_output_dir(in_path, args.semantic, args.output_dir, args.output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "points.npy", points.astype(np.float32))
    print(f"[output] {out_dir.resolve()}")
    if args.use_neo:
        print("[neo]   已启用（仅在所选策略为 spectral 时介入；缺环境/权重时自动回退 scipy 现算）")
    print(f"[split] strategy={args.split_strategy} ask_vlm={args.ask_split_strategy}")
    if args.classify_view_count > 0:
        print(f"[views] 逐块分类固定 {args.classify_view_count} 张视角（角度互补，降幻觉）")
    elif args.select_classify_views:
        print(
            f"[views] 逐块分类视角由 VLM 决定（{args.min_classify_views}~{args.max_classify_views} 张；"
            f"兜底 {args.default_classify_views} 张）"
        )
    else:
        print("[views] 逐块分类用全部 12 视角（未启用视角裁剪）")
    if not args.no_live_report:
        print(
            f"[live]   {(out_dir / 'live_report.html').resolve()}"
            f"  (浏览器打开，每 {args.live_report_refresh}s 自动刷新)"
        )
    print(
        "[hint] 首次运行可能先建 matplotlib 字体缓存（约 1–2 分钟）；"
        "之后每轮迭代含 12 视角渲染 + 多次 VLM 调用，整流程通常 5–15 分钟。"
        "可用 --verbose 看更详细日志。"
    )

    def progress(event: str, payload: Dict[str, Any]) -> None:
        if event == "classified":
            print(f"  [classify] {payload['block_id']} -> {payload['label']}")
        elif event == "split":
            strat = payload.get("strategy") or payload.get("mode") or ""
            strat_tag = f" strategy={strat}" if strat else ""
            print(
                f"  [split]    {payload['block_id']} seeds={payload.get('seed_count')} "
                f"children={len(payload.get('children', []))}{strat_tag}"
            )
        elif event == "abandoned":
            print(f"  [skip]     {payload['block_id']} ({payload.get('reason')})")

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
        ask_split_strategy=args.ask_split_strategy,
        split_strategy=args.split_strategy,
        split_strategy_model=args.split_strategy_model or provider_cfg.default_model,
        use_neo=args.use_neo,
        neo_ckpt=args.neo_ckpt,
        neo_python=args.neo_python,
        neo_k=args.neo_k,
        neo_device=args.neo_device,
        neo_no_mass=args.neo_no_mass,
        image_size=args.image_size,
        panel_cols=args.panel_cols,
        point_size=args.point_size,
        classification_model=provider_cfg.default_model,
        seed_model=provider_cfg.default_model,
        accept_partial_at_max_depth=args.accept_partial_at_max_depth,
        accept_partial_min_points=args.accept_partial_min_points,
        select_best_view=not args.no_view_select,
        view_select_model=args.view_select_model or provider_cfg.default_model,
        select_classify_views=args.select_classify_views,
        classify_view_count=max(0, int(args.classify_view_count)),
        default_classify_view_count=max(1, int(args.default_classify_views)),
        min_classify_views=max(1, int(args.min_classify_views)),
        max_classify_views=max(1, int(args.max_classify_views)),
        single_view_image_size=args.single_view_size,
        parallel_vlm_workers=max(1, int(args.parallel_vlm)),
        postprocess=args.postprocess,
        postprocess_max_rounds=max(1, int(args.postprocess_rounds)),
        postprocess_model=args.postprocess_model or provider_cfg.default_model,
        live_report=not args.no_live_report,
        live_report_refresh_seconds=args.live_report_refresh,
    )
    pipeline = PointCloudRefinementPipeline(client, config)
    result = pipeline.run(points, out_dir)

    print()
    print(f"[result] {result.message}")
    print(f"  iterations         : {result.num_iterations}")
    print(f"  final blocks       : {result.num_final_blocks}")
    print(f"  full points        : {len(result.full_point_ids)} / {len(points)}")
    print(f"  none points        : {len(result.none_point_ids)}")
    pp = result.extra.get("postprocess")
    if pp:
        print(
            f"  postprocess        : full {pp['num_full_before']} -> {pp['num_full_after']} "
            f"(-{pp['num_removed']} / +{pp['num_added']}, {pp['rounds']} 轮)"
        )
    if result.pending_point_ids:
        print(f"  pending points     : {len(result.pending_point_ids)}")
    if args.use_neo:
        neo_ok = result.extra.get("neo_used", False)
        print(
            f"  NEO 特征空间       : "
            f"{'已就绪（spectral 策略切片复用）' if neo_ok else '失败/回退到 scipy 现算'}"
        )
    if result.best_view is not None:
        bv = result.best_view
        tag = "fallback" if bv.fallback else "VLM"
        print(
            f"  best view ({tag:>8}) : #{bv.index + 1}  "
            f"elev={bv.elev:.1f}° azim={bv.azim:.1f}°"
        )
        if bv.classify_view_indices:
            nums = ", ".join(f"#{i + 1}" for i in bv.classify_view_indices)
            print(f"  classify views     : {nums}  ({len(bv.classify_view_indices)} 张)")
        if bv.reason:
            print(f"                       reason: {bv.reason[:140]}")
    if result.best_view_overview:
        print(f"  best view overview : {result.best_view_overview}")
    if result.semantic_panel:
        print(f"  semantic mask png  : {result.semantic_panel}")
    if result.tree_json:
        print(f"  block_tree.json    : {result.tree_json}")
    if result.mask_json:
        print(f"  semantic_mask.json : {result.mask_json}")
    if result.report_html:
        print(f"  html report        : {result.report_html}")
    print(f"  live report        : {out_dir / 'live_report.html'}")
    print(f"  elapsed            : {result.elapsed_seconds:.1f}s")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
