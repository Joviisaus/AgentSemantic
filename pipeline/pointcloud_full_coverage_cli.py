"""
点云全覆盖 / 全语义分块命令行入口（mesh 版 ``pipeline.full_coverage_cli`` 的点云孪生版）。

与 ``pipeline.full_coverage_cli`` 关系，等同 ``pipeline.pointcloud_cli`` 之于 ``pipeline.cli``：
    - 输入是 mesh（会表面采样成点云）或裸点云 .npy/.ply/.pcd/.xyz；
    - 无初始 face_labels：整朵点云作为根，靠几何谱聚类逐层切分；
    - **默认可选 NEO**：整朵点云一次性前馈全局 LBO 特征空间，spectral 子块分裂切片复用；
    - 去掉 mesh 专属的凸包边界优化 / handle-loop / 基于面邻接的拓扑修复。

流程：VLM 枚举全覆盖语义清单 → 整朵点云逐层几何切分 + 每块 multi-label 命中数 →
      命中 1 个归属该语义、≥2 个继续切、0 个兜底 → 收尾按 kNN 邻居把未标注点传播到已定语义。

输出（默认 output/<模型>/pointcloud_vlm_full/）：
    enumeration.json  semantics_overview.png  panels/  responses/
    point_semantic_labels.json  full_coverage_mask.png  block_tree.json
    live_report.html  iterations/iter_NN.png

用法：
    python -m pipeline.pointcloud_full_coverage_cli testmodels/Cat.stl --provider qwen
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from vlm.client import build_vlm_client
from vlm.config import (
    DEFAULT_CONFIG_PATH,
    get_provider_config,
    load_vlm_config,
    resolve_model_name,
)


def _safe_dir_name(name: str) -> str:
    out = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
    return out or "full"


def _default_model_dir_name(input_path: Path, base: str = "output") -> str:
    """根据输入路径推断"模型名"子目录（与 pointcloud_cli 同构）。"""
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
    explicit: Optional[str],
    base: str = "output",
) -> Path:
    if explicit:
        return Path(explicit)
    return Path(base) / _default_model_dir_name(input_path, base) / "pointcloud_vlm_full"


def _parse_list_arg(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    raw = arg.strip()
    if not raw:
        return None
    for sep in ("、", "，", ";"):
        raw = raw.replace(sep, ",")
    parts = [x.strip().strip("/") for x in raw.split(",")]
    return [x for x in parts if x] or None


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
        prog="python -m pipeline.pointcloud_full_coverage_cli",
        description=(
            "点云全覆盖语义分块 CLI（mesh full_coverage_cli 的点云孪生版）："
            "VLM 枚举全覆盖语义清单 + multi-label 逐层切分，默认可选 NEO 加速。"
        ),
    )
    p.add_argument(
        "input",
        type=str,
        help="点云来源：mesh（.off/.glb/.obj/.stl… 会表面采样）或裸点云 .npy/.ply/.pcd/.xyz",
    )

    # ---- 全覆盖：语义清单 / 枚举 ----
    p.add_argument(
        "--enumerate-hint", type=str, default="",
        help="VLM 枚举阶段补充说明（宜短），引导粗粒度全覆盖分区。例如「头部是两个蘑菇的马」。",
    )
    p.add_argument(
        "--semantic-parts", type=str, default=None,
        help="直接给定全覆盖语义清单（逗号/顿号分隔），跳过 VLM 枚举。",
    )
    p.add_argument(
        "--semantic-parts-file", type=str, default=None,
        help="从 JSON 文件读语义清单（list 或含 semantic_parts 字段）。",
    )
    p.add_argument(
        "--enumerate-min-parts", type=int, default=2,
        help="枚举最少部位数（默认 2）",
    )
    p.add_argument(
        "--enumerate-max-parts", type=int, default=12,
        help="枚举最多部位数（默认 12）",
    )
    p.add_argument(
        "--enumerate-model", type=str, default=None,
        help="枚举阶段单独指定模型；不传则沿用 --model。",
    )

    # ---- 输出 ----
    p.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="输出目录。默认 <output-base>/<模型>/pointcloud_vlm_full/",
    )
    p.add_argument(
        "--output-base", type=str, default="output",
        help="输出根目录（默认 output）",
    )

    # ---- VLM provider（与 pointcloud_cli 一致）----
    p.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help="VLM 配置 YAML 路径",
    )
    p.add_argument("--provider", type=str, default=None, help="覆盖 active_provider，如 qwen")
    p.add_argument("--model", type=str, default=None, help="覆盖该 provider 的 default_model")
    p.add_argument("--env-file", type=str, default=None, help="额外 .env 文件路径")

    # ---- 点云加载 ----
    p.add_argument(
        "--num-sample-points", type=int, default=15000,
        help="输入是 mesh 时表面采样点数（裸点云输入忽略，默认 15000）",
    )
    p.add_argument("--seed", type=int, default=0, help="采样 / KMeans 随机种子")

    # ---- 主循环 / 几何切分 ----
    p.add_argument(
        "--max-depth", type=int, default=6,
        help="几何切分递归最大深度（默认 6）；达上限时 VLM 必须给唯一最终归属。",
    )
    p.add_argument(
        "--min-block-size", type=int, default=60,
        help="小于此点数的块不再切，直接归属（默认 60）",
    )
    p.add_argument(
        "--root-seed-count", type=int, default=6,
        help="无初始分块时第一刀切多少块（默认 6）",
    )
    p.add_argument(
        "--max-seeds-per-split", type=int, default=4,
        help="单次分裂最多种子数（默认 4）",
    )
    p.add_argument(
        "--split-strategy", type=str, default="auto",
        choices=["auto", "spectral", "cvt", "sdf", "skeleton", "geodesic_fps"],
        help=(
            "分块策略：auto=每个待切块由 VLM 看高亮图自选策略 + 参数（默认）；"
            "也可强制指定其一。NEO 仅在所选策略为 spectral 时介入。"
        ),
    )
    p.add_argument(
        "--no-ask-split-strategy", action="store_false", dest="ask_split_strategy",
        help="关闭 VLM 选策略，统一用 --split-strategy（auto 时退化为 spectral）。",
    )
    p.set_defaults(ask_split_strategy=True)
    p.add_argument(
        "--split-strategy-model", type=str, default=None,
        help="选策略步骤单独指定模型；不传则沿用分类模型。",
    )
    p.add_argument("--split-n-neighbors", type=int, default=16, help="kNN 图 / 法向曲率的邻居数")
    p.add_argument(
        "--no-split-enforce-connectivity", action="store_true",
        help="关闭 KMeans 后逐 label 连通清洗（默认开启，保证每块单连通）",
    )
    p.add_argument(
        "--split-curvature-weight", type=float, default=0.6, help="曲率指标权重（spectral）"
    )
    p.add_argument(
        "--split-convexity-weight", type=float, default=0.0, help="凹凸/凸包指标权重（spectral，>0 启用）"
    )
    p.add_argument(
        "--split-normal-weight", type=float, default=1.0, help="法向权重（spectral）"
    )

    # ---- NEO 后端：默认关闭（与 pointcloud_semantic_cli 一致，避免缺环境时误导）----
    p.add_argument(
        "--use-neo", action="store_true",
        help=(
            "启用 NEO：整朵点云一次性前馈全局 LBO 特征空间，spectral 子块分裂切片复用"
            "（需 .conda-neo 环境 + 权重；缺失时自动回退 scipy 现算）。"
        ),
    )
    p.add_argument(
        "--neo-ckpt", type=str, default=None,
        help="NEO 权重 .ckpt（默认：环境变量 NEO_CKPT，或 weights/ 下唯一 .ckpt）",
    )
    p.add_argument(
        "--neo-python", type=str, default=None,
        help="NEO 环境 python（默认 .conda-neo/bin/python，或环境变量 NEO_PYTHON）",
    )
    p.add_argument("--neo-k", type=int, default=0, help="NEO 取多少个模式；0=模型默认")
    p.add_argument("--neo-device", type=str, default="cpu", help="NEO 推理设备 cpu/cuda/mps")
    p.add_argument("--neo-no-mass", action="store_true", help="NEO 消融 mass-aware attention")

    # ---- 渲染 / 视角 / 并发 / 报告 ----
    p.add_argument("--image-size", type=int, default=512, help="单视角分辨率")
    p.add_argument("--panel-cols", type=int, default=4, help="拼图列数（默认 4×3=12）")
    p.add_argument("--point-size", type=float, default=None, help="点渲染半径；默认按点数自适应")
    p.add_argument(
        "--single-view-size", type=int, default=720, metavar="PX",
        help="单视角（live_report / 时间线 / overview）渲染分辨率",
    )
    p.add_argument(
        "--no-view-select", action="store_true",
        help="关闭 VLM 选最佳展示视角，直接用默认 overview 视角。",
    )
    p.add_argument(
        "--view-select-model", type=str, default=None,
        help="视角选择模型（默认 --model）",
    )
    p.add_argument("--view-select-hint", type=str, default="", help="视角选择 VLM 简短提示")
    p.add_argument(
        "--parallel-vlm", type=int, default=4, metavar="N",
        help="每轮 multi-label / 选策略并发 VLM 数（默认 4；注意各家 QPS/RPM 限额）",
    )
    p.add_argument("--no-live-report", action="store_true", help="关闭 live_report.html")
    p.add_argument(
        "--live-report-refresh", type=int, default=10, metavar="SEC",
        help="live_report 刷新间隔（秒）",
    )
    p.add_argument("--no-report", action="store_true", help="关闭最终 report.html")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--dry-run", action="store_true",
        help="只渲染整点云 overview，不调用 VLM",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("pipeline.pointcloud_full_coverage").setLevel(logging.INFO)

    from geometry.pointcloud_io import load_pointcloud
    from pipeline.pointcloud_full_coverage import (
        PointCloudFullCoverageConfig,
        PointCloudFullCoveragePipeline,
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

    # ---- 输出目录 ----
    out_dir = _resolve_output_dir(in_path, args.output_dir, args.output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        np.save(out_dir / "points.npy", points.astype(np.float32))
    except OSError as exc:
        print(f"[warn] points.npy 保存失败: {exc}", file=sys.stderr)
    print(f"[output] {out_dir.resolve()}")

    semantic_parts = (
        _parse_semantic_parts_file(args.semantic_parts_file)
        or _parse_list_arg(args.semantic_parts)
    )
    if semantic_parts:
        print(f"[semantic_parts] {len(semantic_parts)} 项: {semantic_parts}")

    # ---- dry-run：只渲染 overview ----
    if args.dry_run:
        from rendering.pointcloud_render import (
            compute_point_normals,
            render_pointcloud_overview,
        )

        try:
            normals = compute_point_normals(points)
        except Exception:  # noqa: BLE001
            normals = None
        ov = out_dir / "semantics_overview.png"
        render_pointcloud_overview(
            points, ov,
            elev=PointCloudFullCoverageConfig.display_elev,
            azim=PointCloudFullCoverageConfig.display_azim,
            image_size=args.single_view_size,
            background="white",
            point_size=args.point_size,
            normals=normals,
        )
        print(f"[dry-run] overview: {ov}")
        return 0

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
    vlm_model = resolve_model_name(args.model, provider_cfg)
    print(
        f"[vlm] provider={provider_cfg.name} model={vlm_model} "
        f"base_url={provider_cfg.base_url or '<n/a>'}"
    )
    client = build_vlm_client(provider_cfg)

    if args.use_neo:
        print("[neo]   已启用（仅 spectral 策略介入；缺环境/权重时自动回退 scipy 现算）")
    print(f"[split] strategy={args.split_strategy} ask_vlm={args.ask_split_strategy}")
    if not args.no_live_report:
        print(
            f"[live]   {(out_dir / 'live_report.html').resolve()} "
            f"(每 {args.live_report_refresh}s 刷新)"
        )
    print(
        "[hint] 首次运行可能先建 matplotlib 字体缓存；之后每轮含 12 视角渲染 + 多次 VLM 调用，"
        "整流程通常 5–15 分钟。可用 --verbose 看更详细日志。"
    )

    def progress(event: str, payload: Dict[str, Any]) -> None:
        if event == "classified":
            matched = payload.get("matched", []) or []
            print(f"  [classify] {payload['block_id']} -> matched={matched}")
        elif event == "split_start":
            print(
                f"  [split…]   {payload['block_id']} "
                f"pts={payload.get('faces')} k={payload.get('seed_count')} "
                f"matched={payload.get('matched')}"
            )
        elif event == "split":
            strat = payload.get("strategy") or ""
            strat_tag = f" strategy={strat}" if strat else ""
            print(
                f"  [split]    {payload['block_id']} "
                f"k={payload.get('seed_count')} "
                f"children={len(payload.get('children', []))} "
                f"matched={payload.get('matched', [])}{strat_tag}"
            )
        elif event == "split_failed":
            print(f"  [splfail]  {payload['block_id']} → assigned={payload.get('dominant')}")
        elif event == "max_depth":
            print(f"  [maxdep]   {payload['block_id']} → assigned={payload.get('dominant')}")
        elif event == "iteration_end":
            print(
                f"  [iter]     round={payload.get('iteration')} "
                f"queue_after={payload.get('queue_after')}"
            )
        elif event == "main_loop_done":
            print(
                f"  [done]     main loop finished after {payload.get('iterations')} "
                f"iterations ({payload.get('num_blocks')} blocks) → finalizing"
            )

    config = PointCloudFullCoverageConfig(
        enumerate_hint=args.enumerate_hint,
        semantic_parts=semantic_parts,
        enumerate_min_parts=args.enumerate_min_parts,
        enumerate_max_parts=args.enumerate_max_parts,
        max_depth=args.max_depth,
        min_block_size=args.min_block_size,
        root_seed_count=args.root_seed_count,
        max_seeds_per_split=args.max_seeds_per_split,
        image_size=args.image_size,
        panel_cols=args.panel_cols,
        point_size=args.point_size,
        classification_model=vlm_model,
        enumerate_model=resolve_model_name(args.enumerate_model or args.model, provider_cfg),
        ask_split_strategy=args.ask_split_strategy,
        split_strategy=args.split_strategy,
        split_strategy_model=resolve_model_name(
            args.split_strategy_model or args.model, provider_cfg
        ),
        split_n_neighbors=args.split_n_neighbors,
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
        parallel_vlm_workers=max(1, int(args.parallel_vlm)),
        live_report=not args.no_live_report,
        live_report_refresh_seconds=args.live_report_refresh,
        select_best_view=not args.no_view_select,
        view_select_model=resolve_model_name(args.view_select_model or args.model, provider_cfg),
        view_select_semantic_hint=args.view_select_hint,
        single_view_image_size=args.single_view_size,
        generate_report=not args.no_report,
    )
    pipeline = PointCloudFullCoveragePipeline(client, config)
    result = pipeline.run(points, out_dir, progress_cb=progress)

    print()
    print(f"[result] {result.message}")
    print(f"  iterations         : {result.num_iterations}")
    print(f"  initial / final    : {result.num_initial_blocks} / {result.num_final_blocks}")
    print(f"  semantic parts     : {len(result.semantic_parts)}")
    for name in result.semantic_parts:
        cnt = len(result.semantic_point_groups.get(name, []))
        print(f"    - {name:<20s} {cnt} pts")
    if result.pending_point_ids:
        print(f"  pending points     : {len(result.pending_point_ids)}")
    if args.use_neo:
        neo_ok = result.extra.get("neo_used", False)
        print(
            f"  NEO 特征空间       : "
            f"{'已就绪（spectral 切片复用）' if neo_ok else '失败/回退 scipy 现算'}"
        )
    if result.best_view is not None:
        bv = result.best_view
        tag = "fallback" if bv.fallback else "VLM"
        print(f"  best view ({tag:>8}) : #{bv.index + 1}  elev={bv.elev:.1f}° azim={bv.azim:.1f}°")
    if result.coverage_panel:
        print(f"  full coverage png  : {result.coverage_panel}")
    if result.labels_json:
        print(f"  point_semantic_labels.json : {result.labels_json}")
    if result.tree_json:
        print(f"  block_tree.json    : {result.tree_json}")
    if result.report_html:
        print(f"  report.html        : {result.report_html}")
    print(f"  live report        : {out_dir / 'live_report.html'}")
    print(f"  elapsed            : {result.elapsed_seconds:.1f}s")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
