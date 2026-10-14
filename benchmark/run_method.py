"""Run one method over a subset and write per-point predictions.

Each method writes ``benchmark/results/<method>/<key>.npy`` (per-point 0-based
part ids aligned to GT point order; -1 = unlabeled).

Currently implemented: ``ours`` / ``ours_pointcloud`` (run under the agentsematic env).
Baselines (cosmo3d / pointclip / unim / pointcept) are dispatched to their own
adapter modules.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from benchmark.dataset_registry import get_dataset


def _run_ours(args, keys, by_key, C) -> int:
    from benchmark.methods import ours
    from vlm.config import DEFAULT_CONFIG_PATH

    config_path = args.config or str(DEFAULT_CONFIG_PATH)
    client, provider_cfg = ours.build_client(
        config_path, args.provider, args.model, args.env_file
    )
    work_dir = C.RESULTS_DIR / "ours" / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for i, key in enumerate(keys):
        meta = by_key[key]
        pred = C.RESULTS_DIR / "ours" / f"{key}.npy"
        if pred.is_file() and not args.overwrite:
            print(f"[{i+1}/{len(keys)}] skip {key} (exists)")
            continue
        print(f"[{i+1}/{len(keys)}] {key} ({meta.category}, {len(meta.part_names)} parts)")
        summary = ours.run_instance(
            meta, client, work_dir,
            prompt_template=args.prompt_template,
            max_depth=args.max_depth,
            max_seeds=args.max_seeds,
            min_block_size=args.min_block_size,
            model=args.model,
            select_best_view=not args.no_view_select,
            parallel_vlm=args.parallel_vlm,
        )
        print(f"    done in {summary['elapsed_seconds']:.1f}s "
              f"(parts: {[m['num_full_faces'] for m in summary['per_part']]})")
    print(f"[ours] subset complete in {time.perf_counter() - t0:.1f}s")
    return 0


def _run_ours_pointcloud(args, keys, by_key, C) -> int:
    from benchmark.methods import ours_pointcloud
    from vlm.config import DEFAULT_CONFIG_PATH

    config_path = args.config or str(DEFAULT_CONFIG_PATH)
    client, provider_cfg = ours_pointcloud.build_client(
        config_path, args.provider, args.model, args.env_file
    )
    # oracle 重跑写入独立子目录，skip 检查也必须看同一个子目录，
    # 否则会误判 token 版 ours_pointcloud 的 pred 已存在而全部跳过。
    subdir = os.environ.get("AGENTSEMANTIC_RESULTS_SUBDIR") or "ours_pointcloud"
    work_dir = C.RESULTS_DIR / subdir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for i, key in enumerate(keys):
        meta = by_key[key]
        pred = C.RESULTS_DIR / subdir / f"{key}.npy"
        if pred.is_file() and not args.overwrite:
            print(f"[{i+1}/{len(keys)}] skip {key} (exists)")
            continue
        print(f"[{i+1}/{len(keys)}] {key} ({meta.category}, {len(meta.part_names)} parts)")
        summary = ours_pointcloud.run_instance(
            meta, client, work_dir,
            dataset_module=C,
            prompt_template=args.prompt_template,
            max_depth=args.max_depth,
            max_seeds=args.max_seeds,
            min_block_size=args.min_block_size,
            model=args.model,
            select_best_view=not args.no_view_select,
            parallel_vlm=args.parallel_vlm,
            use_neo=args.use_neo,
            neo_ckpt=args.neo_ckpt,
            neo_python=args.neo_python,
            neo_k=args.neo_k,
            neo_device=args.neo_device,
            neo_no_mass=args.neo_no_mass,
            neo_densify_target=args.neo_densify_target,
        )
        print(f"    done in {summary['elapsed_seconds']:.1f}s "
              f"(parts: {[m['num_full_points'] for m in summary['per_part']]})")
    print(f"[ours_pointcloud] subset complete in {time.perf_counter() - t0:.1f}s")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a segmentation method over a benchmark subset")
    ap.add_argument("--dataset", default="compat200",
                    help="dataset name: compat200 (default, CoSMo3D comparison) or shapenetpart")
    ap.add_argument("--method", required=True,
                    choices=("ours", "ours_pointcloud", "ours_pointcloud_oracle", "ours_pointcloud_fair", "cosmo3d", "pointclip", "unim", "pointcept"))
    ap.add_argument("--subset", default="tiny", help="subset name (tiny|subset|full)")
    ap.add_argument("--limit", type=int, default=0, help="only process first N keys (debug)")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing preds")
    ap.add_argument("--verbose", action="store_true")

    # ours / VLM options
    ap.add_argument("--config", default=None, help="VLM config yaml (default: vlm.config.DEFAULT_CONFIG_PATH)")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--env-file", default=None)
    ap.add_argument("--prompt-template", default="{part}",
                    help="semantic text template, placeholders {part} {category}")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--max-seeds", type=int, default=4)
    ap.add_argument("--min-block-size", type=int, default=20)
    ap.add_argument("--no-view-select", action="store_true")
    ap.add_argument("--parallel-vlm", type=int, default=1)

    # ours_pointcloud：稀疏 benchmark 默认 NEO + mesh 辅助增强 + spectral
    ap.add_argument(
        "--no-neo", action="store_false", dest="use_neo",
        help="关闭 NEO（spectral 回退 scipy 现算；默认启用 NEO + proxy mesh 稀疏增强）",
    )
    ap.set_defaults(use_neo=True)
    ap.add_argument("--neo-ckpt", type=str, default=None, help="NEO 权重 .ckpt")
    ap.add_argument("--neo-python", type=str, default=None, help="NEO 环境 python")
    ap.add_argument("--neo-k", type=int, default=0, help="NEO 模式数；0=模型默认")
    ap.add_argument("--neo-device", type=str, default="cpu", help="NEO 设备 cpu/cuda/mps")
    ap.add_argument("--neo-no-mass", action="store_true", help="NEO 消融 mass-aware attention")
    ap.add_argument(
        "--neo-densify-target", type=int, default=20000,
        help="稀疏点云 mesh 辅助增强时 NEO 输入目标总点数",
    )

    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    try:
        C = get_dataset(args.dataset)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    dataset_name = args.dataset.strip().lower().replace("-", "_")
    if dataset_name not in {"compat200", "compat_200", "3dcompat200", "3d_compat200"}:
        if args.method not in {"ours_pointcloud", "ours_pointcloud_oracle", "ours_pointcloud_fair", "pointclip"}:
            print(
                f"[error] method={args.method} is only wired for compat200; "
                "supplementary point-cloud datasets currently support ours_pointcloud and pointclip.",
                file=sys.stderr,
            )
            return 2

    keys = C.load_subset(args.subset)
    by_key = C.manifest_by_key()
    missing = [k for k in keys if k not in by_key]
    if missing:
        print(f"[warn] {len(missing)} subset keys missing from manifest", file=sys.stderr)
        keys = [k for k in keys if k in by_key]
    if args.limit:
        keys = keys[: args.limit]
    print(f"[run] dataset={args.dataset} method={args.method} subset={args.subset} n={len(keys)}")

    if args.method == "ours":
        return _run_ours(args, keys, by_key, C)
    if args.method == "ours_pointcloud":
        return _run_ours_pointcloud(args, keys, by_key, C)
    if args.method == "ours_pointcloud_oracle":
        # 人工 oracle 重跑：写入独立结果子目录，便于与 token 版 ours_pointcloud 对比
        os.environ["AGENTSEMANTIC_RESULTS_SUBDIR"] = "ours_pointcloud_oracle"
        return _run_ours_pointcloud(args, keys, by_key, C)
    if args.method == "ours_pointcloud_fair":
        # 公平版：GT 仅提供部件名（语义），分类由助手肉眼看渲染面板判定，绝不读 GT 掩码/高亮图。
        # 几何分裂由块大小驱动（vlm_oracle 的 FAIR 分支），与 GT 无关。
        os.environ["AGENTSEMANTIC_VLM_ORACLE"] = "1"
        os.environ["AGENTSEMANTIC_VLM_FAIR"] = "1"
        os.environ["AGENTSEMANTIC_RESULTS_SUBDIR"] = "ours_pointcloud_fair"
        return _run_ours_pointcloud(args, keys, by_key, C)
    if args.method == "cosmo3d":
        from benchmark.methods import cosmo3d
        return cosmo3d.run_subset(keys, by_key, overwrite=args.overwrite)
    if args.method == "pointclip":
        from benchmark.methods import pointclip
        return pointclip.run_subset(keys, by_key, dataset_module=C, overwrite=args.overwrite)
    if args.method == "unim":
        from benchmark.methods import unim
        return unim.run_subset(keys, by_key, overwrite=args.overwrite)
    if args.method == "pointcept":
        from benchmark.methods import pointcept
        return pointcept.run_subset(keys, by_key, overwrite=args.overwrite)
    return 2


if __name__ == "__main__":
    sys.exit(main())
