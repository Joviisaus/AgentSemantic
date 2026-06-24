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
    work_dir = C.RESULTS_DIR / "ours_pointcloud" / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for i, key in enumerate(keys):
        meta = by_key[key]
        pred = C.RESULTS_DIR / "ours_pointcloud" / f"{key}.npy"
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
                    choices=("ours", "ours_pointcloud", "cosmo3d", "pointclip", "unim", "pointcept"))
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
        if args.method != "ours_pointcloud":
            print(
                f"[error] method={args.method} is only wired for compat200; "
                "supplementary point-cloud datasets currently support ours_pointcloud only.",
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
    if args.method == "cosmo3d":
        from benchmark.methods import cosmo3d
        return cosmo3d.run_subset(keys, by_key, overwrite=args.overwrite)
    if args.method == "pointclip":
        from benchmark.methods import pointclip
        return pointclip.run_subset(keys, by_key, overwrite=args.overwrite)
    if args.method == "unim":
        from benchmark.methods import unim
        return unim.run_subset(keys, by_key, overwrite=args.overwrite)
    if args.method == "pointcept":
        from benchmark.methods import pointcept
        return pointcept.run_subset(keys, by_key, overwrite=args.overwrite)
    return 2


if __name__ == "__main__":
    sys.exit(main())
