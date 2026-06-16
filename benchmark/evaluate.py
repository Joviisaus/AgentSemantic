"""Evaluate saved predictions against GT and emit an mIoU comparison report.

    conda run -n agentsematic python -m benchmark.evaluate --subset tiny --methods ours cosmo3d

Reads ``benchmark/results/<method>/<key>.npy`` and ``benchmark/data/gt/<key>.npz``;
writes ``benchmark/results/summary_<subset>.json`` and ``.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from benchmark import compat200 as C
from benchmark import metrics as M


def _eval_method(method: str, keys: List[str]) -> Dict:
    pred_dir = C.RESULTS_DIR / method
    samples: List[M.SampleScore] = []
    missing = 0
    for key in keys:
        pp = pred_dir / f"{key}.npy"
        if not pp.is_file():
            missing += 1
            continue
        pred = np.load(pp)
        gt = C.load_gt(key)
        if pred.shape[0] != gt.labels.shape[0]:
            print(f"  [warn] {method}/{key}: pred {pred.shape} vs gt {gt.labels.shape}",
                  file=sys.stderr)
            continue
        samples.append(
            M.sample_scores(pred, gt.labels, gt.part_names,
                            key=key, category=gt.category)
        )
    agg = M.aggregate(samples)
    return {
        "method": method,
        "n_subset": len(keys),
        "n_evaluated": len(samples),
        "n_missing": missing,
        "aggregate": {
            "instance_miou_skip": agg.instance_miou_skip,
            "instance_miou_empty1": agg.instance_miou_empty1,
            "category_miou_skip": agg.category_miou_skip,
            "instance_accuracy": agg.instance_accuracy,
        },
        "per_category": agg.per_category,
        "per_sample": [
            {"key": s.key, "category": s.category, "miou_skip": s.miou_skip,
             "accuracy": s.accuracy, "num_parts": s.num_parts}
            for s in samples
        ],
    }


def _render_md(subset: str, results: List[Dict]) -> str:
    lines = [
        f"# CoMPaT-200 coarse — semantic part segmentation (subset: `{subset}`)",
        "",
        "Per-point mIoU. `miou_skip` = CoSMo3D convention (empty-union parts skipped); "
        "`category_miou` = macro mean over object categories (CoSMo3D headline metric).",
        "",
        "| method | n_eval | instance mIoU | category mIoU | mIoU(empty=1) | point acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(results, key=lambda x: -x["aggregate"]["instance_miou_skip"]):
        a = r["aggregate"]
        lines.append(
            f"| {r['method']} | {r['n_evaluated']}/{r['n_subset']} | "
            f"{a['instance_miou_skip']:.4f} | {a['category_miou_skip']:.4f} | "
            f"{a['instance_miou_empty1']:.4f} | {a['instance_accuracy']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate method predictions (mIoU)")
    ap.add_argument("--subset", default="tiny")
    ap.add_argument("--methods", nargs="+",
                    default=["ours", "cosmo3d", "pointclip", "unim", "pointcept"])
    args = ap.parse_args(argv)

    keys = C.load_subset(args.subset)
    results = []
    for method in args.methods:
        if not (C.RESULTS_DIR / method).is_dir():
            continue
        r = _eval_method(method, keys)
        if r["n_evaluated"] == 0:
            print(f"[skip] {method}: no predictions found")
            continue
        results.append(r)
        a = r["aggregate"]
        print(f"[{method}] n={r['n_evaluated']}/{r['n_subset']}  "
              f"instance mIoU={a['instance_miou_skip']:.4f}  "
              f"category mIoU={a['category_miou_skip']:.4f}  "
              f"acc={a['instance_accuracy']:.4f}")

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = C.RESULTS_DIR / f"summary_{args.subset}.json"
    out_json.write_text(json.dumps({"subset": args.subset, "results": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = C.RESULTS_DIR / f"summary_{args.subset}.md"
    out_md.write_text(_render_md(args.subset, results), encoding="utf-8")
    print(f"\n[report] {out_json}\n[report] {out_md}")
    if results:
        print("\n" + _render_md(args.subset, results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
