"""Test GT-visual -> VLM semantic on HuggingFace CoMPaT/3DCoMPaT200 2D rows.

Example::

    python -m benchmark.hf_compat200.run_test --offset 0 --length 100 --limit 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

AGENTSEMATIC_ROOT = Path(__file__).resolve().parents[2]
if str(AGENTSEMATIC_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTSEMATIC_ROOT))

from benchmark.hf_compat200.gt_semantic import describe_gt_part_2d  # noqa: E402
from benchmark.hf_compat200.load import (  # noqa: E402
    category_for_model,
    dominant_category_part,
    fetch_rows,
    group_rows_by_shape,
    load_image_field,
    load_vocab,
)
from benchmark.methods.ours import build_client, _humanize  # noqa: E402
from vlm.config import DEFAULT_CONFIG_PATH  # noqa: E402

LOGGER = logging.getLogger("benchmark.hf_compat200.run_test")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="HF 3DCoMPaT200 GT-semantic smoke test")
    ap.add_argument("--split", default="train", choices=["train", "validation", "test"])
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--limit", type=int, default=3, help="how many unique shapes to test")
    ap.add_argument("--out", type=Path,
                    default=Path("benchmark/results/hf_compat200_test"))
    ap.add_argument("--provider", default="qwen")
    ap.add_argument("--model", default=None)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args(argv)

    vocab = load_vocab()
    print(f"[hf] fetching split={args.split} offset={args.offset} length={args.length} ...",
          flush=True)
    rows = fetch_rows(args.split, args.offset, args.length)
    if not rows:
        print("[hf] no rows returned", file=sys.stderr)
        return 1

    groups = group_rows_by_shape(rows)
    print(f"[hf] {len(rows)} rows -> {len(groups)} shape/style groups", flush=True)

    client, _ = build_client(args.config, args.provider, args.model, None)
    results = []
    tested = 0

    for group_key, group_rows in sorted(groups.items()):
        if tested >= args.limit:
            break
        model_id = group_rows[0]["model_id.txt"].strip()
        cat_id, cat_name = category_for_model(model_id, vocab)
        target_cls = int(group_rows[0]["target.cls"])
        if target_cls != cat_id:
            LOGGER.warning("%s: target.cls=%d != category_id=%d", model_id, target_cls, cat_id)

        # pick GT part from first view mask
        mask0 = load_image_field(group_rows[0]["mask.png"])
        import numpy as np
        stored, gt_name, px = dominant_category_part(np.array(mask0), model_id, vocab)
        if stored <= 0 or px <= 0:
            print(f"[skip] {group_key}: no in-category part pixels", flush=True)
            continue

        out_dir = args.out / group_key / f"cls{stored}"
        semantic, panel, raw = describe_gt_part_2d(
            client, group_rows, stored, out_dir,
            object_category=_humanize(cat_name), model=args.model,
        )
        rec = {
            "group_key": group_key,
            "model_id": model_id,
            "category": cat_name,
            "target_stored": stored,
            "gt_part_name": gt_name,
            "gt_pixels_view0": px,
            "n_views": len(group_rows),
            "vlm_semantic": semantic,
            "vlm_raw": raw,
            "panel": str(panel) if panel else "",
            "match": semantic.lower() in gt_name.lower() or gt_name.lower() in semantic.lower(),
        }
        results.append(rec)
        tested += 1
        print(
            f"\n=== [{tested}/{args.limit}] {model_id} ({cat_name}) "
            f"GT part={gt_name!r} ({px}px) ===",
            flush=True,
        )
        print(f"  panel: {panel}", flush=True)
        print(f"  VLM semantic: {semantic!r}", flush=True)
        print(f"  raw: {raw[:120]!r}", flush=True)

    summary_path = args.out / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    n_match = sum(1 for r in results if r.get("match"))
    print(f"\n[hf] tested {len(results)} shapes, name overlap={n_match}/{len(results)}")
    print(f"[hf] summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
