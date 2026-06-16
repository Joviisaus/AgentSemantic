"""One-time GT export: CoSMo3D ``.pt`` tensors -> numpy ``.npz``.

Run under the **cosmo3d** conda env (needs torch)::

    conda run -n cosmo3d python -m benchmark.export_gt

Produces ``benchmark/data/manifest.json`` and one ``benchmark/data/gt/<key>.npz``
per instance, each with arrays: points (N,3), labels (N,), part_names, etc.
Downstream code (ours adapter, evaluator) only needs numpy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from benchmark import compat200 as C


def _derive_labels(mask2points: np.ndarray, n_points: int) -> np.ndarray:
    """mask2points -> per-point 0-based part id (-1 if not exactly one active)."""
    m = np.asarray(mask2points)
    # orient to (num_parts, n_points)
    if m.shape[0] == n_points and m.shape[1] != n_points:
        m = m.T
    elif m.shape[1] != n_points and m.shape[0] != n_points:
        raise ValueError(f"mask2points shape {m.shape} incompatible with {n_points} points")
    m = (m != 0).astype(np.int32)            # (P, N)
    active = m.sum(axis=0)                    # per point
    labels = np.full(n_points, C.IGNORE_LABEL, dtype=np.int64)
    valid = active == 1
    if valid.any():
        labels[valid] = np.argmax(m[:, valid], axis=0)
    return labels


def export_one(meta: C.InstanceMeta) -> bool:
    gt_dir = Path(meta.gt_dir)
    try:
        points = torch.load(gt_dir / "points.pt").cpu().numpy().astype(np.float32)
        m2p = torch.load(gt_dir / "mask2points.pt").cpu().numpy()
    except Exception as exc:  # noqa: BLE001
        print(f"  [skip] {meta.key}: load failed ({exc})")
        return False
    n = points.shape[0]
    labels = _derive_labels(m2p, n)
    # sanity: GT part count should be >= max label index + 1
    out = C.gt_path(meta.key)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        points=points,
        labels=labels,
        part_names=np.array(meta.part_names, dtype=object),
        category=meta.category,
        instance=meta.instance,
        proxy_ply=meta.proxy_ply,
    )
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export CoMPaT-200 coarse GT to npz")
    ap.add_argument("--limit", type=int, default=0, help="only export first N (debug)")
    ap.add_argument("--build-subsets", action="store_true",
                    help="also write tiny/subset/full key lists")
    ap.add_argument("--subset", default=None,
                    help="only export GT for this subset's keys (default: all)")
    args = ap.parse_args(argv)

    print("[discover] scanning bundle proxies + CoSMo3D GT dirs ...")
    instances = C.discover_instances()
    print(f"[discover] {len(instances)} instances with both mesh proxy and GT")
    C.write_manifest(instances)
    print(f"[manifest] -> {C.MANIFEST_PATH}")

    if args.build_subsets:
        for name in ("tiny", "subset", "full"):
            keys = C.build_subset(name, instances)
            C.save_subset(name, keys)
            print(f"[subset] {name}: {len(keys)} keys -> {C.SUBSET_DIR / (name + '.txt')}")

    if args.subset:
        wanted = set(C.load_subset(args.subset))
        todo = [m for m in instances if m.key in wanted]
        print(f"[export] limiting GT export to subset '{args.subset}': {len(todo)} instances")
    else:
        todo = instances
    if args.limit:
        todo = todo[: args.limit]
    ok = 0
    for i, meta in enumerate(todo):
        if export_one(meta):
            ok += 1
        if (i + 1) % 100 == 0:
            print(f"  exported {i + 1}/{len(todo)} ...")
    print(f"[export] {ok}/{len(todo)} npz written -> {C.GT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
