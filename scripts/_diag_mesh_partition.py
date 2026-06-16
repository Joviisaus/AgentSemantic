"""Diagnose mesh size / CC count / partition timing for STL inputs."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry.partition.connected_components import split_block_by_connected_components
from geometry.partition.geodesic_fps import split_block_by_geodesic_fps
from geometry.partition.graph_cut import split_block_by_graph_cut
from geometry.partition.sdf import split_block_by_sdf
from geometry.partition.spectral import split_block_by_spectral_seeds

MAX_SPECTRAL_FACES = 80_000


def load_mesh(path: Path):
    m = trimesh.load(str(path))
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(
            [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def bench(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out


def analyze(path: Path) -> None:
    if not path.is_file():
        print(f"MISSING {path}")
        return

    t_load = time.perf_counter()
    v, f = load_mesh(path)
    load_s = time.perf_counter() - t_load

    cc_groups = split_block_by_connected_components(
        v, f, list(range(len(f))), min_component_faces=1
    )
    n_cc = len(cc_groups) if cc_groups else 1
    print(f"\n=== {path.name} ===")
    print(f"  load={load_s:.1f}s  V={len(v):,}  F={len(f):,}  CC={n_cc}")

    if cc_groups:
        block = sorted(cc_groups, key=len, reverse=True)[0]
        cc_sizes = sorted((len(g) for g in cc_groups), reverse=True)
        print(f"  CC face sizes (top 5): {cc_sizes[:5]}")
    else:
        block = list(range(len(f)))

    print(f"  benchmark block: {len(block):,} faces")

    elapsed, split = bench(
        split_block_by_geodesic_fps, v, f, block, 4, lloyd_iter=4
    )
    print(f"  geodesic_fps   k=4: {elapsed:7.2f}s  clusters={split.num_clusters}")

    if len(block) <= MAX_SPECTRAL_FACES:
        for name, fn, kwargs in (
            ("spectral", split_block_by_spectral_seeds, {}),
            ("graph_cut", split_block_by_graph_cut, {}),
        ):
            try:
                elapsed, split = bench(fn, v, f, block, 4, **kwargs)
                print(
                    f"  {name:14s} k=4: {elapsed:7.2f}s  clusters={split.num_clusters}"
                )
            except Exception as exc:
                print(f"  {name:14s} k=4: FAILED {exc}")
        try:
            elapsed, split = bench(
                split_block_by_sdf, v, f, block, 4, num_rays=8
            )
            print(
                f"  {'sdf':14s} k=4: {elapsed:7.2f}s  clusters={split.num_clusters}"
            )
        except Exception as exc:
            print(f"  {'sdf':14s} k=4: FAILED {exc}")
    else:
        print(
            f"  spectral/graph_cut/sdf: skipped (block>{MAX_SPECTRAL_FACES:,} faces; "
            "spectral ARPACK would take minutes+)"
        )


if __name__ == "__main__":
    for rel in (
        r"C:\Users\jirenouyang\Downloads\Troll_10.stl",
        r"C:\Users\jirenouyang\Downloads\gecko_even_better2.STL",
    ):
        analyze(Path(rel))
