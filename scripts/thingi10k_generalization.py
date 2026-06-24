"""Utilities for a small Thingi10K point-cloud generalization probe.

This script is intentionally lightweight: it samples random STL meshes from a
Thingiverse/Thingi10K raw_meshes directory, renders multi-view point-cloud
previews, and stores the sampled points so downstream CLIs can reuse identical
inputs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from geometry.pointcloud_io import load_pointcloud
from rendering.pointcloud_render import (
    compute_point_normals,
    render_pointcloud_segmentation_panel,
    render_pointcloud_thumbnails_panel,
)


def render_candidates(args: argparse.Namespace) -> int:
    root = Path(args.raw_meshes)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("*.stl"))
    if args.max_file_mb > 0:
        max_bytes = int(args.max_file_mb * 1024 * 1024)
        files = [p for p in files if p.stat().st_size <= max_bytes]
    if not files:
        raise FileNotFoundError(f"no .stl files found in {root}")

    rng = random.Random(args.seed)
    candidates = rng.sample(files, min(args.count, len(files)))
    rows = []
    for mesh_path in candidates:
        try:
            points, source = load_pointcloud(
                mesh_path,
                num_sample_points=args.num_sample_points,
                seed=args.sample_seed,
            )
            normals = compute_point_normals(points)
            preview_path = out_dir / f"{mesh_path.stem}.png"
            points_path = out_dir / f"{mesh_path.stem}.points.npy"
            render_pointcloud_thumbnails_panel(
                points,
                preview_path,
                thumb_size=args.thumb_size,
                cols=4,
                title=f"Thingi10K {mesh_path.stem}",
                normals=normals,
            )
            np.save(points_path, points.astype(np.float32))
            row = {
                "id": mesh_path.stem,
                "mesh": str(mesh_path),
                "preview": str(preview_path),
                "points": str(points_path),
                "n_points": int(len(points)),
                "source": source,
            }
            rows.append(row)
            print(f"[ok] {mesh_path.name} -> {preview_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {mesh_path.name}: {exc}")

    manifest = out_dir / "candidates.json"
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] rendered {len(rows)} candidates -> {manifest}")
    return 0


def _load_mesh(path: Path):
    import trimesh

    obj = trimesh.load(str(path), file_type=path.suffix.lower().lstrip("."), process=False)
    if isinstance(obj, trimesh.Scene):
        meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"no mesh geometry in {path}")
        obj = trimesh.util.concatenate(meshes)
    if not isinstance(obj, trimesh.Trimesh) or len(obj.faces) == 0:
        raise ValueError(f"not a triangle mesh: {path}")
    return obj


def _face_components(mesh) -> np.ndarray:
    import trimesh

    n_faces = len(mesh.faces)
    comps = trimesh.graph.connected_components(
        mesh.face_adjacency,
        nodes=np.arange(n_faces),
        min_len=1,
    )
    face_labels = np.full(n_faces, -1, dtype=np.int64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    order = sorted(
        range(len(comps)),
        key=lambda i: float(np.sum(areas[np.asarray(comps[i], dtype=np.int64)])),
        reverse=True,
    )
    for out_label, comp_idx in enumerate(order):
        face_labels[np.asarray(comps[comp_idx], dtype=np.int64)] = out_label
    return face_labels


def render_component_candidates(args: argparse.Namespace) -> int:
    """Render meshes whose natural connected components can serve as pseudo GT."""
    root = Path(args.raw_meshes)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("*.stl"))
    if args.max_file_mb > 0:
        max_bytes = int(args.max_file_mb * 1024 * 1024)
        files = [p for p in files if p.stat().st_size <= max_bytes]
    if not files:
        raise FileNotFoundError(f"no .stl files found in {root}")

    rng = random.Random(args.seed)
    shuffled = files[:]
    rng.shuffle(shuffled)
    rows = []
    scanned = 0
    for mesh_path in shuffled:
        scanned += 1
        if args.max_scan and scanned > args.max_scan:
            break
        if len(rows) >= args.count:
            break
        try:
            mesh = _load_mesh(mesh_path)
            face_comp = _face_components(mesh)
            n_comp = int(face_comp.max() + 1)
            if n_comp < args.min_components or n_comp > args.max_components:
                continue
            areas = np.bincount(face_comp, weights=np.asarray(mesh.area_faces), minlength=n_comp)
            ratios = areas / max(float(areas.sum()), 1e-12)
            # Avoid dust-only extras and strongly fragmented models.
            meaningful = np.flatnonzero(ratios >= args.min_component_ratio)
            if len(meaningful) < args.min_components or len(meaningful) > args.max_components:
                continue

            if args.sample_seed is not None:
                np.random.seed(int(args.sample_seed))
            points, sampled_faces = __import__("trimesh").sample.sample_surface(
                mesh,
                int(args.num_sample_points),
            )
            labels = face_comp[np.asarray(sampled_faces, dtype=np.int64)]
            normals = compute_point_normals(points)

            stem = mesh_path.stem
            points_path = out_dir / f"{stem}.points.npy"
            labels_path = out_dir / f"{stem}.component_labels.npy"
            preview_path = out_dir / f"{stem}.components.png"
            np.save(points_path, np.asarray(points, dtype=np.float32))
            np.save(labels_path, labels.astype(np.int64))
            render_pointcloud_segmentation_panel(
                points,
                labels,
                preview_path,
                title=f"Thingi10K {stem} connected components",
                image_size=(args.thumb_size, args.thumb_size),
                cols=4,
                normals=normals,
            )
            row = {
                "id": stem,
                "mesh": str(mesh_path),
                "preview": str(preview_path),
                "points": str(points_path),
                "component_labels": str(labels_path),
                "n_points": int(len(points)),
                "num_components": n_comp,
                "component_area_ratios": [float(x) for x in ratios[: args.max_components]],
                "meaningful_components": [int(x) for x in meaningful.tolist()],
            }
            rows.append(row)
            print(
                f"[ok] {mesh_path.name}: comps={n_comp} "
                f"ratios={[round(float(x), 3) for x in ratios[:args.max_components]]}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            if args.verbose:
                print(f"[skip] {mesh_path.name}: {exc}", flush=True)

    manifest = out_dir / "component_candidates.json"
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] rendered {len(rows)} component candidates -> {manifest}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-candidates",
        action="store_true",
        help="screen by natural mesh connected components instead of plain random previews",
    )
    parser.add_argument(
        "--raw-meshes",
        default=r"E:\data\Thingi10K\Thingi10K\raw_meshes",
        help="Thingi10K raw_meshes directory",
    )
    parser.add_argument(
        "--output-dir",
        default="output/thingi10k_generalization/candidates",
        help="Directory for preview PNGs, sampled point clouds, and candidates.json",
    )
    parser.add_argument("--count", type=int, default=24, help="number of random meshes")
    parser.add_argument("--seed", type=int, default=20260623, help="mesh sampling seed")
    parser.add_argument("--sample-seed", type=int, default=7, help="surface sampling seed")
    parser.add_argument("--num-sample-points", type=int, default=6000)
    parser.add_argument("--thumb-size", type=int, default=220)
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=5)
    parser.add_argument("--min-component-ratio", type=float, default=0.02)
    parser.add_argument("--max-scan", type=int, default=1000)
    parser.add_argument("--max-file-mb", type=float, default=0.0)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.component_candidates:
        return render_component_candidates(args)
    return render_candidates(args)


if __name__ == "__main__":
    raise SystemExit(main())
