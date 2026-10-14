"""Prepare and finalize multi-connected animated-asset segmentation tests.

This helper keeps the existing segmentation CLIs unchanged:
- simple meshes are sent to pipeline.full_coverage_cli;
- complex meshes are sampled once to points.npy, then sent to
  pipeline.pointcloud_full_coverage_cli;
- point labels are mapped back to mesh faces by per-face majority vote, with
  same-connected-component nearest-face propagation for unsampled faces.

Animated GLB assets are evaluated at their default static pose. Scene-node
transforms are baked into a PLY snapshot before connectivity analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ASSET_NAMES = ("RiggedFigure", "CesiumMan", "CesiumMilkTruck")
PALETTE = np.asarray(
    [
        (230, 57, 70, 255),
        (41, 128, 185, 255),
        (243, 156, 18, 255),
        (39, 174, 96, 255),
        (142, 68, 173, 255),
        (22, 160, 133, 255),
        (211, 84, 0, 255),
        (127, 140, 141, 255),
        (241, 196, 15, 255),
        (52, 73, 94, 255),
        (231, 76, 60, 255),
        (46, 204, 113, 255),
    ],
    dtype=np.uint8,
)


def _load_static_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene")
    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"Expected Scene for {path}, got {type(loaded)}")
    mesh = loaded.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Could not flatten scene {path}, got {type(mesh)}")
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def _face_components(mesh: trimesh.Trimesh) -> tuple[int, np.ndarray]:
    n_faces = len(mesh.faces)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency):
        rows = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
        cols = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(n_faces, n_faces),
        ).tocsr()
    else:
        graph = coo_matrix((n_faces, n_faces), dtype=np.uint8).tocsr()
    return connected_components(graph, directed=False)


def _complexity(mesh: trimesh.Trimesh) -> dict[str, Any]:
    count, labels = _face_components(mesh)
    area_by_component = np.bincount(
        labels,
        weights=np.asarray(mesh.area_faces, dtype=np.float64),
        minlength=count,
    )
    area_by_component = np.sort(area_by_component)[::-1]
    total_area = float(area_by_component.sum()) or 1.0
    ratios = area_by_component / total_area
    significant = int(np.count_nonzero(ratios >= 0.02))
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(count),
        "significant_components_ge_2pct": significant,
        "largest_component_area_ratio": round(float(ratios[0]), 6) if len(ratios) else 0.0,
        "top_component_area_ratios": [round(float(x), 6) for x in ratios[:20]],
    }


def _render_overview(mesh: trimesh.Trimesh, output: Path, title: str) -> None:
    from rendering.render_component import render_overview_only_panel

    render_overview_only_panel(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        output,
        title=title,
        image_size=480,
        cols=4,
    )


def prepare(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "policy": {
            "simple_if_components_lte": int(args.max_total_components),
            "simple_if_significant_components_lte": int(args.max_significant_components),
            "significant_area_ratio": 0.02,
            "complex_route": "pointcloud_full_coverage_cli",
            "simple_route": "full_coverage_cli",
            "sample_points": int(args.sample_points),
            "seed": int(args.seed),
        },
        "assets": {},
    }

    for name in ASSET_NAMES:
        source = source_dir / f"{name}.glb"
        if not source.is_file():
            raise FileNotFoundError(source)
        asset_dir = output_root / name
        asset_dir.mkdir(parents=True, exist_ok=True)
        mesh = _load_static_mesh(source)
        static_mesh = asset_dir / f"{name}_default_pose.ply"
        mesh.export(static_mesh)
        overview = asset_dir / "source_overview.png"
        _render_overview(mesh, overview, f"{name} default pose")

        stats = _complexity(mesh)
        is_simple = (
            stats["components"] <= int(args.max_total_components)
            and stats["significant_components_ge_2pct"]
            <= int(args.max_significant_components)
        )
        route = "mesh" if is_simple else "pointcloud"
        entry: dict[str, Any] = {
            "name": name,
            "source_glb": str(source),
            "static_mesh": str(static_mesh),
            "source_overview": str(overview),
            "route": route,
            "complexity": stats,
        }
        if route == "pointcloud":
            np.random.seed(int(args.seed))
            points, face_idx = trimesh.sample.sample_surface(mesh, int(args.sample_points))
            points_path = asset_dir / "points.npy"
            face_idx_path = asset_dir / "sample_face_idx.npy"
            np.save(points_path, np.asarray(points, dtype=np.float32))
            np.save(face_idx_path, np.asarray(face_idx, dtype=np.int64))
            entry.update(
                {
                    "points": str(points_path),
                    "sample_face_idx": str(face_idx_path),
                    "num_sample_points": int(len(points)),
                }
            )
        manifest["assets"][name] = entry

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(manifest_path)


def _fill_unlabeled_faces(
    mesh: trimesh.Trimesh, labels: np.ndarray
) -> tuple[np.ndarray, int]:
    result = np.asarray(labels, dtype=np.int64).copy()
    unlabeled_before = int(np.count_nonzero(result < 0))
    if unlabeled_before == 0:
        return result, 0
    face_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    _, component_labels = _face_components(mesh)
    global_known = result >= 0
    if not np.any(global_known):
        raise ValueError("No labeled face is available for propagation")
    global_tree = cKDTree(face_centers[global_known])
    global_values = result[global_known]

    for component_id in np.unique(component_labels):
        in_component = component_labels == component_id
        missing = in_component & (result < 0)
        if not np.any(missing):
            continue
        known = in_component & (result >= 0)
        if np.any(known):
            tree = cKDTree(face_centers[known])
            _, nearest = tree.query(face_centers[missing], k=1)
            result[missing] = result[known][nearest]
        else:
            _, nearest = global_tree.query(face_centers[missing], k=1)
            result[missing] = global_values[nearest]
    return result, unlabeled_before


def _point_labels_to_faces(
    mesh: trimesh.Trimesh,
    point_labels: np.ndarray,
    sample_face_idx: np.ndarray,
    num_semantics: int,
) -> tuple[np.ndarray, dict[str, int]]:
    point_labels = np.asarray(point_labels, dtype=np.int64)
    sample_face_idx = np.asarray(sample_face_idx, dtype=np.int64)
    if len(point_labels) != len(sample_face_idx):
        raise ValueError(
            f"point labels ({len(point_labels)}) != sample_face_idx ({len(sample_face_idx)})"
        )
    valid = (
        (point_labels >= 0)
        & (point_labels < int(num_semantics))
        & (sample_face_idx >= 0)
        & (sample_face_idx < len(mesh.faces))
    )
    votes = np.zeros((len(mesh.faces), int(num_semantics)), dtype=np.int32)
    np.add.at(votes, (sample_face_idx[valid], point_labels[valid]), 1)
    directly_labeled = votes.sum(axis=1) > 0
    face_labels = np.full(len(mesh.faces), -1, dtype=np.int64)
    face_labels[directly_labeled] = np.argmax(votes[directly_labeled], axis=1)
    face_labels, propagated = _fill_unlabeled_faces(mesh, face_labels)
    return face_labels, {
        "sample_points": int(len(point_labels)),
        "valid_sample_labels": int(np.count_nonzero(valid)),
        "directly_voted_faces": int(np.count_nonzero(directly_labeled)),
        "propagated_faces": int(propagated),
    }


def _label_metrics(mesh: trimesh.Trimesh, labels: np.ndarray, semantics: list[str]) -> dict[str, Any]:
    _, mesh_components = _face_components(mesh)
    metrics: dict[str, Any] = {}
    for label_id, name in enumerate(semantics):
        mask = labels == label_id
        touched_mesh_components = np.unique(mesh_components[mask]) if np.any(mask) else []
        sub_faces = np.flatnonzero(mask)
        islands = 0
        if len(sub_faces):
            local_index = np.full(len(mesh.faces), -1, dtype=np.int64)
            local_index[sub_faces] = np.arange(len(sub_faces))
            adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
            keep = mask[adjacency[:, 0]] & mask[adjacency[:, 1]] if len(adjacency) else np.zeros(0, dtype=bool)
            local_edges = local_index[adjacency[keep]] if np.any(keep) else np.empty((0, 2), dtype=np.int64)
            rows = np.concatenate([local_edges[:, 0], local_edges[:, 1]]) if len(local_edges) else np.array([], dtype=np.int64)
            cols = np.concatenate([local_edges[:, 1], local_edges[:, 0]]) if len(local_edges) else np.array([], dtype=np.int64)
            graph = coo_matrix(
                (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                shape=(len(sub_faces), len(sub_faces)),
            ).tocsr()
            islands, _ = connected_components(graph, directed=False)
        metrics[name] = {
            "faces": int(np.count_nonzero(mask)),
            "face_ratio": round(float(np.mean(mask)), 6),
            "semantic_islands": int(islands),
            "mesh_components_touched": int(len(touched_mesh_components)),
        }
    return metrics


def _export_colored_mesh(
    mesh: trimesh.Trimesh, labels: np.ndarray, output_path: Path
) -> None:
    colored = mesh.copy()
    colors = np.tile(np.asarray((210, 215, 225, 255), dtype=np.uint8), (len(mesh.faces), 1))
    for label_id in np.unique(labels[labels >= 0]):
        colors[labels == label_id] = PALETTE[int(label_id) % len(PALETTE)]
    colored.visual.face_colors = colors
    colored.export(output_path)


def finalize(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = args.asset
    entry = manifest["assets"][name]
    asset_dir = output_root / name
    mesh = trimesh.load(entry["static_mesh"], force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(type(mesh))

    labels_path = Path(args.labels_json)
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    semantics = [str(x) for x in payload["semantic_parts"]]
    if entry["route"] == "pointcloud":
        point_labels = np.asarray(payload["point_semantic_index"], dtype=np.int64)
        sample_face_idx = np.load(entry["sample_face_idx"])
        face_labels, mapping_stats = _point_labels_to_faces(
            mesh, point_labels, sample_face_idx, len(semantics)
        )
    else:
        face_labels = np.asarray(payload["face_semantic_index"], dtype=np.int64)
        if len(face_labels) != len(mesh.faces):
            raise ValueError(
                f"face labels ({len(face_labels)}) != mesh faces ({len(mesh.faces)})"
            )
        mapping_stats = {
            "sample_points": 0,
            "valid_sample_labels": 0,
            "directly_voted_faces": int(len(mesh.faces)),
            "propagated_faces": 0,
        }

    final_dir = asset_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    metrics = _label_metrics(mesh, face_labels, semantics)
    final_payload = {
        "unit": "face",
        "num_faces": int(len(mesh.faces)),
        "semantic_parts": semantics,
        "face_semantic_index": face_labels.tolist(),
        "route": entry["route"],
        "source_labels_json": str(labels_path),
        "mapping_stats": mapping_stats,
        "label_metrics": metrics,
    }
    face_json = final_dir / "face_semantic_labels.json"
    face_json.write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_path = final_dir / "label_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    colored_mesh = final_dir / f"{name}_semantic_colored.ply"
    _export_colored_mesh(mesh, face_labels, colored_mesh)

    from rendering.render_component import render_full_coverage_panel

    panel = final_dir / "mesh_semantic_panel.png"
    render_full_coverage_panel(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        face_labels,
        semantics,
        panel,
        title=f"{name} mesh semantic labels ({entry['route']} route)",
        image_size=480,
        cols=4,
    )
    entry["segmentation_labels_json"] = str(labels_path)
    entry["final_face_labels"] = str(face_json)
    entry["final_colored_mesh"] = str(colored_mesh)
    entry["final_mesh_panel"] = str(panel)
    entry["mapping_stats"] = mapping_stats
    entry["label_metrics"] = metrics
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(face_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--source-dir", required=True)
    prep.add_argument("--output-root", required=True)
    prep.add_argument("--sample-points", type=int, default=8000)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--max-total-components", type=int, default=5)
    prep.add_argument("--max-significant-components", type=int, default=5)
    prep.set_defaults(func=prepare)

    fin = sub.add_parser("finalize")
    fin.add_argument("--output-root", required=True)
    fin.add_argument("--asset", choices=ASSET_NAMES, required=True)
    fin.add_argument("--labels-json", required=True)
    fin.set_defaults(func=finalize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
