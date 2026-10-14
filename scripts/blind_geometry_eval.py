"""Blind geometry-only batch evaluation for anonymous animated assets.

This is intentionally separate from the human-adjudicated report. It never
loads GLB metadata: only geometry.ply, points.npy, sample_face_idx.npy and an
anonymous overview are consumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from geometry.pointcloud_partition import split_points_by_strategy
from rendering.pointcloud_render import (
    compute_point_normals,
    render_pointcloud_segmentation_panel,
)
from rendering.render_component import render_full_coverage_panel
from vlm.client import build_vlm_client, parse_json_response
from vlm.config import DEFAULT_CONFIG_PATH, get_provider_config, load_vlm_config

PALETTE = np.asarray(
    [
        (230, 57, 70, 255), (41, 128, 185, 255), (243, 156, 18, 255),
        (39, 174, 96, 255), (142, 68, 173, 255), (22, 160, 133, 255),
        (211, 84, 0, 255), (127, 140, 141, 255),
    ], dtype=np.uint8,
)


def face_components(mesh: trimesh.Trimesh) -> np.ndarray:
    n = len(mesh.faces)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency):
        rows = np.r_[adjacency[:, 0], adjacency[:, 1]]
        cols = np.r_[adjacency[:, 1], adjacency[:, 0]]
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    else:
        graph = coo_matrix((n, n)).tocsr()
    return connected_components(graph, directed=False)[1]


def fill_faces(mesh: trimesh.Trimesh, labels: np.ndarray) -> tuple[np.ndarray, int]:
    result = np.asarray(labels, dtype=np.int64).copy()
    missing_before = int(np.count_nonzero(result < 0))
    if not missing_before:
        return result, 0
    centers = np.asarray(mesh.triangles_center)
    cc = face_components(mesh)
    globally_known = result >= 0
    if not np.any(globally_known):
        raise ValueError("No face received a point vote")
    global_tree = cKDTree(centers[globally_known])
    global_values = result[globally_known]
    for component in np.unique(cc):
        in_component = cc == component
        missing = in_component & (result < 0)
        if not np.any(missing):
            continue
        known = in_component & (result >= 0)
        if np.any(known):
            tree = cKDTree(centers[known])
            _, nearest = tree.query(centers[missing], k=1)
            result[missing] = result[known][nearest]
        else:
            _, nearest = global_tree.query(centers[missing], k=1)
            result[missing] = global_values[nearest]
    return result, missing_before


def points_to_faces(
    mesh: trimesh.Trimesh,
    point_labels: np.ndarray,
    sample_face_idx: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, dict[str, int]]:
    point_labels = np.asarray(point_labels, dtype=np.int64)
    sample_face_idx = np.asarray(sample_face_idx, dtype=np.int64)
    valid = (
        (point_labels >= 0) & (point_labels < class_count)
        & (sample_face_idx >= 0) & (sample_face_idx < len(mesh.faces))
    )
    votes = np.zeros((len(mesh.faces), class_count), dtype=np.int32)
    np.add.at(votes, (sample_face_idx[valid], point_labels[valid]), 1)
    direct = votes.sum(1) > 0
    labels = np.full(len(mesh.faces), -1, dtype=np.int64)
    labels[direct] = np.argmax(votes[direct], axis=1)
    labels, propagated = fill_faces(mesh, labels)
    return labels, {
        "sample_points": int(len(point_labels)),
        "valid_sample_labels": int(np.count_nonzero(valid)),
        "directly_voted_faces": int(np.count_nonzero(direct)),
        "propagated_faces": int(propagated),
    }


def classify_groups(
    client,
    model: str,
    panels: list[Path],
    semantics: list[str],
) -> tuple[list[str], str]:
    from PIL import Image, ImageDraw

    tile_width, tile_height = 520, 400
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    montage = Image.new("RGB", (tile_width * cols, tile_height * rows), "white")
    for index, panel in enumerate(panels):
        image = Image.open(panel).convert("RGB")
        image.thumbnail((tile_width - 20, tile_height - 38))
        tile = Image.new("RGB", (tile_width, tile_height), "white")
        ImageDraw.Draw(tile).text((12, 8), f"group {index}", fill=(20, 30, 50))
        tile.paste(image, ((tile_width - image.width) // 2, 30))
        montage.paste(tile, ((index % cols) * tile_width, (index // cols) * tile_height))
    montage_path = panels[0].parent.parent / "root_groups_montage.png"
    montage.save(montage_path)

    numbered = ", ".join(f"{i}={name}" for i, name in enumerate(semantics))
    prompt = (
        "You are evaluating anonymous 3D geometry. The single montage contains numbered tiles "
        "for the same object. Each tile is a 12-view panel with one geometric block highlighted. "
        "Do not infer filenames, source assets, materials, skeletons or metadata. "
        f"Allowed labels are: {numbered}. For groups 0 through {len(panels) - 1}, choose exactly "
        "one allowed label per group based only on highlighted geometry. Labels may repeat. "
        f'Return JSON only with exactly {len(panels)} labels in group order: '
        '{"labels":["..."],"notes":"brief uncertainty note"}'
    )
    response = client.chat(
        user_prompt=prompt,
        image_paths=[montage_path],
        temperature=0.0,
        json_mode=True,
        model=model,
    )
    data = parse_json_response(response.text)
    labels: list[str] = []
    for item in data.get("labels", []):
        value = str(item).strip()
        if value in semantics:
            labels.append(value)
            continue
        try:
            index = int(value)
        except ValueError:
            labels.append(value)
        else:
            labels.append(semantics[index] if 0 <= index < len(semantics) else value)
    return labels, response.text


def run_asset(
    asset_dir: Path,
    semantics: list[str],
    client,
    provider_name: str,
    model: str,
    image_size: int,
) -> dict:
    points = np.load(asset_dir / "points.npy")
    face_idx = np.load(asset_dir / "sample_face_idx.npy")
    mesh = trimesh.load(asset_dir / "geometry.ply", force="mesh", process=False)
    out = asset_dir / "blind_auto"
    panels_dir = out / "root_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    split = split_points_by_strategy(
        points, range(len(points)), len(semantics), "geodesic_fps",
        n_neighbors=16, lloyd_iter=0, random_state=0,
    )
    groups = [np.asarray(group, dtype=np.int64) for group in split.sub_point_groups]
    group_ids = np.full(len(points), -1, dtype=np.int64)
    normals = compute_point_normals(points)
    panels: list[Path] = []
    for group_id, group in enumerate(groups):
        group_ids[group] = group_id
        highlight = np.full(len(points), -1, dtype=np.int64)
        highlight[group] = 0
        panel = panels_dir / f"group_{group_id:02d}.png"
        render_pointcloud_segmentation_panel(
            points, highlight.tolist(), panel,
            title=f"anonymous root group {group_id}",
            image_size=(image_size, image_size), cols=4, normals=normals,
        )
        panels.append(panel)
    np.save(out / "root_group_id.npy", group_ids)

    predicted_names, raw = classify_groups(client, model, panels, semantics)
    valid_names = set(semantics)
    predicted_names = [x if x in valid_names else "__invalid__" for x in predicted_names]
    while len(predicted_names) < len(groups):
        predicted_names.append("__invalid__")
    predicted_names = predicted_names[:len(groups)]
    name_to_id = {name: i for i, name in enumerate(semantics)}
    point_labels = np.full(len(points), -1, dtype=np.int64)
    for group_id, group in enumerate(groups):
        name = predicted_names[group_id]
        if name in name_to_id:
            point_labels[group] = name_to_id[name]

    point_payload = {
        "unit": "point",
        "num_points": int(len(points)),
        "semantic_parts": semantics,
        "point_semantic_index": point_labels.tolist(),
        "group_predictions": predicted_names,
        "root_group_sizes": [int(len(x)) for x in groups],
        "provider": provider_name,
        "model": model,
        "raw_response": raw,
        "protocol": "anonymous geometry-only; untouched root-block automatic classification",
    }
    (out / "point_semantic_labels.json").write_text(
        json.dumps(point_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_pointcloud_segmentation_panel(
        points, point_labels.tolist(), out / "point_result.png",
        title="blind automatic point labels", image_size=(image_size, image_size),
        cols=4, normals=normals,
    )

    if np.any(point_labels >= 0):
        face_labels, mapping = points_to_faces(mesh, point_labels, face_idx, len(semantics))
        colors = np.tile(np.asarray((210, 215, 225, 255), dtype=np.uint8), (len(mesh.faces), 1))
        for label_id in range(len(semantics)):
            colors[face_labels == label_id] = PALETTE[label_id % len(PALETTE)]
        colored = mesh.copy()
        colored.visual.face_colors = colors
        colored.export(out / "semantic_colored.ply")
        render_full_coverage_panel(
            np.asarray(mesh.vertices), np.asarray(mesh.faces), face_labels, semantics,
            out / "mesh_result.png", title="blind automatic mesh labels",
            image_size=image_size, cols=4,
        )
        face_payload = {
            "unit": "face", "num_faces": int(len(mesh.faces)),
            "semantic_parts": semantics,
            "face_semantic_index": face_labels.tolist(),
            "mapping_stats": mapping,
            "protocol": "mapped from untouched automatic point predictions",
        }
        (out / "face_semantic_labels.json").write_text(
            json.dumps(face_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        mapping = None

    result = {
        "semantic_parts": semantics,
        "root_groups": len(groups),
        "group_predictions": predicted_names,
        "valid_groups": int(sum(x in valid_names for x in predicted_names)),
        "unique_predicted_labels": sorted({x for x in predicted_names if x in valid_names}),
        "mapping_stats": mapping,
        "point_result": str(out / "point_result.png"),
        "mesh_result": str(out / "mesh_result.png") if mapping else None,
    }
    (out / "result_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--provider", default="qwen")
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    root = Path(args.root)
    enumeration = json.loads((root / "blind_enumeration.json").read_text(encoding="utf-8"))
    semantics = [str(x) for x in enumeration[args.asset]["semantic_parts"]]
    config = load_vlm_config(DEFAULT_CONFIG_PATH, load_env=True)
    provider = get_provider_config(config, args.provider)
    client = build_vlm_client(provider)
    result = run_asset(
        root / args.asset, semantics, client, provider.name,
        provider.default_model, args.image_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
