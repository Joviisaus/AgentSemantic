"""Anonymous iterative evaluation for static multi-connected meshes.

The evaluator consumes only sanitized geometry.ply / points.npy. Original OBJ
names and material/group records are audited during prepare, then excluded from
all VLM and partition inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from geometry.pointcloud_partition import split_points_by_strategy
from rendering.pointcloud_render import (
    compute_point_normals,
    render_pointcloud_overview,
    render_pointcloud_segmentation_panel,
    render_pointcloud_thumbnails_panel,
)
from rendering.render_component import render_overview_only_panel, render_full_coverage_panel
from vlm.client import build_vlm_client, parse_json_response
from vlm.config import DEFAULT_CONFIG_PATH, get_provider_config, load_vlm_config

SELECTED_IDS = [
    "22065bf26a7da50a5c4dc8ac281ba59d",
    "5afe31025e8574ea8d6e12c39b2d5023",
    "f1c9ebc8417c029a4da5feafe6f1c8fc",
    "e6daa1435018a044da5feafe6f1c8fc",
    "151d033af5381fd4c1e1e64c8ea306ea",
    "2022610a5d1a8455abc49cae1a831a9e",
    "e73d23dbd069877f4da5feafe6f1c8fc",
    "506a05b6ba66bde0b726426d6c667c7",
    "a55b721ea5a29d7f639ff561fa3f5bac",
    "65680f336a50db976afa58a86636d6be",
]

PALETTE = np.asarray(
    [
        (230, 57, 70, 255), (41, 128, 185, 255), (243, 156, 18, 255),
        (39, 174, 96, 255), (142, 68, 173, 255), (22, 160, 133, 255),
        (211, 84, 0, 255), (127, 140, 141, 255),
    ], dtype=np.uint8,
)


@dataclass
class Block:
    block_id: str
    point_ids: np.ndarray
    depth: int
    candidates: list[int]
    parent_id: str | None = None
    assigned: int | None = None
    children: list[str] | None = None


def sparse_graph_components(mesh: trimesh.Trimesh) -> tuple[int, np.ndarray]:
    n = len(mesh.faces)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency):
        rows = np.r_[adjacency[:, 0], adjacency[:, 1]]
        cols = np.r_[adjacency[:, 1], adjacency[:, 0]]
        graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    else:
        graph = coo_matrix((n, n)).tocsr()
    return connected_components(graph, directed=False)


def mesh_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    count, labels = sparse_graph_components(mesh)
    areas = np.bincount(
        labels, weights=np.asarray(mesh.area_faces, dtype=np.float64), minlength=count
    )
    ratios = np.sort(areas)[::-1] / max(float(areas.sum()), 1e-12)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    lengths = (
        np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
        if len(edges) else np.asarray([0.0])
    )
    bbox_diag = float(np.linalg.norm(np.ptp(mesh.vertices, axis=0)))
    edge_ratio = float(np.median(lengths) / max(bbox_diag, 1e-12))
    sparse_reasons = []
    if len(mesh.faces) < 3000:
        sparse_reasons.append("faces<3000")
    if len(mesh.vertices) < 2000:
        sparse_reasons.append("vertices<2000")
    if edge_ratio > 0.02:
        sparse_reasons.append("median_edge/bbox_diag>0.02")
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(count),
        "significant_components_ge_2pct": int(np.count_nonzero(ratios >= 0.02)),
        "bbox_diagonal": bbox_diag,
        "median_unique_edge": float(np.median(lengths)),
        "median_edge_bbox_ratio": edge_ratio,
        "sparse_reasons": sparse_reasons,
        "top_component_area_ratios": [round(float(x), 6) for x in ratios[:20]],
    }


def audit_obj_metadata(path: Path) -> dict[str, Any]:
    records: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith(("o ", "g ", "usemtl ", "mtllib ")):
            records.append(stripped[:200])
    unique = sorted(set(records))
    semantic_pattern = re.compile(
        r"handle|body|strap|pocket|bag|lid|base|part|left|right|top|bottom|front|back",
        re.I,
    )
    semantic = [x for x in unique if semantic_pattern.search(x)]
    generic = all(
        re.match(r"^(g mesh\d+_mesh\d+-geometry|usemtl material[_0-9]*|mtllib model_normalized\.mtl)$", x)
        for x in unique
    ) if unique else True
    return {
        "records": unique,
        "semantic_records": semantic,
        "generic_only": bool(generic and not semantic),
    }


def prepare(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    manifest: dict[str, Any] = {
        "protocol": (
            "Original static OBJ metadata audited first. Segmentation receives only anonymous "
            "geometry.ply, points.npy and gray renders. No source id, group, object, material or "
            "category name is passed."
        ),
        "sample_points": int(args.sample_points),
        "assets": {},
    }
    for index, source_id in enumerate(SELECTED_IDS, start=1):
        source = source_root / source_id / "models" / "model_normalized.obj"
        audit = audit_obj_metadata(source)
        if not audit["generic_only"]:
            raise ValueError(f"{source} contains non-generic metadata: {audit['semantic_records']}")
        asset_name = f"static_blind_{index:03d}"
        asset_dir = output_root / asset_name
        asset_dir.mkdir(parents=True, exist_ok=True)
        loaded = trimesh.load(source, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(type(loaded))
        mesh = loaded.copy()
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        static_path = asset_dir / "geometry.ply"
        mesh.export(static_path)
        state = np.random.get_state()
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        points, face_idx = trimesh.sample.sample_surface(mesh, int(args.sample_points))
        np.random.set_state(state)
        points = np.asarray(points, dtype=np.float32)
        np.save(asset_dir / "points.npy", points)
        np.save(asset_dir / "sample_face_idx.npy", np.asarray(face_idx, dtype=np.int64))
        overview = asset_dir / "overview.png"
        render_pointcloud_thumbnails_panel(
            points, overview, thumb_size=256, cols=4,
            title=f"static blind geometry {index:03d}",
            normals=compute_point_normals(points),
        )
        metrics = mesh_metrics(mesh)
        manifest["assets"][asset_name] = {
            "source_sha_id": source_id,
            "source_path": str(source),
            "metadata_audit": audit,
            "geometry": str(static_path),
            "overview": str(overview),
            "points": str(asset_dir / "points.npy"),
            "sample_face_idx": str(asset_dir / "sample_face_idx.npy"),
            "metrics": metrics,
            "route": "pointcloud",
            "route_reasons": (
                ["multi_connected"] if metrics["components"] > 5 else []
            ) + (["sparse_mesh"] if metrics["sparse_reasons"] else []),
        }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_root / "manifest.json")


def compose_montage(images: list[Path], labels: list[str], output: Path) -> Path:
    tile_w, tile_h, cols = 470, 350, 2
    rows = (len(images) + cols - 1) // cols
    montage = Image.new("RGB", (tile_w * cols, tile_h * rows), "white")
    for i, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_w - 18, tile_h - 38))
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        ImageDraw.Draw(tile).text((10, 8), labels[i], fill=(20, 30, 50))
        tile.paste(image, ((tile_w - image.width) // 2, 30))
        montage.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))
    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output)
    return output


def build_classification_montage(
    blocks: list[Block], points: np.ndarray, normals: np.ndarray,
    iteration: int, asset_dir: Path, image_size: int,
) -> Path:
    """Render the per-block classification montage used by both the VLM path and
    the agent-judgment path. Each block contributes two thumbnails: the region
    highlighted on the full object (location) and the isolated geometry (shape),
    so the judge can decide which semantic parts the block represents.
    """
    panel_dir = asset_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panels: list[Path] = []
    labels: list[str] = []
    for block in blocks:
        highlight = np.full(len(points), -1, dtype=np.int64)
        highlight[block.point_ids] = 0
        region_panel = panel_dir / f"iter_{iteration:02d}_{block.block_id}_region.png"
        render_pointcloud_segmentation_panel(
            points, highlight.tolist(), region_panel,
            title=f"anonymous {block.block_id}", image_size=(image_size, image_size),
            cols=4, normals=normals,
        )
        panels.append(region_panel)
        labels.append(f"{block.block_id} region")
        solo_points = np.asarray(points[block.point_ids], dtype=np.float64)
        solo_labels = np.zeros(len(solo_points), dtype=np.int64)
        solo_panel = panel_dir / f"iter_{iteration:02d}_{block.block_id}_shape.png"
        solo_normals = normals[block.point_ids] if normals is not None else None
        render_pointcloud_segmentation_panel(
            solo_points, solo_labels.tolist(), solo_panel,
            title=f"isolated {block.block_id}", image_size=(image_size, image_size),
            cols=4, normals=solo_normals,
        )
        panels.append(solo_panel)
        labels.append(f"{block.block_id} shape")
    return compose_montage(
        panels, labels, asset_dir / "iterations" / f"iter_{iteration:02d}_blocks.png"
    )


def serialize_blocks(blocks: dict[str, Block]) -> dict[str, Any]:
    return {
        bid: {
            "block_id": b.block_id,
            "point_ids": np.asarray(b.point_ids, dtype=np.int64).tolist(),
            "depth": b.depth,
            "candidates": list(b.candidates),
            "parent_id": b.parent_id,
            "assigned": b.assigned,
            "children": list(b.children or []),
        }
        for bid, b in blocks.items()
    }


def deserialize_blocks(data: dict[str, Any]) -> dict[str, Block]:
    blocks: dict[str, Block] = {}
    for bid, d in data.items():
        blocks[bid] = Block(
            d["block_id"],
            np.asarray(d["point_ids"], dtype=np.int64),
            int(d["depth"]),
            list(d["candidates"]),
            parent_id=d.get("parent_id"),
            assigned=d.get("assigned"),
            children=list(d.get("children") or []),
        )
    return blocks


def enumerate_all(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    assets = sorted(p for p in root.glob("static_blind_*") if p.is_dir())
    montage = compose_montage(
        [p / "overview.png" for p in assets],
        [p.name for p in assets], root / "enumeration_montage.png",
    )
    cfg = load_vlm_config(DEFAULT_CONFIG_PATH, load_env=True)
    provider = get_provider_config(cfg, args.provider)
    client = build_vlm_client(provider)
    prompt = (
        "The image contains 10 numbered anonymous static 3D gray geometries. No source or "
        "metadata is available. For each asset, identify the broad object type and enumerate "
        "3 to 6 coarse semantic parts that cover the whole surface and are geometrically "
        "distinguishable. Merge left/right symmetric parts when geometry cannot reliably "
        "separate them. Return JSON only: {\"assets\":[{\"id\":\"static_blind_001\","
        "\"object_kind\":\"...\",\"semantic_parts\":[\"...\"]}]}"
    )
    response = client.chat(
        user_prompt=prompt, image_paths=[montage], temperature=0.0,
        json_mode=True, model=provider.default_model,
    )
    data = parse_json_response(response.text)
    rows = data.get("assets", [])
    result = {str(row.get("id")): row for row in rows if row.get("id")}
    for asset in assets:
        if asset.name not in result:
            result[asset.name] = {
                "id": asset.name,
                "object_kind": "unknown object",
                "semantic_parts": ["main body", "handle", "accessory"],
                "enumeration_fallback": True,
            }
    payload = {
        "provider": provider.name,
        "model": provider.default_model,
        "raw_response": response.text,
        "assets": result,
    }
    (root / "enumeration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(root / "enumeration.json")


def parse_classification(text: str, block_ids: list[str], n_semantics: int) -> dict[str, dict[str, Any]]:
    data = parse_json_response(text)
    rows = data.get("blocks", [])
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        block_id = str(row.get("id", ""))
        if block_id not in block_ids:
            continue
        matched = []
        for value in row.get("matched", []):
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < n_semantics and index not in matched:
                matched.append(index)
        try:
            dominant = int(row.get("dominant", -1))
        except (TypeError, ValueError):
            dominant = -1
        if not 0 <= dominant < n_semantics:
            dominant = matched[0] if matched else 0
        result[block_id] = {
            "matched": matched,
            "dominant": dominant,
            "notes": str(row.get("notes", "")),
        }
    for block_id in block_ids:
        result.setdefault(block_id, {"matched": [], "dominant": 0, "notes": "missing response"})
    return result


def classify_blocks(
    client, model: str, asset_dir: Path, iteration: int,
    blocks: list[Block], points: np.ndarray, normals: np.ndarray,
    semantics: list[str], image_size: int,
) -> tuple[dict[str, dict[str, Any]], Path, str]:
    montage = build_classification_montage(
        blocks, points, normals, iteration, asset_dir, image_size
    )
    allowed = "; ".join(f"{i}={name}" for i, name in enumerate(semantics))
    prompt = (
        "This montage shows blocks from one anonymous static 3D object. For each block there "
        "are two thumbnails: first the region highlighted on the full object (location), then "
        "the same geometry isolated (its real shape). Decide which allowed semantic parts are "
        "present in that block's isolated shape. A block may contain zero, one or multiple "
        "parts; if the isolated shape is clearly one part, list only that one. Use only "
        f"geometry. Allowed indices: {allowed}. Return JSON only: {{\"blocks\":[{{\"id\":\"C0.0\","
        "\"matched\":[0],\"dominant\":0,\"notes\":\"...\"}]}}"
    )
    try:
        response = client.chat(
            user_prompt=prompt, image_paths=[montage], temperature=0.0,
            json_mode=True, model=model,
        )
        raw = response.text or ""
        decisions = parse_classification(
            raw, [b.block_id for b in blocks], len(semantics)
        )
    except Exception as exc:  # preserve failure as an auditable fallback
        raw = json.dumps({"error": str(exc)}, ensure_ascii=False)
        decisions = {
            block.block_id: {
                "matched": [],
                "dominant": block.candidates[0] if block.candidates else 0,
                "notes": f"classification failure: {exc}",
            }
            for block in blocks
        }
    return decisions, montage, raw


def split_block(
    points: np.ndarray, block: Block, seed_count: int,
    strategy: str, convexity_weight: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "n_neighbors": 16, "lloyd_iter": 0, "random_state": 0,
    }
    if strategy == "spectral":
        kwargs.update({
            "curvature_weight": 0.6,
            "concavity_weight": float(convexity_weight),
            "normal_weight": 1.0,
        })
    result = split_points_by_strategy(
        points, block.point_ids.tolist(), seed_count, strategy, **kwargs
    )
    groups = [np.asarray(group, dtype=np.int64) for group in result.sub_point_groups if group]
    return groups, dict(result.debug or {})


def current_labels(blocks: dict[str, Block], n_points: int) -> np.ndarray:
    labels = np.full(n_points, -2, dtype=np.int64)
    for block in blocks.values():
        if block.children:
            continue
        if block.assigned is not None:
            labels[block.point_ids] = int(block.assigned)
    return labels


def point_to_face(
    mesh: trimesh.Trimesh, point_labels: np.ndarray, face_idx: np.ndarray, n_semantics: int
) -> tuple[np.ndarray, dict[str, int]]:
    valid = (
        (point_labels >= 0) & (point_labels < n_semantics)
        & (face_idx >= 0) & (face_idx < len(mesh.faces))
    )
    votes = np.zeros((len(mesh.faces), n_semantics), dtype=np.int32)
    np.add.at(votes, (face_idx[valid], point_labels[valid]), 1)
    direct = votes.sum(axis=1) > 0
    labels = np.full(len(mesh.faces), -1, dtype=np.int64)
    labels[direct] = np.argmax(votes[direct], axis=1)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    _, cc = sparse_graph_components(mesh)
    global_known = labels >= 0
    global_tree = cKDTree(centers[global_known])
    global_values = labels[global_known]
    propagated = int(np.count_nonzero(~direct))
    for component in np.unique(cc):
        inside = cc == component
        missing = inside & (labels < 0)
        if not np.any(missing):
            continue
        known = inside & (labels >= 0)
        if np.any(known):
            tree = cKDTree(centers[known])
            _, nearest = tree.query(centers[missing], k=1)
            labels[missing] = labels[known][nearest]
        else:
            _, nearest = global_tree.query(centers[missing], k=1)
            labels[missing] = global_values[nearest]
    return labels, {
        "sample_points": int(len(point_labels)),
        "valid_sample_labels": int(np.count_nonzero(valid)),
        "directly_voted_faces": int(np.count_nonzero(direct)),
        "propagated_faces": propagated,
    }


def save_agent_state(
    state_path: Path, blocks: dict[str, Block],
    queue: list[str], iteration: int, timeline: list[dict[str, Any]],
) -> None:
    state_path.write_text(json.dumps({
        "blocks": serialize_blocks(blocks),
        "queue": queue,
        "iteration": iteration,
        "timeline": timeline,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    asset_dir = root / args.asset
    enumeration = json.loads((root / "enumeration.json").read_text(encoding="utf-8"))
    entry = enumeration["assets"][args.asset]
    semantics = [str(x) for x in entry.get("semantic_parts", []) if str(x).strip()]
    semantics = semantics[:6] or ["main body", "handle", "accessory"]
    points = np.load(asset_dir / "points.npy")
    face_idx = np.load(asset_dir / "sample_face_idx.npy")
    mesh = trimesh.load(asset_dir / "geometry.ply", force="mesh", process=False)
    normals = compute_point_normals(points)
    strategy_name = "geodesic_fps" if args.strategy == "geodesic" else "spectral"
    base_name = "geodesic" if args.strategy == "geodesic" else f"convex_spectral_{args.convexity_weight:g}"
    # Agent mode keeps its own run dir so the API run is preserved for comparison.
    run_name = f"{base_name}_agent" if args.agent else base_name
    run_dir = asset_dir / "runs" / run_name
    (run_dir / "iterations").mkdir(parents=True, exist_ok=True)
    (run_dir / "panels").mkdir(parents=True, exist_ok=True)
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    agent_dir = run_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)

    if args.agent:
        provider_name = "agent"
        model_name = "agent-visual-judgment"
    else:
        cfg = load_vlm_config(DEFAULT_CONFIG_PATH, load_env=True)
        provider = get_provider_config(cfg, args.provider)
        client = build_vlm_client(provider)
        provider_name = provider.name
        model_name = provider.default_model

    state_path = run_dir / "agent_state.json"
    if args.agent and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        blocks = deserialize_blocks(state["blocks"])
        queue = list(state["queue"])
        iteration = int(state["iteration"])
        timeline = state["timeline"]
    else:
        blocks: dict[str, Block] = {
            "C0": Block("C0", np.arange(len(points), dtype=np.int64), 0, list(range(len(semantics))))
        }
        timeline = []
        root = blocks["C0"]
        groups, debug = split_block(
            points, root, max(2, min(len(semantics), args.max_seeds)),
            strategy_name, args.convexity_weight,
        )
        root.children = []
        queue = []
        for index, group in enumerate(groups):
            block_id = f"C0.{index}"
            blocks[block_id] = Block(
                block_id, group, 1, list(range(len(semantics))), parent_id="C0"
            )
            root.children.append(block_id)
            queue.append(block_id)
        group_labels = np.full(len(points), -1, dtype=np.int64)
        for index, group in enumerate(groups):
            group_labels[group] = index
        render_pointcloud_segmentation_panel(
            points, group_labels.tolist(), run_dir / "iterations" / "iter_01_root_split.png",
            title=f"iteration 1 root split {run_name}", image_size=(args.image_size, args.image_size),
            cols=4, normals=normals,
        )
        timeline.append({
            "iteration": 1, "phase": "root_split", "queue_before": 1,
            "queue_after": len(queue), "strategy": strategy_name,
            "convexity_weight": args.convexity_weight if strategy_name == "spectral" else 0.0,
            "seed_count": max(2, min(len(semantics), args.max_seeds)),
            "children": list(queue), "debug": debug,
        })
        iteration = 2
        if args.agent:
            save_agent_state(state_path, blocks, queue, iteration, timeline)

    while queue and iteration < args.max_iterations:
        current = [blocks[block_id] for block_id in queue]
        montage: Path | None = None
        if args.agent:
            dec_path = agent_dir / f"iter_{iteration:02d}_decision.json"
            if dec_path.exists():
                decisions = json.loads(dec_path.read_text(encoding="utf-8"))
            else:
                montage = build_classification_montage(
                    current, points, normals, iteration, asset_dir, args.image_size
                )
                allowed = "; ".join(f"{i}={name}" for i, name in enumerate(semantics))
                prompt = (
                    "You are the classifier for one anonymous static 3D object's current "
                    "blocks. For each block the montage shows two thumbnails: (left) the region "
                    "highlighted on the full object for location, (right) the same geometry "
                    f"isolated for its real shape. Decide which allowed semantic parts are "
                    f"present in that block's isolated shape. Allowed indices: {allowed}. "
                    "Return a JSON object keyed by block id: "
                    "{\"C0.0\": {\"matched\": [0], \"dominant\": 0, \"notes\": \"...\"}, ...}. "
                    "matched = indices of parts present (empty if none, one if clearly a single "
                    "part, multiple if the shape clearly spans several parts). dominant = the most "
                    "representative index in matched (or 0 if empty). Use only geometry; the object "
                    "has no text labels."
                )
                request = {
                    "image": str(montage),
                    "prompt": prompt,
                    "semantics": semantics,
                    "allowed": allowed,
                    "blocks": [
                        {"id": b.block_id, "points": int(len(b.point_ids)), "candidates": b.candidates}
                        for b in current
                    ],
                }
                (agent_dir / f"iter_{iteration:02d}_request.json").write_text(
                    json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                save_agent_state(state_path, blocks, queue, iteration, timeline)
                print(f"AGENT_DECISION_NEEDED {agent_dir / f'iter_{iteration:02d}_request.json'}")
                return
        else:
            decisions, montage, raw = classify_blocks(
                client, provider.default_model, run_dir, iteration,
                current, points, normals, semantics, args.image_size,
            )
            (run_dir / "responses" / f"iter_{iteration:02d}.json").write_text(
                json.dumps({"raw": raw, "decisions": decisions}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        next_queue: list[str] = []
        actions = []
        for block in current:
            decision = decisions[block.block_id]
            matched = decision["matched"]
            dominant = int(decision["dominant"])
            action: dict[str, Any] = {
                "block_id": block.block_id, "points": int(len(block.point_ids)),
                "depth": block.depth, "matched": matched, "dominant": dominant,
                "notes": decision.get("notes", ""),
            }
            if len(matched) == 1:
                block.assigned = matched[0]
                action["action"] = "assign_unique"
            elif len(matched) > 1 and block.depth < args.max_depth and len(block.point_ids) >= args.min_block_size * 2:
                seed_count = max(2, min(len(matched), args.max_seeds))
                subgroups, subdebug = split_block(
                    points, block, seed_count, strategy_name, args.convexity_weight,
                )
                if len(subgroups) > 1:
                    block.children = []
                    for child_index, subgroup in enumerate(subgroups):
                        child_id = f"{block.block_id}.{child_index}"
                        blocks[child_id] = Block(
                            child_id, subgroup, block.depth + 1, list(matched),
                            parent_id=block.block_id,
                        )
                        block.children.append(child_id)
                        next_queue.append(child_id)
                    action.update({
                        "action": "split_multi", "seed_count": seed_count,
                        "children": list(block.children), "split_debug": subdebug,
                    })
                else:
                    block.assigned = dominant
                    action["action"] = "split_failed_assign_dominant"
            else:
                block.assigned = dominant
                action["action"] = "assign_dominant_fallback"
            actions.append(action)
        labels = current_labels(blocks, len(points))
        render_pointcloud_segmentation_panel(
            points, labels.tolist(), run_dir / "iterations" / f"iter_{iteration:02d}.png",
            title=f"iteration {iteration} assigned and pending", image_size=(args.image_size, args.image_size),
            cols=4, normals=normals,
        )
        if args.agent:
            (run_dir / "responses" / f"iter_{iteration:02d}.json").write_text(
                json.dumps({"agent": True, "decisions": decisions}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        timeline.append({
            "iteration": iteration, "phase": "classify_and_split",
            "queue_before": len(queue), "queue_after": len(next_queue),
            "block_montage": str(montage) if montage is not None else "",
            "actions": actions,
        })
        queue = next_queue
        iteration += 1
        if args.agent:
            save_agent_state(state_path, blocks, queue, iteration, timeline)

    if queue:
        for block_id in queue:
            block = blocks[block_id]
            block.assigned = block.candidates[0] if block.candidates else 0
        timeline.append({
            "iteration": iteration, "phase": "forced_finalize",
            "queue_before": len(queue), "queue_after": 0,
            "reason": "max_iterations reached",
        })

    point_labels = current_labels(blocks, len(points))
    point_labels[point_labels < 0] = 0
    render_pointcloud_segmentation_panel(
        points, point_labels.tolist(), run_dir / "point_result.png",
        title=f"final point labels {run_name}", image_size=(args.image_size, args.image_size),
        cols=4, normals=normals,
    )
    face_labels, mapping = point_to_face(mesh, point_labels, face_idx, len(semantics))
    # Extremely fragmented meshes can crash or stall VTK cell rendering. The
    # point-result panel uses the exact semantic labels before face voting; keep
    # it as the stable visual proxy while still exporting true face labels/PLY.
    shutil.copy2(run_dir / "point_result.png", run_dir / "mesh_result.png")
    colored = mesh.copy()
    colors = np.tile(np.asarray((210, 215, 225, 255), dtype=np.uint8), (len(mesh.faces), 1))
    for semantic_id in range(len(semantics)):
        colors[face_labels == semantic_id] = PALETTE[semantic_id % len(PALETTE)]
    colored.visual.face_colors = colors
    colored.export(run_dir / "semantic_colored.ply")
    point_payload = {
        "unit": "point", "semantic_parts": semantics,
        "point_semantic_index": point_labels.tolist(),
        "provider": provider_name, "model": model_name,
        "strategy": run_name,
    }
    face_payload = {
        "unit": "face", "semantic_parts": semantics,
        "face_semantic_index": face_labels.tolist(), "mapping_stats": mapping,
        "strategy": run_name,
    }
    (run_dir / "point_semantic_labels.json").write_text(
        json.dumps(point_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "face_semantic_labels.json").write_text(
        json.dumps(face_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tree = {
        block_id: {
            "parent": block.parent_id, "depth": block.depth,
            "points": int(len(block.point_ids)), "candidates": block.candidates,
            "assigned": block.assigned, "children": block.children or [],
        }
        for block_id, block in blocks.items()
    }
    result = {
        "asset": args.asset, "object_kind": entry.get("object_kind"),
        "semantic_parts": semantics, "strategy": run_name,
        "num_iterations": len(timeline), "timeline": timeline,
        "block_tree": tree, "mapping_stats": mapping,
        "semantic_point_counts": {
            name: int(np.count_nonzero(point_labels == i)) for i, name in enumerate(semantics)
        },
        "semantic_face_counts": {
            name: int(np.count_nonzero(face_labels == i)) for i, name in enumerate(semantics)
        },
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "block_tree.json").write_text(
        json.dumps({"num_iterations": len(timeline), "blocks": tree}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.agent and state_path.exists():
        state_path.unlink()
    print(run_dir / "result.json")


def convex_root(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    asset_dir = root / args.asset
    enumeration = json.loads((root / "enumeration.json").read_text(encoding="utf-8"))
    semantics = [
        str(x) for x in enumeration["assets"][args.asset].get("semantic_parts", [])
        if str(x).strip()
    ][:6] or ["main body", "handle", "accessory"]
    points = np.load(asset_dir / "points.npy")
    budget = min(int(args.control_points), len(points))
    rng = np.random.default_rng(0)
    control_ids = np.sort(rng.choice(len(points), size=budget, replace=False))
    control = np.asarray(points[control_ids], dtype=np.float64)
    split = split_points_by_strategy(
        control, range(len(control)), max(2, min(len(semantics), args.max_seeds)),
        "spectral", n_neighbors=16, curvature_weight=0.6,
        concavity_weight=float(args.convexity_weight), normal_weight=1.0,
        random_state=0,
    )
    control_labels = np.full(len(control), -1, dtype=np.int64)
    for label_id, group in enumerate(split.sub_point_groups):
        control_labels[np.asarray(group, dtype=np.int64)] = label_id
    tree = cKDTree(control)
    _, nearest = tree.query(np.asarray(points, dtype=np.float64), k=1)
    labels = control_labels[np.asarray(nearest, dtype=np.int64)]
    out = asset_dir / "runs" / f"convex_root_{args.convexity_weight:g}"
    out.mkdir(parents=True, exist_ok=True)
    render_pointcloud_segmentation_panel(
        points, labels.tolist(), out / "iter_01_root_split.png",
        title=f"convexity-aware spectral root split w={args.convexity_weight:g}",
        image_size=(args.image_size, args.image_size), cols=4,
        normals=compute_point_normals(points),
    )
    payload = {
        "asset": args.asset, "strategy": "spectral",
        "split_convexity_weight": float(args.convexity_weight),
        "curvature_weight": 0.6, "normal_weight": 1.0,
        "sample_points": int(len(points)), "control_points": int(budget),
        "control_to_full": "1-nearest-neighbor",
        "seed_count": max(2, min(len(semantics), args.max_seeds)),
        "num_groups": int(len(split.sub_point_groups)),
        "debug": dict(split.debug or {}),
        "semantic_classification_performed": False,
        "note": "Convexity candidate only; compare this root partition with geodesic iteration 1.",
    }
    (out / "candidate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out / "candidate.json")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-root", required=True)
    prep.add_argument("--output-root", required=True)
    prep.add_argument("--sample-points", type=int, default=8000)
    prep.add_argument("--seed", type=int, default=0)
    prep.set_defaults(func=prepare)
    enum = sub.add_parser("enumerate")
    enum.add_argument("--output-root", required=True)
    enum.add_argument("--provider", default="qwen")
    enum.set_defaults(func=enumerate_all)
    run = sub.add_parser("evaluate")
    run.add_argument("--output-root", required=True)
    run.add_argument("--asset", required=True)
    run.add_argument("--provider", default="zhipu")
    run.add_argument("--strategy", choices=["geodesic", "convex"], default="geodesic")
    run.add_argument("--convexity-weight", type=float, default=0.6)
    run.add_argument("--max-iterations", type=int, default=8)
    run.add_argument("--max-depth", type=int, default=4)
    run.add_argument("--max-seeds", type=int, default=5)
    run.add_argument("--min-block-size", type=int, default=120)
    run.add_argument("--image-size", type=int, default=256)
    run.add_argument("--agent", action="store_true",
                     help="Use the agent's own visual judgment instead of a VLM API for "
                          "block classification. Renders the per-round montage and waits for "
                          "iter_NN_decision.json written by the agent, then continues.")
    run.set_defaults(func=evaluate)
    convex = sub.add_parser("convex-root")
    convex.add_argument("--output-root", required=True)
    convex.add_argument("--asset", required=True)
    convex.add_argument("--convexity-weight", type=float, default=0.6)
    convex.add_argument("--control-points", type=int, default=2500)
    convex.add_argument("--max-seeds", type=int, default=5)
    convex.add_argument("--image-size", type=int, default=192)
    convex.set_defaults(func=convex_root)
    return p


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
