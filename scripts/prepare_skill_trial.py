from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from rendering.pointcloud_render import compute_point_normals, render_pointcloud_thumbnails_panel
from scripts.static_multicc_iterative_eval import mesh_metrics

OUTPUT = Path("E:/Proj/AgentSematic/output/skill_trial_20260730")
SOURCES = [
    Path("E:/Proj/AgentSematic/testmodels/models/Link.stl"),
    Path("E:/Proj/AgentSematic/testmodels/models/Paimon_Fixed.stl"),
]
SAMPLE_POINTS = 8000


def audit_binary_stl(path: Path) -> dict:
    raw = path.read_bytes()
    header = raw[:80]
    is_binary = not header.lstrip().lower().startswith(b"solid")
    return {
        "file_type": "stl",
        "binary": is_binary,
        "generic_only": is_binary,
        "semantic_records": [],
        "sha256": hashlib.sha256(raw).hexdigest(),
        "note": "Binary STL has no group/object/material/joint/comment namespace.",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": (
            "Skill trial. Source metadata is audited before geometry processing. "
            "Segmentation receives only anonymous geometry.ply, points.npy, "
            "sample_face_idx.npy and gray overview images; source filenames are not passed."
        ),
        "sample_points": SAMPLE_POINTS,
        "assets": {},
    }
    for index, source in enumerate(SOURCES, start=1):
        print(f"[{index}/{len(SOURCES)}] audit {source.name}", flush=True)
        audit = audit_binary_stl(source)
        if not audit["generic_only"]:
            raise ValueError(f"Non-anonymous source rejected: {source}")
        print(f"[{index}/{len(SOURCES)}] load", flush=True)
        loaded = trimesh.load(source, force="mesh", process=False)
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(type(loaded))
        mesh = loaded.copy()
        print(f"[{index}/{len(SOURCES)}] merge vertices", flush=True)
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        print(f"[{index}/{len(SOURCES)}] compute metrics", flush=True)
        metrics = mesh_metrics(mesh)
        asset = f"skill_blind_{index:03d}"
        asset_dir = OUTPUT / asset
        asset_dir.mkdir(parents=True, exist_ok=True)
        geometry = asset_dir / "geometry.ply"
        print(f"[{index}/{len(SOURCES)}] export and sample", flush=True)
        mesh.export(geometry)
        state = np.random.get_state()
        np.random.seed(20260730 + index)
        points, face_idx = trimesh.sample.sample_surface(mesh, SAMPLE_POINTS)
        np.random.set_state(state)
        points = np.asarray(points, dtype=np.float32)
        face_idx = np.asarray(face_idx, dtype=np.int64)
        np.save(asset_dir / "points.npy", points)
        np.save(asset_dir / "sample_face_idx.npy", face_idx)
        print(f"[{index}/{len(SOURCES)}] render overview", flush=True)
        overview = asset_dir / "overview.png"
        render_pointcloud_thumbnails_panel(
            points,
            overview,
            thumb_size=256,
            cols=4,
            title=f"anonymous complex asset {index:03d}",
            normals=compute_point_normals(points),
        )
        reasons = ["multi_connected"] if metrics["components"] > 5 else []
        reasons += ["sparse_mesh"] if metrics["sparse_reasons"] else []
        if not reasons:
            reasons = ["skill_trial_pointcloud_baseline"]
        manifest["assets"][asset] = {
            "source_audit": audit,
            "geometry": str(geometry),
            "points": str(asset_dir / "points.npy"),
            "sample_face_idx": str(asset_dir / "sample_face_idx.npy"),
            "overview": str(overview),
            "metrics": metrics,
            "route": "pointcloud",
            "route_reasons": reasons,
        }
        print(asset, metrics)
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUTPUT / "manifest.json")


if __name__ == "__main__":
    main()
