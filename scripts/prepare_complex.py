import re, json, time
from pathlib import Path
import numpy as np
import trimesh

from rendering.pointcloud_render import (
    compute_point_normals, render_pointcloud_thumbnails_panel,
)
from scripts.static_multicc_iterative_eval import mesh_metrics

OUTPUT = Path("E:/Proj/AgentSematic/output/static_complex_blind_20260730")
TARGET_FACES = 15000
SAMPLE_POINTS = 8000

SOURCES = [
    ("E:/Proj/AgentSematic/testmodels/Mushroom_Horse_-_1a_-_Mirror_-_WIP.stl", "creature"),
    ("E:/Proj/AgentSematic/testmodels/cetan_emissary11_reduced.stl", "creature"),
    ("E:/Proj/AgentSematic/testmodels/femaleScuba.obj", "human_figure"),
    ("E:/Proj/AgentSematic/testmodels/Cat.stl", "animal"),
    ("E:/Proj/AgentSematic/testmodels/male_scuba.obj", "human_figure"),
    ("E:/Proj/AgentSematic/testmodels/models/Ancient_Red_Dragon.stl", "creature"),
]

# semantic keywords that would indicate leaked part labels in the source text
SEMANTIC_KW = re.compile(
    r"\b(head|body|torso|arm|leg|wing|tail|horn|eye|ear|hand|foot|fin|tank|"
    r"mask|helmet|weapon|sword|shield|left|right|top|bottom|front|back|"
    r"part|group|object|material|mesh|component|segment)\b", re.I)

GROUP_LINE = re.compile(r"^(g\s+(\S+)|o\s+(\S+)|usemtl\s+(\S+)|mtllib\s+(\S+))", re.I)


def audit_source(path: Path) -> dict:
    """Return metadata audit. STL (binary) carries no text -> always generic.
    OBJ is scanned for named groups/objects/materials that leak semantics."""
    if path.suffix.lower() == ".stl":
        with open(path, "rb") as f:
            head = f.read(80)
        is_binary = not head[:5].lower().startswith(b"solid")
        return {
            "file_type": "stl", "binary": bool(is_binary),
            "generic_only": True, "semantic_records": [],
            "note": "binary STL carries no group/object/material text",
        }
    text = path.read_text(encoding="utf-8", errors="ignore")
    records = []
    for line in text.splitlines():
        m = GROUP_LINE.match(line.strip())
        if m:
            name = m.group(2) or m.group(3) or m.group(4) or m.group(5) or ""
            if SEMANTIC_KW.search(name):
                records.append(name)
    return {
        "file_type": "obj", "binary": False,
        "generic_only": len(records) == 0,
        "semantic_records": records,
        "note": "scanned g/o/usemtl/mtllib lines",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": (
            "Complex organic/character source meshes. Audit first; only geometry "
            "(anonymous geometry.ply + points.npy) is passed to segmentation. The "
            "agent judges part semantics visually without any source, group, object "
            "or material name. Functional parts only (no directional labels)."
        ),
        "target_faces": TARGET_FACES, "sample_points": SAMPLE_POINTS, "assets": {},
    }
    accepted = []
    for idx, (src, kind) in enumerate(SOURCES, start=1):
        src = Path(src)
        t0 = time.time()
        audit = audit_source(src)
        if not audit["generic_only"]:
            print(f"SKIP {src.name}: contains semantic metadata {audit['semantic_records']}")
            continue
        loaded = trimesh.load(src, force="mesh", process=False, skip_materials=True)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        mesh = loaded.copy()
        # simplify to a manageable, manifold-leaning resolution
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
        except Exception as e:
            print(f"  simplify failed for {src.name}: {e}; using as-is")
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
        asset_name = f"static_blind_{idx:03d}"
        asset_dir = OUTPUT / asset_name
        asset_dir.mkdir(parents=True, exist_ok=True)
        geom = asset_dir / "geometry.ply"
        mesh.export(geom)
        state = np.random.get_state()
        np.random.seed(1234 + idx)
        points, face_idx = trimesh.sample.sample_surface(mesh, SAMPLE_POINTS)
        np.random.set_state(state)
        points = np.asarray(points, dtype=np.float32)
        np.save(asset_dir / "points.npy", points)
        np.save(asset_dir / "sample_face_idx.npy", np.asarray(face_idx, dtype=np.int64))
        normals = compute_point_normals(points)
        overview = asset_dir / "overview.png"
        render_pointcloud_thumbnails_panel(
            points, overview, thumb_size=256, cols=4,
            title=f"blind geometry {idx:03d}", normals=normals,
        )
        metrics = mesh_metrics(mesh)
        manifest["assets"][asset_name] = {
            "source_path": str(src), "object_kind": kind,
            "metadata_audit": audit, "geometry": str(geom),
            "overview": str(overview),
            "points": str(asset_dir / "points.npy"),
            "sample_face_idx": str(asset_dir / "sample_face_idx.npy"),
            "metrics": metrics, "route": "pointcloud",
        }
        accepted.append((asset_name, src.name, metrics))
        print(f"OK {asset_name} <- {src.name} faces={metrics['faces']} "
              f"cc={metrics.get('components')} t={time.time()-t0:4.1f}s")
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest ->", OUTPUT / "manifest.json")
    print("accepted:", [a[0] for a in accepted])


if __name__ == "__main__":
    main()
