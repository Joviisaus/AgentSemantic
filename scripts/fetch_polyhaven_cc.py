"""Fetch CC0 game assets from Poly Haven, audit + anonymize their node names,
and compute connected components for the blind connected-components labeling route.

Poly Haven API is keyless. Models are served as glTF (JSON) + external .bin.
We download the 1k glTF + its buffers, load via trimesh, STRIP all semantic
node/geometry names (blind-test protocol), concatenate geometries into one
anonymized mesh, and compute face-level connected components.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import trimesh

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _scipy_cc
except Exception as exc:  # noqa: BLE001
    print(f"[fatal] scipy unavailable: {exc}", file=sys.stderr)
    raise

UA = {"User-Agent": "Mozilla/5.0 (AgentSematic-eval/1.0)"}

SEMANTIC_HINT = (
    "head|body|arm|leg|torso|tail|wing|hand|foot|wheel|door|window|seat|back|"
    "lid|cover|handle|knob|button|screen|lens|crate|barrel|chest|lid|blade|"
    "guard|shelf|leg|top|base|frame|panel|ring|cap|tip|grip|strap|strap"
)


def http_get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def http_download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def fetch_model(slug: str, out_dir: Path) -> Path:
    """Download glTF(1k) + buffers; return path to the .gltf file."""
    files = http_get_json(f"https://api.polyhaven.com/files/{slug}?type=models")
    if "gltf" not in files:
        raise KeyError(f"no gltf for {slug} (available: {list(files.keys())})")
    gltf_info = files["gltf"]["1k"]["gltf"]
    gltf_url = gltf_info["url"]
    model_dir = out_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    gltf_path = model_dir / f"{slug}_1k.gltf"
    http_download(gltf_url, gltf_path)
    # parse gltf to pull external buffers (.bin) referenced relative to the gltf
    gj = json.loads(gltf_path.read_text(encoding="utf-8"))
    base = gltf_url.rsplit("/", 1)[0]
    for buf in gj.get("buffers", []):
        uri = buf.get("uri", "")
        if uri and not uri.startswith("data:"):
            burl = base + "/" + uri if not uri.startswith("http") else uri
            http_download(burl, model_dir / uri)
    # also pull images referenced if any (not required for geometry, skip)
    return gltf_path


def connected_components_faces(mesh):
    """Face-level connected components via SciPy on face-adjacency graph."""
    n_faces = len(mesh.faces)
    adj = mesh.face_adjacency
    row = np.concatenate([adj[:, 0], adj[:, 1]])
    col = np.concatenate([adj[:, 1], adj[:, 0]])
    data = np.ones(len(row), dtype=bool)
    graph = coo_matrix((data, (row, col)), shape=(n_faces, n_faces))
    n_comp, labels = _scipy_cc(graph, directed=False)
    return int(n_comp), labels


def process(slug: str, out_dir: Path, save_ply: bool = True) -> dict:
    gltf_path = fetch_model(slug, out_dir)
    scene = trimesh.load(str(gltf_path), process=False)
    geoms = []
    original_names = []
    if isinstance(scene, trimesh.Scene):
        for name, g in scene.geometry.items():
            if isinstance(g, trimesh.Trimesh) and len(g.faces):
                geoms.append(g)
                original_names.append(str(name))
    elif isinstance(scene, trimesh.Trimesh):
        geoms.append(scene)
        original_names.append("<single>")

    if not geoms:
        raise ValueError(f"no triangle geometry in {slug}")

    # Poly Haven glTF geometry is usually non-indexed (per-triangle vertices),
    # so face_adjacency would treat every triangle as isolated. Merge coincident
    # vertices PER GEOMETRY so each real surface shell becomes connected, while
    # separate geometries stay disconnected (true connected components).
    cleaned = []
    for g in geoms:
        gg = g.copy()
        try:
            gg.merge_vertices()
        except Exception:
            pass
        cleaned.append(gg)

    # Anonymize: drop all names, concatenate into one mesh.
    merged = trimesh.util.concatenate(cleaned)
    merged.metadata = {}

    n_comp, face_labels = connected_components_faces(merged)
    areas = np.asarray(merged.area_faces, dtype=np.float64)
    comp_area = np.bincount(face_labels, weights=areas, minlength=n_comp)
    total_area = float(comp_area.sum()) + 1e-12

    bbox = merged.bounding_box.extents
    components = []
    for c in range(n_comp):
        mask = face_labels == c
        cf = merged.faces[mask]
        cidx = np.unique(cf)
        sub = merged.submesh([c]) if False else None
        # build submesh from faces of this component
        try:
            sub_mesh = trimesh.Trimesh(
                vertices=merged.vertices,
                faces=cf,
                process=False,
            )
        except Exception:
            sub_mesh = None
        comp_area_ratio = float(comp_area[c] / total_area)
        if sub_mesh is not None:
            bbox_c = [round(float(x), 4) for x in sub_mesh.bounding_box.extents]
            centroid_c = [round(float(x), 4) for x in sub_mesh.centroid.tolist()]
            is_wt = bool(sub_mesh.is_watertight)
            vol = round(float(sub_mesh.volume), 4) if is_wt else None
        else:
            bbox_c = [0.0, 0.0, 0.0]
            centroid_c = [0.0, 0.0, 0.0]
            is_wt = False
            vol = None
        components.append(
            {
                "component": int(c),
                "n_faces": int(mask.sum()),
                "n_vertices": int(len(cidx)),
                "area_ratio": round(comp_area_ratio, 4),
                "bbox": bbox_c,
                "centroid": centroid_c,
                "is_watertight": is_wt,
                "volume": vol,
            }
        )

    # audit: did original node names carry semantics?
    joined = " ".join(original_names).lower()
    import re

    semantic_hits = sorted(set(re.findall(SEMANTIC_HINT, joined, flags=re.I)))
    has_semantic = len(semantic_hits) > 0

    out = {
        "slug": slug,
        "source": "Poly Haven (CC0)",
        "gltf": str(gltf_path),
        "n_geometries": len(geoms),
        "original_geometry_names": original_names,
        "audit_semantic_names_found": semantic_hits,
        "audit_has_semantic_names": has_semantic,
        "anonymized": True,
        "n_connected_components": int(n_comp),
        "merged_faces": int(len(merged.faces)),
        "merged_vertices": int(len(merged.vertices)),
        "bbox_diagonal": round(float(np.linalg.norm(bbox)), 4),
        "components": components,
    }

    if save_ply:
        ply_path = out_dir / slug / f"{slug}_anonymized.ply"
        merged.export(str(ply_path))
        out["anonymized_ply"] = str(ply_path)
        # also save face-level component labels
        lbl_path = out_dir / slug / f"{slug}_face_components.npy"
        np.save(lbl_path, face_labels.astype(np.int64))
        out["face_component_labels"] = str(lbl_path)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="testmodels/cc_game_assets")
    ap.add_argument(
        "--slugs",
        nargs="*",
        default=[
            "decorative_book_set_01",
            "book_encyclopedia_set_01",
            "bananas",
            "wicker_basket_01",
            "vintage_radio_transceiver",
            "electric_stove",
            "plastic_crate_03",
            "metal_jug",
            "wooden_cutting_board",
            "yellow_onion",
        ],
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for slug in args.slugs:
        try:
            print(f"[fetch] {slug} ...", flush=True)
            res = process(slug, out_dir)
            results.append(res)
            print(
                f"  -> geoms={res['n_geometries']} components={res['n_connected_components']} "
                f"faces={res['merged_faces']} semantic_names={res['audit_semantic_names_found']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {slug}: {exc}", flush=True)

    manifest = out_dir / "manifest_components.json"
    manifest.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {len(results)} models -> {manifest}")


if __name__ == "__main__":
    main()
