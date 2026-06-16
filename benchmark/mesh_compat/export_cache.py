"""Stage 1 (run in the **MeshReduction** conda env).

Export a shared, method-agnostic mesh+GT cache for a set of 3DCoMPaT++ shapes,
so that *every* method (AgentSematic "ours", LLM_Spectral_Search, ...) is scored
on byte-identical geometry + ground truth.

For each selected shape we dump ``<cache>/<shape_id>.npz`` with:
    vertices        (V,3) float64   - original (unstylized) mesh vertices
    faces           (F,3) int64     - original mesh faces
    gt_xyz          (P,3) float32   - GT point cloud sampled FROM the mesh
    gt_part_ids     (P,)  int64     - per-point coarse class id (global space)
    target_face_idx (P,)  int64     - per-point face index on this mesh

and a single ``<cache>/index.json`` listing, per shape:
    {category_id, category_name, part_names[], part_class_ids[]}

This reuses ``scripts/eval_3dcompat_spectral.py`` helpers (ShapeLoader +
``_load_mesh_and_gt_from_loader``) so the GT is exactly what the spectral
baseline consumes.

Example::

    D:\\...\\envs\\MeshReduction\\python.exe ^
        E:\\Proj\\AgentSematic\\benchmark\\mesh_compat\\export_cache.py ^
        --n 5 --split valid --cache E:\\Proj\\AgentSematic\\benchmark\\results\\mesh_cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

MRA_ROOT = Path(r"E:\Proj\MeshReductionAgent")
for p in (MRA_ROOT, MRA_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Importing the spectral eval module wires up the vendored 3DCoMPaT loaders on
# sys.path and gives us the exact mesh/GT helpers the baseline uses.
import eval_3dcompat_spectral as S  # noqa: E402


def _diversified_pool(loader, explicit: str) -> List[str]:
    if explicit.strip():
        return [s.strip() for s in explicit.split(",") if s.strip()]
    by_cat: Dict[int, str] = {}
    for sid in loader.shape_ids:
        by_cat.setdefault(int(sid[:2], 16), sid)
    diversified = list(dict.fromkeys(by_cat.values()))
    rest = [s for s in loader.shape_ids if s not in diversified]
    return diversified + rest


def _weld_via_obj(mesh, tmp_dir: Path):
    """Round-trip through OBJ (with vertex normals) so coincident vertices are
    welded into a consistent connectivity graph.

    3DCoMPaT GLB scenes are "triangle soups": faces at material seams don't
    share vertex indices, so the raw face-adjacency graph is empty (0 valid
    components) and any connectivity-based partition degenerates. This mirrors
    ``evaluation/spectral_predictor.py`` exactly, guaranteeing that AgentSematic
    and the spectral baseline operate on identical topology.
    """
    import trimesh
    m = mesh.copy() if hasattr(mesh, "copy") else mesh
    try:
        m.visual = None
    except Exception:  # noqa: BLE001
        pass
    try:
        _ = m.vertex_normals  # materialise so the OBJ writer emits ``vn``
    except Exception:  # noqa: BLE001
        pass
    tmp_dir.mkdir(parents=True, exist_ok=True)
    obj_path = tmp_dir / "weld.obj"
    m.export(obj_path)
    welded = trimesh.load(str(obj_path))
    if isinstance(welded, trimesh.Scene):
        welded = trimesh.util.concatenate(list(welded.geometry.values()))
    try:
        obj_path.unlink()
    except Exception:  # noqa: BLE001
        pass
    return welded


def _count_components(mesh) -> int:
    """Total connected-component count of the (welded) face-adjacency graph.

    A clean single-piece object => 1; a fragmented "triangle soup" => many.
    We count *all* components (no face-size filter) so dust-fragmented meshes
    rank as hard, not easy. OOM-safe (graph labels, no submesh copies).
    """
    try:
        from trimesh.graph import connected_component_labels
        n_faces = int(len(mesh.faces))
        if n_faces == 0:
            return 10 ** 9
        adj = mesh.face_adjacency
        if adj is None or len(adj) == 0:
            return n_faces  # fully disconnected: every face its own component
        labels = connected_component_labels(adj, node_count=n_faces)
        return int(np.unique(labels).size)
    except Exception:  # noqa: BLE001
        return 10 ** 9  # bury un-probeable shapes at the bottom


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export 3DCoMPaT mesh+GT cache")
    ap.add_argument("--cache", required=True, type=Path,
                    help="output cache directory")
    ap.add_argument("--zip", dest="zip_path", type=Path, default=S.DEFAULT_ZIP)
    ap.add_argument("--vocab", type=Path, default=S.DEFAULT_VOCAB)
    ap.add_argument("--meta-dir", type=Path, default=S.COMPAT_META)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--n", type=int, default=5,
                    help="how many fewest-component shapes to keep in the cache")
    ap.add_argument("--probe-cap", type=int, default=0,
                    help="how many candidate shapes to load+component-probe "
                         "before keeping the easiest --n (0 -> 3*n)")
    ap.add_argument("--no-sort-by-components", action="store_true",
                    help="keep pool order instead of fewest-components-first")
    ap.add_argument("--shape-ids", type=str, default="")
    ap.add_argument("--n-points", type=int, default=2048)
    args = ap.parse_args(argv)

    import trimesh

    vocab = S._load_json(args.vocab)
    parts_list: List[str] = vocab["part_names"]
    classes = vocab["categories"]
    cat_to_parts = {int(k): list(v) for k, v in vocab["category_to_parts"].items()}

    print(f"building ShapeLoader (split={args.split}) ...", flush=True)
    loader = S._build_loader(args.zip_path, args.meta_dir, split=args.split)
    print(f"  loader.shape_ids = {len(loader.shape_ids)} shapes", flush=True)

    pool = _diversified_pool(loader, args.shape_ids)
    if args.shape_ids.strip():
        probe_cap = len(pool)
    else:
        probe_cap = args.probe_cap if args.probe_cap > 0 else max(3 * args.n, args.n)
    pool = pool[:probe_cap]
    print(f"probing {len(pool)} candidate shapes (keep easiest {args.n} by "
          f"connected-component count) ...", flush=True)

    args.cache.mkdir(parents=True, exist_ok=True)

    # Probe: load each candidate, compute component count, hold arrays in RAM.
    probed: List[dict] = []
    for sid in pool:
        class_id = int(sid[:2], 16)
        class_name = classes.get(str(class_id), "object")
        part_ids = cat_to_parts.get(class_id, [])
        part_names = [parts_list[p] for p in part_ids]
        part_class_ids = [int(p) for p in part_ids]
        if not part_names:
            print(f"  skip {sid}: empty category vocabulary", flush=True)
            continue
        try:
            mesh, xyz, gt = S._load_mesh_and_gt_from_loader(
                loader, sid, n_points=int(args.n_points))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {sid}: mesh/GT load failed ({e})", flush=True)
            continue

        # Weld coincident vertices so connectivity (and component count) is
        # correct and identical to what the spectral baseline reloads.
        try:
            mesh = _weld_via_obj(mesh, args.cache / "_weld_tmp" / sid)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN {sid}: weld failed ({e}); using raw mesh", flush=True)

        n_comp = _count_components(mesh)
        xyz = np.asarray(xyz, dtype=np.float64)
        try:
            _, _, tfi = trimesh.proximity.closest_point(mesh, xyz)
            target_face_idx = np.asarray(tfi, dtype=np.int64)
        except Exception:  # noqa: BLE001
            from scipy.spatial import cKDTree
            fc = np.asarray(mesh.vertices)[np.asarray(mesh.faces)].mean(axis=1)
            _, target_face_idx = cKDTree(fc).query(xyz, k=1)
            target_face_idx = np.asarray(target_face_idx, dtype=np.int64)

        probed.append({
            "sid": sid, "n_comp": int(n_comp),
            "vertices": np.asarray(mesh.vertices, dtype=np.float64),
            "faces": np.asarray(mesh.faces, dtype=np.int64),
            "gt_xyz": np.asarray(xyz, dtype=np.float32),
            "gt_part_ids": np.asarray(gt, dtype=np.int64),
            "target_face_idx": target_face_idx,
            "meta": {
                "category_id": class_id, "category_name": class_name,
                "part_names": part_names, "part_class_ids": part_class_ids,
                "n_faces": int(len(mesh.faces)), "n_points": int(xyz.shape[0]),
                "n_components": int(n_comp),
            },
        })
        print(f"  probed {sid} ({class_name}) faces={len(mesh.faces)} "
              f"components={n_comp}", flush=True)

    if not args.no_sort_by_components:
        probed.sort(key=lambda d: (d["n_comp"], int(d["sid"][:2], 16)))
    kept = probed[: args.n] if args.n > 0 else probed

    # Write npz + index.json (kept order == ascending component count).
    index: Dict[str, dict] = {}
    for d in kept:
        sid = d["sid"]
        np.savez_compressed(
            args.cache / f"{sid}.npz",
            vertices=d["vertices"], faces=d["faces"], gt_xyz=d["gt_xyz"],
            gt_part_ids=d["gt_part_ids"], target_face_idx=d["target_face_idx"],
        )
        index[sid] = d["meta"]

    (args.cache / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[cache] kept {len(kept)} shapes (fewest components first):")
    for d in kept:
        print(f"    {d['sid']:12} {d['meta']['category_name']:22} "
              f"components={d['n_comp']} faces={d['meta']['n_faces']}")
    print(f"[cache] index.json -> {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
