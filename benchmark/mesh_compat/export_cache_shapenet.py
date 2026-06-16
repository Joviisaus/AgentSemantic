"""Stage 1 (ShapeNetPart) — export a shared mesh+GT cache, same schema as
``export_cache.py`` so ``run_ours.py`` / ``run_spectral.py`` / ``eval_unified``
consume it unchanged.

ShapeNetPart ships *point clouds* with per-point part labels but **no mesh**, so
this adapter pairs each shape with its **ShapeNetCore** mesh (by synset + model
id) and snaps the labelled points to mesh faces (``target_face_idx``), exactly
like the 3DCoMPaT exporter.

Inputs
------
--part-root : ShapeNetPart "_normal" root, i.e. the folder that contains
              ``<synset>/<model_id>.txt`` (each line: x y z nx ny nz label) and
              ``train_test_split/shuffled_<split>_file_list.json``.
--core-root : ShapeNetCore root, i.e. the folder that contains
              ``<synset>/<model_id>/...`` with either
                v2: ``models/model_normalized.obj``  or
                v1: ``model.obj``.

Per kept shape we dump ``<cache>/<sid>.npz`` with::
    vertices        (V,3) float64
    faces           (F,3) int64
    gt_xyz          (P,3) float32   - the labelled ShapeNetPart points
    gt_part_ids     (P,)  int64     - per-point GLOBAL part class id (0..49)
    target_face_idx (P,)  int64     - per-point face index on the mesh

and ``<cache>/index.json`` mapping ``sid -> {category_id, category_name,
part_names, part_class_ids, n_faces, n_points, n_components}``.

Example::

    D:\\...\\envs\\agentsematic\\python.exe ^
      E:\\Proj\\AgentSematic\\benchmark\\mesh_compat\\export_cache_shapenet.py ^
      --cache E:\\Proj\\AgentSematic\\benchmark\\results\\mesh_cache_shapenet ^
      --part-root E:\\data\\shapenetcore_partanno_segmentation_benchmark_v0_normal ^
      --core-root E:\\data\\ShapeNetCore.v2 ^
      --split test --n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_split_list(part_root: Path, split: str) -> List[Tuple[str, str]]:
    """Return [(synset, model_id), ...] for a split, from train_test_split JSON."""
    fmap = {
        "train": "shuffled_train_file_list.json",
        "val": "shuffled_val_file_list.json",
        "valid": "shuffled_val_file_list.json",
        "test": "shuffled_test_file_list.json",
    }
    fname = fmap.get(split, f"shuffled_{split}_file_list.json")
    fpath = part_root / "train_test_split" / fname
    pairs: List[Tuple[str, str]] = []
    if fpath.is_file():
        for entry in _load_json(fpath):
            parts = str(entry).replace("\\", "/").strip("/").split("/")
            if len(parts) >= 2:
                pairs.append((parts[-2], parts[-1]))
        return pairs
    # Fallback: scan all <synset>/<model>.txt
    for syn_dir in sorted(p for p in part_root.iterdir() if p.is_dir()):
        if syn_dir.name == "train_test_split":
            continue
        for txt in sorted(syn_dir.glob("*.txt")):
            pairs.append((syn_dir.name, txt.stem))
    return pairs


def _load_points_labels(part_root: Path, synset: str, model_id: str
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    txt = part_root / synset / f"{model_id}.txt"
    if not txt.is_file():
        return None
    arr = np.loadtxt(txt).astype(np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    xyz = arr[:, :3]
    labels = arr[:, -1].astype(np.int64)
    return xyz, labels


def _find_core_obj(core_root: Path, synset: str, model_id: str) -> Optional[Path]:
    candidates = [
        core_root / synset / model_id / "models" / "model_normalized.obj",  # v2
        core_root / synset / model_id / "model.obj",                        # v1
        core_root / synset / model_id / f"{model_id}.obj",
    ]
    for c in candidates:
        if c.is_file():
            return c
    hits = list((core_root / synset / model_id).rglob("*.obj"))
    return hits[0] if hits else None


def _load_mesh(obj_path: Path):
    import trimesh
    m = trimesh.load(str(obj_path), force="mesh", process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    try:
        m.merge_vertices()
    except Exception:  # noqa: BLE001
        pass
    return m


def _count_components(mesh) -> int:
    try:
        from trimesh.graph import connected_component_labels
        n_faces = int(len(mesh.faces))
        if n_faces == 0:
            return 10 ** 9
        adj = mesh.face_adjacency
        if adj is None or len(adj) == 0:
            return n_faces
        labels = connected_component_labels(adj, node_count=n_faces)
        return int(np.unique(labels).size)
    except Exception:  # noqa: BLE001
        return 10 ** 9


def _align_points_to_mesh(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Center + uniform-scale the labelled points into the mesh's AABB frame so
    ``closest_point`` is meaningful even if the two normalizations differ."""
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(verts, dtype=np.float64)
    p_min, p_max = p.min(0), p.max(0)
    v_min, v_max = v.min(0), v.max(0)
    p_ext = float((p_max - p_min).max())
    v_ext = float((v_max - v_min).max())
    if p_ext <= 1e-12 or v_ext <= 1e-12:
        return p
    scale = v_ext / p_ext
    return (p - (p_min + p_max) / 2.0) * scale + (v_min + v_max) / 2.0


def _resolve_labels(labels: np.ndarray, part_block: List[int],
                    label_mode: str) -> Tuple[np.ndarray, str]:
    """Return per-point GLOBAL part ids (0..49) and the detected mode string."""
    uniq = sorted(set(int(x) for x in np.unique(labels)))
    block = list(part_block)
    block_set = set(block)
    if label_mode == "global" or (label_mode == "auto" and set(uniq) <= block_set):
        return labels.astype(np.int64), "global"
    # local -> map sorted local ids onto the category block in order.
    local_sorted = uniq
    mapping = {}
    for i, lv in enumerate(local_sorted):
        mapping[int(lv)] = block[i] if i < len(block) else block[-1]
    out = np.array([mapping[int(x)] for x in labels], dtype=np.int64)
    return out, f"local->global({local_sorted}->{[mapping[v] for v in local_sorted]})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export ShapeNetPart mesh+GT cache")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--part-root", required=True, type=Path,
                    help="ShapeNetPart _normal root (<synset>/<model>.txt)")
    ap.add_argument("--core-root", required=True, type=Path,
                    help="ShapeNetCore root (<synset>/<model>/...obj)")
    ap.add_argument("--vocab", type=Path,
                    default=Path(r"E:\Proj\MeshReductionAgent\configs\vocab\shapenetpart.json"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=5,
                    help="how many fewest-component shapes to keep")
    ap.add_argument("--probe-cap", type=int, default=0,
                    help="candidates to probe before keeping easiest --n (0 -> 4*n)")
    ap.add_argument("--categories", default="",
                    help="comma list of category names to restrict (e.g. chair,table)")
    ap.add_argument("--shape-ids", default="",
                    help="comma list of '<synset>_<model_id>' or '<model_id>' to force")
    ap.add_argument("--label-mode", default="auto",
                    choices=["auto", "global", "local"])
    ap.add_argument("--no-sort-by-components", action="store_true")
    args = ap.parse_args(argv)

    import trimesh  # noqa: F401  (ensure available early)

    vocab = _load_json(args.vocab)
    cat_names: Dict[int, str] = {int(k): v for k, v in vocab["categories"].items()}
    cat_to_parts = {int(k): list(v) for k, v in vocab["category_to_parts"].items()}
    part_names_all: List[str] = vocab["part_names"]
    syn_to_cat: Dict[str, int] = {
        str(k): int(v) for k, v in vocab.get("synset_to_category", {}).items()
    }

    want_cats = {c.strip().lower() for c in args.categories.split(",") if c.strip()}
    want_ids = {s.strip() for s in args.shape_ids.split(",") if s.strip()}

    pairs = _read_split_list(args.part_root, args.split)
    if not pairs:
        print(f"[shapenet] no shapes found under {args.part_root} (split={args.split})",
              file=sys.stderr)
        return 1

    # filter by category / explicit ids
    filtered: List[Tuple[str, str]] = []
    for syn, mid in pairs:
        cat_id = syn_to_cat.get(syn)
        if cat_id is None:
            continue
        if want_cats and cat_names.get(cat_id, "").lower() not in want_cats:
            continue
        if want_ids and (mid not in want_ids and f"{syn}_{mid}" not in want_ids):
            continue
        filtered.append((syn, mid))

    if want_ids:
        probe_cap = len(filtered)
    else:
        probe_cap = args.probe_cap if args.probe_cap > 0 else max(4 * args.n, args.n)
    pool = filtered[:probe_cap]
    print(f"[shapenet] split={args.split} pool={len(pool)} "
          f"(keep easiest {args.n} by components)", flush=True)

    args.cache.mkdir(parents=True, exist_ok=True)
    probed: List[dict] = []
    for syn, mid in pool:
        sid = f"{syn}_{mid}"
        cat_id = syn_to_cat.get(syn)
        if cat_id is None:
            continue
        block = cat_to_parts.get(cat_id, [])
        if not block:
            print(f"  skip {sid}: empty category vocabulary", flush=True)
            continue

        pl = _load_points_labels(args.part_root, syn, mid)
        if pl is None:
            print(f"  skip {sid}: no point/label txt", flush=True)
            continue
        xyz, labels = pl
        gt, lab_mode = _resolve_labels(labels, block, args.label_mode)

        obj = _find_core_obj(args.core_root, syn, mid)
        if obj is None:
            print(f"  skip {sid}: no ShapeNetCore obj", flush=True)
            continue
        try:
            mesh = _load_mesh(obj)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {sid}: mesh load failed ({e})", flush=True)
            continue
        if len(mesh.faces) == 0:
            print(f"  skip {sid}: empty mesh", flush=True)
            continue

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        xyz_aligned = _align_points_to_mesh(xyz, verts)
        try:
            _, _, tfi = trimesh.proximity.closest_point(mesh, xyz_aligned)
            target_face_idx = np.asarray(tfi, dtype=np.int64)
        except Exception:  # noqa: BLE001
            from scipy.spatial import cKDTree
            fc = verts[np.asarray(mesh.faces)].mean(axis=1)
            _, target_face_idx = cKDTree(fc).query(xyz_aligned, k=1)
            target_face_idx = np.asarray(target_face_idx, dtype=np.int64)

        n_comp = _count_components(mesh)
        present = [p for p in block if int((gt == p).sum()) > 0]
        part_names = [part_names_all[p] for p in present]
        part_class_ids = [int(p) for p in present]
        if not present:
            print(f"  skip {sid}: no labelled points in vocab block", flush=True)
            continue

        probed.append({
            "sid": sid, "n_comp": int(n_comp),
            "vertices": verts,
            "faces": np.asarray(mesh.faces, dtype=np.int64),
            "gt_xyz": xyz_aligned.astype(np.float32),
            "gt_part_ids": gt.astype(np.int64),
            "target_face_idx": target_face_idx,
            "meta": {
                "category_id": int(cat_id),
                "category_name": cat_names.get(cat_id, "object"),
                "part_names": part_names,
                "part_class_ids": part_class_ids,
                "n_faces": int(len(mesh.faces)),
                "n_points": int(xyz.shape[0]),
                "n_components": int(n_comp),
            },
        })
        print(f"  probed {sid} ({cat_names.get(cat_id)}) faces={len(mesh.faces)} "
              f"comp={n_comp} parts={len(present)} labels={lab_mode}", flush=True)

    if not probed:
        print("[shapenet] nothing probed successfully", file=sys.stderr)
        return 1

    if not args.no_sort_by_components:
        probed.sort(key=lambda d: (d["n_comp"], d["sid"]))
    kept = probed[: args.n] if args.n > 0 else probed

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
        print(f"    {d['sid']:40} {d['meta']['category_name']:12} "
              f"comp={d['n_comp']} faces={d['meta']['n_faces']} "
              f"parts={len(d['meta']['part_class_ids'])}")
    print(f"[cache] index.json -> {args.cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
