"""Export ShapeNet-Part point clouds into the benchmark GT schema.

Input root can be either the common HDF5 release used by CoSMo3D/PointCLIP:

    <root>/test0.h5
    <root>/test1.h5

with datasets ``data`` (B,N,3), ``label`` (B,1), and ``seg`` (B,N), or the
standard ``*_normal`` ShapeNet-Part folder:

    <root>/<synset>/<model_id>.txt
    <root>/train_test_split/shuffled_<split>_file_list.json

The exporter maps global ShapeNet-Part labels (0..49) into category-local labels
(0..num_parts-1), matching the evaluator's per-shape part-name order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from benchmark import shapenetpart as S

DEFAULT_SYNSET_TO_CATEGORY = {
    "02691156": "airplane",
    "02773838": "bag",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03261776": "earphone",
    "03467517": "guitar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03790512": "motorbike",
    "03797390": "mug",
    "03948459": "pistol",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04379243": "table",
}

DEFAULT_CATEGORY_ID_TO_NAME = [
    "airplane", "bag", "cap", "car", "chair", "earphone", "guitar", "knife",
    "lamp", "laptop", "motorbike", "mug", "pistol", "rocket", "skateboard", "table",
]

DEFAULT_CATEGORY_TO_PART_IDS = {
    "airplane": [0, 1, 2, 3],
    "bag": [4, 5],
    "cap": [6, 7],
    "car": [8, 9, 10, 11],
    "chair": [12, 13, 14, 15],
    "earphone": [16, 17, 18],
    "guitar": [19, 20, 21],
    "knife": [22, 23],
    "lamp": [24, 25, 26, 27],
    "laptop": [28, 29],
    "motorbike": [30, 31, 32, 33, 34, 35],
    "mug": [36, 37],
    "pistol": [38, 39, 40],
    "rocket": [41, 42, 43],
    "skateboard": [44, 45, 46],
    "table": [47, 48, 49],
}

DEFAULT_PART_NAMES = [
    "airplane_body", "airplane_wing", "airplane_tail", "airplane_engine",
    "bag_handle", "bag_body",
    "cap_body", "cap_visor",
    "car_roof", "car_hood", "car_wheel", "car_body",
    "chair_back", "chair_seat", "chair_leg", "chair_arm",
    "earphone_earcup", "earphone_headband", "earphone_data_wire",
    "guitar_head", "guitar_neck", "guitar_body",
    "knife_blade", "knife_handle",
    "lamp_base", "lamp_shade", "lamp_fixing_bracket", "lamp_stem",
    "laptop_keyboard", "laptop_screen",
    "motorbike_gas_tank", "motorbike_seat", "motorbike_wheel",
    "motorbike_handle", "motorbike_light", "motorbike_engine",
    "mug_handle", "mug_body",
    "pistol_barrel", "pistol_handle", "pistol_trigger",
    "rocket_body", "rocket_fin", "rocket_nose",
    "skateboard_wheel", "skateboard_deck", "skateboard_foot_belt",
    "table_top", "table_leg", "table_drawer",
]


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_split_list(root: Path, split: str) -> List[Tuple[str, str]]:
    fmap = {
        "train": "shuffled_train_file_list.json",
        "val": "shuffled_val_file_list.json",
        "valid": "shuffled_val_file_list.json",
        "test": "shuffled_test_file_list.json",
    }
    split_path = root / "train_test_split" / fmap.get(split, f"shuffled_{split}_file_list.json")
    pairs: List[Tuple[str, str]] = []
    if split_path.is_file():
        for entry in _load_json(split_path):
            parts = str(entry).replace("\\", "/").strip("/").split("/")
            if len(parts) >= 2:
                pairs.append((parts[-2], parts[-1]))
        return pairs

    for syn_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if syn_dir.name == "train_test_split":
            continue
        for txt in sorted(syn_dir.glob("*.txt")):
            pairs.append((syn_dir.name, txt.stem))
    return pairs


def _vocab_from_json(path: Path | None):
    if not path:
        return DEFAULT_SYNSET_TO_CATEGORY, DEFAULT_CATEGORY_TO_PART_IDS, DEFAULT_PART_NAMES
    payload = _load_json(path)
    part_names = [str(x) for x in payload.get("part_names", DEFAULT_PART_NAMES)]
    categories = {int(k): str(v) for k, v in payload.get("categories", {}).items()}
    syn_to_cat_id = {str(k): int(v) for k, v in payload.get("synset_to_category", {}).items()}
    cat_to_parts_raw = payload.get("category_to_parts", {})

    syn_to_category = {
        syn: categories.get(cat_id, str(cat_id))
        for syn, cat_id in syn_to_cat_id.items()
    } or DEFAULT_SYNSET_TO_CATEGORY
    category_to_ids = {
        categories.get(int(cat_id), str(cat_id)): [int(x) for x in ids]
        for cat_id, ids in cat_to_parts_raw.items()
    } or DEFAULT_CATEGORY_TO_PART_IDS
    return syn_to_category, category_to_ids, part_names


def _load_points_labels(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 4:
        raise ValueError(f"expected x y z ... label columns, got shape {arr.shape}")
    return arr[:, :3].astype(np.float32), arr[:, -1].astype(np.int64)


def _localize_labels(labels: np.ndarray, part_ids: List[int]) -> np.ndarray:
    out = np.full(labels.shape, S.IGNORE_LABEL, dtype=np.int64)
    global_to_local = {int(pid): i for i, pid in enumerate(part_ids)}
    # Some mirrors store category-local labels instead of global labels.
    valid_values = set(int(x) for x in np.unique(labels) if int(x) >= 0)
    if valid_values <= set(range(len(part_ids))):
        global_to_local.update({i: i for i in range(len(part_ids))})
    for src, dst in global_to_local.items():
        out[labels == src] = dst
    return out


def _h5_files(root: Path, split: str) -> List[Path]:
    search_roots = [root, root / "hdf5_data", root / "shapenet_part_seg_hdf5_data" / "hdf5_data"]
    files: List[Path] = []
    if split == "all":
        patterns = ["*.h5"]
    elif split in {"val", "valid"}:
        patterns = ["*val*.h5"]
    else:
        patterns = [f"*{split}*.h5"]
    for base in search_roots:
        if not base.is_dir():
            continue
        for pattern in patterns:
            files.extend(sorted(base.glob(pattern)))
    # Preserve order while removing duplicates from overlapping roots.
    seen = set()
    unique: List[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _part_names_for(category: str, category_to_ids: Dict[str, List[int]], all_part_names: List[str]) -> List[str]:
    part_ids = category_to_ids[category]
    return [
        all_part_names[pid] if 0 <= pid < len(all_part_names) else f"part_{pid}"
        for pid in part_ids
    ]


def _export_hdf5(
    args,
    files: List[Path],
    category_to_ids: Dict[str, List[int]],
    all_part_names: List[str],
    want_cats: set[str],
) -> List[S.InstanceMeta]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to export ShapeNetPart HDF5. Run with an env that has h5py "
            "(the go_pointcloud_benchmark.ps1 script uses -ExportEnv, default cosmo3d)."
        ) from exc

    instances: List[S.InstanceMeta] = []
    for h5_path in files:
        with h5py.File(h5_path, "r") as f:
            points_all = f["data"][:].astype(np.float32)
            cat_all = f["label"][:].reshape(-1).astype(np.int64)
            seg_all = f["seg"][:].astype(np.int64)
        for i in range(points_all.shape[0]):
            cat_id = int(cat_all[i])
            if cat_id < 0 or cat_id >= len(DEFAULT_CATEGORY_ID_TO_NAME):
                print(f"  [skip] {h5_path.name}:{i}: unknown category id {cat_id}")
                continue
            category = DEFAULT_CATEGORY_ID_TO_NAME[cat_id]
            if want_cats and category.lower() not in want_cats:
                continue
            part_ids = category_to_ids.get(category)
            if not part_ids:
                continue
            labels = _localize_labels(seg_all[i], part_ids)
            if not np.any(labels >= 0):
                print(f"  [skip] {h5_path.name}:{i}: no labels in category part ids")
                continue

            instance = f"{h5_path.stem}_{i:05d}"
            key = S.safe_key(category, instance)
            part_names = _part_names_for(category, category_to_ids, all_part_names)
            np.savez_compressed(
                S.gt_path(key),
                points=points_all[i],
                labels=labels,
                part_names=np.array(part_names, dtype=object),
                category=category,
                instance=instance,
                source_path=str(h5_path),
                h5_index=int(i),
                category_id=int(cat_id),
                global_part_ids=np.array(part_ids, dtype=np.int64),
            )
            instances.append(S.InstanceMeta(key, category, instance, str(h5_path), part_names))
            if args.limit and len(instances) >= args.limit:
                return instances
    return instances


def _export_txt(
    args,
    pairs: List[Tuple[str, str]],
    syn_to_category: Dict[str, str],
    category_to_ids: Dict[str, List[int]],
    all_part_names: List[str],
    want_cats: set[str],
) -> List[S.InstanceMeta]:
    instances: List[S.InstanceMeta] = []
    for synset, model_id in pairs:
        category = syn_to_category.get(synset)
        if not category:
            continue
        if want_cats and category.lower() not in want_cats:
            continue
        part_ids = category_to_ids.get(category)
        if not part_ids:
            continue
        txt = args.root / synset / f"{model_id}.txt"
        if not txt.is_file():
            print(f"  [skip] {synset}/{model_id}: missing txt")
            continue
        try:
            points, raw_labels = _load_points_labels(txt)
            labels = _localize_labels(raw_labels, part_ids)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {synset}/{model_id}: {exc}")
            continue
        if not np.any(labels >= 0):
            print(f"  [skip] {synset}/{model_id}: no labels in category part ids")
            continue

        part_names = _part_names_for(category, category_to_ids, all_part_names)
        key = S.safe_key(category, model_id)
        np.savez_compressed(
            S.gt_path(key),
            points=points,
            labels=labels,
            part_names=np.array(part_names, dtype=object),
            category=category,
            instance=model_id,
            source_path=str(txt),
            synset=synset,
            global_part_ids=np.array(part_ids, dtype=np.int64),
        )
        instances.append(S.InstanceMeta(key, category, model_id, str(txt), part_names))
        if args.limit and len(instances) >= args.limit:
            break
    return instances


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export ShapeNet-Part GT to benchmark npz files")
    ap.add_argument("--root", type=Path, default=S.SHAPENETPART_ROOT)
    ap.add_argument("--vocab", type=Path, default=None, help="optional ShapeNetPart vocab JSON")
    ap.add_argument("--split", default="test", choices=["train", "val", "valid", "test", "all"])
    ap.add_argument("--categories", default="", help="comma-separated category filter")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--build-subsets", action="store_true")
    args = ap.parse_args(argv)

    if not args.root.exists():
        print(f"[shapenetpart] root not found: {args.root}", file=sys.stderr)
        return 1

    syn_to_category, category_to_ids, all_part_names = _vocab_from_json(args.vocab)
    want_cats = {c.strip().lower() for c in args.categories.split(",") if c.strip()}
    S.GT_DIR.mkdir(parents=True, exist_ok=True)
    h5_paths = _h5_files(args.root, args.split)
    if h5_paths:
        print(f"[shapenetpart] detected HDF5 release: {len(h5_paths)} file(s)")
        instances = _export_hdf5(args, h5_paths, category_to_ids, all_part_names, want_cats)
    else:
        pairs = _read_split_list(args.root, args.split)
        if not pairs:
            print(f"[shapenetpart] no HDF5 or *_normal txt shapes found under {args.root}", file=sys.stderr)
            return 1
        instances = _export_txt(
            args, pairs, syn_to_category, category_to_ids, all_part_names, want_cats
        )

    if not instances:
        print("[shapenetpart] no samples exported", file=sys.stderr)
        return 1

    S.write_manifest(instances)
    print(f"[shapenetpart] manifest -> {S.MANIFEST_PATH}")
    if args.build_subsets:
        for name in ("tiny", "subset", "full"):
            keys = S.build_subset(name, instances)
            S.save_subset(name, keys)
            print(f"[shapenetpart] subset {name}: {len(keys)} -> {S.SUBSET_DIR / (name + '.txt')}")
    print(f"[shapenetpart] exported {len(instances)} samples -> {S.GT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

