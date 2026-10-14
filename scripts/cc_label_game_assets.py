"""Connected-components labeling route for blind CC0 game assets.

Phase 1 (--render): load the anonymized mesh, compute face connected components,
keep significant parts (area >= 2%), render a per-part contact sheet + a
full-coverage 12-view panel, and write agent_request.json (part stats).

Phase 2 (--apply): read agent_labels.json (part_index -> functional label from
agent visual judgment), export face_semantic_labels.json + semantic_colored.ply
+ enumeration.json + result.json.

Segmentation is PURELY geometric (connected components); labeling is assigned by
agent visual judgment of each isolated part (no semantic names exist in source).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _scipy_cc
except Exception as exc:  # noqa: BLE401
    raise

# import rendering helpers
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rendering.render_component import FULL_COVERAGE_PALETTE  # noqa: E402
from rendering.heatmap_renderer import (  # noqa: E402
    VIEW_GRID,
    _apply_camera,
    _to_pyvista_faces,
    stitch_views,
)

SIG_THRESHOLD = 0.02  # project standard: keep components >= 2% area

# Categorical palette for SEMANTIC labels (not part indices): distinct, high
# chroma, friendly to legends. Dust/unknown stays gray.
SEMANTIC_PALETTE = [
    (231, 76, 60),    # red
    (46, 134, 193),   # blue
    (39, 174, 96),    # green
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
    (26, 188, 156),   # teal
    (214, 68, 122),   # pink
    (149, 165, 166),  # gray-blue
    (108, 52, 131),   # deep violet
    (22, 160, 133),   # sea green
    (211, 84, 0),     # pumpkin
    (100, 181, 246),  # light blue
    (240, 147, 43),   # carrot
    (133, 193, 233),  # sky
    (200, 247, 197),  # mint
]

DUST_COLOR = (210, 210, 215)


def assign_label_colors(unique_labels):
    """Map each semantic label -> (r,g,b). Stable, distinct per label."""
    colors = {}
    for i, lab in enumerate(sorted(unique_labels)):
        colors[lab] = SEMANTIC_PALETTE[i % len(SEMANTIC_PALETTE)]
    return colors


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def connected_components_faces(mesh):
    n_faces = len(mesh.faces)
    adj = mesh.face_adjacency
    row = np.concatenate([adj[:, 0], adj[:, 1]])
    col = np.concatenate([adj[:, 1], adj[:, 0]])
    data = np.ones(len(row), dtype=bool)
    graph = coo_matrix((data, (row, col)), shape=(n_faces, n_faces))
    n_comp, labels = _scipy_cc(graph, directed=False)
    return int(n_comp), labels


def area_sorted_components(mesh, labels, n_comp):
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    comp_area = np.bincount(labels, weights=areas, minlength=n_comp)
    order = np.argsort(-comp_area)  # largest first
    remap = np.empty(n_comp, dtype=np.int64)
    remap[order] = np.arange(n_comp)
    face_labels = remap[labels]  # now label 0 = largest
    comp_area_sorted = comp_area[order]
    return face_labels, comp_area_sorted


def full_coverage_assignment(mesh, labels, n_comp, sig_idx):
    """Assign EVERY raw connected component to a significant part index so that
    no face is left unlabeled.

    - Significant components keep their part index (1-based).
    - Dust components are merged into the nearest significant component by
      centroid distance.

    Returns rawcomp_to_part: int array of length n_comp mapping each raw
    component id -> a part index (>=1). Always -1 only if there are no
    significant components at all.
    """
    face_centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    comp_centroid = np.zeros((n_comp, 3), dtype=np.float64)
    comp_count = np.zeros(n_comp, dtype=np.int64)
    np.add.at(comp_centroid, labels, face_centroids)
    np.add.at(comp_count, labels, 1)
    comp_count = np.maximum(comp_count, 1)
    comp_centroid /= comp_count[:, None]

    sig_part = {int(cid): pi for pi, cid in enumerate(sig_idx, start=1)}
    sig_cids = list(sig_part.keys())
    sig_centroids = comp_centroid[sig_cids]  # (K, 3)

    rawcomp_to_part = np.full(n_comp, -1, dtype=np.int64)
    for cid in sig_cids:
        rawcomp_to_part[cid] = sig_part[cid]
    for c in range(n_comp):
        if rawcomp_to_part[c] != -1:
            continue
        d = comp_centroid[c][None, :] - sig_centroids
        dist = np.sqrt((d ** 2).sum(axis=1))
        nearest = sig_cids[int(dist.argmin())]
        rawcomp_to_part[c] = sig_part[nearest]
    return rawcomp_to_part


def load_anonymized(ply_path: Path):
    mesh = trimesh.load(str(ply_path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        # fallback: concatenate scene
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms)
    return mesh


# --------------------------------------------------------------------------- #
# rendering helpers (pyvista)
# --------------------------------------------------------------------------- #
def _render_isolated(vertices, faces, mask, size=300, elev=25.0, azim=45.0):
    import pyvista as pv

    f_sub = faces[mask]
    used = np.unique(f_sub.reshape(-1))
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    v_sub = vertices[used]
    pv_faces = _to_pyvista_faces(remap[f_sub])
    pdata = pv.PolyData(v_sub, pv_faces)
    plotter = pv.Plotter(off_screen=True, window_size=(size, size))
    plotter.set_background("white")
    plotter.add_mesh(pdata, color="silver", smooth_shading=True, lighting=True)
    _apply_camera(plotter, v_sub, elev, azim)
    plotter.enable_anti_aliasing("ssaa")
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(img)


def render_contact_sheet(ply_path, face_labels, comp_area_sorted, sig_mask,
                         out_path, cols=4, thumb=300):
    mesh = load_anonymized(ply_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    from PIL import Image, ImageDraw

    total_area = float(comp_area_sorted.sum()) + 1e-12
    sig_idx = np.flatnonzero(sig_mask)  # component ids that are significant
    n = len(sig_idx)
    rows = int(np.ceil(n / cols))
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 26)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for cell, cid in enumerate(sig_idx):
        r, c = divmod(cell, cols)
        x0, y0 = c * thumb, r * (thumb + 26)
        m = face_labels == int(cid)
        img = _render_isolated(vertices, faces, m, size=thumb,
                               elev=25.0, azim=45.0 + (cell % 3) * 35.0)
        im = Image.fromarray(img).convert("RGB")
        sheet.paste(im, (x0, y0))
        ratio = comp_area_sorted[cid] / total_area
        d.text((x0 + 4, y0 + thumb + 4),
               f"P{cell + 1}  {ratio * 100:.1f}%",
               fill=(20, 20, 20))
        d.text((x0 + 4, y0 + thumb + 16),
               f"faces={int(m.sum())}",
               fill=(90, 90, 90))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def render_coverage_by_part(vertices, faces, face_part, out_path,
                            title="", cols=4, image_size=360):
    """Fast full-coverage panel: one PolyData colored per part (no per-CC loop)."""
    import pyvista as pv
    from PIL import Image, ImageDraw

    palette = np.array(FULL_COVERAGE_PALETTE, dtype=np.int64)
    n_parts = int(face_part.max()) + 1 if int((face_part >= 0).sum()) > 0 else 0
    face_rgb = np.tile(np.array([225, 225, 230], dtype=np.uint8), (len(faces), 1))
    for pi in range(n_parts):
        m = face_part == pi
        if np.any(m):
            face_rgb[m] = palette[pi % len(palette)]

    pv_faces = _to_pyvista_faces(faces)
    pdata = pv.PolyData(vertices, pv_faces)
    pdata.cell_data["RGB"] = face_rgb

    views = []
    for elev, azim in VIEW_GRID:
        plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size))
        plotter.set_background("white")
        plotter.add_mesh(pdata, scalars="RGB", rgb=True, show_scalar_bar=False,
                         smooth_shading=True, lighting=True, ambient=0.45,
                         diffuse=0.55)
        _apply_camera(plotter, vertices, elev, azim)
        plotter.enable_anti_aliasing("ssaa")
        views.append(plotter.screenshot(return_img=True))
        plotter.close()

    labels = [f"e{e:.0f}a{a:.0f}" for e, a in VIEW_GRID]
    stitched = stitch_views(list(views), cols=cols, labels=labels)
    grid = Image.fromarray(stitched)
    title_h = 40
    out = Image.new("RGB", (grid.width, grid.height + title_h), (255, 255, 255))
    ImageDraw.Draw(out).rectangle((0, 0, out.width, title_h), fill=(245, 245, 245))
    ImageDraw.Draw(out).text((8, 10), title, fill=(20, 20, 20))
    out.paste(grid, (0, title_h))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path


def render_coverage_from_face_rgb(vertices, faces, face_rgb, out_path,
                                  title="", cols=4, image_size=360):
    """Render full-coverage panel from a precomputed per-face RGB array
    (used for semantic-class coloring where same label -> same color)."""
    import pyvista as pv
    from PIL import Image, ImageDraw

    pv_faces = _to_pyvista_faces(faces)
    pdata = pv.PolyData(np.asarray(vertices, dtype=np.float64), pv_faces)
    pdata.cell_data["RGB"] = np.asarray(face_rgb, dtype=np.uint8)

    views = []
    for elev, azim in VIEW_GRID:
        plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size))
        plotter.set_background("white")
        plotter.add_mesh(pdata, scalars="RGB", rgb=True, show_scalar_bar=False,
                         smooth_shading=True, lighting=True, ambient=0.45,
                         diffuse=0.55)
        _apply_camera(plotter, np.asarray(vertices, dtype=np.float64), elev, azim)
        plotter.enable_anti_aliasing("ssaa")
        views.append(plotter.screenshot(return_img=True))
        plotter.close()

    labels = [f"e{e:.0f}a{a:.0f}" for e, a in VIEW_GRID]
    stitched = stitch_views(list(views), cols=cols, labels=labels)
    grid = Image.fromarray(stitched)
    title_h = 40
    out = Image.new("RGB", (grid.width, grid.height + title_h), (255, 255, 255))
    ImageDraw.Draw(out).rectangle((0, 0, out.width, title_h), fill=(245, 245, 245))
    ImageDraw.Draw(out).text((8, 10), title, fill=(20, 20, 20))
    out.paste(grid, (0, title_h))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# phases
# --------------------------------------------------------------------------- #
def phase_render(slug, base_dir, out_dir):
    ply = base_dir / slug / f"{slug}_anonymized.ply"
    if not ply.exists():
        raise FileNotFoundError(ply)
    mesh = load_anonymized(ply)
    n_comp, raw_labels = connected_components_faces(mesh)
    face_labels, comp_area_sorted = area_sorted_components(mesh, raw_labels, n_comp)
    total_area = float(comp_area_sorted.sum()) + 1e-12
    ratios = comp_area_sorted / total_area
    sig_mask = ratios >= SIG_THRESHOLD
    sig_idx = np.flatnonzero(sig_mask)  # 0-based component ids (area-sorted)
    # part_index (1-based) for human; mapping part_index-1 -> component id
    parts = []
    for pi, cid in enumerate(sig_idx, start=1):
        m = face_labels == int(cid)
        sub = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[m],
                              process=False)
        parts.append({
            "part_index": pi,
            "component_id": int(cid),
            "area_ratio": round(float(ratios[cid]), 4),
            "n_faces": int(m.sum()),
            "bbox": [round(float(x), 4) for x in sub.bounding_box.extents],
            "is_watertight": bool(sub.is_watertight),
        })

    panels_dir = out_dir / slug / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    contact = render_contact_sheet(ply, face_labels, comp_area_sorted, sig_mask,
                                   panels_dir / f"{slug}_parts.png", cols=4)
    # full coverage panel
    face_part = np.where(sig_mask[face_labels], np.searchsorted(sig_idx, face_labels), -1)
    # face_part: -1 for dust, else 0-based part index among significant
    cov = render_coverage_by_part(
        np.asarray(mesh.vertices), np.asarray(mesh.faces),
        face_part,
        panels_dir / f"{slug}_coverage.png",
        title=f"{slug} connected-components coverage",
        cols=4, image_size=360,
    )

    request = {
        "slug": slug,
        "source": "Poly Haven (CC0 game asset), node names stripped",
        "n_raw_components": int(n_comp),
        "n_significant_parts": int(len(sig_idx)),
        "significance_threshold": SIG_THRESHOLD,
        "contact_sheet": str(contact),
        "coverage_panel": str(cov),
        "anonymized_ply": str(ply),
        "parts": parts,
        "instruction": (
            "Each part P{i} is a connected component with NO semantic name. "
            "Assign a functional label (no directional words like front/back/"
            "left/right). Parts that are clearly the same object type may share "
            "a label. Write agent_labels.json = {part_index: {label, notes}}."
        ),
    }
    req_path = out_dir / slug / "agent_request.json"
    req_path.write_text(json.dumps(request, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[render] {slug}: rawCC={n_comp} sigParts={len(sig_idx)} "
          f"contact={contact.name} coverage={cov.name}")
    print(f"  AGENT_DECISION_NEEDED {req_path}")
    return request


def phase_apply(slug, base_dir, out_dir):
    req_path = out_dir / slug / "agent_request.json"
    labels_path = out_dir / slug / "agent_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"missing {labels_path} (agent must assign labels)")
    request = json.loads(req_path.read_text(encoding="utf-8"))
    agent_labels = json.loads(labels_path.read_text(encoding="utf-8"))

    ply = base_dir / slug / f"{slug}_anonymized.ply"
    mesh = load_anonymized(ply)
    n_comp, raw_labels = connected_components_faces(mesh)
    face_labels, comp_area_sorted = area_sorted_components(mesh, raw_labels, n_comp)
    total_area = float(comp_area_sorted.sum()) + 1e-12
    ratios = comp_area_sorted / total_area
    sig_mask = ratios >= SIG_THRESHOLD
    sig_idx = np.flatnonzero(sig_mask)
    # part_index (1-based) -> component id
    part_to_comp = {pi: int(cid) for pi, cid in enumerate(sig_idx, start=1)}

    # build per-face part index: significant -> 0-based part index, dust -> -1
    face_part = np.full(len(face_labels), -1, dtype=np.int64)
    for pi, cid in part_to_comp.items():
        face_part[face_labels == cid] = pi - 1

    # functional label per part
    label_of_part = {}
    for pi_str, val in agent_labels.items():
        pi = int(pi_str)
        label_of_part[pi] = val.get("label", f"part_{pi}") if isinstance(val, dict) else val

    unique_labels = set(label_of_part.values())
    label_colors = assign_label_colors(unique_labels)

    # per-face functional label (-1 dust)
    face_functional = np.array(
        [label_of_part.get(pi, "dust") if pi >= 0 else "dust" for pi in face_part],
        dtype=object,
    )

    # export face_semantic_labels.json (compact, per-face arrays)
    face_out = {
        "slug": slug,
        "source": request["source"],
        "n_faces": int(len(face_labels)),
        "n_raw_components": int(n_comp),
        "n_significant_parts": int(len(sig_idx)),
        "significance_threshold": SIG_THRESHOLD,
        "face_part_index": face_part.tolist(),         # -1 = dust
        "face_functional_label": face_functional.tolist(),
        "part_index_to_label": {str(k): v for k, v in label_of_part.items()},
        "label_colors": {k: list(v) for k, v in sorted(label_colors.items())},
    }
    (out_dir / slug / "face_semantic_labels.json").write_text(
        json.dumps(face_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # colored PLY — by SEMANTIC CLASS (same label -> same color)
    n_parts = len(sig_idx)
    face_rgb = np.tile(np.array(DUST_COLOR, dtype=np.uint8), (len(face_labels), 1))
    for pi in range(n_parts):
        m = face_part == pi
        lab = label_of_part.get(pi, "dust")
        face_rgb[m] = label_colors.get(lab, DUST_COLOR)
    # per-vertex color = mean of incident face colors
    n_v = len(mesh.vertices)
    v_acc = np.zeros((n_v, 3), dtype=np.float64)
    v_cnt = np.zeros(n_v, dtype=np.int32)
    for vi in range(3):
        np.add.at(v_acc, mesh.faces[:, vi], face_rgb)
        np.add.at(v_cnt, mesh.faces[:, vi], 1)
    vertex_rgb = (v_acc / np.maximum(v_cnt, 1)[:, np.newaxis]).astype(np.uint8)
    colored = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                              vertex_colors=vertex_rgb, process=False)
    colored_path = out_dir / slug / "semantic_colored.ply"
    colored.export(str(colored_path))

    # semantic-class coverage panel
    panels_dir = out_dir / slug / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    sem_cov = render_coverage_from_face_rgb(
        np.asarray(mesh.vertices), np.asarray(mesh.faces), face_rgb,
        panels_dir / f"{slug}_semantic_coverage.png",
        title=f"{slug} semantic-class coloring",
        cols=4, image_size=360,
    )

    # ---- FULL-COVERAGE semantic coloring (every face gets a label) ----
    # Dust connected components are merged into the nearest significant part so
    # the whole mesh is colored and no gray gaps remain.
    rawcomp_to_part = full_coverage_assignment(mesh, raw_labels, n_comp, sig_idx)
    face_part_full = rawcomp_to_part[raw_labels]  # every face -> part index (>=1)
    face_label_full = np.array(
        [label_of_part.get(int(p), "dust") for p in face_part_full], dtype=object)

    face_rgb_full = np.tile(np.array(DUST_COLOR, dtype=np.uint8), (len(face_labels), 1))
    for pi in range(1, n_parts + 1):
        m = face_part_full == pi
        lab = label_of_part.get(pi, "dust")
        face_rgb_full[m] = label_colors.get(lab, DUST_COLOR)

    # in-situ mesh highlight (whole mesh colored, significant parts emphasized)
    mesh_hl = render_coverage_from_face_rgb(
        np.asarray(mesh.vertices), np.asarray(mesh.faces), face_rgb_full,
        panels_dir / f"{slug}_mesh_highlight.png",
        title=f"{slug} mesh highlight (full coverage)",
        cols=4, image_size=360,
    )

    # full-coverage colored PLY
    v_acc2 = np.zeros((n_v, 3), dtype=np.float64)
    v_cnt2 = np.zeros(n_v, dtype=np.int32)
    for vi in range(3):
        np.add.at(v_acc2, mesh.faces[:, vi], face_rgb_full)
        np.add.at(v_cnt2, mesh.faces[:, vi], 1)
    vertex_rgb_full = (v_acc2 / np.maximum(v_cnt2, 1)[:, np.newaxis]).astype(np.uint8)
    colored_full = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                                   vertex_colors=vertex_rgb_full, process=False)
    colored_full_path = out_dir / slug / "semantic_colored_fullcover.ply"
    colored_full.export(str(colored_full_path))

    # full-coverage per-face labels JSON
    face_out_full = {
        "slug": slug,
        "mode": "full_coverage",
        "n_faces": int(len(face_labels)),
        "n_raw_components": int(n_comp),
        "n_significant_parts": int(len(sig_idx)),
        "every_face_labeled": True,
        "dust_merged_to_nearest_significant": True,
        "face_part_index": face_part_full.tolist(),
        "face_functional_label": face_label_full.tolist(),
        "part_index_to_label": {str(k): v for k, v in label_of_part.items()},
        "label_colors": {k: list(v) for k, v in sorted(label_colors.items())},
    }
    (out_dir / slug / "face_semantic_labels_fullcover.json").write_text(
        json.dumps(face_out_full, ensure_ascii=False, indent=2), encoding="utf-8")

    # enumeration.json
    enum = {
        "slug": slug,
        "object_kind": "unknown (anonymized CC0 game asset)",
        "n_parts": n_parts,
        "part_index_to_label": {str(k): v for k, v in label_of_part.items()},
        "label_colors": {k: list(v) for k, v in sorted(label_colors.items())},
        "part_stats": [
            {
                "part_index": pi,
                "label": label_of_part.get(pi, f"part_{pi}"),
                "area_ratio": round(float(ratios[part_to_comp[pi]] / total_area), 4),
                "n_faces": int((face_labels == part_to_comp[pi]).sum()),
            }
            for pi in range(1, n_parts + 1)
        ],
        "dust_components": int(n_comp - len(sig_idx)),
        "labeling_mode": "Agent visual judgment (connected-component segmentation, no source semantics)",
    }
    (out_dir / slug / "enumeration.json").write_text(
        json.dumps(enum, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "slug": slug,
        "source": request["source"],
        "n_raw_components": int(n_comp),
        "n_significant_parts": n_parts,
        "significance_threshold": SIG_THRESHOLD,
        "part_labels": label_of_part,
        "verification": {
            "faces_labeled": int((face_part >= 0).sum()),
            "faces_total": int(len(face_labels)),
            "colored_ply_faces": int(len(colored.faces)),
            "colored_ply_matches_mesh": int(len(colored.faces)) == int(len(mesh.faces)),
            "faces_labeled_full": int(len(face_part_full)),
            "full_coverage": bool(np.all(face_part_full >= 1)),
            "fullcover_ply_faces": int(len(colored_full.faces)),
            "fullcover_ply_matches_mesh": int(len(colored_full.faces)) == int(len(mesh.faces)),
        },
        "outputs": {
            "face_semantic_labels": str(out_dir / slug / "face_semantic_labels.json"),
            "semantic_colored_ply": str(colored_path),
            "enumeration": str(out_dir / slug / "enumeration.json"),
            "contact_sheet": request["contact_sheet"],
            "coverage_panel": request["coverage_panel"],
            "semantic_coverage_panel": str(sem_cov),
            "mesh_highlight": str(mesh_hl),
            "semantic_colored_fullcover_ply": str(colored_full_path),
            "face_semantic_labels_fullcover": str(out_dir / slug / "face_semantic_labels_fullcover.json"),
        },
    }
    (out_dir / slug / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[apply] {slug}: parts={n_parts} -> {colored_path.name} + labels")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="testmodels/cc_game_assets")
    ap.add_argument("--out-dir", default="output/cc_game_assets_20260730")
    ap.add_argument(
        "--slugs",
        nargs="*",
        default=["bananas", "electric_stove", "vintage_radio_transceiver",
                 "book_encyclopedia_set_01"],
    )
    ap.add_argument("--phase", choices=["render", "apply", "all"], default="render")
    args = ap.parse_args()
    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for slug in args.slugs:
        if args.phase in ("render", "all"):
            phase_render(slug, base_dir, out_dir)
        if args.phase in ("apply", "all"):
            results.append(phase_apply(slug, base_dir, out_dir))
    if args.phase in ("apply", "all"):
        (out_dir / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] {len(results)} results -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
