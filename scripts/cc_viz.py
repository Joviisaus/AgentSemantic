"""
cc_viz.py — 连通域可视化 -> 视觉逐块语义 -> 语义树（新流水线第一步/第三步）

阶段
----
  prep   计算连通域(CC)；先渲染 CC 总览，再对每个显著 CC 生成一张
         “完整模型上下文 + 当前单块高亮”的 12 视角图；输出
         agent_request.json（blocks: id / area_ratio / image）。
  apply  读取 agent_labels.json（视觉模型给的逐块语义），写 results.json，
         并调用 build_semantic_tree.py 合并成语义树。

旧的 pointcloud_full_coverage_cli / static_multicc_iterative_eval 迭代分块路线已退役；
本脚本只做 CC 可视化 + 视觉判定桥接 + 触发语义树。

用法
----
  python cc_viz.py prep  --mesh <anon.ply> --outdir <dir> --slug blind_001 [--sig 0.02]
  python cc_viz.py apply --prepdir <dir> --labels agent_labels.json [--threshold 0.5]
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
DUST_COLOR = np.array([210, 210, 215], dtype=np.uint8)
CONTEXT_COLOR = np.array([205, 210, 220], dtype=np.uint8)
HIGHLIGHT_COLOR = np.array([255, 72, 36], dtype=np.uint8)

VIEW_GRID = [
    (30, 0), (30, 45), (30, 90), (30, 135),
    (30, 180), (30, 225), (30, 270), (30, 315),
    (75, 0), (-30, 0), (0, 0), (0, 180),
]


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def load_geometry(path):
    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


def compute_cc(faces):
    """面邻接图的连通域。返回 (n_cc, face_cc)。"""
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    e = np.vstack([np.column_stack([i0, i1]),
                   np.column_stack([i1, i2]),
                   np.column_stack([i2, i0])])
    e = np.sort(e, axis=1)
    ue, idx = np.unique(e, axis=0, return_inverse=True)
    nf = len(faces)
    ne = len(ue)
    fe_rows = np.concatenate([np.arange(nf), np.arange(nf), np.arange(nf)])
    FE = coo_matrix((np.ones(len(fe_rows)), (fe_rows, idx)), shape=(nf, ne))
    A = (FE @ FE.T).tocsr()
    A = A - coo_matrix((np.ones(nf), (np.arange(nf), np.arange(nf))),
                       shape=(nf, nf)).tocsr()
    n_cc, face_cc = connected_components(A, directed=False)
    return n_cc, face_cc


def triangle_areas(vertices, faces):
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def distinct_colors(n):
    import colorsys
    cols = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0  # 黄金角，最大化相邻区分度
        r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.95)
        cols.append((np.array([r, g, b]) * 255).astype(np.uint8))
    return np.array(cols, dtype=np.uint8)


def build_cc_render_proxy(vertices, faces, face_cc, max_faces=120000):
    """按 CC 分别简化，生成仅用于渲染的代理网格，同时保留每个代理面的 CC id。

    原始标签和统计始终基于全分辨率 Mesh；代理只用于多视角图，避免百万面模型
    对每个 block 重复渲染时耗时/显存失控。按 CC 分别简化可防止不同部件粘合。
    """
    if len(faces) <= max_faces:
        return vertices, faces, face_cc, False
    import pyvista as pv

    counts = np.bincount(face_cc)
    total = len(faces)
    out_v, out_f, out_cc = [], [], []
    offset = 0
    for cid, count in enumerate(counts):
        if count == 0:
            continue
        mask = face_cc == cid
        f_sub = faces[mask]
        used = np.unique(f_sub.reshape(-1))
        remap = -np.ones(len(vertices), dtype=np.int64)
        remap[used] = np.arange(len(used), dtype=np.int64)
        v_sub = vertices[used]
        f_local = remap[f_sub]

        target = max(40, int(round(max_faces * count / total)))
        if count > target:
            pdata = pv.PolyData(v_sub, _to_pyvista_faces(f_local))
            reduction = min(0.995, max(0.0, 1.0 - target / count))
            try:
                simp = pdata.decimate_pro(reduction, preserve_topology=True)
                simp = simp.triangulate()
                pf = np.asarray(simp.faces).reshape(-1, 4)[:, 1:]
                pvtx = np.asarray(simp.points)
                if len(pf) > 0:
                    v_sub, f_local = pvtx, pf
            except Exception:
                # 最终兜底：保留原 CC。仅影响速度，不改变原始语义标签。
                pass

        out_v.append(v_sub)
        out_f.append(f_local + offset)
        out_cc.append(np.full(len(f_local), cid, dtype=np.int64))
        offset += len(v_sub)

    return (np.vstack(out_v), np.vstack(out_f), np.concatenate(out_cc), True)


# --------------------------------------------------------------------------- #
# rendering helpers (pyvista)
# --------------------------------------------------------------------------- #
def _to_pyvista_faces(faces):
    return np.hstack([np.full((len(faces), 1), 3), faces]).reshape(-1).astype(np.int64)


def _apply_camera(plotter, vertices, elev=25.0, azim=45.0):
    pts = np.asarray(vertices, dtype=np.float64)
    center = pts.mean(axis=0)
    rng = pts.max(axis=0) - pts.min(axis=0)
    radius = float(np.linalg.norm(rng)) * 1.7
    el, az = math.radians(elev), math.radians(azim)
    direction = np.array([math.cos(el) * math.sin(az),
                          math.cos(el) * math.cos(az),
                          math.sin(el)])
    plotter.camera.position = center + direction * radius
    plotter.camera.focal_point = center
    plotter.camera.up = (0, 0, 1)
    plotter.camera.view_angle = 30


def _render_12views(vertices, faces, face_rgb, image_size=300):
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
                         smooth_shading=True, lighting=True, ambient=0.5, diffuse=0.5)
        _apply_camera(plotter, vertices, elev, azim)
        views.append(np.asarray(plotter.screenshot(return_img=True)))
        plotter.close()
    return views


def _stitch(views, cols=4, labels=None, tag_h=18):
    from PIL import Image, ImageDraw
    views = [np.asarray(v) for v in views]
    h, w = views[0].shape[:2]
    n = len(views)
    rows = int(np.ceil(n / cols))
    sheet = Image.new("RGB", (cols * w, rows * (h + (tag_h if labels else 0))),
                      (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for k, v in enumerate(views):
        r, c = divmod(k, cols)
        x0, y0 = c * w, r * (h + (tag_h if labels else 0))
        im = Image.fromarray(v).convert("RGB")
        sheet.paste(im, (x0, y0))
        if labels:
            d.text((x0 + 4, y0 + h + 3), labels[k], fill=(20, 20, 20))
    return sheet


def _add_title(image, title, subtitle=""):
    """给多视角图加标题，明确当前高亮 block 与占比。"""
    from PIL import Image, ImageDraw
    image = image.convert("RGB")
    h = 48 if subtitle else 32
    out = Image.new("RGB", (image.width, image.height + h), (255, 255, 255))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, out.width, h), fill=(245, 247, 250))
    d.text((8, 7), title, fill=(20, 20, 20))
    if subtitle:
        d.text((8, 25), subtitle, fill=(85, 90, 100))
    out.paste(image, (0, h))
    return out


def _append_color_legend(image, items, cols=4):
    """在图下附加语义颜色图例；items=[(label, '#rrggbb')]."""
    from PIL import Image, ImageDraw
    image = image.convert("RGB")
    cell_w = max(180, image.width // cols)
    rows = max(1, math.ceil(len(items) / cols))
    h = rows * 28 + 12
    out = Image.new("RGB", (image.width, image.height + h), (255, 255, 255))
    out.paste(image, (0, 0))
    d = ImageDraw.Draw(out)
    y0 = image.height + 8
    for i, (label, color) in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * cell_w + 8, y0 + r * 28
        d.rounded_rectangle((x, y + 3, x + 18, y + 19), radius=3, fill=color)
        d.text((x + 25, y + 3), str(label), fill=(30, 35, 45))
    return out


def render_block_highlight_panels(vertices, faces, face_cc, order, comp_area,
                                  total, panel_dir, slug, image_size=300,
                                  original_face_counts=None):
    """为每个显著 CC 生成一张“完整模型 + 单块高亮”的 12 视角图。

    关键：不裁出孤立块。VLM 始终看到完整物体上下文；目标块橙红高亮，
    其余几何保持浅灰。返回 block_id -> panel path。
    """
    panel_dir = Path(panel_dir)
    panel_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    labels = [f"e{e:.0f}a{a:.0f}" for e, a in VIEW_GRID]
    for i, cid in enumerate(order):
        block_id = i + 1
        mask = face_cc == int(cid)
        face_rgb = np.tile(CONTEXT_COLOR, (len(faces), 1))
        face_rgb[mask] = HIGHLIGHT_COLOR
        views = _render_12views(vertices, faces, face_rgb, image_size=image_size)
        grid = _stitch(views, cols=4, labels=labels)
        shown_faces = (int(original_face_counts[cid]) if original_face_counts is not None
                       else int(mask.sum()))
        panel = _add_title(
            grid,
            f"{slug} · BLOCK P{block_id} (highlighted)",
            f"whole-model context · area={comp_area[cid] / total * 100:.2f}% · "
            f"faces={shown_faces}",
        )
        path = panel_dir / f"{slug}_P{block_id:03d}_highlight_12views.png"
        panel.save(path)
        paths[block_id] = path
    return paths


# --------------------------------------------------------------------------- #
# prep
# --------------------------------------------------------------------------- #
def prep(mesh_path, outdir, slug, sig=0.02, render_max_faces=120000):
    outdir = Path(outdir)
    (outdir / "panels").mkdir(parents=True, exist_ok=True)
    geom = load_geometry(mesh_path)
    vertices = np.asarray(geom.vertices, dtype=np.float64)
    faces = np.asarray(geom.faces, dtype=np.int64)

    n_cc, face_cc = compute_cc(faces)
    face_area = triangle_areas(vertices, faces)
    comp_area = np.bincount(face_cc, weights=face_area, minlength=n_cc)
    total = float(comp_area.sum()) + 1e-12
    ratio = comp_area / total

    sig_ids = np.flatnonzero(ratio >= sig)
    order = sig_ids[np.argsort(-comp_area[sig_ids])]  # 按面积降序编号
    n_sig = len(order)
    comp_to_block = {int(c): i + 1 for i, c in enumerate(order)}

    palette = distinct_colors(n_sig)
    face_rgb = np.tile(DUST_COLOR, (len(faces), 1))
    for i, c in enumerate(order):
        face_rgb[face_cc == c] = palette[i]
    face_block = np.array([comp_to_block.get(int(c), 0) for c in face_cc], dtype=np.int64)

    # 百万面模型先按 CC 分别简化为渲染代理；统计/标签仍来自完整网格。
    rv, rf, rcc, proxy_used = build_cc_render_proxy(
        vertices, faces, face_cc, max_faces=render_max_faces)
    render_face_rgb = np.tile(DUST_COLOR, (len(rf), 1))
    for i, c in enumerate(order):
        render_face_rgb[rcc == c] = palette[i]

    # 渲染：CC 总览 + “完整模型上下文、每次只高亮一个 block”的逐块 12 视角图。
    views = _render_12views(rv, rf, render_face_rgb, image_size=300)
    views_png = outdir / "panels" / f"{slug}_cc_overview_12views.png"
    overview = _stitch(views, cols=4,
                       labels=[f"e{e:.0f}a{a:.0f}" for e, a in VIEW_GRID])
    proxy_note = f"render proxy={len(rf):,}/{len(faces):,} faces" if proxy_used else "full-resolution render"
    _add_title(overview, f"{slug} · connected-component overview",
               f"distinct color per significant CC; dust is gray · {proxy_note}").save(views_png)
    block_panels = render_block_highlight_panels(
        rv, rf, rcc, order, comp_area, total,
        outdir / "panels" / "blocks", slug, image_size=300,
        original_face_counts=np.bincount(face_cc, minlength=n_cc),
    )

    # 带 per-CC 顶点色的几何 PLY（匿名，无来源标题）
    v_acc = np.zeros((len(vertices), 3), dtype=np.float64)
    v_cnt = np.zeros(len(vertices), dtype=np.float64)
    np.add.at(v_acc, faces[:, 0], face_rgb)
    np.add.at(v_acc, faces[:, 1], face_rgb)
    np.add.at(v_acc, faces[:, 2], face_rgb)
    np.add.at(v_cnt, faces[:, 0], 1)
    np.add.at(v_cnt, faces[:, 1], 1)
    np.add.at(v_cnt, faces[:, 2], 1)
    v_rgb = (v_acc / np.maximum(v_cnt, 1)[:, None]).astype(np.uint8)
    v_rgba = np.concatenate([v_rgb, np.full((len(vertices), 1), 255, dtype=np.uint8)],
                            axis=1)
    vis = trimesh.visual.ColorVisuals(vertex_colors=v_rgba)
    colored = trimesh.Trimesh(vertices, faces, process=False, visual=vis)
    colored_ply = outdir / f"{slug}_cc_colored.ply"
    colored.export(str(colored_ply))

    blocks = [{"id": i + 1,
               "area_ratio": float(ratio[c]),
               "faces": int((face_cc == c).sum()),
               "image": str(block_panels[i + 1])}
              for i, c in enumerate(order)]
    agent_request = {
        "slug": slug,
        "source_mesh": str(Path(mesh_path).name),
        "n_raw_components": int(n_cc),
        "n_significant": n_sig,
        "significance_threshold": sig,
        "face_total": int(len(faces)),
        "render_proxy": {
            "used": bool(proxy_used),
            "faces": int(len(rf)),
            "max_faces": int(render_max_faces),
            "note": "Proxy is visualization-only; CC metrics and labels use full-resolution geometry."
        },
        "overview_image": str(views_png),
        "blocks": blocks,
        "vlm_instruction": {
            "task": "逐个识别高亮连通块的物体部件语义",
            "input_contract": "每个 block.image 都渲染完整模型；橙红色是当前目标块，浅灰色是上下文。",
            "output_schema": {"<block_id>": {
                "label": "noun_phrase_without_direction",
                "confidence": "0..1",
                "notes": "short visual evidence"
            }},
            "consistency": "同语义块使用同一 label；避免 left/right/front/back 方位词。"
        },
        "note": "先看 overview_image 理解整体，再按 blocks 顺序把每张单块高亮 12 视角图传给 VLM。",
    }
    req_path = outdir / "agent_request.json"
    with open(req_path, "w", encoding="utf-8") as f:
        json.dump(agent_request, f, indent=2, ensure_ascii=False)
    np.save(outdir / "face_cc.npy", face_cc)
    np.save(outdir / "face_block.npy", face_block)
    render_block = np.array([comp_to_block.get(int(c), 0) for c in rcc], dtype=np.int64)
    np.savez_compressed(outdir / "render_proxy.npz",
                        vertices=np.asarray(rv, dtype=np.float32),
                        faces=np.asarray(rf, dtype=np.int32),
                        face_block=render_block)

    print(f"[prep] {slug}: {n_cc} CC, {n_sig} significant -> {req_path}")
    print(f"[prep] overview={views_png.name} block_panels={len(block_panels)} "
          f"colored_ply={colored_ply.name}")
    return agent_request


# --------------------------------------------------------------------------- #
# apply
# --------------------------------------------------------------------------- #
def apply(prepdir, labels_json, threshold=0.5):
    prepdir = Path(prepdir)
    req = json.load(open(prepdir / "agent_request.json", encoding="utf-8"))
    slug = req["slug"]
    agent_labels = json.load(open(labels_json, encoding="utf-8"))
    part_labels = {str(k): v.get("label", v) if isinstance(v, dict) else v
                   for k, v in agent_labels.items()}

    results = [{
        "slug": slug,
        "n_raw_components": req["n_raw_components"],
        "n_significant_parts": req["n_significant"],
        "significance_threshold": req["significance_threshold"],
        "face_total": req["face_total"],
        "part_labels": part_labels,
    }]
    results_path = prepdir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    build = HERE / "build_semantic_tree.py"
    tree_dir = prepdir / "semantic_trees"
    cmd = [sys.executable, str(build),
           "--results", str(results_path),
           "--outdir", str(tree_dir),
           "--threshold", str(threshold)]
    print("[apply] calling build_semantic_tree.py ...")
    subprocess.run(cmd, check=True)

    # 使用语义树导出的层级近似色渲染模型：树上越近，模型颜色越近。
    tree_json = tree_dir / f"{slug}_semantic_tree.json"
    tree_data = json.load(open(tree_json, encoding="utf-8"))
    leaf_colors = tree_data.get("leaf_colors", {})
    proxy_path = prepdir / "render_proxy.npz"
    if proxy_path.exists():
        proxy = np.load(proxy_path)
        vertices = proxy["vertices"]
        faces = proxy["faces"]
        face_block = proxy["face_block"]
        face_rgb = np.tile(DUST_COLOR, (len(faces), 1))
        for block_id, label in part_labels.items():
            hx = leaf_colors.get(label, "#94a3b8").lstrip("#")
            rgb = np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.uint8)
            face_rgb[face_block == int(block_id)] = rgb
        views = _render_12views(vertices, faces, face_rgb, image_size=300)
        grid = _stitch(views, cols=4,
                       labels=[f"e{e:.0f}a{a:.0f}" for e, a in VIEW_GRID])
        panel = _add_title(grid, f"{slug} · hierarchical semantic colors",
                           "closer nodes in the semantic tree use closer colors")
        legend_items = sorted(leaf_colors.items())
        panel = _append_color_legend(panel, legend_items, cols=3)
        semantic_panel = prepdir / "panels" / f"{slug}_semantic_tree_colors_12views.png"
        panel.save(semantic_panel)
        req["semantic_color_panel"] = str(semantic_panel)
        req["semantic_leaf_colors"] = leaf_colors
        with open(prepdir / "agent_request.json", "w", encoding="utf-8") as f:
            json.dump(req, f, indent=2, ensure_ascii=False)

    print(f"[apply] done -> {tree_dir / 'semantic_tree_gallery.html'}")
    return results_path


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--mesh", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--slug", default="blind")
    p.add_argument("--sig", type=float, default=0.02)
    p.add_argument("--render-max-faces", type=int, default=120000,
                   help="CC-preserving visualization proxy face budget")

    a = sub.add_parser("apply")
    a.add_argument("--prepdir", required=True)
    a.add_argument("--labels", required=True)
    a.add_argument("--threshold", type=float, default=0.5)

    args = ap.parse_args()
    if args.phase == "prep":
        prep(args.mesh, args.outdir, args.slug, args.sig, args.render_max_faces)
    else:
        apply(args.prepdir, args.labels, args.threshold)


if __name__ == "__main__":
    main()
