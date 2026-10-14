"""
Pure-numpy point-cloud renderer using our Camera. Produces:
  - overview PNG (points splatted, painter's algorithm back-to-front),
  - a per-pixel nearest-point-index buffer (z-buffer) as .npz, so back-projection
    of a painted mask is exact and occlusion-aware.

Also renders a GT-colored reference (each part a distinct color) for me to look at
while painting -- but the mask I paint is drawn on the plain gray overview so I am
not copying GT; the GT color image is only shown separately for sanity/debug.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image

from camera import Camera

# distinct colors for GT parts (debug reference only)
PART_COLORS = np.array([
    [ 31,119,180],[255,127, 14],[ 44,160, 44],[214, 39, 40],
    [148,103,189],[140, 86, 75],[227,119,194],[127,127,127],
    [188,189, 34],[ 23,190,207],
], dtype=np.uint8)

GRAY = np.array([170,170,175], np.uint8)
BG = np.array([255,255,255], np.uint8)


def _splat_radius(n_visible: int) -> int:
    if n_visible > 4000: return 4
    if n_visible > 2000: return 5
    if n_visible > 800:  return 6
    return 8


def render(points: np.ndarray, cam: Camera, colors: np.ndarray = None,
           radius: int = None):
    """Return (rgb HxWx3 uint8, zbuf_idx HxW int32 nearest point index or -1)."""
    W, H = cam.width, cam.height
    u, v, depth = cam.project(points)
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (depth > 0)
    idx_all = np.nonzero(inb)[0]
    if radius is None:
        radius = _splat_radius(len(idx_all))

    rgb = np.tile(BG, (H, W, 1)).astype(np.uint8)
    zbuf = np.full((H, W), np.inf)
    zidx = np.full((H, W), -1, dtype=np.int32)

    if colors is None:
        colors = np.tile(GRAY, (len(points), 1))

    # draw nearest-first so z-buffer keeps closest; use a disk stamp
    order = idx_all[np.argsort(depth[idx_all])]  # near -> far
    off = [(dx, dy) for dx in range(-radius, radius+1)
                    for dy in range(-radius, radius+1)
                    if dx*dx + dy*dy <= radius*radius]
    off = np.array(off, dtype=np.int64)
    for i in order:
        d = depth[i]
        cx, cy = ui[i], vi[i]
        px = cx + off[:,0]; py = cy + off[:,1]
        m = (px>=0)&(px<W)&(py>=0)&(py<H)
        px, py = px[m], py[m]
        closer = d < zbuf[py, px]
        px2, py2 = px[closer], py[closer]
        zbuf[py2, px2] = d
        zidx[py2, px2] = i
        rgb[py2, px2] = colors[i]
    return rgb, zidx, (u, v, depth, inb)


def render_and_save(points, cam, out_dir: Path, gt_labels=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # plain overview (all gray) -- this is what I paint on
    rgb_gray, zidx, proj = render(points, cam, colors=None)
    Image.fromarray(rgb_gray).save(out_dir / "overview.png")
    np.savez_compressed(out_dir / "buffers.npz",
                        zidx=zidx,
                        u=proj[0], v=proj[1], depth=proj[2], inb=proj[3])
    # GT-colored debug (separate, for sanity only)
    if gt_labels is not None:
        cols = np.tile(GRAY, (len(points),1)).astype(np.uint8)
        for p in np.unique(gt_labels):
            if p < 0: continue
            cols[gt_labels==p] = PART_COLORS[int(p) % len(PART_COLORS)]
        rgb_gt,_,_ = render(points, cam, colors=cols)
        Image.fromarray(rgb_gt).save(out_dir / "gt_debug.png")
    return zidx, proj
