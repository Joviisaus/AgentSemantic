"""
My (the VLM's) hand-painted part masks for the 15 tiny-subset shapes.

For each key I look at work/<key>/overview.png (plain gray point cloud rendered
from the chosen important view) and paint, per part, the pixel region I judge to
belong to that part. Regions are expressed as image-space primitives at 720x720:
  ("poly", [(x,y),...]) | ("ellipse",(cx,cy,rx,ry)) | ("rect",(x0,y0,x1,y1))

Painting order matters: earlier = background, later = foreground (drawn on top).
Each part gets its own single-channel mask PNG in work/<key>/mask_<partidx>.png.

part index == index in meta.json['part_names'] (0-based), matching GT labels.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

WORK = Path(r"E:/Proj/AgentSematic/benchmark/ai_vlm_exp/work")
S = 720

# For each key: ordered list of (part_index, [regions]).
# "silhouette" as a region means: fill the whole visible-point silhouette
# (computed from the render buffers) -- used for the dominant/background part.
MASKS = {
 "airplane__coarse_b00_004": [  # parts: 0 engine,1 body,2 wheel
    (1, ["silhouette"]),                                   # body = whole plane
    (0, [("ellipse",(300,345,18,14))]),                    # engine speck (small)
    (2, [("ellipse",(455,438,26,26))]),                    # wheel cluster bottom
 ],
 "belt__coarse_b2f_025": [  # 0 body,1 buckle
    (0, ["silhouette"]),                                   # whole strap
    (1, [("poly",[(150,255),(340,205),(390,280),(250,360),(140,320)])]),  # buckle upper-left
 ],
 "bed__coarse_b04_017": [  # 0 base,1 body
    (1, ["silhouette"]),                                   # mattress slab = almost everything
    (0, [("ellipse",(455,565,16,16)),("ellipse",(130,410,12,12)),
         ("ellipse",(560,360,12,12))]),                    # base = visible leg tips only
 ],
 "balance_scale__coarse_b2d_026": [  # 0 base,1 body,2 string
    (1, ["silhouette"]),                                   # post+beam
    (0, [("poly",[(360,385),(600,435),(430,550),(175,505)])]),  # flat ground platform
    (2, [("ellipse",(305,300,32,95)),("ellipse",(475,300,45,80))]),  # strings drawn last -> pans win
 ],
 "bench__coarse_b05_030": [  # 0 base,1 seat
    (1, ["silhouette"]),                                   # seat + back
    (0, [("rect",(150,360,560,470))]),                     # legs region bottom
 ],
 "arcade_machine__coarse_b2b_026": [  # 0 base,1 body,2 glass_panel,3 door
    (1, ["silhouette"]),                                   # body dominant
    (2, [("poly",[(255,165),(490,155),(475,345),(300,350)])]),  # glass screen upper band
    (3, [("poly",[(380,445),(490,470),(470,565),(370,545)])]),  # door diagonal lower-right
    (0, [("poly",[(220,515),(490,510),(470,625),(300,630)])]),  # base bottom (drawn last)
 ],
 "axe__coarse_b2c_038": [  # 0 body(head),1 handle
    (1, ["silhouette"]),                                   # handle shaft dominant length
    (0, [("poly",[(250,250),(380,240),(400,340),(280,360),(240,330)])]),  # head wedge top
 ],
 "bath_tub__coarse_b2e_029": [  # 0 body,1 spout
    (0, ["silhouette"]),                                   # basin
    (1, [("ellipse",(360,205,10,30))]),                    # spout speck top-center
 ],
 "bbq_grill__coarse_b03_00c": [  # 0 base,1 grate,2 handle
    (0, ["silhouette"]),                                   # body/base dominant
    (1, [("poly",[(250,255),(470,235),(490,300),(270,320)])]),  # grate lid top
    (2, [("ellipse",(500,270,16,10))]),                    # handle small
 ],
 "bag__coarse_b01_002": [  # 0 body,1 handle
    (0, ["silhouette"]),                                   # body rectangle
    (1, [("poly",[(540,235),(640,300),(650,400),(590,410),(560,340),(520,270)])]),  # handle loop
 ],
 "air_conditioning_unit__coarse_b2a_01d": [  # 0 body,1 condenser_unit
    (0, ["silhouette"]),                                   # body dominant
    (1, [("ellipse",(360,300,70,55))]),                    # condenser patch middle
 ],
 "bird_house__coarse_b07_003": [  # 0 body,1 roof
    (0, ["silhouette"]),                                   # body dominant
    (1, [("rect",(190,150,560,235))]),                     # roof top band
 ],
 "bicycle__coarse_b06_007": [  # 0 body,1 seat,2 wheel
    (0, ["silhouette"]),                                   # frame
    (2, [("ellipse",(200,445,110,110)),("ellipse",(475,350,98,98))]),  # rear + front wheels
    (1, [("ellipse",(285,235,28,20))]),                    # seat nub top-left (drawn last)
 ],
 "basket__coarse_b02_000": [  # 0 body,1 lid
    (0, ["silhouette"]),                                   # body
    (1, [("poly",[(210,235),(520,235),(530,300),(200,300)])]),  # lid top rim
 ],
 "binoculars__coarse_b30_024": [  # 0 body,1 lens
    (0, ["silhouette"]),                                   # body
    (1, [("ellipse",(320,300,45,45))]),                    # lens patch
 ],
}


def load_silhouette(key):
    """Dilated silhouette of visible points, as a boolean HxW."""
    buf = np.load(WORK / key / "buffers.npz")
    zidx = buf["zidx"]
    sil = zidx >= 0
    # dilate a bit to fill splat gaps
    from scipy.ndimage import binary_dilation  # noqa
    return sil


def dilate(mask, it=3):
    m = mask.copy()
    for _ in range(it):
        p = np.zeros_like(m)
        p[1:,:]  |= m[:-1,:]; p[:-1,:] |= m[1:,:]
        p[:,1:]  |= m[:,:-1]; p[:,:-1] |= m[:,1:]
        m = m | p
    return m


def draw_regions(regions, sil):
    img = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(img)
    use_sil = False
    for reg in regions:
        if reg == "silhouette":
            use_sil = True
            continue
        kind, spec = reg
        if kind == "poly":
            d.polygon(spec, fill=255)
        elif kind == "rect":
            d.rectangle(spec, fill=255)
        elif kind == "ellipse":
            cx, cy, rx, ry = spec
            d.ellipse((cx-rx, cy-ry, cx+rx, cy+ry), fill=255)
    arr = np.array(img) > 0
    if use_sil:
        arr = arr | sil
    return arr


def main():
    for key, parts in MASKS.items():
        buf = np.load(WORK / key / "buffers.npz")
        sil = dilate(buf["zidx"] >= 0, it=3)
        for pidx, regions in parts:
            m = draw_regions(regions, sil)
            # intersect every mask with the silhouette: only points can be labeled
            m = m & sil
            Image.fromarray((m*255).astype(np.uint8)).save(
                WORK / key / f"mask_{pidx}.png")
        print(f"{key}: painted {len(parts)} part masks")


if __name__ == "__main__":
    main()
