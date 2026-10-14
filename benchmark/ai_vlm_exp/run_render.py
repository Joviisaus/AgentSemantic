"""Render all 15 tiny-subset point clouds from a chosen important view."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from camera import Camera
from render import render, render_and_save

GT_DIR = Path(r"E:/Proj/AgentSematic/benchmark/data/gt")
OUT = Path(r"E:/Proj/AgentSematic/benchmark/ai_vlm_exp/work")

KEYS = [
    "airplane__coarse_b00_004","belt__coarse_b2f_025","bed__coarse_b04_017",
    "balance_scale__coarse_b2d_026","bench__coarse_b05_030",
    "arcade_machine__coarse_b2b_026","axe__coarse_b2c_038",
    "bath_tub__coarse_b2e_029","bbq_grill__coarse_b03_00c","bag__coarse_b01_002",
    "air_conditioning_unit__coarse_b2a_01d","bird_house__coarse_b07_003",
    "bicycle__coarse_b06_007","basket__coarse_b02_000","binoculars__coarse_b30_024",
]

# candidate views (elev, azim): 3/4 fronts + top + sides -> pick max visible
CANDS = [(20,45),(20,135),(20,-45),(20,-135),(35,45),(35,-45),(10,0),(10,90),(55,30)]


def count_visible(points, cam):
    _, zidx, _ = render(points, cam)
    return len(np.unique(zidx[zidx>=0]))


def main():
    for key in KEYS:
        z = np.load(GT_DIR / f"{key}.npz", allow_pickle=True)
        pts = z["points"].astype(np.float64)
        labels = z["labels"].astype(np.int64)
        pnames = [str(x) for x in z["part_names"].tolist()]

        best = None
        for (el, az) in CANDS:
            cam = Camera(pts, el, az)
            vis = count_visible(pts, cam)
            if best is None or vis > best[0]:
                best = (vis, el, az, cam)
        vis, el, az, cam = best
        out_dir = OUT / key
        render_and_save(pts, cam, out_dir, gt_labels=labels)
        meta = {"key": key, "category": str(z["category"]),
                "part_names": pnames, "n_points": int(len(pts)),
                "view": {"elev": el, "azim": az, "visible_pts": int(vis)},
                "camera": cam.to_dict()}
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"{key:42s} view=({el:>4},{az:>5}) visible={vis}/{len(pts)} parts={pnames}")


if __name__ == "__main__":
    main()
