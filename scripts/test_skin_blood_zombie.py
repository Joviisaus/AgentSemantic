"""对比 optimize_skin 前后蒙皮与行走关键帧变形（Blood_Zombie）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geometry.semantic_rig import build_semantic_skeleton, resolve_rig_spec, build_animations
from geometry.skin_optimize import (
    SkinOptimizeConfig,
    optimize_semantic_skin,
    weight_edge_discontinuity,
)
from pipeline.semantic_rig_cli import _load_mesh, _load_labels


def _weight_stats(w: np.ndarray, thr: float = 0.14) -> dict:
    nz = (w > thr).sum(axis=1)
    dom = w.max(axis=1)
    return {
        "mean_bones": float(nz.mean()),
        "min_bones": int(nz.min()),
        "max_bones": int(nz.max()),
        "mean_dominant": float(dom.mean()),
        "frac_single_dominant_0.9": float((dom > 0.9).mean()),
        "frac_multi_0.1": float((w > 0.1).sum(axis=1).mean()),
    }


def main() -> int:
    mesh = Path(r"C:\Users\jirenouyang\Downloads\Blood_Zombie.stl")
    labels = ROOT / "output" / "Blood_Zombie" / "vlm_full" / "face_semantic_labels.refined.json"
    rig_dir = ROOT / "output" / "Blood_Zombie" / "vlm_full" / "rig"
    if not labels.is_file():
        labels = ROOT / "output" / "Blood_Zombie" / "vlm_full" / "face_semantic_labels.json"
    if not mesh.is_file():
        print(f"missing mesh: {mesh}")
        return 1
    if not labels.is_file():
        print(f"missing labels: {labels}")
        return 1

    verts, faces = _load_mesh(mesh)
    parts, face_idx = _load_labels(labels)
    with open(rig_dir / "rig_design.json", encoding="utf-8") as f:
        rd = json.load(f)
    spec = rd.get("spec", rd)
    resolved = resolve_rig_spec(spec, parts)

    skel = build_semantic_skeleton(
        verts, faces, face_idx, parts,
        parent_override=resolved.parent_override,
        axis_override=resolved.axis_override,
    )
    w_build = skel.weights.copy()
    st0 = _weight_stats(w_build)

    with open(rig_dir / "anim_design.json", encoding="utf-8") as f:
        ad = json.load(f)
    anim_spec = ad.get("spec", ad)
    anims = build_animations(resolve_rig_spec(anim_spec, parts), skel)
    angles = dict(anims[0].keyframes[1].angles) if anims else {}

    def seam_jump(w: np.ndarray) -> float:
        posed = skel.pose_from_angles(angles)
        edges = set()
        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, v in ((a, b), (b, c), (c, a)):
                if u > v:
                    u, v = v, u
                edges.add((u, v))
        jumps = []
        for u, v in edges:
            jumps.append(np.linalg.norm(posed[u] - posed[v]))
        return float(np.mean(jumps))

    skel.weights = w_build
    jump0 = seam_jump(w_build)

    cfg = SkinOptimizeConfig(weight_mode="smooth", verbose=True, optimize_joints=False)
    optimize_semantic_skin(skel, cfg, animations=anims)
    w_opt = skel.weights.copy()
    st1 = _weight_stats(w_opt)
    jump1 = seam_jump(w_opt)

    diff = np.abs(w_opt - w_build)
    print("\n=== build vs optimized ===")
    print("build ", st0)
    print("opt   ", st1)
    print(f"|w|_mean_delta={diff.mean():.6f} max={diff.max():.4f}")
    print(f"edge_len_mean posed@kf1: build={jump0:.4f} opt={jump1:.4f}")
    print(
        f"weight_edge_jump: build={weight_edge_discontinuity(w_build, faces):.4f} "
        f"opt={weight_edge_discontinuity(w_opt, faces):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
