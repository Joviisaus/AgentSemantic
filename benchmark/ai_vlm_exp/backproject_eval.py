"""
Back-project my painted 2D masks to 3D per-point labels (occlusion-aware),
then compute mIoU against GT with CoSMo3D's skip convention.

Rule per point:
  1. Forward-project all 5000 points with the SAME camera used for rendering.
  2. A point is "directly visible" if it is the nearest point (z-buffer winner)
     within a small pixel tolerance at its projected pixel.
  3. For a directly-visible point at pixel (u,v): its predicted part = the mask
     that covers (u,v). Priority when masks overlap: the part painted LATER in
     paint_masks.MASKS wins (smaller foreground parts override the background).
  4. Occluded / not-directly-visible points get no 2D evidence -> label by the
     nearest directly-labeled point in 3D (kNN propagation, k=1). This is the
     honest "blind-spot" fill: a single view cannot see them, so we propagate.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from camera import Camera
from paint_masks import MASKS, WORK

GT_DIR = Path(r"E:/Proj/AgentSematic/benchmark/data/gt")
IGNORE = -1


def cam_from_meta(meta, points):
    c = meta["camera"]
    cam = Camera(points, c["elev"], c["azim"],
                 width=c["width"], height=c["height"],
                 fov_deg=c["fov_deg"], zoom=c["zoom"])
    return cam


def sample_score(pred, gt, part_names):
    pred = pred.astype(np.int64); gt = gt.astype(np.int64)
    labeled = gt != IGNORE
    acc = float(np.mean(pred[labeled] == gt[labeled])) if labeled.any() else 0.0
    ious = []
    per = {}
    for p in range(len(part_names)):
        inter = int(np.sum((pred == p) & (gt == p)))
        union = int(np.sum((pred == p) | (gt == p)))
        if union == 0:
            per[part_names[p]] = None
            continue
        iou = inter / union
        ious.append(iou); per[part_names[p]] = iou
    miou = float(np.mean(ious)) if ious else 0.0
    return miou, acc, per


def predict_key(key):
    z = np.load(GT_DIR / f"{key}.npz", allow_pickle=True)
    pts = z["points"].astype(np.float64)
    gt = z["labels"].astype(np.int64)
    part_names = [str(x) for x in z["part_names"].tolist()]
    meta = json.load(open(WORK / key / "meta.json", encoding="utf-8"))
    cam = cam_from_meta(meta, pts)

    buf = np.load(WORK / key / "buffers.npz")
    zidx = buf["zidx"]                       # HxW nearest point idx or -1
    u, v, depth, inb = buf["u"], buf["v"], buf["depth"], buf["inb"]
    H, Wd = zidx.shape

    # load masks with paint priority: order in MASKS list; later wins
    order = [pidx for pidx, _ in MASKS[key]]
    masks = {}
    for pidx in order:
        m = np.array(Image.open(WORK / key / f"mask_{pidx}.png")) > 127
        masks[pidx] = m

    N = len(pts)
    pred = np.full(N, IGNORE, dtype=np.int64)

    ui = np.clip(np.round(u).astype(np.int64), 0, Wd - 1)
    vi = np.clip(np.round(v).astype(np.int64), 0, H - 1)

    # directly visible = this point owns its pixel in the z-buffer
    visible = np.zeros(N, dtype=bool)
    for i in np.nonzero(inb)[0]:
        if zidx[vi[i], ui[i]] == i:
            visible[i] = True

    # assign visible points by mask, later-in-order wins
    for i in np.nonzero(visible)[0]:
        px, py = ui[i], vi[i]
        lab = IGNORE
        for pidx in order:           # later overrides earlier
            if masks[pidx][py, px]:
                lab = pidx
        pred[i] = lab

    # propagate to occluded/unlabeled points via nearest labeled point in 3D
    labeled_mask = pred != IGNORE
    if labeled_mask.any() and (~labeled_mask).any():
        src = pts[labeled_mask]; src_lab = pred[labeled_mask]
        tgt = pts[~labeled_mask]
        # brute-force nearest (N small)
        # chunk to limit memory
        idx_tgt = np.nonzero(~labeled_mask)[0]
        step = 1024
        for s in range(0, len(tgt), step):
            chunk = tgt[s:s+step]
            d2 = ((chunk[:, None, :] - src[None, :, :]) ** 2).sum(-1)
            nn = d2.argmin(1)
            pred[idx_tgt[s:s+step]] = src_lab[nn]

    miou, acc, per = sample_score(pred, gt, part_names)
    n_vis = int(visible.sum())
    return {
        "key": key, "category": meta["category"], "part_names": part_names,
        "miou_skip": miou, "accuracy": acc, "part_ious": per,
        "num_parts": len(part_names), "n_visible": n_vis,
        "view": meta["view"]["view"] if "view" in meta.get("view", {}) else meta["view"],
    }, pred


def main():
    results = []
    for key in MASKS:
        r, pred = predict_key(key)
        np.save(WORK / key / "pred_labels.npy", pred)
        results.append(r)
        print(f"{key:42s} mIoU={r['miou_skip']:.4f} acc={r['accuracy']:.4f} "
              f"vis={r['n_visible']:4d} parts={r['num_parts']}")

    inst_miou = float(np.mean([r["miou_skip"] for r in results]))
    inst_acc = float(np.mean([r["accuracy"] for r in results]))
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["miou_skip"])
    per_cat = {c: float(np.mean(v)) for c, v in by_cat.items()}
    cat_miou = float(np.mean(list(per_cat.values())))

    summary = {
        "method": "ai_vlm_handpaint_singleview",
        "n": len(results),
        "instance_miou_skip": inst_miou,
        "category_miou_skip": cat_miou,
        "instance_accuracy": inst_acc,
        "per_category": per_cat,
        "per_sample": results,
    }
    out = WORK.parent / "ai_vlm_result.json"
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n=== AI-VLM (hand-paint, single important view) ===")
    print(f"instance mIoU(skip) = {inst_miou:.4f}")
    print(f"category mIoU(skip) = {cat_miou:.4f}")
    print(f"instance accuracy   = {inst_acc:.4f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
