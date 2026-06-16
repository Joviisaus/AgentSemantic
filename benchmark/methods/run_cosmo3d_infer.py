"""CoSMo3D inference -> per-point predictions on the CoMPaT-200 coarse subset.

Run under the **cosmo3d** env, with cwd = E:\\Proj\\CoSMo3D (the repo uses
relative resource paths) and AgentSematic on PYTHONPATH::

    conda run -n cosmo3d python E:\\Proj\\AgentSematic\\benchmark\\methods\\run_cosmo3d_infer.py ^
        --subset tiny --checkpoint dataset/checkpoints/ours_final.pth

Reuses CoSMo3D's Eval3dcom loader + PointSemSegWithDecoder, reproduces the
reference prediction (net_out = model(data)[0], NN-upsample to full cloud), and
writes 0-based per-point labels (-1 = unlabeled) to
``benchmark/results/cosmo3d/<key>.npy`` aligned to points.pt order.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

AGENTSEMATIC = Path(r"E:\Proj\AgentSematic")
if str(AGENTSEMATIC) not in sys.path:
    sys.path.insert(0, str(AGENTSEMATIC))

from benchmark.compat200 import safe_key, RESULTS_DIR  # noqa: E402


def _scalar(v):
    if isinstance(v, (list, tuple)):
        return v[0]
    return v


@torch.no_grad()
def predict_full(model, data, n_chunks: int = 20) -> np.ndarray:
    """Replicate CoSMo3D compute_3d_iou_upsample prediction (panoptic=False).

    Returns per-point labels over xyz_full (points.pt order): 0 = unlabeled,
    1..M = part index (1-based, matching mask_labels.txt line order).
    """
    temperature = float(np.exp(model.ln_logit_scale.item()))
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and "full" not in k:
            data[k] = v.cuda(non_blocking=True)
    if "mask_offset" not in data:
        n_masks = int(data["label_embeds"].shape[0])
        data["mask_offset"] = torch.tensor([n_masks], device=data["label_embeds"].device)

    model_output = model(data)
    net_out = model_output[0] if isinstance(model_output, (tuple, list)) else model_output

    label_embeds = data["label_embeds"]
    logits = net_out @ label_embeds.T  # (n_sub, n_parts)
    logits0 = torch.cat([torch.zeros(logits.shape[0], 1).cuda(), logits], dim=1)
    pred_softmax = torch.nn.Softmax(dim=1)(logits0 * temperature)

    xyz_sub = data["coord"]
    xyz_full = data["xyz_full"].squeeze()
    chunk_len = xyz_full.shape[0] // n_chunks + 1
    idx_list = []
    for i in range(n_chunks):
        cur = xyz_full[chunk_len * i: chunk_len * (i + 1)]
        if cur.shape[0] == 0:
            continue
        d = ((xyz_sub.unsqueeze(0) - cur.cuda().unsqueeze(1)) ** 2).sum(dim=-1) ** 0.5
        idx_list.append(torch.min(d, 1)[1])
    all_nn = torch.cat(idx_list, dim=0)
    pred_full = pred_softmax[all_nn].argmax(dim=1).cpu().numpy()  # 0=unlabeled
    return pred_full.astype(np.int64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="CoSMo3D inference -> per-point preds")
    ap.add_argument("--subset", default="tiny")
    ap.add_argument("--checkpoint", default="dataset/checkpoints/ours_final.pth")
    ap.add_argument("--data-root", default="dataset/test_3dcompat200")
    ap.add_argument("--datatype", default="coarse", choices=["coarse", "fine"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    from model.data.data import Eval3dcom, collate_fn
    from model.evaluation.utils import set_seed
    from release_module.network.canoncolor_bbox_pre import PointSemSegWithDecoder
    from torch.utils.data import DataLoader

    set_seed(123)
    torch.manual_seed(123)

    # subset keys (read the list AgentSematic wrote)
    subset_txt = AGENTSEMATIC / "benchmark" / "data" / "subsets" / f"{args.subset}.txt"
    wanted = {ln.strip() for ln in subset_txt.read_text(encoding="utf-8").splitlines() if ln.strip()}
    categories = sorted({k.split("__", 1)[0] for k in wanted})
    print(f"[cosmo3d] subset={args.subset} keys={len(wanted)} categories={len(categories)}")

    print(f"[cosmo3d] loading model from {args.checkpoint}")
    dummy = argparse.Namespace()
    model = PointSemSegWithDecoder(args=dummy)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state_dict"], strict=True)
    model = model.eval().cuda()

    out_dir = RESULTS_DIR / "cosmo3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    for category in categories:
        cat_dir = os.path.join(args.data_root, category)
        if not os.path.isdir(cat_dir):
            print(f"  [skip] no dir for category {category}")
            continue
        dataset = Eval3dcom(data_root=args.data_root, category=category,
                            datatype=args.datatype, apply_rotation=False, decorated=True)
        if len(dataset) == 0:
            continue
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
        for data in loader:
            stem = _scalar(data.get("sample_stem"))
            key = safe_key(category, str(stem))
            if key not in wanted:
                continue
            out_path = out_dir / f"{key}.npy"
            if out_path.is_file() and not args.overwrite:
                print(f"  [skip] {key} (exists)")
                done += 1
                continue
            pred_full = predict_full(model, data, n_chunks=20)   # 0=unlabeled, 1..M
            pred0 = pred_full - 1                                 # -1=unlabeled, 0..M-1
            np.save(out_path, pred0.astype(np.int64))
            n_lab = int((pred0 >= 0).sum())
            print(f"  [ok] {key}: {pred0.shape[0]} pts, {n_lab} labeled")
            done += 1
    print(f"[cosmo3d] wrote/found {done} predictions -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
