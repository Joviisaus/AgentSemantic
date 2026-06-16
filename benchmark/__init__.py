"""3D CoMPaT-200 coarse semantic part-segmentation benchmark.

Compares AgentSematic (mesh + VLM semantic search) against point-cloud
segmentation baselines (CoSMo3D, PointCLIP_V2, UniM-OV3D, Pointcept) on a
unified per-point mIoU metric.

Pipeline:
    1. export_gt.py   (cosmo3d env): .pt tensors -> numpy .npz ground truth
    2. methods/*      : each method writes per-point predictions (.npy) aligned
                        to the GT point order (length 5000, -1 = unlabeled)
    3. evaluate.py    : load GT + preds -> mIoU report

All paths are resolved relative to this package via :mod:`benchmark.compat200`.
"""
