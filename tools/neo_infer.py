"""
NEO 推理适配脚本——**在 NEO 的隔离环境（third_party/NEO + .conda-neo, py3.12）里运行**。

输入一个点云 .npy（(N,3)），用预训练的 NEO（Neural Eigenspace Operator）前馈预测
低频 Laplace-Beltrami 特征空间，经 Rayleigh-Ritz 投影得到特征对，落盘为 .npy。

主流程（py3.14 主环境）通过子进程调用本脚本（见 geometry/neo_backend.py），
以此跨越「NEO 需要 torch2.6/py3.12」与「主项目用 py3.14」的依赖鸿沟。

CPU 友好：forward(return_time=False) 不触发 torch.cuda.synchronize；
不依赖 triton / liger-kernel（g2pt 的 norm/softmax 会自动回退到纯 PyTorch）。

用法：
    python tools/neo_infer.py \
        --ckpt /path/to/model.ckpt \
        --points /tmp/points.npy \
        --out   /tmp/eigvecs.npy \
        [--eigvals-out /tmp/eigvals.npy] [--k 32] [--device cpu] [--no-mass] [--no-normalize]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _add_neo_to_path() -> None:
    """把 third_party/NEO/src 加进 sys.path，使 ``import g2pt`` 可用（无需 pip install）。"""
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    neo_src = repo_root / "third_party" / "NEO" / "src"
    if not neo_src.is_dir():
        raise SystemExit(f"[neo_infer] 找不到 NEO 源码: {neo_src}（git submodule 是否已 checkout？）")
    sys.path.insert(0, str(neo_src))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="NEO 点云特征空间推理（输出 eigvecs.npy）")
    parser.add_argument("--ckpt", required=True, help="NEO 预训练权重 .ckpt")
    parser.add_argument("--points", required=True, help="输入点云 .npy，形状 (N,3)")
    parser.add_argument("--out", required=True, help="输出特征向量 .npy，形状 (N,k)")
    parser.add_argument("--eigvals-out", default=None, help="可选：输出特征值 .npy，形状 (k,)")
    parser.add_argument("--k", type=int, default=0, help="特征向量个数；0 = 用模型 targ_dim_model")
    parser.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    parser.add_argument("--no-mass", action="store_true", help="消融 mass-aware attention")
    parser.add_argument("--no-normalize", action="store_true", help="不做单位球归一化（默认会做）")
    args = parser.parse_args(argv)

    _add_neo_to_path()

    import torch  # noqa: E402  延迟到 path 设置后
    from g2pt.training.pretrain import PretrainTraining  # noqa: E402
    from g2pt.utils.gev import solve_gev_from_subspace  # noqa: E402
    from g2pt.data.transforms import normalize_pc  # noqa: E402
    from robust_laplacian import point_cloud_laplacian  # noqa: E402
    import scipy.sparse as sp  # noqa: E402

    pc_raw = np.load(args.points)
    pc_raw = np.asarray(pc_raw, dtype=np.float64)
    if pc_raw.ndim != 2 or pc_raw.shape[1] < 3:
        raise SystemExit(f"[neo_infer] points 形状非法: {pc_raw.shape}，需要 (N,>=3)")
    pc = pc_raw[:, :3]
    if not args.no_normalize:
        pc = normalize_pc(pc, 0)

    # 点云 Laplacian（与 NEO infer.py 一致）。
    L, M = point_cloud_laplacian(np.ascontiguousarray(pc))
    mass = M.diagonal().astype(np.float32)
    L = sp.csr_matrix(L).astype(np.float64)
    M = sp.csr_matrix(M).astype(np.float64)

    # 加载模型（CPU）。
    model = PretrainTraining.load_from_checkpoint(
        args.ckpt, weights_only=False, strict=False
    )
    device = torch.device(args.device)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    k = args.k if args.k > 0 else int(model.targ_dim_model)

    pc_tensor = torch.tensor(pc, dtype=torch.float32, device=device).view(1, -1, 3)
    mass_tensor = torch.tensor(mass, dtype=torch.float32, device=device).view(1, -1, 1)

    with torch.no_grad():
        outputs, _pred_original = model.forward(
            pc_tensor, mass_tensor, args.no_mass, return_time=False
        )
    subspace_vector = outputs[0].float().cpu().numpy()  # (N, d)

    # Rayleigh-Ritz：把离散 Laplacian 投影到子空间求小特征对（CPU numpy 路径）。
    eigvals, eigvecs = solve_gev_from_subspace(
        subspace_vector, pc_tensor[0], mass_tensor[0], k=k, L=L, M=M
    )
    eigvals = np.asarray(eigvals, dtype=np.float64)
    eigvecs = np.asarray(eigvecs, dtype=np.float64)

    # 按特征值升序排列，保证第 0 列是近零常数模（与 scipy 后端约定一致）。
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, eigvecs.astype(np.float32))
    if args.eigvals_out:
        np.save(Path(args.eigvals_out), eigvals.astype(np.float32))

    print(
        f"[neo_infer] OK N={pc.shape[0]} k={eigvecs.shape[1]} "
        f"-> {out_path} (eigvals[:4]={np.round(eigvals[:4], 5).tolist()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
