"""
NEO 特征空间后端桥（主环境 py3.14 侧）。

NEO 依赖 torch2.6 / py3.12 / g2pt，与主项目（py3.14）的依赖栈不兼容，因此 NEO 推理
运行在独立环境（``.conda-neo``）里。本模块把「点云 → 低频 LBO 特征向量」封装成一次
子进程调用：

    主进程: 点云 ndarray
      → 写临时 points.npy
      → subprocess: <neo_python> tools/neo_infer.py --ckpt ... --points ... --out eigvecs.npy
      → 读回 eigvecs.npy (N, k)
      → 交给 geometry.pointcloud_cluster.spectral_cluster_points(precomputed_eigvecs=...)

环境定位优先级（均可被显式参数 / 环境变量覆盖）：
  - NEO python:  ``--neo-python`` / env ``NEO_PYTHON`` / 默认 ``.conda-neo/bin/python``
  - 权重 ckpt:   ``--neo-ckpt``   / env ``NEO_CKPT``
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_NEO_PYTHON = _REPO_ROOT / ".conda-neo" / "bin" / "python"
_NEO_INFER_SCRIPT = _REPO_ROOT / "tools" / "neo_infer.py"


class NEOBackendError(RuntimeError):
    pass


def resolve_neo_python(explicit: Optional[str] = None) -> Path:
    cand = explicit or os.environ.get("NEO_PYTHON") or str(_DEFAULT_NEO_PYTHON)
    p = Path(cand)
    if not p.is_file():
        raise NEOBackendError(
            f"找不到 NEO 环境的 python: {p}\n"
            f"请先创建 NEO 环境（conda create -p .conda-neo python=3.12 + 装 NEO 依赖），"
            f"或用 --neo-python / 环境变量 NEO_PYTHON 指定。"
        )
    return p


def resolve_neo_ckpt(explicit: Optional[str] = None) -> Path:
    cand = explicit or os.environ.get("NEO_CKPT")
    if not cand:
        raise NEOBackendError(
            "未指定 NEO 预训练权重。请用 --neo-ckpt 或环境变量 NEO_CKPT 指定 .ckpt 路径。\n"
            "权重下载（作者 Google Drive）："
            "https://drive.google.com/drive/folders/12026qTECiGywESPz6WSbpgD8BPMd41Ha"
        )
    p = Path(cand)
    if not p.is_file():
        raise NEOBackendError(f"NEO 权重文件不存在: {p}")
    return p


def neo_eigenspace(
    points: np.ndarray,
    *,
    ckpt: Optional[str] = None,
    neo_python: Optional[str] = None,
    k: int = 0,
    device: str = "cpu",
    no_mass: bool = False,
    normalize: bool = True,
    timeout_seconds: int = 1200,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    调用 NEO 子进程，返回 (eigvecs[N,k], eigvals[k] 或 None)。

    eigvecs 按特征值升序排列，第 0 列为近零常数模——与 scipy 后端约定一致，
    可直接作为 ``spectral_cluster_points(precomputed_eigvecs=...)`` 的输入。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"points 必须是 (N,3)，得到 {pts.shape}")

    py = resolve_neo_python(neo_python)
    ckpt_path = resolve_neo_ckpt(ckpt)

    with tempfile.TemporaryDirectory(prefix="neo_infer_") as td:
        td_path = Path(td)
        pts_npy = td_path / "points.npy"
        evec_npy = td_path / "eigvecs.npy"
        eval_npy = td_path / "eigvals.npy"
        np.save(pts_npy, pts[:, :3].astype(np.float32))

        cmd = [
            str(py),
            str(_NEO_INFER_SCRIPT),
            "--ckpt", str(ckpt_path),
            "--points", str(pts_npy),
            "--out", str(evec_npy),
            "--eigvals-out", str(eval_npy),
            "--k", str(int(k)),
            "--device", device,
        ]
        if no_mass:
            cmd.append("--no-mass")
        if not normalize:
            cmd.append("--no-normalize")

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise NEOBackendError(f"NEO 推理超时（>{timeout_seconds}s）") from exc

        if proc.returncode != 0:
            raise NEOBackendError(
                f"NEO 推理子进程失败（exit {proc.returncode}）。\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout[-2000:]}\n"
                f"STDERR:\n{proc.stderr[-2000:]}"
            )

        if not evec_npy.is_file():
            raise NEOBackendError(
                f"NEO 推理未产出 {evec_npy}。STDOUT:\n{proc.stdout[-1000:]}"
            )
        eigvecs = np.load(evec_npy).astype(np.float64)
        eigvals = np.load(eval_npy).astype(np.float64) if eval_npy.is_file() else None

    if eigvecs.shape[0] != pts.shape[0]:
        raise NEOBackendError(
            f"NEO 返回特征向量行数 {eigvecs.shape[0]} 与点数 {pts.shape[0]} 不一致"
        )
    return eigvecs, eigvals
