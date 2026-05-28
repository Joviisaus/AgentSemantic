"""
SEG-MAT 集成：在 medoid 输出姿态 mesh 上运行分割流水线。

依赖 third_party/SEG-MAT（git submodule）与本地构建的 QMAT CLI。
initial.ma 会自动在常见路径中查找，无需命令行指定。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SEG_MAT_EXE = (
    PROJECT_ROOT
    / "third_party"
    / "SEG-MAT"
    / "exe program"
    / "SEG-MAT-Release"
    / "SEG-MAT.exe"
)
DEFAULT_QMAT_CLI = PROJECT_ROOT / "third_party" / "QMAT" / "build" / "Release" / "QMAT.exe"
DEFAULT_BASE_MAT_VERTICES = 2000


@dataclass
class SegMatResult:
    ok: bool
    work_dir: Path
    mesh_off: Optional[Path] = None
    initial_ma: Optional[Path] = None
    base_ma: Optional[Path] = None
    smat_ma: Optional[Path] = None
    seg_off: Optional[Path] = None
    num_segments: int = 0
    face_labels: List[int] = field(default_factory=list)
    message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def _win_path(path: Path) -> str:
    """SEG-MAT 在 Windows 上依赖反斜杠路径解析输出目录。"""
    return os.path.normpath(str(path.resolve()))


def export_off(vertices: np.ndarray, faces: np.ndarray, path: Path) -> Path:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(path), file_type="off")
    return path


def _resolve_executable(explicit: Optional[str | Path], default: Path, env_var: str) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    env = os.environ.get(env_var, "").strip()
    if env and Path(env).is_file():
        return Path(env)
    return default if default.is_file() else None


def _run_checked(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def _find_exported_ma(work_dir: Path) -> Path:
    matches = sorted(work_dir.glob("*___v_*___e_*___f_*.ma"))
    if not matches:
        raise FileNotFoundError(f"QMAT 未在 {work_dir} 生成简化后的 .ma 文件")
    return matches[-1]


def run_qmat_simplify(
    qmat_exe: Path,
    mesh_off: Path,
    initial_ma: Path,
    target_spheres: int,
    work_dir: Path,
    output_name: str,
) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    _run_checked(
        [str(qmat_exe), _win_path(mesh_off), _win_path(initial_ma), str(int(target_spheres))],
        cwd=work_dir,
        timeout=600,
    )
    out = _find_exported_ma(work_dir)
    dest = work_dir / f"{output_name}.ma"
    if dest.exists():
        dest.unlink()
    shutil.move(str(out), str(dest))
    return dest


def compute_mat_files(
    mesh_off: Path,
    work_dir: Path,
    qmat_exe: Path,
    initial_ma: Path,
    base_vertices: int = DEFAULT_BASE_MAT_VERTICES,
) -> Tuple[Path, Path]:
    """从完整 initial.ma 生成 SEG-MAT 所需的 base / structure MAT。"""
    with open(initial_ma, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split()
    if len(header) < 3:
        raise ValueError(f"无效的 .ma 文件头: {initial_ma}")
    initial_v = int(header[0])
    smat_vertices = max(int(round(0.001 * initial_v)), 15)

    base_dir = work_dir / "qmat_base"
    smat_dir = work_dir / "qmat_smat"
    if base_dir.exists():
        shutil.rmtree(base_dir)
    if smat_dir.exists():
        shutil.rmtree(smat_dir)

    base_ma = run_qmat_simplify(qmat_exe, mesh_off, initial_ma, base_vertices, base_dir, "base_mat")
    smat_ma = run_qmat_simplify(qmat_exe, mesh_off, initial_ma, smat_vertices, smat_dir, "smat_mat")
    return base_ma, smat_ma


def run_seg_mat(
    seg_mat_exe: Path,
    mesh_off: Path,
    base_ma: Path,
    smat_ma: Path,
    grow: float = 0.015,
    min_region: float = 0.002,
    prim: int = 0,
    timeout: float = 600,
) -> Path:
    mesh_off = Path(mesh_off)
    mesh_off_win = _win_path(mesh_off)
    _run_checked(
        [
            str(seg_mat_exe),
            "-m",
            mesh_off_win,
            "-b",
            _win_path(base_ma),
            "-s",
            _win_path(smat_ma),
            "-g",
            str(grow),
            "-n",
            str(min_region),
            "-p",
            str(prim),
        ],
        cwd=mesh_off.parent,
        timeout=timeout,
    )
    stem = mesh_off.stem
    seg_off = mesh_off.parent / f"{stem}_seg.off"
    if not seg_off.is_file():
        raise FileNotFoundError(f"SEG-MAT 未生成 {seg_off}")
    return seg_off


def parse_colored_seg_off(path: Path) -> Tuple[List[int], int]:
    """
    解析 SEG-MAT 输出的彩色 OFF（每面末尾 R G B 0），按颜色映射 segment id。
    返回 face_labels（与 mesh.faces 行序一致）与 segment 数量。
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        if first != "OFF":
            raise ValueError(f"非 OFF 文件: {path}")
        counts = f.readline().strip().split()
        n_verts, n_faces = int(counts[0]), int(counts[1])
        for _ in range(n_verts):
            f.readline()
        color_to_label: Dict[Tuple[int, int, int], int] = {}
        face_labels: List[int] = []
        for _ in range(n_faces):
            parts = f.readline().strip().split()
            if len(parts) < 7:
                face_labels.append(-1)
                continue
            rgb = (int(parts[4]), int(parts[5]), int(parts[6]))
            if rgb == (0, 0, 0):
                face_labels.append(-1)
                continue
            if rgb not in color_to_label:
                color_to_label[rgb] = len(color_to_label)
            face_labels.append(color_to_label[rgb])
    return face_labels, len(color_to_label)


def _mesh_stem(path: Optional[Path]) -> str:
    if not path:
        return ""
    return path.stem.strip() or "mesh"


def _valid_ma(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split()
        if not (len(header) >= 3 and header[0].isdigit()):
            return False
        from skeleton_mat import validate_ma_topology

        return validate_ma_topology(path)
    except OSError:
        return False


def find_initial_ma(
    mesh_off: Path,
    work_dir: Path,
    source_mesh: Optional[Path] = None,
    explicit: Optional[Path] = None,
) -> Optional[Path]:
    """
    自动查找完整 initial.ma，按优先级：
      1. 显式路径 / 环境变量 SEG_MAT_INITIAL_MA
      2. 输出目录缓存 medoid_pose/render_pose.ma、initial.ma
      3. 与 render_pose.off 同名的 .ma
      4. 与源 mesh 同目录：{stem}.ma、{stem}_initial.ma、{stem}_mat.ma
      5. config/mat/{stem}.ma、data/mat/{stem}.ma

    若均未命中，调用方可用蒙皮骨架导出 render_pose.ma（见 run_seg_mat_on_render_pose）。
    """
    candidates: List[Path] = []

    if explicit:
        candidates.append(Path(explicit))
    env_ma = os.environ.get("SEG_MAT_INITIAL_MA", "").strip()
    if env_ma:
        candidates.append(Path(env_ma))

    stem = _mesh_stem(source_mesh) if source_mesh else mesh_off.stem
    src_dir = source_mesh.parent if source_mesh else None

    candidates.extend(
        [
            work_dir / "render_pose.ma",
            work_dir / "initial.ma",
            mesh_off.with_suffix(".ma"),
        ]
    )
    if src_dir is not None:
        candidates.extend(
            [
                src_dir / f"{stem}.ma",
                src_dir / f"{stem}_initial.ma",
                src_dir / f"{stem}_mat.ma",
            ]
        )
    candidates.extend(
        [
            PROJECT_ROOT / "config" / "mat" / f"{stem}.ma",
            PROJECT_ROOT / "data" / "mat" / f"{stem}.ma",
        ]
    )

    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.is_absolute() or path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if _valid_ma(path):
            return path.resolve()
    return None


def _resolve_initial_ma(
    mesh_off: Path,
    work_dir: Path,
    initial_ma: Optional[Path],
    source_mesh: Optional[Path] = None,
) -> Optional[Path]:
    return find_initial_ma(mesh_off, work_dir, source_mesh, explicit=initial_ma)


def _export_initial_ma_from_skeleton(
    rig: Any,
    pose_vertices: np.ndarray,
    anim_name: str,
    anim_time: float,
    output_path: Path,
) -> Path:
    from mesh_distortion_analyzer import evaluate_skeleton_joint_positions
    from skeleton_mat import export_skeleton_initial_ma

    joint_positions = evaluate_skeleton_joint_positions(rig, anim_name, anim_time)
    return export_skeleton_initial_ma(rig, joint_positions, pose_vertices, output_path)


def run_seg_mat_on_render_pose(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_dir: str | Path,
    *,
    source_mesh: Optional[str | Path] = None,
    seg_mat_exe: Optional[str | Path] = None,
    qmat_exe: Optional[str | Path] = None,
    initial_ma: Optional[str | Path] = None,
    base_ma: Optional[str | Path] = None,
    smat_ma: Optional[str | Path] = None,
    base_vertices: int = DEFAULT_BASE_MAT_VERTICES,
    grow: float = 0.015,
    min_region: float = 0.002,
    mesh_name: str = "render_pose",
    skeleton: Any = None,
    skeleton_anim_name: str = "",
    skeleton_time: float = 0.0,
) -> SegMatResult:
    """
    在 medoid 输出姿态上执行 SEG-MAT 分割。
    initial.ma 自动查找；若无现成文件且提供 skeleton，则从蒙皮骨架导出。
    若提供 base_ma / smat_ma 则跳过 QMAT 简化。
    """
    work_dir = Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    result = SegMatResult(ok=False, work_dir=work_dir)
    src_mesh = Path(source_mesh) if source_mesh else None

    seg_exe = _resolve_executable(seg_mat_exe, DEFAULT_SEG_MAT_EXE, "SEG_MAT_EXE")
    if not seg_exe:
        result.message = "未找到 SEG-MAT.exe"
        return result

    mesh_off = export_off(vertices, faces, work_dir / f"{mesh_name}.off")
    result.mesh_off = mesh_off

    base_path = Path(base_ma) if base_ma else None
    smat_path = Path(smat_ma) if smat_ma else None

    if not (base_path and base_path.is_file() and smat_path and smat_path.is_file()):
        qmat = _resolve_executable(qmat_exe, DEFAULT_QMAT_CLI, "QMAT_EXE")
        init_ma = _resolve_initial_ma(
            mesh_off,
            work_dir,
            Path(initial_ma) if initial_ma else None,
            source_mesh=src_mesh,
        )
        if not init_ma and skeleton is not None and getattr(skeleton, "joint_nodes", None):
            try:
                skel_ma = work_dir / f"{mesh_name}.ma"
                _export_initial_ma_from_skeleton(
                    skeleton,
                    vertices,
                    skeleton_anim_name,
                    skeleton_time,
                    skel_ma,
                )
                if _valid_ma(skel_ma):
                    init_ma = skel_ma.resolve()
                    result.extra["initial_ma_source"] = "skeleton"
            except (OSError, ValueError) as exc:
                result.extra["skeleton_ma_error"] = str(exc)
        if not init_ma:
            result.message = "未找到 initial.ma 且无法从骨架导出，已跳过分割"
            return result
        if not qmat:
            result.message = "未找到 QMAT CLI"
            result.initial_ma = init_ma
            return result
        result.initial_ma = init_ma
        try:
            base_path, smat_path = compute_mat_files(
                mesh_off, work_dir, qmat, init_ma, base_vertices=base_vertices
            )
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            result.message = f"QMAT 简化 MAT 失败: {exc}"
            return result

    result.base_ma = base_path
    result.smat_ma = smat_path

    try:
        seg_off = run_seg_mat(seg_exe, mesh_off, base_path, smat_path, grow=grow, min_region=min_region)
    except (subprocess.CalledProcessError, OSError, FileNotFoundError) as exc:
        result.message = f"SEG-MAT 执行失败: {exc}"
        return result

    result.seg_off = seg_off
    try:
        face_labels, num_segments = parse_colored_seg_off(seg_off)
        result.face_labels = face_labels
        result.num_segments = num_segments
    except (OSError, ValueError) as exc:
        result.message = f"SEG-MAT 已运行但解析结果失败: {exc}"
        return result

    labels_path = work_dir / "face_segment_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_segments": num_segments,
                "num_faces": len(face_labels),
                "face_labels": face_labels,
                "seg_off": str(seg_off),
                "base_ma": str(base_path),
                "smat_ma": str(smat_path),
            },
            f,
            indent=2,
        )
    result.extra["labels_json"] = str(labels_path)
    result.ok = True
    src = result.extra.get("initial_ma_source")
    src_note = "（骨架导出）" if src == "skeleton" else ""
    result.message = f"SEG-MAT 完成: {num_segments} 个分割部件{src_note}, 输出 {seg_off.name}"
    return result
