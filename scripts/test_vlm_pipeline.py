"""
重构后的 4 包结构 smoke / 集成测试。

不调用任何 VLM API（不烧 token、不联网），覆盖：

    1.  4 个新顶层包 (vlm / geometry / rendering / pipeline) 全部可正常 import
        且公共符号齐全；
    2.  `python -m vlm.cli` 与 `python -m pipeline.cli` 都能展示 --help（验证
        薄转发与真正入口都活着）；
    3.  纯几何链路：在一个程序生成的小立方体 mesh 上跑：
            geometry.spectral_cluster.spectral_cluster_vertices
                + geometry.seed_split._extract_submesh
                + geometry.blocks.build_initial_blocks
            → 验证拆分后包能跑通；
    4.  纯渲染链路：在同一个立方体上跑：
            rendering.heatmap_renderer.render_segmentation_view
                + rendering.render_component.render_single_view_overview
                + rendering.view_thumbnails.render_thumbnails_panel
            → 验证 pyvista / PIL 可调，产物落到 output/smoke_test/。

通过 = exit 0，任何环节失败 = exit 1。

用法：

    python -m scripts.test_vlm_pipeline                # 全部 case
    python -m scripts.test_vlm_pipeline --imports-only # 只验 import
    python -m scripts.test_vlm_pipeline --no-render    # 跳过 pyvista 渲染部分
    python -m scripts.test_vlm_pipeline --cli-only     # 只验 CLI --help

CI 友好：所有产物都写到 output/smoke_test/，失败时把堆栈打到 stderr。
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

CHECK_OK = "[ OK ]"
CHECK_FAIL = "[FAIL]"
CHECK_SKIP = "[SKIP]"


def _run_case(name: str, fn: Callable[[], None]) -> Tuple[str, str, float, str]:
    """跑一个测试用例，返回 (name, status, seconds, detail)."""
    t0 = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 — 测试脚本要捕获一切
        elapsed = time.perf_counter() - t0
        detail = f"{type(exc).__name__}: {exc}"
        print(f"{CHECK_FAIL}  {name:<46} {elapsed:6.2f}s  {detail}",
              file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return name, "fail", elapsed, detail
    elapsed = time.perf_counter() - t0
    print(f"{CHECK_OK}  {name:<46} {elapsed:6.2f}s")
    return name, "ok", elapsed, ""


# ---------------------------------------------------------------------------
# 1. import 检查
# ---------------------------------------------------------------------------

EXPECTED_EXPORTS = {
    "vlm": {
        "VLMClient", "build_vlm_client", "classify_component",
        "ComponentClassification", "review_progress", "ProgressReview",
        "select_best_view", "BestViewSelection",
        "load_dotenv", "resolve_env_placeholder",
        "load_vlm_config", "get_provider_config", "VLMProviderConfig",
        "parse_json_response", "VLMResponse",
    },
    "pipeline": {
        "SemanticRefinementConfig", "SemanticRefinementPipeline",
        "SemanticRefinementResult", "run_semantic_refinement",
        "generate_report",
    },
}


def case_imports() -> None:
    """所有 4 个顶层包都能 import，且公共符号齐全。"""
    for pkg_name in ("vlm", "geometry", "rendering", "pipeline"):
        mod = importlib.import_module(pkg_name)
        if pkg_name in EXPECTED_EXPORTS:
            missing = EXPECTED_EXPORTS[pkg_name] - set(getattr(mod, "__all__", []))
            assert not missing, f"{pkg_name}.__all__ 缺少: {sorted(missing)}"

    # vlm/ 内部不应再 import 任何几何 / 渲染模块（结构正确性自检）
    forbidden_in_vlm = {
        "vlm.blocks", "vlm.seed_split", "vlm.grouped_split",
        "vlm.render_component", "vlm.pipeline", "vlm.report",
    }
    for mod_name in forbidden_in_vlm:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            continue
        raise AssertionError(
            f"{mod_name} 仍能 import — 说明文件没搬干净 / vlm/ 还残留几何模块"
        )

    # 进一步直接 import 关键深层模块
    importlib.import_module("geometry.blocks")
    importlib.import_module("geometry.seed_split")
    importlib.import_module("geometry.spectral_cluster")
    importlib.import_module("rendering.heatmap_renderer")
    importlib.import_module("rendering.render_component")
    importlib.import_module("rendering.view_thumbnails")
    importlib.import_module("pipeline.pipeline")
    importlib.import_module("pipeline.report")
    importlib.import_module("pipeline.cli")


# ---------------------------------------------------------------------------
# 2. CLI --help 检查
# ---------------------------------------------------------------------------


def _run_help(module: str) -> None:
    """跑 `python -m <module> --help`，超时或非 0 直接抛。"""
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`python -m {module} --help` 退出码 {proc.returncode}\n"
            f"STDERR:\n{proc.stderr[:2000]}"
        )
    if "usage" not in proc.stdout.lower():
        raise RuntimeError(
            f"`python -m {module} --help` 输出里没 'usage'：\n{proc.stdout[:500]}"
        )


def case_cli_pipeline_help() -> None:
    """`python -m pipeline.cli --help` 应正常输出 usage。"""
    _run_help("pipeline.cli")


def case_cli_vlm_help_shim() -> None:
    """`python -m vlm.cli --help` 经薄转发后仍正常工作。"""
    _run_help("vlm.cli")


# ---------------------------------------------------------------------------
# 3. 几何链路 (无 VLM)
# ---------------------------------------------------------------------------


def _make_cube_mesh():
    """生成一个 8 顶点 12 三角形的立方体，作为最小测试 mesh。"""
    import numpy as np

    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    f = np.array([
        [0, 1, 2], [0, 2, 3],          # bottom
        [4, 6, 5], [4, 7, 6],          # top
        [0, 5, 1], [0, 4, 5],          # front
        [2, 1, 6], [1, 5, 6],          # right
        [3, 2, 7], [2, 6, 7],          # back
        [0, 3, 4], [3, 7, 4],          # left
    ], dtype=np.int64)
    return v, f


def case_geometry_chain() -> None:
    """spectral_cluster + seed_split._extract_submesh + blocks.build_initial_blocks。"""
    import numpy as np

    from geometry.blocks import build_initial_blocks
    from geometry.seed_split import _extract_submesh
    from geometry.spectral_cluster import (
        spectral_cluster_vertices,
        vertex_labels_to_face_labels,
    )

    v, f = _make_cube_mesh()

    vlabels, k, debug = spectral_cluster_vertices(v, f, n_clusters=2)
    assert vlabels.shape == (len(v),)
    assert k >= 1
    assert isinstance(debug, dict)

    flabels = np.asarray(vertex_labels_to_face_labels(vlabels, f))
    assert flabels.shape == (len(f),), (
        f"face label 数应等于 face 数 {len(f)}, 得到 {flabels.shape}"
    )
    assert flabels.min() >= 0

    # 初始 block 树
    blocks = build_initial_blocks(flabels, id_prefix="C")
    assert isinstance(blocks, list) and len(blocks) >= 1, (
        "build_initial_blocks 至少要产生 1 个根块"
    )
    first = blocks[0]
    first_face_ids = np.asarray(first.face_ids, dtype=np.int64)
    assert first_face_ids.size > 0

    # _extract_submesh 应给出一个非空子 mesh
    sub_v, sub_f, _v_remap = _extract_submesh(v, f, first_face_ids)
    assert sub_v.shape[0] > 0 and sub_f.shape[0] > 0


# ---------------------------------------------------------------------------
# 4. 渲染链路 (无 VLM)
# ---------------------------------------------------------------------------


def case_rendering_chain(out_dir: Path) -> None:
    """heatmap / render_component / view_thumbnails 都能跑出有效 PNG。"""
    import numpy as np
    from PIL import Image

    from rendering.heatmap_renderer import render_segmentation_view
    from rendering.render_component import render_single_view_overview
    from rendering.view_thumbnails import render_thumbnails_panel

    v, f = _make_cube_mesh()
    face_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) heatmap 分块单视角 (返回 ndarray，要自己存盘)
    seg_arr = render_segmentation_view(
        v, f, face_labels,
        elevation_deg=22.6,
        azimuth_deg=45.0,
        image_size=(256, 256),
    )
    assert seg_arr.ndim == 3 and seg_arr.shape[-1] in (3, 4), (
        f"render_segmentation_view 应返回 HxWxC ndarray, 得到 {seg_arr.shape}"
    )
    seg_path = out_dir / "cube_segmentation_view.png"
    Image.fromarray(seg_arr).save(seg_path)
    assert seg_path.exists() and seg_path.stat().st_size > 256

    # 2) render_component 单视角透明背景
    overview_path = out_dir / "cube_overview_transparent.png"
    render_single_view_overview(
        v, f,
        output_path=overview_path,
        elev=22.6, azim=45.0,
        image_size=256,
    )
    assert overview_path.exists() and overview_path.stat().st_size > 256

    # 3) view_thumbnails 12 张缩略图拼图
    thumbs_path = out_dir / "cube_thumbnails_panel.png"
    render_thumbnails_panel(
        v, f, thumbs_path,
        thumb_size=128, cols=4,
        title="smoke: 12 candidate views",
    )
    assert thumbs_path.exists() and thumbs_path.stat().st_size > 256


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.test_vlm_pipeline",
        description="重构后的 4 包结构 smoke 测试（不调用 VLM API）。",
    )
    p.add_argument(
        "--imports-only", action="store_true",
        help="只验 import，不跑 CLI / 几何 / 渲染。",
    )
    p.add_argument(
        "--cli-only", action="store_true",
        help="只验 `python -m vlm.cli` / `python -m pipeline.cli` 的 --help。",
    )
    p.add_argument(
        "--no-render", action="store_true",
        help="跳过 pyvista 渲染部分（无显卡 / 无 pyvista 时使用）。",
    )
    p.add_argument(
        "--no-cli", action="store_true",
        help="跳过子进程 CLI --help 检查（在受限 sandbox 中使用）。",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "output" / "smoke_test",
        help="渲染产物输出目录（默认 output/smoke_test/）。",
    )
    p.add_argument(
        "--keep", action="store_true",
        help="保留 --out-dir 内容（默认会先清空）。",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if not args.keep and args.out_dir.exists():
        shutil.rmtree(args.out_dir, ignore_errors=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cases: List[Tuple[str, Callable[[], None]]] = []

    if args.cli_only:
        if not args.no_cli:
            cases.append(("CLI: python -m pipeline.cli --help",
                          case_cli_pipeline_help))
            cases.append(("CLI: python -m vlm.cli --help (shim)",
                          case_cli_vlm_help_shim))
    else:
        cases.append(("imports: vlm/geometry/rendering/pipeline",
                      case_imports))
        if not args.imports_only:
            if not args.no_cli:
                cases.append(("CLI: python -m pipeline.cli --help",
                              case_cli_pipeline_help))
                cases.append(("CLI: python -m vlm.cli --help (shim)",
                              case_cli_vlm_help_shim))
            cases.append(("geometry chain: spectral + extract_submesh + blocks",
                          case_geometry_chain))
            if not args.no_render:
                cases.append(
                    ("rendering chain: heatmap + component + thumbnails",
                     lambda: case_rendering_chain(args.out_dir)),
                )

    print(f"smoke test root: {REPO_ROOT}")
    print(f"render out_dir : {args.out_dir}")
    print()
    print(f"{'case':<46} {'time':>7s}  detail")
    print("-" * 90)

    results: List[Tuple[str, str, float, str]] = []
    for name, fn in cases:
        results.append(_run_case(name, fn))

    n_total = len(results)
    n_ok = sum(1 for _n, st, *_ in results if st == "ok")
    n_fail = sum(1 for _n, st, *_ in results if st == "fail")
    print()
    print(f"summary: {n_ok}/{n_total} ok, {n_fail} failed")

    if n_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
