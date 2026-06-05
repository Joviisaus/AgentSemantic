"""
语义分块 → 可变形骨架 命令行入口。

把 full_coverage 产出的 ``face_semantic_labels.json`` + 对应 mesh 适配成一具
语义骨架（每个语义部位一根骨，接缝处建关节，自动蒙皮），施加一个检验姿态，
并导出可在任意 3D 查看器里打开的产物：

    <out>/semantic_skeleton.json   骨架层级 / 关节 / 权重摘要
    <out>/rest.off                 静止姿态（应与输入一致）
    <out>/posed.off                施加检验姿态后的形变网格
    <out>/rig_rest.png             (可选 --render) 按骨上色的静止 12 视角
    <out>/rig_posed.png            (可选 --render) 按骨上色的形变 12 视角

示例：
    python -m pipeline.semantic_rig_cli output/Armor_v2/medoid_pose/render_pose.off \
        --labels output/Armor_v2/vlm_full/face_semantic_labels.json \
        --root-part 躯干 --angle 25 --render
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh

from geometry.semantic_rig import (
    RigAnimation,
    SemanticSkeleton,
    build_animations,
    build_part_adjacency,
    build_semantic_skeleton,
    make_test_animation,
    make_test_pose,
    resolve_rig_spec,
)


# 一组区分度较高的上色（按骨索引取模）
_BONE_PALETTE: List[Tuple[int, int, int]] = [
    (230, 60, 40), (46, 158, 94), (52, 120, 220), (242, 168, 43),
    (150, 90, 200), (40, 190, 200), (220, 90, 160), (120, 130, 60),
    (200, 120, 60), (90, 200, 120), (120, 100, 220), (200, 200, 60),
]


def _load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    # 必须与 full_coverage_cli._load_mesh 用同样的加载方式（trimesh 默认
    # process=True，会合并重复顶点）。否则：
    #   1) STL 等格式 process=False 时每个三角形各自独占 3 个顶点、面间无共享
    #      边，face_adjacency 为空 -> 算不出接缝/关节；
    #   2) 面的数量/顺序与生成 face_semantic_labels.json 时不一致，标签错位。
    ext = path.suffix.lstrip(".").lower() or None
    try:
        mesh = trimesh.load(str(path), file_type=ext)
    except (ValueError, OSError):
        mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"{path} 中找不到 trimesh.Trimesh")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"无法把 {path} 解析为 Trimesh，得到 {type(mesh)}")
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def _load_labels(path: Path) -> Tuple[List[str], List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "semantic_parts" not in data or "face_semantic_index" not in data:
        raise ValueError(
            f"{path} 需含 semantic_parts 与 face_semantic_index 字段"
        )
    return list(data["semantic_parts"]), [int(x) for x in data["face_semantic_index"]]


def _model_dir_name(mesh_path: Path, base: Path) -> Optional[str]:
    """mesh 若位于 <base>/<model>/... 则返回 <model>，否则返回 stem。"""
    try:
        rel = mesh_path.resolve().relative_to(base.resolve())
        if rel.parts and rel.parts[0] not in ("", "."):
            return rel.parts[0]
    except (ValueError, OSError):
        pass
    return mesh_path.stem or None


def _discover_label_files(
    output_base: Path,
    mesh_path: Path,
    n_faces: int,
) -> List[Tuple[Path, bool, List[str]]]:
    """在 output_base 下找面数匹配的 face_semantic_labels.json。

    返回 [(路径, 是否同属该 mesh 的模型目录, 部位名)]，按
    "同模型目录优先 → 修改时间新→旧" 排序。
    """
    if not output_base.is_dir():
        return []
    model_dir = _model_dir_name(mesh_path, output_base)
    found: List[Tuple[Path, bool, float, List[str]]] = []
    for c in output_base.glob("**/face_semantic_labels.json"):
        try:
            with open(c, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if int(data.get("num_faces", -1)) != n_faces:
            continue
        in_model = bool(model_dir) and model_dir in c.parts
        found.append((c, in_model, c.stat().st_mtime, list(data.get("semantic_parts", []))))
    found.sort(key=lambda t: (t[1], t[2]), reverse=True)
    return [(c, in_model, parts) for c, in_model, _mt, parts in found]


def _prompt_yes_no(question: str, *, default: bool, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        ans = input(f"{question} {suffix} ").strip().lower()
    except EOFError:
        print("(非交互环境，默认退出；用 -y 可强制先跑分块)")
        return False
    if not ans:
        return default
    return ans in ("y", "yes")


def _run_segmentation(
    mesh_path: Path,
    seg_extra: Sequence[str],
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[str] = None,
    env_file: Optional[str] = None,
) -> int:
    """调用 full_coverage_cli 先把语义分块跑出来。

    把 rig CLI 上的 VLM 连接参数（provider/model/config/env-file）一并转发，
    这样单条命令里的 ``--provider qwen`` 对分块和 LLM rig 设计都生效。
    """
    from pipeline import full_coverage_cli

    argv: List[str] = [str(mesh_path)]
    if provider:
        argv += ["--provider", provider]
    if model:
        argv += ["--model", model]
    if config:
        argv += ["--config", config]
    if env_file:
        argv += ["--env-file", env_file]
    argv += list(seg_extra)
    print(f"[segment] 运行: python -m pipeline.full_coverage_cli {' '.join(argv)}")
    return full_coverage_cli.main(argv)


def _bone_face_colors(skel: SemanticSkeleton) -> np.ndarray:
    """每个面按其语义部位对应骨的调色，返回 (F, 3) uint8。"""
    part_to_bone = {b.part_index: b.index for b in skel.bones}
    n_faces = len(skel.faces)
    colors = np.full((n_faces, 3), 210, dtype=np.uint8)
    # 面的部位：用骨的 face_ids 反查更稳（与建骨一致）
    for b in skel.bones:
        rgb = np.asarray(_BONE_PALETTE[b.index % len(_BONE_PALETTE)], dtype=np.uint8)
        fids = np.asarray(b.face_ids, dtype=np.int64)
        if fids.size:
            colors[fids] = rgb
    _ = part_to_bone  # 保留以便将来按 face_semantic_index 直接上色
    return colors


def _render_colored(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    out_path: Path,
    title: str,
    image_size: int,
    cols: int,
) -> Optional[Path]:
    try:
        from rendering.render_component import (
            _compose_panel,
            _render_views_with_color_table,
        )
    except ImportError as exc:
        print(f"[render] 跳过（缺少渲染依赖）: {exc}", file=sys.stderr)
        return None
    views = _render_views_with_color_table(
        vertices, faces, face_colors, image_size=(image_size, image_size)
    )
    return _compose_panel(views, output_path=out_path, title=title, cols=cols)


def _part_color_table(face_idx: np.ndarray, n_parts: int) -> np.ndarray:
    """按 face_semantic_index 直接上色，返回 (F, 3) uint8。"""
    pal = np.asarray(_BONE_PALETTE, dtype=np.uint8)
    safe = np.clip(face_idx, 0, max(0, n_parts - 1))
    colors = pal[safe % len(pal)]
    colors[face_idx < 0] = np.array([210, 210, 215], dtype=np.uint8)
    return colors.astype(np.uint8)


def _adjacency_name_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_idx: np.ndarray,
    semantic_parts: Sequence[str],
) -> List[Tuple[str, str]]:
    """相邻部位的名字对（供 LLM 推层级）。"""
    seams = build_part_adjacency(vertices, faces, face_idx)
    pairs: List[Tuple[str, str]] = []
    for (a, b) in sorted(seams.keys()):
        if 0 <= a < len(semantic_parts) and 0 <= b < len(semantic_parts):
            pairs.append((semantic_parts[a], semantic_parts[b]))
    return pairs


def _build_vlm_client(args: argparse.Namespace, *, tag: str = "vlm"):
    """按 CLI 上的 provider/model/config/env-file 造一个 VLM client。失败返回 None。"""
    from vlm.client import build_vlm_client
    from vlm.config import DEFAULT_CONFIG_PATH, get_provider_config, load_vlm_config

    cfg = load_vlm_config(
        args.config or str(DEFAULT_CONFIG_PATH),
        load_env=True,
        env_paths=[args.env_file] if args.env_file else None,
    )
    provider_cfg = get_provider_config(cfg, args.provider, model=args.model)
    if not provider_cfg.has_key() and not provider_cfg.is_local():
        print(f"[错误] provider '{provider_cfg.name}' 缺少 API key", file=sys.stderr)
        return None
    print(f"[{tag}]    provider={provider_cfg.name} model={provider_cfg.default_model}")
    return build_vlm_client(provider_cfg)


def _get_rig_spec(
    args: argparse.Namespace,
    out_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_idx: np.ndarray,
    semantic_parts: Sequence[str],
) -> Optional[Dict]:
    """得到 rig-spec：优先 --rig-spec 文件；否则 --llm-rig 调 VLM。无则 None。"""
    if args.rig_spec:
        sp = Path(args.rig_spec)
        if not sp.is_file():
            print(f"[错误] rig-spec 不存在: {sp}", file=sys.stderr)
            return None
        with open(sp, "r", encoding="utf-8") as f:
            spec = json.load(f)
        print(f"[rig]    读取 rig-spec: {sp}")
        return spec

    if not args.llm_rig:
        return None

    pairs = _adjacency_name_pairs(vertices, faces, face_idx, semantic_parts)
    print(f"[rig]    部位邻接 {len(pairs)} 对，调 LLM 设计骨架层级 + 动画…")

    image_path: Optional[Path] = None
    if not args.no_rig_image:
        image_path = _render_colored(
            vertices, faces, _part_color_table(face_idx, len(semantic_parts)),
            out_dir / "rig_overview.png", "整模型（按部位上色）",
            args.image_size, args.panel_cols,
        )

    from vlm.rig_designer import design_rig

    client = _build_vlm_client(args, tag="rig")
    if client is None:
        return None

    design = design_rig(
        client,
        semantic_parts,
        pairs,
        object_kind_hint=args.rig_hint,
        image_path=image_path,
        model=args.model,
    )
    with open(out_dir / "rig_design.json", "w", encoding="utf-8") as f:
        json.dump(design.to_dict(), f, ensure_ascii=False, indent=2)
    if not design.ok:
        print(f"[错误] LLM rig 设计失败: {design.error}", file=sys.stderr)
        if design.raw_text:
            print(f"        原始返回: {design.raw_text[:200]}", file=sys.stderr)
        return None
    print(f"[rig]    LLM 设计完成（{design.elapsed_seconds:.1f}s），"
          f"object_kind={design.object_kind!r}")
    return design.spec


def _get_anim_spec(
    args: argparse.Namespace,
    out_dir: Path,
    skel: SemanticSkeleton,
    semantic_parts: Sequence[str],
    *,
    rig_spec: Optional[Dict] = None,
    overview_image: Optional[Path] = None,
) -> Optional[Dict]:
    """得到 animations spec：优先 --anim-spec；否则调动画 agent。"""
    if args.anim_spec:
        sp = Path(args.anim_spec)
        if not sp.is_file():
            print(f"[错误] anim-spec 不存在: {sp}", file=sys.stderr)
            return None
        with open(sp, "r", encoding="utf-8") as f:
            data = json.load(f)
        anims = data.get("animations", data if isinstance(data, list) else [])
        print(f"[anim]   读取 anim-spec: {sp}（{len(anims)} 段）")
        return {"animations": anims}

    names = [str(n).strip() for n in (args.anim_name or []) if str(n).strip()]
    if not names:
        return None

    use_llm = args.llm_anim or not args.anim_spec
    if not use_llm:
        return None

    from vlm.anim_designer import (
        design_animations,
        format_hierarchy_table,
        hierarchy_from_rig_spec,
    )

    if rig_spec and rig_spec.get("bones"):
        parent_map = hierarchy_from_rig_spec(rig_spec)
        root = str(rig_spec.get("root") or "").strip()
        hierarchy = format_hierarchy_table(semantic_parts, parent_map, root=root)
    else:
        hierarchy = "\n".join(skel.hierarchy_lines())

    print(f"[anim]   调 LLM 生成动作: {names} …")

    image_path = overview_image
    if image_path is None and not args.no_anim_image:
        image_path = out_dir / "rig_overview.png"
        if not image_path.is_file():
            image_path = _render_colored(
                skel.vertices, skel.faces,
                _bone_face_colors(skel),
                image_path, "整模型（按骨上色）",
                args.image_size, args.panel_cols,
            )

    client = _build_vlm_client(args, tag="anim")
    if client is None:
        return None

    hint = args.anim_hint.strip() or args.rig_hint.strip()
    design = design_animations(
        client,
        names,
        semantic_parts,
        hierarchy,
        object_kind_hint=args.rig_hint,
        anim_hint=hint,
        image_path=image_path if not args.no_anim_image else None,
        model=args.model,
    )
    with open(out_dir / "anim_design.json", "w", encoding="utf-8") as f:
        json.dump(design.to_dict(), f, ensure_ascii=False, indent=2)
    if not design.ok:
        print(f"[错误] 动画 agent 失败: {design.error}", file=sys.stderr)
        if design.raw_text:
            print(f"        原始返回: {design.raw_text[:200]}", file=sys.stderr)
        return None
    print(f"[anim]   LLM 生成 {len(design.animations)} 段动画"
          f"（{design.elapsed_seconds:.1f}s）")
    for a in design.animations:
        n_kf = len(a.get("keyframes") or [])
        print(f"[anim]     - {a.get('name', '?')}: {n_kf} 关键帧")
    return {"animations": design.animations}


def _export_animation(
    skel: SemanticSkeleton,
    anim: RigAnimation,
    out_dir: Path,
    *,
    bake_frames: bool,
    anim_gif: bool,
    face_colors: np.ndarray,
    image_size: int,
) -> None:
    """导出一段动画：关键帧 .off（默认）/ 全插值帧（--bake-frames）/ GIF（--anim-gif）。"""
    adir = out_dir / f"anim_{anim.name}"
    adir.mkdir(parents=True, exist_ok=True)
    faces = skel.faces

    # 关键帧角度 -> 顶点
    kf_angles = [(k.t, dict(k.angles)) for k in sorted(anim.keyframes, key=lambda x: x.t)]
    for i, (t, angles) in enumerate(kf_angles):
        verts = skel.pose_from_angles(angles)
        fp = adir / f"keyframe_{i:02d}_t{t:.2f}.off"
        trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(str(fp))
    print(f"[anim]   {anim.name}: {len(kf_angles)} 关键帧 -> {adir}")

    frames: Optional[List[Tuple[float, np.ndarray]]] = None
    if bake_frames or anim_gif:
        frames = skel.sample_animation(anim)

    if bake_frames and frames is not None:
        fdir = adir / "frames"
        fdir.mkdir(exist_ok=True)
        for fi, (t, verts) in enumerate(frames):
            trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(
                str(fdir / f"frame_{fi:03d}.off")
            )
        print(f"[anim]   {anim.name}: 烘焙 {len(frames)} 帧 -> {fdir}")

    if anim_gif and frames is not None:
        gif = _render_anim_gif(
            faces, frames, face_colors, adir / f"{anim.name}.gif", image_size
        )
        if gif:
            print(f"[anim]   {anim.name}: GIF -> {gif}")


def _render_anim_gif(
    faces: np.ndarray,
    frames: Sequence[Tuple[float, np.ndarray]],
    face_colors: np.ndarray,
    out_gif: Path,
    image_size: int,
    *,
    elev: float = 15.0,
    azim: float = 35.0,
) -> Optional[Path]:
    try:
        import pyvista as pv
        from PIL import Image

        from rendering.heatmap_renderer import _apply_camera, _to_pyvista_faces
    except ImportError as exc:
        print(f"[anim]   GIF 跳过（缺少 pyvista/PIL）: {exc}", file=sys.stderr)
        return None

    pv_faces = _to_pyvista_faces(faces)
    imgs: List["Image.Image"] = []
    # 固定相机：用首帧确定取景，避免每帧跳动
    ref = frames[0][1]
    for _t, verts in frames:
        pdata = pv.PolyData(verts, pv_faces)
        pdata.cell_data["RGB"] = face_colors
        plotter = pv.Plotter(off_screen=True, window_size=(image_size, image_size))
        plotter.set_background("white")
        plotter.add_mesh(pdata, scalars="RGB", rgb=True, show_scalar_bar=False,
                         smooth_shading=True)
        _apply_camera(plotter, ref, elev, azim)
        arr = plotter.screenshot(return_img=True)
        plotter.close()
        imgs.append(Image.fromarray(arr))
    if not imgs:
        return None
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(
        str(out_gif), save_all=True, append_images=imgs[1:],
        duration=max(20, int(1000 / max(1, len(frames) / max(1, frames[-1][0] or 1)))),
        loop=0,
    )
    return out_gif


def _resolve_animations(
    args: argparse.Namespace,
    out_dir: Path,
    skel: SemanticSkeleton,
    semantic_parts: Sequence[str],
    resolved,
    *,
    spec: Optional[Dict],
) -> Tuple[List[RigAnimation], bool]:
    """先解析给定动画关键帧，供 R_mo 平滑与后续导出复用。

    返回 (animations, from_user_spec)。后者为 True 表示来自 anim-spec/LLM/rig 内嵌，
    False 表示回退到检验姿态 ``make_test_animation``。
    """
    want_anim = bool(args.anim_name or args.anim_spec or args.llm_anim)
    overview_png = out_dir / "rig_overview.png"
    overview = overview_png if overview_png.is_file() else None
    animations: List[RigAnimation] = []
    from_user = False

    if want_anim and args.anim_spec:
        sp = Path(args.anim_spec)
        if sp.is_file():
            with open(sp, "r", encoding="utf-8") as f:
                anim_spec = json.load(f)
            ar = resolve_rig_spec(anim_spec, semantic_parts)
            for w in ar.warnings:
                print(f"[anim]   注意: {w}")
            animations = build_animations(ar, skel)
            from_user = bool(animations)
    elif want_anim and args.anim_name:
        anim_spec = _get_anim_spec(
            args, out_dir, skel, semantic_parts,
            rig_spec=spec, overview_image=overview,
        )
        if anim_spec:
            ar = resolve_rig_spec(anim_spec, semantic_parts)
            for w in ar.warnings:
                print(f"[anim]   注意: {w}")
            animations = build_animations(ar, skel)
            from_user = bool(animations)
    elif resolved is not None and getattr(resolved, "animations", None):
        animations = build_animations(resolved, skel)
        from_user = bool(animations)

    if not animations:
        animations = [make_test_animation(skel, angle_deg=args.angle)]
        if args.optimize_skin:
            print(
                f"[anim]   未指定动画，R_mo 使用检验姿态 "
                f"(make_test_animation @ {args.angle}°)"
            )
    elif args.optimize_skin:
        print(
            f"[anim]   R_mo 使用给定动画 {len(animations)} 段: "
            f"{[a.name for a in animations]}"
        )
    return animations, from_user


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pipeline.semantic_rig_cli",
        description="语义分块 → 可变形骨架（接缝关节 + 自动蒙皮 + 检验姿态）",
    )
    p.add_argument("mesh", type=str, help="mesh 路径（.off/.obj/.glb…）")
    p.add_argument(
        "--labels",
        type=str,
        default=None,
        help=(
            "face_semantic_labels.json 路径。省略时在 --output-base 下自动查找"
            "与该 mesh 面数匹配的残留分块结果；找不到则提示是否先跑全覆盖分块。"
        ),
    )
    p.add_argument(
        "--output-base", type=str, default="output",
        help="自动查找残留分块结果 / 跑分块时的输出根目录（默认 output）",
    )
    p.add_argument(
        "-y", "--yes", action="store_true",
        help="未找到分块结果时不再询问，直接先跑全覆盖分块。",
    )
    p.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="输出目录（默认 <labels 所在目录>/rig）",
    )
    p.add_argument(
        "--root-part", type=str, default=None,
        help="根部位名（如 躯干 / 底座）；不传则自动取邻接度最高的部位",
    )
    p.add_argument(
        "--blend-radius-ratio", type=float, default=0.06,
        help="接缝过渡带半径占 bbox 对角线比例（默认 0.06）",
    )
    p.add_argument(
        "--no-split-components", action="store_true",
        help="关闭【同部位多连通块拆成多根骨】（默认开启，如狗的 4 条腿各成一骨）",
    )
    p.add_argument(
        "--min-component-faces", type=int, default=20,
        help="小于此面数的连通块并入相邻块去噪（默认 20）",
    )
    p.add_argument(
        "--refine-split-names", action="store_true",
        help="对同部位的多个不连通块调 VLM 细分命名（如「腿」→ 左前腿/右前腿…）",
    )
    p.add_argument(
        "--angle", type=float, default=25.0,
        help="检验姿态每根叶子骨的弯曲角度（度，默认 25）",
    )
    p.add_argument(
        "--render", action="store_true",
        help="额外渲染按骨上色的 rest/posed 12 视角 PNG（需 pyvista）",
    )
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--panel-cols", type=int, default=4)

    # ---- LLM 给定骨架层级 + 动画 ----
    g = p.add_argument_group("LLM rig / 动画")
    g.add_argument(
        "--llm-rig", action="store_true",
        help="让 LLM 给定骨架层级 + 关节轴 + 动画关键帧（而非自动层级 + 检验姿态）。",
    )
    g.add_argument(
        "--rig-spec", type=str, default=None,
        help=(
            "直接读一份 rig-spec JSON（含 root/bones/animations），跳过 LLM。"
            "schema 见 vlm/rig_designer.py / geometry.semantic_rig.resolve_rig_spec。"
        ),
    )
    g.add_argument(
        "--rig-hint", type=str, default="",
        help="给 LLM 的物体类型提示（如「类人机甲」「四足动物」）。",
    )
    g.add_argument(
        "--no-rig-image", action="store_true",
        help="--llm-rig 时不给 LLM 看渲染图（默认会渲一张整模型按部位上色图）。",
    )
    g.add_argument(
        "--bake-frames", action="store_true",
        help="导出每段动画的全部插值帧 .off（默认只导关键帧，省磁盘）。",
    )
    g.add_argument(
        "--anim-gif", action="store_true",
        help="为每段动画渲染一个 GIF（需 pyvista，且与建骨同环境）。",
    )
    g.add_argument(
        "--export-glb", action="store_true",
        help=(
            "导出带蒙皮 + 动画的 GLB（rig.glb），可在任意 glTF 查看器点播放。"
            "无 spec 动画时用自动检验动画。"
        ),
    )
    g.add_argument(
        "--optimize-skin", action="store_true",
        help=(
            "导出前在接缝蒙皮 u0 上网格扩散（连续、自然多骨，1–4 骨/点）。"
            "默认不调关节；加 --joint-opt 才做关节步。"
        ),
    )
    g.add_argument(
        "--opt-cycles", type=int, default=1,
        help="--optimize-skin 且 --joint-opt 时的外轮次数（默认 1）",
    )
    g.add_argument(
        "--part-opt-iter", type=int, default=12,
        help="--optimize-skin 每轮分块+调和迭代步数（默认 12）",
    )
    g.add_argument(
        "--joint-opt-iter", type=int, default=60,
        help="--optimize-skin 关节微调步数（默认 60）",
    )
    g.add_argument(
        "--joint-opt", action="store_true",
        help="在平滑蒙皮基础上微调关节（默认仅更新权重，较快）",
    )
    g.add_argument(
        "--no-joint-opt", action="store_true",
        help=argparse.SUPPRESS,
    )
    g.add_argument(
        "--fast-skin-opt", action="store_true",
        help="蒙皮优化快速模式（降迭代/边采样/调和缓存，约 3–8× 加速）",
    )
    g.add_argument(
        "--anim-name", action="append", default=None, metavar="NAME",
        help=(
            "语义动作名，可重复。例：--anim-name 走路 --anim-name 摇尾巴。"
            "与 --llm-anim 联用时由动画 agent 生成关键帧；也可配合 --anim-spec。"
        ),
    )
    g.add_argument(
        "--anim-hint", type=str, default="",
        help="给动画 agent 的补充说明（如「缓慢四足步态」「幅度小一点」）。",
    )
    g.add_argument(
        "--llm-anim", action="store_true",
        help="用 LLM 动画 agent 根据 --anim-name 生成关键帧（有 anim-name 时默认开启）。",
    )
    g.add_argument(
        "--anim-spec", type=str, default=None,
        help="直接读动画 JSON（含 animations 数组），跳过动画 agent。",
    )
    g.add_argument(
        "--no-anim-image", action="store_true",
        help="动画 agent 不看渲染图（默认会复用 rig 概览图或现渲一张）。",
    )
    # VLM 连接参数（--llm-rig / --llm-anim / --refine-split-names 共用）
    g.add_argument("--config", type=str, default=None, help="VLM 配置 YAML（默认内置路径）")
    g.add_argument("--provider", type=str, default=None, help="覆盖 active_provider")
    g.add_argument("--model", type=str, default=None, help="覆盖 default_model")
    g.add_argument("--env-file", type=str, default=None, help="额外 .env 路径")
    return p


def _resolve_labels(
    args: argparse.Namespace,
    mesh_path: Path,
    n_faces: int,
    seg_extra: Sequence[str],
) -> Tuple[Optional[Path], int]:
    """确定要用的 labels 文件。

    返回 (labels_path, exit_code)。labels_path 为 None 时按 exit_code 退出
    （0 = 用户主动选择退出，>0 = 出错）。
    """
    # 1) 显式指定
    if args.labels:
        lp = Path(args.labels)
        if not lp.is_file():
            print(f"[错误] labels 不存在: {lp}", file=sys.stderr)
            return None, 2
        if seg_extra:
            print(f"[注意] 已指定 --labels，忽略多余参数: {list(seg_extra)}")
        return lp, 0

    base = Path(args.output_base)
    # 2) 自动查找残留分块结果
    matches = _discover_label_files(base, mesh_path, n_faces)
    if matches:
        chosen, in_model, parts = matches[0]
        tag = "同模型目录" if in_model else "面数匹配"
        print(f"[labels] 自动选用残留分块（{tag}）: {chosen}")
        print(f"[labels] {len(parts)} 部位: {parts}")
        if len(matches) > 1:
            print(f"[labels] 另有 {len(matches) - 1} 个候选（已选最新/同目录的）:")
            for c, _im, _p in matches[1:]:
                print(f"           - {c}")
        return chosen, 0

    # 3) 没有 → 询问是否先跑分块
    print(f"[labels] 在 {base}/ 下未找到与该 mesh（F={n_faces}）匹配的分块结果。")
    if not _prompt_yes_no(
        "是否先跑全覆盖语义分块（会调用 VLM）？",
        default=True,
        assume_yes=args.yes,
    ):
        print("[exit]   已选择退出。可手动指定 --labels，或先运行 "
              "python -m pipeline.full_coverage_cli <mesh> ...")
        return None, 0

    code = _run_segmentation(
        mesh_path, seg_extra,
        provider=args.provider, model=args.model,
        config=args.config, env_file=args.env_file,
    )
    if code != 0:
        print(f"[错误] 分块失败（exit={code}）", file=sys.stderr)
        return None, code

    matches = _discover_label_files(base, mesh_path, n_faces)
    if not matches:
        print("[错误] 分块已运行但仍未找到 face_semantic_labels.json", file=sys.stderr)
        return None, 1
    chosen, _im, parts = matches[0]
    print(f"[labels] 分块完成，选用: {chosen}")
    print(f"[labels] {len(parts)} 部位: {parts}")
    return chosen, 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args, seg_extra = _build_arg_parser().parse_known_args(argv)

    mesh_path = Path(args.mesh)
    if not mesh_path.is_file():
        print(f"[错误] mesh 不存在: {mesh_path}", file=sys.stderr)
        return 2

    print(f"[mesh]   {mesh_path}")
    vertices, faces = _load_mesh(mesh_path)
    print(f"[mesh]   V={len(vertices)} F={len(faces)}")

    labels_path, code = _resolve_labels(args, mesh_path, len(faces), seg_extra)
    if labels_path is None:
        return code

    out_dir = Path(args.output_dir) if args.output_dir else labels_path.parent / "rig"
    out_dir.mkdir(parents=True, exist_ok=True)

    semantic_parts, face_idx = _load_labels(labels_path)
    if len(face_idx) != len(faces):
        print(
            f"[错误] face_semantic_index 长度 {len(face_idx)} 与 faces "
            f"{len(faces)} 不一致（mesh 与 labels 不匹配？）",
            file=sys.stderr,
        )
        return 2
    print(f"[labels] 使用: {labels_path}（{len(semantic_parts)} 部位）")
    face_idx_arr = np.asarray(face_idx, dtype=np.int64)

    # 可选：对同部位的多个不连通块调 VLM 细分命名（腿 → 左前腿/右前腿…）
    if args.refine_split_names:
        client = _build_vlm_client(args, tag="split")
        if client is None:
            return 1
        from vlm.component_namer import refine_disconnected_parts

        print("[split]  检测同部位不连通块并调 VLM 细分命名…")
        naming = refine_disconnected_parts(
            client, vertices, faces, semantic_parts, face_idx,
            out_dir=out_dir, image_size=args.image_size, panel_cols=args.panel_cols,
            model=args.model, min_component_faces=args.min_component_faces,
        )
        for w in naming.warnings:
            print(f"[split]  注意: {w}")
        for orig, names in naming.renames.items():
            print(f"[split]  「{orig}」{len(names)} 块 → {names}")
        if naming.changed:
            semantic_parts = naming.semantic_parts
            face_idx = naming.face_semantic_index
            face_idx_arr = np.asarray(face_idx, dtype=np.int64)
            refined_path = out_dir / "face_semantic_labels.refined.json"
            with open(refined_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"semantic_parts": semantic_parts,
                     "face_semantic_index": face_idx},
                    f, ensure_ascii=False,
                )
            with open(out_dir / "component_naming.json", "w", encoding="utf-8") as f:
                json.dump(naming.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"[split]  细分后 {len(semantic_parts)} 部位 → {refined_path.name}")
        else:
            print("[split]  VLM 未给出更细名字，沿用原标签")

    # 可选：LLM / 文件给定骨架层级 + 关节轴 + 动画
    spec = _get_rig_spec(args, out_dir, vertices, faces, face_idx_arr, semantic_parts)
    parent_override = None
    axis_override = None
    resolved = None
    if spec is not None:
        resolved = resolve_rig_spec(spec, semantic_parts)
        parent_override = resolved.parent_override or None
        axis_override = resolved.axis_override or None
        for w in resolved.warnings:
            print(f"[rig]    注意: {w}")
    elif args.llm_rig or args.rig_spec:
        # 用户要 LLM/spec 但没拿到 → 直接退出，避免悄悄回退到自动层级
        return 1

    print("[build]  构造语义骨架…")
    skel = build_semantic_skeleton(
        vertices,
        faces,
        face_idx,
        semantic_parts,
        root_part=args.root_part,
        blend_radius_ratio=args.blend_radius_ratio,
        parent_override=parent_override,
        axis_override=axis_override,
        split_components=not args.no_split_components,
        min_component_faces=args.min_component_faces,
    )

    print(f"[skel]   bones={skel.num_bones} joints={len(skel.joints)} "
          f"roots={skel.roots}"
          + ("  (层级来自 LLM/spec)" if parent_override else "  (层级自动)"))
    print("[skel]   层级:")
    for ln in skel.hierarchy_lines():
        print("    " + ln)
    blended = int(np.count_nonzero((skel.weights > 1e-6).sum(axis=1) > 1))
    print(f"[skel]   过渡带混合顶点 = {blended}/{len(vertices)}")

    # 先给定骨骼运动姿态，再在这些姿态下做 R_mo 平滑优化
    animations, anim_from_user = _resolve_animations(
        args, out_dir, skel, semantic_parts, resolved, spec=spec,
    )

    skin_opt_report = None
    if args.optimize_skin:
        from geometry.skin_optimize import SkinOptimizeConfig, optimize_semantic_skin

        do_joints = bool(args.joint_opt) and not args.no_joint_opt
        opt_cfg = (
            SkinOptimizeConfig.fast()
            if args.fast_skin_opt
            else SkinOptimizeConfig(
                weight_mode="smooth",
                outer_cycles=args.opt_cycles,
                partition_refine_iters=args.part_opt_iter,
                joint_max_iter=args.joint_opt_iter,
                optimize_joints=do_joints,
                verbose=True,
            )
        )
        if args.fast_skin_opt:
            opt_cfg = replace(opt_cfg, optimize_joints=do_joints, verbose=True)
        print(
            "[skin-opt] 指标分块 + u0 退火扩散 + 网格调和蒙皮"
            + (" + 关节 GRPO" if opt_cfg.optimize_joints else "（未加 --joint-opt，跳过关节迭代）")
        )
        part_vlm = None
        if opt_cfg.use_agent_partition_energy:
            part_vlm = _build_vlm_client(args, tag="part-energy")
        overview = out_dir / "rig_overview.png"
        skin_opt_report = optimize_semantic_skin(
            skel,
            opt_cfg,
            animations=animations,
            test_angle_deg=float(args.angle),
            report_dir=out_dir,
            vlm_client=part_vlm,
            partition_image_path=overview if overview.is_file() else None,
        ).to_dict()
        if skin_opt_report.get("report_html"):
            print(f"[skin-opt] 分块报告: {skin_opt_report['report_html']}")
        blended = int(np.count_nonzero((skel.weights > 0.08).sum(axis=1) >= 2))
        print(
            f"[skin-opt] 写回骨架 weights；多骨(>8%)顶点 "
            f"{blended}/{len(vertices)} ({100.0 * blended / max(len(vertices), 1):.1f}%)"
        )

    # 导出 rest 网格 + 骨架 JSON
    rest_off = out_dir / "rest.off"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(str(rest_off))
    skel_payload: Dict = skel.to_dict()
    if skin_opt_report is not None:
        skel_payload["skin_optimize"] = skin_opt_report

    want_anim = bool(args.anim_name or args.anim_spec or args.llm_anim)
    if want_anim and (args.anim_name or args.anim_spec) and not anim_from_user:
        print("[错误] 未能得到动画 spec", file=sys.stderr)
        return 1

    if want_anim and anim_from_user:
        # LLM/spec 动画路径：导出每段动画
        face_colors = _bone_face_colors(skel)
        anim_summ = []
        for anim in animations:
            _export_animation(
                skel, anim, out_dir,
                bake_frames=args.bake_frames,
                anim_gif=args.anim_gif,
                face_colors=face_colors,
                image_size=args.image_size,
            )
            anim_summ.append(anim.to_dict())
        skel_payload["animations"] = anim_summ
        if args.anim_name or args.anim_spec or args.llm_anim:
            skel_payload["rig_source"] = "llm_anim" if (args.anim_name or args.llm_anim) else "anim_spec"
        else:
            skel_payload["rig_source"] = "llm" if args.llm_rig else "spec"
    else:
        # 默认路径：自动检验姿态
        if (args.llm_rig or args.rig_spec):
            print("[rig]    spec 未含可用动画，回退到自动检验姿态。")
        pose = make_test_pose(skel, angle_deg=args.angle)
        posed = skel.pose(pose)
        moved = float(np.linalg.norm(posed - vertices, axis=1).max())
        print(f"[pose]   弯曲 {len(pose)} 根骨 @ {args.angle}°，最大位移 = {moved:.4f}")
        posed_off = out_dir / "posed.off"
        trimesh.Trimesh(vertices=posed, faces=faces, process=False).export(str(posed_off))
        print(f"[out]    {posed_off}")
        skel_payload["test_pose"] = {
            "angle_deg": args.angle,
            "rotated_bones": sorted(int(b) for b in pose.keys()),
            "max_displacement": moved,
        }
        if args.render:
            colors = _bone_face_colors(skel)
            r1 = _render_colored(
                vertices, faces, colors, out_dir / "rig_rest.png",
                "语义骨架 - 静止", args.image_size, args.panel_cols,
            )
            r2 = _render_colored(
                posed, faces, colors, out_dir / "rig_posed.png",
                f"语义骨架 - 检验姿态 {args.angle}°", args.image_size, args.panel_cols,
            )
            for r in (r1, r2):
                if r:
                    print(f"[out]    {r}")

    # 带蒙皮 + 动画的 GLB（任意查看器可点播放）
    if args.export_glb:
        glb_anims = list(animations) if animations else [
            make_test_animation(skel, angle_deg=args.angle)
        ]
        try:
            from geometry.rig_gltf_export import export_skinned_glb

            glb = export_skinned_glb(
                skel, glb_anims, out_dir / "rig.glb",
                optimize_skin=False,
                skin_opt_config=None,
                mo_animations=animations,
                test_angle_deg=float(args.angle),
            )
            print(f"[out]    {glb}  (动画: {[a.name for a in glb_anims]})")
        except ImportError as exc:
            print(f"[glb]    跳过（缺少 pygltflib）: {exc}", file=sys.stderr)

    skel_json = out_dir / "semantic_skeleton.json"
    with open(skel_json, "w", encoding="utf-8") as f:
        json.dump(skel_payload, f, ensure_ascii=False, indent=2)
    print(f"[out]    {rest_off}")
    print(f"[out]    {skel_json}")

    print("[done]   语义块已适配为可变形骨架。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
