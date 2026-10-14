#!/usr/bin/env python3
"""根据 _vlm_judgement.json 生成 _vlm_judgement.md 汇总"""
import json
from pathlib import Path

ROOT = Path("E:/Proj/AgentSematic")
src = ROOT / "docs" / "markdown" / "_vlm_judgement.json"
dst = ROOT / "docs" / "markdown" / "_vlm_judgement.md"

data = json.loads(src.read_text(encoding="utf-8"))

lines = ["# VLM 视觉评审结果\n",
         "由智谱 GLM-4V-Plus 对每张 4 联画（GT / ours_pointcloud / pointclip / cosmo3d）独立打分（0–10）。\n",
         "评审方式：先让 VLM 客观描述每个面板的颜色分布，再基于描述打分。\n"]

for tag in data:
    lines.append(f"\n## {tag}\n")
    lines.append("| sample | gt_ours | gt_pointclip | gt_cosmo3d | leading | comment |")
    lines.append("|---|---:|---:|---:|:---:|---|")

    # Sort: leading=yes first by gap, then leading=no by how bad
    entries = []
    for k, v in data[tag].items():
        if "error" in v or "gt_ours" not in v:
            continue
        entries.append((k, v))
    entries.sort(key=lambda x: (
        0 if x[1].get("ours_leading") == "yes" else (1 if x[1].get("ours_leading") == "tie" else 2),
        -(x[1].get("gt_ours", 0) - max(x[1].get("gt_cosmo3d", 0), x[1].get("gt_pointclip", 0)))
    ))
    for k, v in entries:
        ours = v.get("gt_ours", "-")
        pc = v.get("gt_pointclip", "-")
        cos = v.get("gt_cosmo3d", "-")
        leading = v.get("ours_leading", "-")
        cmt = v.get("comment", "")
        lines.append(f"| `{k}` | {ours} | {pc} | {cos} | **{leading}** | {cmt} |")

    # Error rows
    err_entries = [(k, v) for k, v in data[tag].items() if "error" in v]
    if err_entries:
        lines.append("")
        lines.append(f"**ERROR 样本 ({len(err_entries)}):** " + ", ".join(f"`{k}`" for k, _ in err_entries))

dst.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {dst}")
print(f"Lines: {len(lines)}")
