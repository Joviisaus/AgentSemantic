"""Build a side-by-side comparison table: my AI-VLM vs official 3 methods."""
import json
from pathlib import Path

BASE = Path(r"E:/Proj/AgentSematic/benchmark")
tiny = json.load(open(BASE / "results/summary_tiny.json", encoding="utf-8"))
mine = json.load(open(BASE / "ai_vlm_exp/ai_vlm_result.json", encoding="utf-8"))

off = {r["method"]: {s["key"]: s for s in r["per_sample"]} for r in tiny["results"]}
off_agg = {r["method"]: r["aggregate"] for r in tiny["results"]}
mine_by = {s["key"]: s for s in mine["per_sample"]}

keys = [s["key"] for s in tiny["results"][0]["per_sample"]]

lines = []
lines.append("# AI-VLM 亲手画 mask + 单视角反投影 vs 官方基线（CoMPaT-200 tiny 15）\n")
lines.append("口径：per-part IoU 求平均、union==0 跳过（CoSMo3D miou_skip）；"
             "instance = 15 样本均值。所有方法用**同一份 GT 与同一 15 个 key**。\n")
lines.append("## 每样本 mIoU(skip)\n")
lines.append("| key | 类别 | **AI-VLM(我)** | ours_pointcloud | cosmo3d | ours(mesh) |")
lines.append("|---|---|---|---|---|---|")
for k in keys:
    cat = mine_by[k]["category"]
    m = mine_by[k]["miou_skip"]
    a = off["ours_pointcloud"][k]["miou_skip"]
    c = off["cosmo3d"][k]["miou_skip"]
    o = off["ours"][k]["miou_skip"]
    star = " ✅" if m > a else ""
    lines.append(f"| {k} | {cat} | **{m:.3f}**{star} | {a:.3f} | {c:.3f} | {o:.3f} |")

lines.append("\n## 汇总\n")
lines.append("| 指标 | **AI-VLM(我)** | ours_pointcloud | cosmo3d | ours(mesh) |")
lines.append("|---|---|---|---|---|")
lines.append(f"| instance mIoU(skip) | **{mine['instance_miou_skip']:.4f}** | "
             f"{off_agg['ours_pointcloud']['instance_miou_skip']:.4f} | "
             f"{off_agg['cosmo3d']['instance_miou_skip']:.4f} | "
             f"{off_agg['ours']['instance_miou_skip']:.4f} |")
lines.append(f"| instance accuracy | **{mine['instance_accuracy']:.4f}** | "
             f"{off_agg['ours_pointcloud']['instance_accuracy']:.4f} | "
             f"{off_agg['cosmo3d']['instance_accuracy']:.4f} | "
             f"{off_agg['ours']['instance_accuracy']:.4f} |")

win = sum(1 for k in keys if mine_by[k]["miou_skip"] > off["ours_pointcloud"][k]["miou_skip"])
lines.append(f"\n**15 个样本里，我(单视角手画)在 {win}/15 个上超过 ours_pointcloud。**\n")

lines.append("## 实验设置\n")
lines.append("- **视角**：每个点云在 9 个候选视角里选「可见点最多」的单一重要视角（几何选择，不看 GT）。")
lines.append("- **画 mask**：我作为视觉模型逐 part 在灰度渲染图上画像素级红区（多边形/椭圆/矩形），"
             "背景主 part 用可见轮廓填充，小 part 后画覆盖。**不使用项目 API**。")
lines.append("- **相机**：纯 numpy 复刻 heatmap_renderer 的针孔相机（FOV=30°、up=z、球坐标机位），"
             "保证画在图上的 mask 能一致地反投影回 3D。")
lines.append("- **反投影**：5000 点正投影→带 z-buffer 遮挡判定；可见点取其像素所属 mask，"
             "被遮挡/背面点用 3D 最近已标点传播（这就是单视角盲区的诚实补齐）。")
lines.append("- **只用单视角、零再分块请求。**\n")

out = BASE / "ai_vlm_exp/COMPARISON.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(out)
print("\n".join(lines[:40]))
