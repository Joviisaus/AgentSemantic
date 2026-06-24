"""Render benchmark summaries into a standalone HTML report.

Example:
    conda run -n agentsematic python -m benchmark.render_report --subset tiny --methods ours cosmo3d
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark import compat200 as C
from benchmark.dataset_registry import get_dataset


def _esc(value: object) -> str:
    return html.escape(str(value))


def _load_summary(subset: str, dataset_module=C) -> dict:
    C = dataset_module
    path = C.RESULTS_DIR / f"summary_{subset}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing summary: {path}; run benchmark.evaluate first")
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_methods(summary: dict, methods: Iterable[str] | None) -> list[dict]:
    results = list(summary.get("results", []))
    if methods:
        keep = set(methods)
        results = [r for r in results if r["method"] in keep]
    if not results:
        raise ValueError("no benchmark results to render")
    return results


def _save_aggregate_chart(results: list[dict], out_path: Path) -> None:
    labels = ["instance mIoU", "category mIoU", "point acc"]
    x = np.arange(len(labels))
    width = min(0.34, 0.8 / max(1, len(results)))
    colors = {"ours": "#4C78A8", "cosmo3d": "#F58518"}

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
    for i, result in enumerate(results):
        agg = result["aggregate"]
        values = [
            agg["instance_miou_skip"],
            agg["category_miou_skip"],
            agg["instance_accuracy"],
        ]
        offset = (i - (len(results) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=result["method"],
            color=colors.get(result["method"]),
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Aggregate benchmark metrics")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _delta_methods(method_names: Iterable[str]) -> tuple[str, str] | None:
    names = set(method_names)
    primary = "ours_pointcloud" if "ours_pointcloud" in names else "ours"
    baseline = "cosmo3d"
    if primary not in names or baseline not in names:
        return None
    return primary, baseline


def _save_delta_chart(results_by_method: dict[str, dict], out_path: Path) -> tuple[str, str] | None:
    pair = _delta_methods(results_by_method.keys())
    if pair is None:
        return None
    primary, baseline = pair

    cats = list(results_by_method[primary]["per_category"].keys())
    deltas = [
        results_by_method[primary]["per_category"][cat]
        - results_by_method[baseline]["per_category"].get(cat, float("nan"))
        for cat in cats
    ]
    order = np.argsort(deltas)
    cats_ord = [cats[i] for i in order]
    deltas_ord = [deltas[i] for i in order]

    fig_h = max(5.0, 0.34 * len(cats_ord) + 1.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_h), dpi=150)
    bar_colors = ["#4C78A8" if value >= 0 else "#F58518" for value in deltas_ord]
    ax.barh(cats_ord, deltas_ord, color=bar_colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"{primary} mIoU - {baseline} mIoU")
    ax.set_title("Per-category mIoU delta")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return pair


def _label_colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    cmap = plt.get_cmap("tab20")
    rgba = np.zeros((labels.shape[0], 4), dtype=float)
    rgba[:, 3] = 0.9
    mask = labels >= 0
    if np.any(mask):
        rgba[mask] = cmap((labels[mask] % 20) / 20.0)
        rgba[mask, 3] = 0.92
    rgba[~mask] = (0.72, 0.72, 0.72, 0.35)
    return rgba


def _render_sample_projection(key: str, methods: list[str], out_path: Path, dataset_module=C) -> bool:
    C = dataset_module
    gt_path = C.gt_path(key)
    if not gt_path.is_file():
        return False

    gt = C.load_gt(key)
    panels: list[tuple[str, np.ndarray]] = [("GT", gt.labels)]
    for method in methods:
        pred_path = C.RESULTS_DIR / method / f"{key}.npy"
        if pred_path.is_file():
            panels.append((method, np.load(pred_path).astype(np.int64)))
    if len(panels) < 2:
        return False

    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.0), dpi=130)
    if len(panels) == 1:
        axes = [axes]

    xy = gt.points[:, [0, 2]]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    pad = (hi - lo).max() * 0.04

    for ax, (title, labels) in zip(axes, panels):
        ax.scatter(xy[:, 0], xy[:, 1], s=2.0, c=_label_colors(labels), linewidths=0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(key, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _table_rows(results: list[dict]) -> str:
    rows = []
    for result in sorted(results, key=lambda item: -item["aggregate"]["instance_miou_skip"]):
        agg = result["aggregate"]
        rows.append(
            "<tr>"
            f"<td><strong>{_esc(result['method'])}</strong></td>"
            f"<td>{result['n_evaluated']}/{result['n_subset']}</td>"
            f"<td>{agg['instance_miou_skip']:.4f}</td>"
            f"<td>{agg['category_miou_skip']:.4f}</td>"
            f"<td>{agg['instance_miou_empty1']:.4f}</td>"
            f"<td>{agg['instance_accuracy']:.4f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _sample_metric_rows(results: list[dict], methods: list[str]) -> str:
    sample_by_method = {
        result["method"]: {sample["key"]: sample for sample in result["per_sample"]}
        for result in results
    }
    keys = [sample["key"] for sample in results[0]["per_sample"]]

    rows = []
    for key in keys:
        category = sample_by_method[methods[0]][key]["category"]
        cells = [f"<td><code>{_esc(key)}</code></td>", f"<td>{_esc(category)}</td>"]
        for method in methods:
            sample = sample_by_method[method][key]
            cells.append(f"<td>{sample['miou_skip']:.4f}</td>")
            cells.append(f"<td>{sample['accuracy']:.4f}</td>")
        pair = _delta_methods(sample_by_method.keys())
        if pair is not None:
            primary, baseline = pair
            delta = (
                sample_by_method[primary][key]["miou_skip"]
                - sample_by_method[baseline][key]["miou_skip"]
            )
            cls = "pos" if delta >= 0 else "neg"
            cells.append(f"<td class='{cls}'>{delta:+.4f}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(rows)


def _sample_cards(results: list[dict], methods: list[str], sample_images: dict[str, str]) -> str:
    sample_by_method = {
        result["method"]: {sample["key"]: sample for sample in result["per_sample"]}
        for result in results
    }
    cards = []
    for key, rel_path in sample_images.items():
        category = sample_by_method[methods[0]][key]["category"]
        metric_bits = [
            f"{_esc(method)} mIoU {sample_by_method[method][key]['miou_skip']:.3f}"
            for method in methods
        ]
        cards.append(
            "<section class='sample-card'>"
            f"<h3>{_esc(category)}</h3>"
            f"<p><code>{_esc(key)}</code><br>{' · '.join(metric_bits)}</p>"
            f"<img src='{_esc(rel_path)}' alt='GT and prediction projection for {_esc(key)}'>"
            "</section>"
        )
    return "".join(cards)


def render_report(
    subset: str,
    methods: list[str] | None,
    output_dir: Path,
    *,
    dataset: str = "compat200",
    dataset_module=C,
) -> Path:
    C = dataset_module
    summary = _load_summary(subset, C)
    results = _filter_methods(summary, methods)
    method_names = [result["method"] for result in results]
    by_method = {result["method"]: result for result in results}

    assets = output_dir / "assets"
    sample_dir = assets / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    _save_aggregate_chart(results, assets / "aggregate_metrics.png")
    delta_pair = _save_delta_chart(by_method, assets / "category_delta.png")

    sample_images: dict[str, str] = {}
    keys = [sample["key"] for sample in results[0]["per_sample"]]
    for key in keys:
        out_path = sample_dir / f"{key}.png"
        if _render_sample_projection(key, method_names, out_path, C):
            sample_images[key] = f"assets/samples/{key}.png"

    method_headers = "".join(
        f"<th>{_esc(method)} mIoU</th><th>{_esc(method)} acc</th>"
        for method in method_names
    )
    delta_header = (
        f"<th>{_esc(delta_pair[0])} - {_esc(delta_pair[1])}</th>" if delta_pair else ""
    )
    best = max(result["aggregate"]["instance_miou_skip"] for result in results)

    delta_panel = (
        '<div><img src="assets/category_delta.png" alt="Per-category delta chart">'
        f'<div class="caption">Positive values favor {_esc(delta_pair[0])}; '
        f'negative values favor {_esc(delta_pair[1])}.</div></div>'
        if delta_pair
        else ""
    )
    sample_section = (
        '<h2>Point-cloud Projection Previews</h2>'
        "<p>Each panel uses the same GT point cloud projected to X/Z. Grey points are "
        "unlabeled; colored points are part IDs modulo a shared categorical palette.</p>"
        f'<div class="samples">{_sample_cards(results, method_names, sample_images)}</div>'
        if sample_images
        else ""
    )
    dataset_key = dataset.lower().replace("-", "_")
    dataset_desc = (
        "ShapeNet-Part semantic part segmentation. This report is a supplementary "
        "point-cloud benchmark; CoMPaT-200 remains the primary CoSMo3D comparison."
        if dataset_key in {"shapenetpart", "shapenet_part"}
        else "CoMPaT-200 coarse semantic part segmentation. Metrics use per-point "
        "mIoU with empty-union parts skipped, matching the CoSMo3D convention."
    )
    summary_source = C.RESULTS_DIR / f"summary_{subset}.json"

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AgentSematic Benchmark Report - {_esc(dataset)} - {_esc(subset)}</title>
<style>
:root {{ color-scheme: light; --ink:#1f2937; --muted:#667085; --line:#d0d5dd; --bg:#f8fafc; --card:#ffffff; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
h2 {{ margin: 32px 0 12px; font-size: 21px; }}
h3 {{ margin: 0 0 6px; font-size: 15px; }}
p {{ color: var(--muted); line-height: 1.5; }}
.hero {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 20px; align-items: start; }}
.sample-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 18px; }}
.stats {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.stat {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 16px; }}
.stat .label {{ color: var(--muted); font-size: 13px; }}
.stat .value {{ font-size: 28px; font-weight: 700; margin-top: 6px; }}
.chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
.chart-grid img, .sample-card img {{ width: 100%; height: auto; display: block; border: 1px solid var(--line); border-radius: 8px; background: white; }}
table {{ width: 100%; border-collapse: collapse; background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; font-size: 13px; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #eef2f6; color: #344054; }}
tr:last-child td {{ border-bottom: 0; }}
code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; }}
.pos {{ color: #067647; font-weight: 700; }}
.neg {{ color: #b42318; font-weight: 700; }}
.samples {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
.sample-card p {{ margin: 0 0 12px; font-size: 13px; }}
.caption {{ font-size: 12px; color: var(--muted); margin-top: 8px; }}
@media (max-width: 900px) {{ .hero, .chart-grid, .samples {{ grid-template-columns: 1fr; }} .stats {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <h1>AgentSematic Benchmark Report</h1>
      <p>Dataset <strong>{_esc(dataset)}</strong>, subset <strong>{_esc(subset)}</strong>. {_esc(dataset_desc)}</p>
    </div>
    <div class="stats">
      <div class="stat"><div class="label">Best instance mIoU</div><div class="value">{best:.3f}</div></div>
      <div class="stat"><div class="label">Methods</div><div class="value">{len(results)}</div></div>
      <div class="stat"><div class="label">Samples</div><div class="value">{len(keys)}</div></div>
    </div>
  </section>

  <h2>Aggregate Results</h2>
  <table><thead><tr><th>method</th><th>n_eval</th><th>instance mIoU</th><th>category mIoU</th><th>mIoU empty=1</th><th>point acc</th></tr></thead><tbody>{_table_rows(results)}</tbody></table>

  <h2>Rendered Metric Charts</h2>
  <div class="chart-grid">
    <div><img src="assets/aggregate_metrics.png" alt="Aggregate metrics bar chart"><div class="caption">Source: {_esc(summary_source)}</div></div>
    {delta_panel}
  </div>

  <h2>Per-sample Metrics</h2>
  <table><thead><tr><th>key</th><th>category</th>{method_headers}{delta_header}</tr></thead><tbody>{_sample_metric_rows(results, method_names)}</tbody></table>

  {sample_section}
</main>
</body>
</html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render benchmark comparison report")
    parser.add_argument("--dataset", default="compat200",
                        help="dataset name: compat200 (default, CoSMo3D comparison) or shapenetpart")
    parser.add_argument("--subset", default="tiny")
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args(argv)

    C_selected = get_dataset(args.dataset)
    output_dir = args.output_dir or (C_selected.RESULTS_DIR / "rendered_report")
    report = render_report(
        args.subset,
        args.methods,
        output_dir,
        dataset=args.dataset,
        dataset_module=C_selected,
    )
    print(f"[report] {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
