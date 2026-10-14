"""
build_semantic_tree.py — 近似语义合并 → 语义树

把连通域标注产出的"平面语义标签"通过近似语义合并构建层级语义树。

方法
----
1. 语义亲和度 (affinity, 0..1)：对每对标签计算 soft 相似度
     - 词元 Jaccard（标签按 '_' 切分后的交集/并集）
     - 子串包含奖励
     - 共享 root token 奖励（如 sword_blade / sword_guard 同根 sword）
     - 功能类别同组奖励（curated SEMANTIC_GROUPS，可扩展）
   距离 = 1 - affinity，即"近似"由阈值控制。
2. 层次聚类：scipy UPGMA（平均链接）得到树状图，按距离阈值切出层级。
3. 内部节点命名：公共词根 + SEMANTIC_GROUPS 词典推断（标注为建议名）。

可选升级：把 affinity 换成 sentence-transformers 句向量余弦相似度，
脚本已预留 embedder 接口，无需改动聚类逻辑。

用法
----
  python build_semantic_tree.py --results <results.json> --outdir <dir> [--threshold 0.5]
"""
import json
import os
import re
import argparse
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# 1. 标签解析与语义亲和度
# ---------------------------------------------------------------------------

# 安全的后缀剥离（仅剥已知功能后缀，避免误伤，如 base->bas）
SUFFIXES = (
    "_tip", "_guard", "_hilt", "_blade", "_armor", "_plate",
    "_cover", "_cap", "_top", "_bottom", "_left", "_right",
    "_front", "_back",
)

# 功能类别词典：root token -> 建议组名。可随意扩展；未知 root 退回其本身。
SEMANTIC_GROUPS = {
    # 武器 / 攻击
    "sword": "Weaponry", "dagger": "Weaponry", "shield": "Weaponry",
    "blade": "Weaponry", "spear": "Weaponry", "axe": "Weaponry",
    "bow": "Weaponry", "club": "Weaponry",
    # 护甲
    "torso": "Armor", "chest": "Armor", "legs": "Armor", "greaves": "Armor",
    "helmet": "Armor", "head": "Armor", "shoulder": "Armor",
    "pauldron": "Armor", "glove": "Armor", "gauntlet": "Armor",
    "arms": "Armor", "cuirass": "Armor", "boot": "Armor",
    # 支撑 / 基座
    "base": "Support", "stand": "Support", "pedestal": "Support",
    "mount": "Support", "foot": "Support",
    # 装饰
    "horn": "Decoration", "hair": "Decoration", "crest": "Decoration",
    "ornament": "Decoration", "feather": "Decoration",
    # 解剖（生物形）
    "body": "Anatomy", "torso_b": "Anatomy", "head_a": "Anatomy",
    "tail": "Anatomy", "arm": "Anatomy", "hand": "Anatomy",
    "jaw": "Anatomy", "leg": "Anatomy", "wing": "Anatomy",
    # 服饰
    "cloak": "Apparel", "cape": "Apparel", "robe": "Apparel",
    "cloth": "Apparel", "skirt": "Apparel",
}

# 同组奖励强度（affinity 分量，0..1）
GROUP_BONUS = 0.6
ROOT_BONUS = 0.9
SUBSTRING_BONUS = 0.7


def tokenize(label: str):
    return [t for t in re.split(r"[_\s]+", label.lower()) if t]


def root_token(label: str):
    toks = tokenize(label)
    if not toks:
        return label.lower()
    last = toks[-1]
    for suf in SUFFIXES:
        if last.endswith(suf) and len(last) > len(suf):
            toks[-1] = last[: -len(suf)]
            break
    return toks[0]


def group_of(label: str):
    return SEMANTIC_GROUPS.get(root_token(label))


def similarity(a: str, b: str) -> float:
    """近似语义相似度 0..1（越大越该合并）。"""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = set(ta) & set(tb)
    union = set(ta) | set(tb)
    jac = len(inter) / len(union)

    sub = SUBSTRING_BONUS if (a in b or b in a) else 0.0
    root = ROOT_BONUS if root_token(a) == root_token(b) else 0.0
    grp = GROUP_BONUS if group_of(a) is not None and group_of(a) == group_of(b) else 0.0

    return max(jac, sub, root, grp)


def build_affinity(labels):
    n = len(labels)
    aff = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            s = similarity(labels[i], labels[j])
            aff[i, j] = aff[j, i] = s
    return aff


# ---------------------------------------------------------------------------
# 2. 层次聚类 -> 嵌套语义树
# ---------------------------------------------------------------------------

def name_node(members):
    """为一组叶子标签推断内部节点名（建议名）。"""
    roots = [root_token(m) for m in members]
    common = max(set(roots), key=roots.count)
    grp = SEMANTIC_GROUPS.get(common)
    if grp:
        # 若全体成员同组且共享更细词根，用更具体的名字
        if len(set(roots)) == 1:
            return common  # 如 sword
        return grp
    if len(set(roots)) == 1:
        return common
    return "group:" + common


def collect_leaf_labels(node):
    if node["type"] == "leaf":
        return [node["label"]]
    out = []
    for c in node["children"]:
        out.extend(collect_leaf_labels(c))
    return out


def prune(node, labels, threshold):
    """按阈值切树：merge 距离 <= threshold 的内部节点保留为语义簇；
    高于阈值的节点被展平（其子树提升为兄弟），形成森林。"""
    if node.is_leaf():
        return [{"type": "leaf", "label": labels[node.id], "id": int(node.id)}]
    left = prune(node.left, labels, threshold)
    right = prune(node.right, labels, threshold)
    if node.dist <= threshold:
        members = collect_leaf_labels({"children": left + right, "type": "node"})
        return [{
            "type": "node",
            "height": float(node.dist),
            "name": name_node(members),
            "members": members,
            "children": left + right,
        }]
    return left + right  # 展平：不再形成高层节点


def collapse_redundant(node):
    """若子节点与父节点同名，则把该子节点的子孙提升一层（消除冗余嵌套）。"""
    if node["type"] == "leaf":
        return node
    node["children"] = [collapse_redundant(c) for c in node["children"]]
    new = []
    changed = False
    for c in node["children"]:
        if c["type"] == "node" and c.get("name") == node.get("name"):
            new.extend(c["children"])
            changed = True
        else:
            new.append(c)
    if changed:
        node["children"] = new
        node["members"] = collect_leaf_labels(node)
    return node


def build_tree(labels, threshold):
    n = len(labels)
    if n == 1:
        return {"type": "leaf", "label": labels[0], "id": 0}
    aff = build_affinity(labels)
    dist = (1.0 - aff).clip(0, 1)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    root = to_tree(Z)
    forest = prune(root, labels, threshold)  # 顶层森林（已按阈值切）
    tree = {
        "type": "node",
        "name": "Object",
        "height": float(root.dist),
        "members": list(labels),
        "children": forest,
    }
    tree = collapse_redundant(tree)
    tree = collapse_redundant(tree)
    return tree


def _hls_hex(h, l, s):
    """HLS(0..1) -> #RRGGBB。"""
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h % 1.0, max(0.0, min(1.0, l)),
                                  max(0.0, min(1.0, s)))
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def assign_hierarchical_colors(tree):
    """按树路径分配颜色：共同祖先越深，色相差越小。

    - 根节点为中性灰；
    - 顶层子树使用黄金角分离的主色相；
    - 每下降一级仅在父色相附近做更小偏移，并略微提高饱和度/降低亮度；
    - 因而兄弟节点近色、远亲节点异色，且每一层仍可由亮度辨认。
    颜色写回每个 node 的 color/depth，并返回 label -> color。
    """
    tree["color"] = "#64748b"
    tree["depth"] = 0
    leaf_colors = {}

    def walk(node, hue, depth):
        node["depth"] = depth
        # 深层节点保持父色系，亮度逐层收紧；叶子最饱和。
        is_leaf = node["type"] == "leaf"
        sat = min(0.82, 0.58 + depth * 0.055 + (0.06 if is_leaf else 0.0))
        light = max(0.37, 0.62 - depth * 0.055 - (0.035 if is_leaf else 0.0))
        node["color"] = _hls_hex(hue, light, sat)
        if is_leaf:
            leaf_colors.setdefault(node["label"], node["color"])
            return
        children = node.get("children", [])
        n = len(children)
        # 子代偏移随深度指数缩小：同一深层语义簇颜色高度接近。
        span = 0.085 / (2 ** max(0, depth - 1))
        for i, child in enumerate(children):
            offset = 0.0 if n == 1 else ((i / (n - 1)) - 0.5) * span
            walk(child, hue + offset, depth + 1)

    children = tree.get("children", []) if tree["type"] != "leaf" else [tree]
    if tree["type"] == "leaf":
        walk(tree, 0.58, 1)
    else:
        for i, child in enumerate(children):
            # 黄金角主色 + 固定起点，避免相邻顶层簇近色。
            hue = (0.03 + i * 0.61803398875) % 1.0
            walk(child, hue, 1)
    return leaf_colors


# ---------------------------------------------------------------------------
# 3. 面积统计（每个标签的面数占比）
# ---------------------------------------------------------------------------

def count_faces_per_label(model_dir, part_labels):
    """本流程：face_block.npy(面->block_id) + part_labels(block_id->label) 统计面数。

    旧格式 face_semantic_labels_fullcover.json 不再使用。
    """
    counts = defaultdict(int)
    fb = os.path.join(model_dir, "face_block.npy")
    if not os.path.exists(fb):
        return counts
    face_block = np.load(fb)
    lab_of = {int(k): v for k, v in part_labels.items()}
    for b in face_block:
        lab = lab_of.get(int(b))
        if lab:
            counts[lab] += 1
    return counts


# ---------------------------------------------------------------------------
# 4. 渲染
# ---------------------------------------------------------------------------

DEPTH_COLORS = [
    "#6b7280", "#2563eb", "#0891b2", "#059669",
    "#d97706", "#dc2626", "#7c3aed", "#db2777",
]


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render_tree_html(node, face_counts, total, depth=0):
    color = node.get("color", DEPTH_COLORS[min(depth, len(DEPTH_COLORS) - 1)])
    level = node.get("depth", depth)
    if node["type"] == "leaf":
        lab = node["label"]
        c = face_counts.get(lab, 0)
        pct = (100.0 * c / total) if total else 0.0
        count_text = f'{c:,} faces · {pct:.1f}%' if total else 'semantic leaf'
        return (
            f'<li class="leaf"><span class="branch" style="border-color:{color}"></span>'
            f'<span class="dot" style="background:{color}"></span>'
            f'<b>{esc(lab)}</b> '
            f'<span class="level">L{level}</span>'
            f'<span class="meta">{count_text}</span></li>'
        )
    name = node.get("name", "group")
    inner = "".join(render_tree_html(c, face_counts, total, depth + 1)
                    for c in node["children"])
    return (
        f'<li class="node"><details open><summary>'
        f'<span class="dot node-dot" style="background:{color}"></span>'
        f'<b>{esc(name)}</b> <span class="level">L{level}</span>'
        f'<span class="meta">{len(node["members"])} parts · '
        f'merge@dist {node["height"]:.2f}</span></summary>'
        f'<ul style="border-left-color:{color}">{inner}</ul></details></li>'
    )


def model_block(slug, tree, face_counts, total, threshold):
    html = render_tree_html(tree, face_counts, total, 0)
    return f"""
    <section class="model">
      <h2>{esc(slug)}</h2>
      <p class="sub">approximate-merge threshold = dist {threshold:.2f}
         (affinity ≥ {1-threshold:.2f}) · {total:,} faces</p>
      <ul class="tree">{html}</ul>
    </section>
    """


def build_gallery(models, corpus_tree, out_path, threshold):
    blocks = []
    for m in models:
        blocks.append(model_block(m["slug"], m["tree"], m["face_counts"],
                                  m["total"], threshold))
    corpus_html = render_tree_html(corpus_tree, {}, 0, 0)
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>语义树画廊 · Semantic Trees</title>
<style>
  body {{ font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         margin:0; background:#f7f8fa; color:#1f2937; }}
  header {{ background:#111827; color:#fff; padding:20px 28px; }}
  header h1 {{ margin:0; font-size:20px; }}
  header p {{ margin:6px 0 0; color:#9ca3af; font-size:13px; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:24px; }}
  .model {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px;
           padding:18px 22px; margin-bottom:20px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .model h2 {{ margin:0 0 4px; font-size:17px; }}
  .sub {{ color:#6b7280; font-size:12px; margin:0 0 12px; }}
  .corpus {{ background:#fffbeb; border:1px solid #fde68a; border-radius:10px;
            padding:18px 22px; margin-bottom:20px; }}
  ul.tree, ul.tree ul {{ list-style:none; margin:7px 0 7px 18px; padding-left:12px; }}
  ul.tree ul {{ border-left:3px solid #d1d5db; }}
  li.node > details > summary {{ cursor:pointer; font-weight:600; padding:2px 0; }}
  li.leaf, li.node {{ margin:4px 0; }}
  .dot {{ display:inline-block; width:12px; height:12px; border-radius:50%;
         margin-right:7px; vertical-align:middle; box-shadow:0 0 0 2px #fff,0 0 0 3px #d1d5db; }}
  .node-dot {{ width:14px; height:14px; }}
  .level {{ display:inline-block; font:10px/1.2 ui-monospace,monospace; color:#475569;
            background:#eef2f7; border-radius:4px; padding:2px 5px; margin-left:6px; }}
  .meta {{ color:#6b7280; font-size:12px; margin-left:6px; }}
  b {{ font-weight:600; }}
</style></head><body>
<header><h1>近似语义合并 · 语义树画廊</h1>
<p>connected-component 平面标签 → soft affinity + UPGMA 层次聚类 → 语义树。
跨模型语料树合并反复出现的概念。</p></header>
<div class="wrap">
  <div class="corpus"><h2>跨模型语料语义树（Corpus Tree）</h2>
    <p class="sub">所有模型标签并集后的总树</p>
    <ul class="tree">{corpus_html}</ul></div>
  {''.join(blocks)}
</div></body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.results) as f:
        results = json.load(f)

    base = os.path.dirname(args.results)  # output/<run>/

    models = []
    corpus_labels = []
    for r in results:
        slug = r["slug"]
        part_labels = r.get("part_labels", {})
        labels = list(part_labels.values())
        if not labels:
            continue
        tree = build_tree(labels, args.threshold)
        leaf_colors = assign_hierarchical_colors(tree)

        # 各模型目录即为 results 所在目录（base/slug），读取 face_block.npy 统计面数
        model_dir = base
        face_counts = count_faces_per_label(model_dir, part_labels)
        total = sum(face_counts.values()) or 0

        out_json = os.path.join(args.outdir, f"{slug}_semantic_tree.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"slug": slug, "threshold": args.threshold,
                       "color_rule": "hierarchical: shared deeper ancestor -> closer hue; depth -> lightness",
                       "leaf_colors": leaf_colors,
                       "face_counts": dict(face_counts), "tree": tree},
                      f, indent=2, ensure_ascii=False)

        models.append({"slug": slug, "tree": tree, "leaf_colors": leaf_colors,
                       "face_counts": face_counts, "total": total})
        corpus_labels.extend(labels)

    # 跨模型语料树（去重保序）
    seen, uniq = set(), []
    for l in corpus_labels:
        if l not in seen:
            seen.add(l); uniq.append(l)
    corpus_tree = build_tree(uniq, args.threshold)
    assign_hierarchical_colors(corpus_tree)

    gallery = os.path.join(args.outdir, "semantic_tree_gallery.html")
    build_gallery(models, corpus_tree, gallery, args.threshold)

    print(f"[ok] {len(models)} models -> {gallery}")
    print(f"[ok] corpus tree labels: {len(uniq)}")


if __name__ == "__main__":
    main()
