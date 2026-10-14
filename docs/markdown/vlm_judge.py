#!/usr/bin/env python3
"""
VLM 评分脚本 V5 — 2 阶段：先描述，再评分
- 用智谱 GLM-4V-Plus（已验证能看图）
- 强 JSON 解析
- 缓存结果
"""
import os, sys, base64, json, mimetypes, re, time
from pathlib import Path

ROOT = Path("E:/Proj/AgentSematic")
OUT_JSON = ROOT / "docs" / "markdown" / "_vlm_judgement.json"

# ---- load env ----
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v

ZHIPU_KEY = os.environ.get("ZHIPU_API_KEY", "")
ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPU_MODEL = "glm-4v-plus"

import requests


def img_b64(path: Path) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    with open(path, "rb") as f:
        return mime, base64.b64encode(f.read()).decode("ascii")


DESC_PROMPT = """这是一张 3D 点云语义分割对比图，共有 3 个面板（从左到右）：GT（真实标签）、ours_pointcloud（我们的方法）、cosmo3d（基线）。
颜色相同的点代表同一种语义部件。灰色 = 未分割/未分类。预测越接近 GT 越好。

请**真正仔细看图**后用中文描述每个面板的颜色分布和点云形态。**重点说有没有彩色点、彩色点分布的形状像不像 GT 的形状**。只描述，不要评分。

示例格式：
GT: [...]
ours_pointcloud: [...]
cosmo3d: [..."""

SCORE_PROMPT = """GT 面板的描述: {desc_gt}
ours_pointcloud 面板的描述: {desc_ours}
cosmo3d 面板的描述: {desc_cos}

GT 作为参照。综合考虑各面板和 GT 的颜色形状相似度（0-10，10=几乎一样，0=完全不同或全灰）评估。
注意：如果一个面板基本全灰（没分割出任何部件），分数应该很低；如果彩色分布与 GT 形态吻合，分数应该高。

只输出严格 JSON（不要 markdown 包裹，不要任何解释）：
{{"gt_ours": N, "gt_cosmo3d": N, "ours_leading": "yes/no/tie", "comment": "一句话说明"}}"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    s = text.find("{")
    if s < 0:
        raise ValueError(f"no JSON: {text[:200]!r}")
    depth, in_str, esc = 0, False, False
    for i in range(s, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[s:i+1])
    snippet = text[s:]
    snippet = re.sub(r",\s*$", "", snippet)
    if snippet.count('"') % 2 == 1:
        snippet += '"'
    open_b = snippet.count("{") - snippet.count("}")
    snippet += "}" * max(open_b, 0)
    try:
        return json.loads(snippet)
    except Exception as e:
        raise ValueError(f"unbalanced JSON: {snippet[:200]!r} ({e})")


def call_zhipu(messages, max_tokens=500, max_retry=3) -> str:
    payload = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    last_err = None
    for attempt in range(max_retry):
        try:
            r = requests.post(ZHIPU_URL, json=payload,
                              headers={"Authorization": f"Bearer {ZHIPU_KEY}",
                                       "Content-Type": "application/json"},
                              timeout=60)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    [429] sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            if "choices" not in data:
                raise RuntimeError(f"no choices: {json.dumps(data)[:200]}")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            wait = 2 * (attempt + 1)
            time.sleep(wait)
    raise RuntimeError(f"failed: {last_err}")


def vlm_score(image_path: Path) -> dict:
    """2-step: describe each panel, then score based on description"""
    mime, b64 = img_b64(image_path)
    img_url = f"data:{mime};base64,{b64}"

    # Step 1: describe
    desc = call_zhipu([{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": img_url}},
            {"type": "text", "text": DESC_PROMPT},
        ]
    }], max_tokens=500)

    # Parse description: each line should be "GT: ..." or "ours_pointcloud: ..." or "cosmo3d: ..."
    desc_gt = desc_ours = desc_cos = ""
    for line in desc.splitlines():
        line = line.strip()
        if line.lower().startswith("gt:") or line.lower().startswith("gt："):
            desc_gt = line.split(":", 1)[-1].strip()
        elif "ours" in line.lower() and ":" in line:
            desc_ours = line.split(":", 1)[-1].strip()
        elif "cosmo" in line.lower() and ":" in line:
            desc_cos = line.split(":", 1)[-1].strip()
    if not (desc_gt and desc_ours and desc_cos):
        # Try chinese colon
        for line in desc.splitlines():
            line = line.strip()
            if line.startswith("GT") and "：" in line:
                desc_gt = line.split("：", 1)[-1].strip()
            elif line.startswith("ours") and "：" in line:
                desc_ours = line.split("：", 1)[-1].strip()
            elif line.startswith("cosmo3d") and "：" in line:
                desc_cos = line.split("：", 1)[-1].strip()
    if not (desc_gt and desc_ours and desc_cos):
        # Use whole desc as fallback
        desc_gt = desc_gt or desc[:200]
        desc_ours = desc_ours or desc[:200]
        desc_cos = desc_cos or desc[:200]

    # Step 2: score
    prompt = SCORE_PROMPT.format(desc_gt=desc_gt, desc_ours=desc_ours, desc_cos=desc_cos)
    score_text = call_zhipu([{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": img_url}},
            {"type": "text", "text": prompt},
        ]
    }], max_tokens=300)
    score = _extract_json(score_text)
    score["_desc"] = {"gt": desc_gt, "ours": desc_ours, "cosmo3d": desc_cos}
    return score


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    if OUT_JSON.exists():
        try:
            results = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            ok = sum(1 for tag in results for k, v in results[tag].items()
                     if "error" not in v and "gt_ours" in v)
            print(f"Resumed: {ok} cached")
        except Exception:
            results = {}

    for tag, samples_dir in [
        ("compat200", ROOT / "benchmark/results/rendered_report/assets/samples"),
        ("shapenetpart", ROOT / "benchmark/results_shapenetpart/rendered_report/assets/samples"),
    ]:
        results.setdefault(tag, {})
        pngs = sorted(samples_dir.glob("*.png"))
        print(f"\n=== {tag}: {len(pngs)} samples ===", flush=True)
        for p in pngs:
            key = p.stem
            if key in results[tag] and "error" not in results[tag][key] and "gt_ours" in results[tag][key]:
                v = results[tag][key]
                print(f"  [cached] {key:50s}  ours={v.get('gt_ours'):>2}  cos={v.get('gt_cosmo3d'):>2}  leading={v.get('ours_leading')}", flush=True)
                continue
            try:
                score = vlm_score(p)
                results[tag][key] = score
                print(f"  {key:50s}  ours={score.get('gt_ours'):>2}  cos={score.get('gt_cosmo3d'):>2}  leading={score.get('ours_leading')}  | {score.get('comment', '')[:80]}", flush=True)
            except Exception as e:
                print(f"  {key:50s}  ERROR: {str(e)[:100]}", flush=True)
                results[tag][key] = {"error": str(e)}
            OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.8)

    # Summary
    print("\n\n=== TOP 'ours leading' cases (visual quality, not mIoU) ===", flush=True)
    for tag in results:
        good = []
        for k, v in results[tag].items():
            if v.get("ours_leading") == "yes" and "gt_ours" in v:
                ours = v.get("gt_ours", 0)
                cos = v.get("gt_cosmo3d", 0)
                gap = ours - cos
                good.append((k, ours, cos, gap, v.get("comment", "")))
        good.sort(key=lambda x: -x[3])
        print(f"\n[{tag}] {len(good)} leading cases (sorted by gap):")
        for k, ours, cos, gap, cmt in good:
            print(f"  {k:50s}  ours={ours} cos={cos} gap=+{gap}  | {cmt}")


if __name__ == "__main__":
    main()
