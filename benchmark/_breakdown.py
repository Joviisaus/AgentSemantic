import json
from pathlib import Path

d = json.loads(Path(r"E:\Proj\AgentSematic\benchmark\results\summary_tiny.json").read_text(encoding="utf-8"))
r = {x["method"]: {s["key"]: s for s in x["per_sample"]} for x in d["results"]}
keys = sorted(r["ours"])
print(f'{"category":26} {"ours":>7} {"cosmo":>7}  winner')
print("-" * 52)
ow = cw = tie = 0
for k in keys:
    o = r["ours"][k]["miou_skip"]
    c = r["cosmo3d"][k]["miou_skip"]
    if o > c + 1e-6:
        w = "ours"; ow += 1
    elif c > o + 1e-6:
        w = "cosmo"; cw += 1
    else:
        w = "="; tie += 1
    print(f'{r["ours"][k]["category"]:26} {o:7.3f} {c:7.3f}  {w}')
print("-" * 52)
print(f"per-shape wins -> ours:{ow}  cosmo:{cw}  tie:{tie}")
