"""快速 ping 一个或多个 ARK 模型 ID，看哪个是激活/有配额的。

每个模型发一个极小请求（"hi" 一个字），输出：
    model_id                            status  http  elapsed  detail
对于 200 → 模型可用；4xx → 失败原因（NotFound / SetLimitExceeded / 鉴权 / ...）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vlm.api_keys import load_dotenv  # noqa: E402

CHAT_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

# 已知 / 待验证的多模态 / 文本模型 ID。后续可继续加。
CANDIDATES = [
    "doubao-seed-2-0-pro-260215",
    "doubao-1-5-vision-pro-32k",
    "doubao-1-5-vision-lite-32k",
    "doubao-1-5-vision-pro-250328",
    "doubao-1-5-vision-lite-250315",
    "deepseek-v4-pro-260425",
]


def _resolve_key() -> str:
    for var in ("BYTEDANCE_ARK_API_KEY", "ARK_API_KEY", "SEEDANCE_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    raise SystemExit("未找到 ARK API key")


def ping(api_key: str, model: str, timeout: int = 30) -> tuple[str, int | None, float, str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(CHAT_URL, headers=headers, json=body, timeout=timeout)
    except requests.RequestException as exc:
        return "NETWORK_ERR", None, time.perf_counter() - t0, str(exc)
    elapsed = time.perf_counter() - t0
    if resp.ok:
        try:
            data = resp.json()
            reply = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError):
            reply = "<unparsable>"
        return "OK", resp.status_code, elapsed, f"reply={reply!r}"
    # try to extract structured error
    try:
        err = resp.json().get("error", {})
        code = err.get("code") or "Error"
        msg = (err.get("message") or "")[:200]
    except ValueError:
        code = "Error"
        msg = resp.text[:200]
    return code, resp.status_code, elapsed, msg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*", default=None, help="自定义 model id 列表；缺省=内置候选")
    p.add_argument("--timeout", type=int, default=30)
    args = p.parse_args()

    load_dotenv()
    api_key = _resolve_key()
    models = args.models or CANDIDATES

    print(f"key tail: ...{api_key[-8:]}")
    print(f"models  : {len(models)}")
    print()
    print(f"{'model':<36}  {'status':<24}  {'http':>4}  {'sec':>5}  detail")
    print("-" * 110)
    for m in models:
        status, http, sec, detail = ping(api_key, m, timeout=args.timeout)
        http_str = str(http) if http is not None else "  -"
        print(f"{m:<36}  {status:<24}  {http_str:>4}  {sec:5.1f}  {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
