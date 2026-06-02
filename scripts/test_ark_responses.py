"""快速验证火山方舟 ARK /v3/responses 接口（DeepSeek-V4-Pro + web_search 工具）。

用法:
    python scripts/test_ark_responses.py [--stream]

读取环境变量 ARK_API_KEY 或 BYTEDANCE_ARK_API_KEY（兼容旧的 SEEDANCE_API_KEY）。
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

URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
MODEL = "deepseek-v4-pro-260425"


def _resolve_key() -> str:
    for var in ("BYTEDANCE_ARK_API_KEY", "ARK_API_KEY", "SEEDANCE_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    raise SystemExit("未找到 ARK API key (BYTEDANCE_ARK_API_KEY / ARK_API_KEY)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stream", action="store_true", help="启用流式输出")
    p.add_argument("--prompt", default="今天有什么热点新闻", help="用户提问")
    p.add_argument("--timeout", type=int, default=120, help="单次 HTTP 超时秒数")
    args = p.parse_args()

    load_dotenv()
    api_key = _resolve_key()

    body = {
        "model": MODEL,
        "stream": bool(args.stream),
        "tools": [{"type": "web_search", "max_keyword": 3}],
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": args.prompt}],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"POST {URL}")
    print(f"  model    = {MODEL}")
    print(f"  stream   = {body['stream']}")
    print(f"  tool     = web_search(max_keyword=3)")
    print(f"  prompt   = {args.prompt}")
    print(f"  key tail = ...{api_key[-8:]}")
    print()

    t0 = time.perf_counter()
    resp = requests.post(
        URL, headers=headers, json=body, timeout=args.timeout, stream=body["stream"]
    )

    if not resp.ok:
        elapsed = time.perf_counter() - t0
        print(f"HTTP {resp.status_code} (elapsed {elapsed:.1f}s)")
        print("----- error body -----")
        print(resp.text[:4000])
        return 1

    if body["stream"]:
        print(f"HTTP {resp.status_code} (streaming)")
        print("----- SSE events (first 30) -----")
        n = 0
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            print(line)
            n += 1
            if n >= 30:
                print(f"...(truncated after {n} lines)...")
                break
        elapsed = time.perf_counter() - t0
        print(f"\nelapsed {elapsed:.1f}s")
        return 0

    elapsed = time.perf_counter() - t0
    print(f"HTTP {resp.status_code} (elapsed {elapsed:.1f}s)")
    print("----- response body -----")
    try:
        data = resp.json()
        print(json.dumps(data, ensure_ascii=False, indent=2)[:4000])
    except ValueError:
        print(resp.text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
