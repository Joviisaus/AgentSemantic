"""快速 ping vlm_config.yaml 里所有有 key 的 provider，看哪些能用。

每个 provider 发一个 1-token 的小请求（不带图）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vlm.client import build_vlm_client  # noqa: E402
from vlm.config import get_provider_config, load_vlm_config  # noqa: E402

PROVIDERS = ["doubao", "hy3", "qwen", "google", "openai", "anthropic", "zhipu"]


def main() -> int:
    cfg = load_vlm_config()
    print(f"{'provider':<10}  {'model':<32}  {'has_key':<7}  {'status':<14}  {'sec':>5}  detail")
    print("-" * 110)
    for name in PROVIDERS:
        try:
            pc = get_provider_config(cfg, name)
        except ValueError as exc:
            print(f"{name:<10}  {'-':<32}  {'?':<7}  {'NO_CONFIG':<14}  {'-':>5}  {exc}")
            continue
        if not pc.has_key():
            print(f"{name:<10}  {pc.default_model:<32}  {'no':<7}  {'SKIP':<14}  {'-':>5}  (无 API key)")
            continue
        client = build_vlm_client(pc)
        t0 = time.perf_counter()
        try:
            resp = client.chat(
                user_prompt="hi",
                image_paths=[],
                temperature=0.0,
                max_tokens=4,
            )
            elapsed = time.perf_counter() - t0
            reply = (resp.text or "").replace("\n", " ")[:40]
            print(
                f"{name:<10}  {pc.default_model:<32}  {'yes':<7}  "
                f"{'OK':<14}  {elapsed:5.1f}  reply={reply!r}"
            )
        except (RuntimeError, ValueError) as exc:
            elapsed = time.perf_counter() - t0
            msg = str(exc)
            # 抠出 HTTP code 关键词
            kw = "ERR"
            for tok in ("HTTP 429", "HTTP 404", "HTTP 401", "HTTP 403", "HTTP 400", "HTTP 5"):
                if tok in msg:
                    kw = tok.replace("HTTP ", "")
                    break
            print(
                f"{name:<10}  {pc.default_model:<32}  {'yes':<7}  "
                f"{kw:<14}  {elapsed:5.1f}  {msg[:200]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
