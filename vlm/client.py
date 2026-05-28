"""
VLM 客户端：把不同厂商的视觉对话 API 抽象成统一接口
    client.chat(messages=[...], images=[...]) -> str (raw text reply)

支持三类：
  - OpenAI Chat Completions 兼容（openai / qwen / zhipu / moonshot / deepseek / ollama / custom）
  - Azure OpenAI（同上但 endpoint 不同）
  - Anthropic Messages
  - Google Gemini generateContent

为避免引入官方 SDK 重型依赖，统一用 requests + base64 调 HTTP。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError("VLM client 需要 requests: pip install requests") from exc

from .config import VLMProviderConfig


@dataclass
class VLMResponse:
    text: str
    raw: Dict[str, Any]
    provider: str
    model: str
    elapsed_seconds: float


def _read_image_b64(path: Path) -> tuple[str, str]:
    """返回 (mime_type, base64_str)。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if not mime or not mime.startswith("image/"):
        mime = "image/png"
    with open(path, "rb") as f:
        data = f.read()
    return mime, base64.b64encode(data).decode("ascii")


class VLMClient:
    """子类必须实现 _do_chat。"""

    def __init__(self, config: VLMProviderConfig):
        self.config = config

    # ----- 公共接口 -----
    def chat(
        self,
        user_prompt: str,
        image_paths: Sequence[str | Path] = (),
        *,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
        model: Optional[str] = None,
    ) -> VLMResponse:
        attempts = 0
        last_exc: Optional[Exception] = None
        backoff = 1.0
        max_retries = max(1, int(self.config.max_retries))
        chosen_model = model or self.config.default_model
        if not chosen_model:
            raise ValueError(f"provider {self.config.name} 未指定 model")

        while attempts < max_retries:
            attempts += 1
            t0 = time.perf_counter()
            try:
                text, raw = self._do_chat(
                    user_prompt=user_prompt,
                    image_paths=[Path(p) for p in image_paths],
                    system=system,
                    temperature=temperature if temperature is not None else self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    json_mode=json_mode,
                    model=chosen_model,
                )
                return VLMResponse(
                    text=text,
                    raw=raw,
                    provider=self.config.name,
                    model=chosen_model,
                    elapsed_seconds=time.perf_counter() - t0,
                )
            except (requests.RequestException, RuntimeError) as exc:
                last_exc = exc
                if attempts >= max_retries:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 8.0)
        assert last_exc is not None
        raise RuntimeError(
            f"VLM 调用失败（provider={self.config.name}, 重试 {max_retries} 次）: {last_exc}"
        ) from last_exc

    # ----- 子类实现 -----
    def _do_chat(
        self,
        *,
        user_prompt: str,
        image_paths: List[Path],
        system: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        model: str,
    ) -> tuple[str, Dict[str, Any]]:
        raise NotImplementedError


class OpenAICompatibleClient(VLMClient):
    """OpenAI Chat Completions 协议。"""

    def _do_chat(
        self,
        *,
        user_prompt: str,
        image_paths: List[Path],
        system: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        model: str,
    ) -> tuple[str, Dict[str, Any]]:
        if not self.config.base_url:
            raise RuntimeError(f"provider {self.config.name} 缺少 base_url")
        url = self.config.base_url.rstrip("/") + "/chat/completions"

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        image_detail = (
            (self.config.extra.get("vision") or {}).get("image_detail", "auto")
            if isinstance(self.config.extra.get("vision"), dict)
            else "auto"
        )
        for p in image_paths:
            mime, b64 = _read_image_b64(p)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{b64}",
                        "detail": image_detail,
                    },
                }
            )

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key != "ollama":
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        resp = requests.post(
            url, headers=headers, json=body, timeout=self.config.timeout_seconds
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:512]}")
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"解析响应失败: {data}") from exc
        if isinstance(text, list):  # 某些 provider 返回结构化 content
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        return str(text), data


class AzureOpenAIClient(VLMClient):
    """Azure OpenAI: URL 形如 {endpoint}/openai/deployments/{deployment}/chat/completions?api-version=..."""

    def _do_chat(
        self,
        *,
        user_prompt: str,
        image_paths: List[Path],
        system: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        model: str,
    ) -> tuple[str, Dict[str, Any]]:
        endpoint = str(self.config.extra.get("endpoint") or "").rstrip("/")
        deployment = str(self.config.extra.get("deployment_name") or model)
        api_version = str(self.config.extra.get("api_version") or "2024-02-15-preview")
        if not endpoint:
            raise RuntimeError("azure_openai 缺少 endpoint")
        url = (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for p in image_paths:
            mime, b64 = _read_image_b64(p)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )

        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        body: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json", "api-key": self.config.api_key}
        resp = requests.post(
            url, headers=headers, json=body, timeout=self.config.timeout_seconds
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:512]}")
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return str(text), data


class AnthropicClient(VLMClient):
    """Anthropic Messages API。"""

    def _do_chat(
        self,
        *,
        user_prompt: str,
        image_paths: List[Path],
        system: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        model: str,
    ) -> tuple[str, Dict[str, Any]]:
        base = (self.config.base_url or "https://api.anthropic.com").rstrip("/")
        url = f"{base}/v1/messages"

        content: List[Dict[str, Any]] = []
        for p in image_paths:
            mime, b64 = _read_image_b64(p)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": b64},
                }
            )
        content.append({"type": "text", "text": user_prompt})

        body: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            body["system"] = system

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": str(self.config.extra.get("api_version") or "2023-06-01"),
        }
        resp = requests.post(
            url, headers=headers, json=body, timeout=self.config.timeout_seconds
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:512]}")
        data = resp.json()
        parts: Iterable[Dict[str, Any]] = data.get("content", []) or []
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        return text, data


class GoogleGeminiClient(VLMClient):
    """Google Generative Language API。"""

    def _do_chat(
        self,
        *,
        user_prompt: str,
        image_paths: List[Path],
        system: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        model: str,
    ) -> tuple[str, Dict[str, Any]]:
        base = (self.config.base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = f"{base}/models/{model}:generateContent?key={self.config.api_key}"

        parts: List[Dict[str, Any]] = [{"text": user_prompt}]
        for p in image_paths:
            mime, b64 = _read_image_b64(p)
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})

        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=body,
            timeout=self.config.timeout_seconds,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:512]}")
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini 空响应: {data}")
        cand_parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in cand_parts if isinstance(p, dict))
        return text, data


_OPENAI_COMPATIBLE = {
    "openai",
    "qwen",
    "zhipu",
    "moonshot",
    "deepseek",
    "ollama",
    "custom",
}


def build_vlm_client(config: VLMProviderConfig) -> VLMClient:
    """根据 provider 名字挑选合适的客户端实现。"""
    name = config.name
    if name == "anthropic":
        return AnthropicClient(config)
    if name == "google":
        return GoogleGeminiClient(config)
    if name == "azure_openai":
        return AzureOpenAIClient(config)
    if name in _OPENAI_COMPATIBLE:
        return OpenAICompatibleClient(config)
    return OpenAICompatibleClient(config)


def parse_json_response(text: str) -> Dict[str, Any]:
    """
    宽松解析 VLM 返回的 JSON：
    - 优先直接 json.loads
    - 若失败，尝试用正则提取 {...} 块
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("空响应")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    if text.startswith("```"):
        # 去掉 ```json ... ```
        cleaned = text.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析 JSON: {text[:200]}") from exc
    raise ValueError(f"响应不含 JSON: {text[:200]}")
