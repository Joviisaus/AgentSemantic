"""
VLM 配置加载：解析 config/vlm_config.yaml 并展开 ${ENV_VAR}，
封装为强类型 VLMProviderConfig，给 client / pipeline 使用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .api_keys import load_dotenv, resolve_env_placeholder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "vlm_config.yaml"


@dataclass
class VLMProviderConfig:
    name: str
    enabled: bool
    api_key: str
    base_url: str
    default_model: str
    models: List[str] = field(default_factory=list)
    timeout_seconds: int = 120
    max_retries: int = 3
    temperature: float = 0.2
    max_tokens: int = 4096
    extra: Dict[str, Any] = field(default_factory=dict)

    def has_key(self) -> bool:
        return bool(self.api_key) and self.api_key not in ("", "ollama")

    def is_local(self) -> bool:
        """Ollama 等本地服务不需要 key。"""
        return self.name == "ollama" or self.api_key == "ollama"


def _expand_placeholders(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_placeholders(v) for v in obj]
    return resolve_env_placeholder(obj)


def load_vlm_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    load_env: bool = True,
    env_paths: Optional[List[str | Path]] = None,
) -> Dict[str, Any]:
    """
    读取 YAML 配置，展开所有 ${ENV_VAR} 占位符。
    会先尝试从 .env 文件加载环境变量（除非 load_env=False）。
    """
    if load_env:
        load_dotenv(env_paths)

    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"找不到 VLM 配置: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _expand_placeholders(raw)


def get_provider_config(
    cfg: Dict[str, Any],
    provider: Optional[str] = None,
    *,
    model: Optional[str] = None,
) -> VLMProviderConfig:
    """
    从已展开的配置字典里取出指定 provider。
    provider=None 时使用 cfg["active_provider"]。
    """
    defaults = cfg.get("defaults", {}) or {}
    providers = cfg.get("providers", {}) or {}
    name = provider or cfg.get("active_provider")
    if not name:
        raise ValueError("未指定 provider 且 vlm_config.yaml 没有 active_provider")
    if name not in providers:
        raise ValueError(
            f"provider '{name}' 不在配置中，可选: {sorted(providers.keys())}"
        )

    raw = providers[name] or {}
    return VLMProviderConfig(
        name=name,
        enabled=bool(raw.get("enabled", False)),
        api_key=str(raw.get("api_key", "") or ""),
        base_url=str(raw.get("base_url", "") or ""),
        default_model=str(model or raw.get("default_model") or ""),
        models=list(raw.get("models", []) or []),
        timeout_seconds=int(raw.get("timeout_seconds", defaults.get("timeout_seconds", 120))),
        max_retries=int(raw.get("max_retries", defaults.get("max_retries", 3))),
        temperature=float(raw.get("temperature", defaults.get("temperature", 0.2))),
        max_tokens=int(raw.get("max_tokens", defaults.get("max_tokens", 4096))),
        extra={
            k: v
            for k, v in raw.items()
            if k
            not in {
                "enabled",
                "api_key",
                "base_url",
                "default_model",
                "models",
                "timeout_seconds",
                "max_retries",
                "temperature",
                "max_tokens",
            }
        },
    )


def resolve_model_name(
    explicit: Optional[str],
    provider_cfg: VLMProviderConfig,
) -> str:
    """把 CLI/配置里的 model 参数解析为 API 可用的 model id。

    常见误用：``--provider qwen --model qwen`` 会把 provider 名当成 model，
    此处自动回退到 ``default_model``（如 ``qwen3.7-plus``）。
    """
    default = str(provider_cfg.default_model or "").strip()
    if not explicit:
        if not default:
            raise ValueError(
                f"provider '{provider_cfg.name}' 未配置 default_model，"
                "请用 --model 指定"
            )
        return default
    name = str(explicit).strip()
    if name == provider_cfg.name:
        if default:
            return default
    if provider_cfg.models and name in provider_cfg.models:
        return name
    if default and name == default:
        return default
    # 允许 Ark 等带日期后缀的完整 endpoint id（可能不在 models 列表里）
    return name
