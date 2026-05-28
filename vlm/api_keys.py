"""
API Key 管理：从 .env 文件加载到环境变量，并提供 ${ENV_VAR} 占位符展开工具。

设计原则：
- 永远不要把真实 key 写入仓库。`.env` 已经被 .gitignore 忽略。
- VLM 配置 YAML 里用 ${OPENAI_API_KEY} 之类的占位符引用，运行时再展开。
- 不强依赖 python-dotenv，自己实现轻量解析器，避免额外依赖。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOTENV_PATHS: tuple[Path, ...] = (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / ".env.local",
)

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")
_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _parse_dotenv_file(path: Path) -> dict[str, str]:
    """轻量 .env 解析：支持 KEY=VALUE / export KEY=VALUE / 引号 / # 注释 / 行内注释。"""
    if not path.is_file():
        return {}

    out: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = _LINE_RE.match(raw_line.rstrip("\n"))
            if not match:
                continue
            key, value = match.group(1), match.group(2)

            # 行内注释（仅对未引号包裹的值生效）
            if value and value[0] not in ("'", '"'):
                hash_idx = value.find(" #")
                if hash_idx >= 0:
                    value = value[:hash_idx].rstrip()
            value = _strip_quotes(value)
            out[key] = value
    return out


def load_dotenv(
    paths: Optional[Iterable[str | Path]] = None,
    *,
    override: bool = False,
) -> dict[str, str]:
    """
    将 .env 文件中的键值写入 os.environ。

    参数：
        paths: 自定义 .env 路径列表；为 None 时使用项目根目录下的
               `.env` 与 `.env.local`（后者优先级更高）。
        override: 是否覆盖已经存在的环境变量。默认 False，
                  保留外部 shell 已经设置的真实 key。
    返回：实际写入（或忽略）的键值字典，便于调试。
    """
    if paths is None:
        path_list = list(DEFAULT_DOTENV_PATHS)
    else:
        path_list = [Path(p) for p in paths]

    merged: dict[str, str] = {}
    for path in path_list:
        merged.update(_parse_dotenv_file(path))

    for key, val in merged.items():
        if override or key not in os.environ:
            os.environ[key] = val
    return merged


def resolve_env_placeholder(value: Any) -> Any:
    """
    展开字符串中的 ${ENV_VAR} 与 ${ENV_VAR:-default} 占位符。
    非字符串原样返回。空字符串视为未设置。
    """
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(env_name, default)

    return _ENV_PLACEHOLDER.sub(repl, value)


def mask_secret(secret: Optional[str], *, keep: int = 4) -> str:
    """脱敏显示 API key，仅保留首尾若干字符。"""
    if not secret:
        return "<unset>"
    if len(secret) <= keep * 2:
        return "*" * len(secret)
    return f"{secret[:keep]}…{secret[-keep:]}"
