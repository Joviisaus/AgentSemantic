"""
VLM 语义细化包：在 SEG-MAT / 谱聚类分块基础上，由 VLM 把分块判定为
[完全符合 / 局部符合 / 完全不符合] 用户语义，对“局部符合”的分块再做谱聚类分裂，
直到没有“局部符合”为止。

子模块：
- api_keys     : .env 加载与 ${ENV_VAR} 占位符展开
- config       : 读取 config/vlm_config.yaml
- client       : VLM HTTP 客户端（OpenAI 兼容 / Anthropic / Google）
- prompts      : 三分类与种子数估计的提示模板
- render_component : 单分块高亮的 12 视角渲染
- seed_split   : 在分块内做子谱聚类
- blocks       : SemanticBlock 树结构
- pipeline     : 迭代细化流水线
- cli          : 命令行入口
"""

from __future__ import annotations

from .api_keys import load_dotenv, resolve_env_placeholder
from .blocks import BlockStatus, SemanticBlock
from .classifier import ComponentClassification, classify_component
from .client import VLMClient, build_vlm_client
from .config import VLMProviderConfig, load_vlm_config
from .pipeline import (
    SemanticRefinementResult,
    SemanticRefinementPipeline,
    run_semantic_refinement,
)
from .render_component import render_highlighted_component_panel
from .seed_split import split_block_by_spectral_seeds

__all__ = [
    "BlockStatus",
    "ComponentClassification",
    "SemanticBlock",
    "SemanticRefinementPipeline",
    "SemanticRefinementResult",
    "VLMClient",
    "VLMProviderConfig",
    "build_vlm_client",
    "classify_component",
    "load_dotenv",
    "load_vlm_config",
    "render_highlighted_component_panel",
    "resolve_env_placeholder",
    "run_semantic_refinement",
    "split_block_by_spectral_seeds",
]
