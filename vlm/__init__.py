"""
vlm/ —— **只**包含与 VLM HTTP 请求 / 提示模板 / 响应解析直接相关的代码。

子模块：
- api_keys     : .env 加载与 ``${ENV_VAR}`` 占位符展开
- config       : 读取 ``config/vlm_config.yaml``
- client       : VLM HTTP 客户端（OpenAI 兼容 / Anthropic / Google / 火山方舟 Doubao）
- prompts      : 三分类 / 种子数估计 / 分组 / 进度自审 的提示模板
- classifier   : 分块三分类（full / partial / none）的请求封装
- review       : 全局进度自审请求封装
- view_selector: 最佳视角选择请求封装（缩略图渲染在 ``rendering.view_thumbnails``）

几何 / 渲染 / 主流程编排不在本包内：
- 几何分裂 / 聚类 / mesh 处理 → 见 ``geometry/``
- 离屏渲染（pyvista, PIL）       → 见 ``rendering/``
- 流水线 / HTML 报告 / CLI       → 见 ``pipeline/``
"""

from __future__ import annotations

from .api_keys import load_dotenv, resolve_env_placeholder
from .classifier import ComponentClassification, classify_component
from .client import VLMClient, VLMResponse, build_vlm_client, parse_json_response
from .config import (
    DEFAULT_CONFIG_PATH,
    VLMProviderConfig,
    get_provider_config,
    load_vlm_config,
)
from .full_coverage import (
    EmptySemanticRepair,
    EmptySemanticRepairPlan,
    MultiLabelClassification,
    SemanticEnumeration,
    classify_multi_label,
    enumerate_semantics,
    resolve_empty_semantic_repairs,
)
from .review import ProgressReview, review_progress
from .view_selector import BestViewSelection, select_best_view

__all__ = [
    "BestViewSelection",
    "ComponentClassification",
    "DEFAULT_CONFIG_PATH",
    "EmptySemanticRepair",
    "EmptySemanticRepairPlan",
    "MultiLabelClassification",
    "ProgressReview",
    "SemanticEnumeration",
    "VLMClient",
    "VLMProviderConfig",
    "VLMResponse",
    "build_vlm_client",
    "classify_component",
    "classify_multi_label",
    "enumerate_semantics",
    "resolve_empty_semantic_repairs",
    "get_provider_config",
    "load_dotenv",
    "load_vlm_config",
    "parse_json_response",
    "resolve_env_placeholder",
    "review_progress",
    "select_best_view",
]
