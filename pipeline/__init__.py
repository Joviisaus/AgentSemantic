"""
pipeline/ —— 主流程编排：VLM 三分类 + 几何分裂 + HTML 报告 + CLI。

子模块：
- pipeline : SemanticRefinementPipeline / run_semantic_refinement
- report   : 把 output_dir 渲染成单文件 HTML 报告
- cli      : 命令行入口（python -m pipeline.cli ...）
"""

from .pipeline import (  # noqa: F401
    SemanticRefinementConfig,
    SemanticRefinementPipeline,
    SemanticRefinementResult,
    run_semantic_refinement,
)
from .report import generate_report  # noqa: F401

__all__ = [
    "SemanticRefinementConfig",
    "SemanticRefinementPipeline",
    "SemanticRefinementResult",
    "generate_report",
    "run_semantic_refinement",
]
