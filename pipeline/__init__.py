"""
pipeline/ —— 主流程编排：VLM 三分类 + 几何分裂 + HTML 报告 + CLI。

子模块：
- pipeline                : SemanticRefinementPipeline / run_semantic_refinement
                            （单语义 full/partial/none）
- full_coverage           : FullCoveragePipeline / run_full_coverage
                            （多语义全覆盖 — VLM 枚举全包围语义 + multi-label 分裂）
- report                  : 把单语义 output_dir 渲染成单文件 HTML 报告
- full_coverage_report    : 把全覆盖 output_dir 渲染成单文件 HTML 报告
- cli                     : 单语义命令行入口（python -m pipeline.cli ...）
- full_coverage_cli       : 全覆盖命令行入口
                            （python -m pipeline.full_coverage_cli ...）
"""

from .full_coverage import (  # noqa: F401
    FullCoverageConfig,
    FullCoveragePipeline,
    FullCoverageResult,
    run_full_coverage,
)
from .pipeline import (  # noqa: F401
    SemanticRefinementConfig,
    SemanticRefinementPipeline,
    SemanticRefinementResult,
    run_semantic_refinement,
)
from .report import generate_report  # noqa: F401

__all__ = [
    "FullCoverageConfig",
    "FullCoveragePipeline",
    "FullCoverageResult",
    "SemanticRefinementConfig",
    "SemanticRefinementPipeline",
    "SemanticRefinementResult",
    "generate_full_coverage_report",
    "generate_report",
    "run_full_coverage",
    "run_semantic_refinement",
]


def __getattr__(name: str):
    """
    懒加载 ``generate_full_coverage_report``，避免 ``python -m
    pipeline.full_coverage_report`` 触发 "module imported twice" RuntimeWarning
    （急切 import 会让 ``pipeline.full_coverage_report`` 同时进入 sys.modules 和
    被 runpy 当作 __main__ 执行）。
    """
    if name == "generate_full_coverage_report":
        from .full_coverage_report import generate_full_coverage_report
        return generate_full_coverage_report
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
