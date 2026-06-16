"""Backward-compatible re-exports — implementations live in ``geometry.partition``."""

from geometry.partition import (
    VALID_STRATEGIES,
    split_block_by_cvt,
    split_block_by_geodesic_fps,
    split_block_by_graph_cut,
    split_block_by_sdf,
    split_block_by_skeleton,
    split_block_by_spectral_seeds,
    split_block_by_strategy,
)

__all__ = [
    "VALID_STRATEGIES",
    "split_block_by_graph_cut",
    "split_block_by_cvt",
    "split_block_by_sdf",
    "split_block_by_strategy",
]
