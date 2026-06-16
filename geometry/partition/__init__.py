"""
Block partition strategies — one module per method.

Methods:
- connected_components : topological CC split
- geodesic_fps         : geodesic FPS + Lloyd Voronoi
- spectral             : spectral clustering on sub-mesh
- graph_cut            : concavity-aware graph cut
- cvt                  : centroidal Voronoi (wraps geodesic_fps)
- sdf                  : shape-diameter function + KMeans
- skeleton             : Laplacian contraction skeleton + geodesic Voronoi
- scale                : large-mesh budgets + vertex graph pooling
- strategy             : unified dispatcher
"""

from .common import SubBlockSplit
from .connected_components import split_block_by_connected_components
from .cvt import split_block_by_cvt
from .geodesic_fps import split_block_by_geodesic_fps
from .graph_cut import split_block_by_graph_cut
from .sdf import split_block_by_sdf
from .skeleton import split_block_by_skeleton
from .spectral import split_block_by_spectral_seeds
from .scale import (
    VertexGraphView,
    build_vertex_graph_view,
    effective_graph_vertex_budget,
    resolve_graph_vertex_budget,
    resolve_partition_scale,
)

from .connectivity import (
    build_k_connected_face_groups,
    finalize_submesh_k_connected_split,
)
from .strategy import VALID_STRATEGIES, split_block_by_strategy

__all__ = [
    "SubBlockSplit",
    "VALID_STRATEGIES",
    "VertexGraphView",
    "build_vertex_graph_view",
    "effective_graph_vertex_budget",
    "resolve_graph_vertex_budget",
    "resolve_partition_scale",
    "build_k_connected_face_groups",
    "finalize_submesh_k_connected_split",
    "split_block_by_connected_components",
    "split_block_by_geodesic_fps",
    "split_block_by_spectral_seeds",
    "split_block_by_graph_cut",
    "split_block_by_cvt",
    "split_block_by_sdf",
    "split_block_by_skeleton",
    "split_block_by_strategy",
]
