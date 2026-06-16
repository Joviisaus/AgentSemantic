"""
Geometric topology repair toolkit (post-VLM surgery).

Phases:
1. **illegal_adjacency** — spectral Fiedler + concave seam cut (Error A)
2. **pseudo_bridge**     — handle-loop detection + cut (Error B)
3. **boundary_refine**   — concavity-aware graph-cut energy minimization
4. Pipeline re-audits with VLM after overrides are applied
"""

from .boundary_optimize import optimize_region_labels, refine_surgery_boundary
from .concave_seam import cut_labels_by_seam, find_concave_seam_between_seeds
from .handle_loop import cut_handle_bridge, estimate_genus, find_handle_loop_cycle
from .local_region import extract_boundary_region, semantic_boundary_faces
from .repair import (
    TopologyRepairResult,
    apply_geometric_repairs,
    infer_forbidden_adjacencies,
    repair_illegal_adjacency,
    repair_pseudo_bridge,
    repair_topology_violation,
)
from .spectral_cut import compute_laplacian_eigenvectors, partition_faces_by_fiedler

__all__ = [
    "TopologyRepairResult",
    "apply_geometric_repairs",
    "compute_laplacian_eigenvectors",
    "cut_handle_bridge",
    "cut_labels_by_seam",
    "estimate_genus",
    "extract_boundary_region",
    "find_concave_seam_between_seeds",
    "find_handle_loop_cycle",
    "infer_forbidden_adjacencies",
    "optimize_region_labels",
    "partition_faces_by_fiedler",
    "refine_surgery_boundary",
    "repair_illegal_adjacency",
    "repair_pseudo_bridge",
    "repair_topology_violation",
    "semantic_boundary_faces",
]
