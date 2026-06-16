"""Centroidal Voronoi tessellation via geodesic Lloyd relaxation."""

from __future__ import annotations

from typing import Optional, Sequence

from .common import SubBlockSplit
from .geodesic_fps import split_block_by_geodesic_fps


def split_block_by_cvt(
    vertices,
    faces,
    face_ids: Sequence[int],
    seed_count: int,
    *,
    lloyd_iter: int = 4,
    vertex_scalars: Optional[Sequence[float]] = None,
    scalar_weight: float = 0.35,
) -> SubBlockSplit:
    split = split_block_by_geodesic_fps(
        vertices,
        faces,
        face_ids,
        seed_count=seed_count,
        lloyd_iter=int(lloyd_iter),
    )
    dbg = dict(split.debug) if split.debug else {}
    dbg["strategy"] = "cvt"
    dbg["lloyd_iter"] = int(lloyd_iter)
    return SubBlockSplit(
        sub_face_groups=split.sub_face_groups,
        sub_labels=split.sub_labels,
        num_clusters=split.num_clusters,
        debug=dbg,
    )
