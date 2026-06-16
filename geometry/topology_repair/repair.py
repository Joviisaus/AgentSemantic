"""Orchestrate geometric topology repair: spectral cut, handle loop, boundary refine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from geometry.partition.common import enforce_per_label_face_connectivity

from .boundary_optimize import refine_surgery_boundary
from .concave_seam import cut_labels_by_seam, find_concave_seam_between_seeds
from .handle_loop import cut_handle_bridge, estimate_genus
from .local_region import (
    boundary_face_seeds,
    extract_boundary_region,
    expand_face_rings,
    semantic_boundary_faces,
)
from .spectral_cut import partition_faces_by_fiedler

LOGGER = logging.getLogger("geometry.topology_repair")


@dataclass
class TopologyRepairResult:
    """Face-level semantic overrides produced by geometric repair."""

    face_overrides: Dict[int, int] = field(default_factory=dict)
    strategy: str = ""
    ok: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)

    def apply_to(self, face_semantic_index: np.ndarray) -> np.ndarray:
        out = np.asarray(face_semantic_index, dtype=np.int64).copy()
        for fid, sem in self.face_overrides.items():
            if 0 <= int(fid) < len(out):
                out[int(fid)] = int(sem)
        return out


def _constraint_field(constraint: Any, name: str, default: Any = "") -> Any:
    """Read a field from ``TopologyConstraint`` or plain dict."""
    if isinstance(constraint, dict):
        return constraint.get(name, default)
    return getattr(constraint, name, default)


def infer_forbidden_adjacencies(
    topology_constraints: Sequence[Any],
    semantic_parts: Sequence[str],
) -> List[Tuple[str, str]]:
    """
    Infer illegal adjacency pairs from attach_to chains.

    A part should only neighbor its attach_to parents (and siblings sharing parent).
    """
    parts = [str(p).strip() for p in semantic_parts if str(p).strip()]
    allowed: Dict[str, Set[str]] = {p: set() for p in parts}
    for c in topology_constraints:
        part = str(_constraint_field(c, "part", "")).strip()
        attach = _constraint_field(c, "attach_to", []) or []
        if part not in allowed:
            continue
        for parent in attach:
            ps = str(parent).strip()
            if ps in parts:
                allowed[part].add(ps)
                allowed[ps].add(part)

    forbidden: List[Tuple[str, str]] = []
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            if b not in allowed.get(a, set()) and a not in allowed.get(b, set()):
                if (a, b) not in forbidden and (b, a) not in forbidden:
                    forbidden.append((a, b))
    return forbidden


_MAX_REPAIR_REGION_FACES = 16_000


def repair_illegal_adjacency(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    semantic_parts: Sequence[str],
    part_a: str,
    part_b: str,
    *,
    ring_rings: int = 2,
    refine_boundary: bool = False,
    include_core: bool = False,
) -> TopologyRepairResult:
    """
    Error A: cut illegal adjacency (e.g. head welded to arm).

    Pipeline: local submesh → Fiedler partition → concave seam → boundary refine.
    """
    parts = [str(p).strip() for p in semantic_parts]
    if part_a not in parts or part_b not in parts:
        return TopologyRepairResult(strategy="illegal_adjacency", debug={"reason": "unknown parts"})

    idx_a = parts.index(part_a)
    idx_b = parts.index(part_b)
    fpi = np.asarray(face_semantic_index, dtype=np.int64)

    boundary = semantic_boundary_faces(faces, fpi, idx_a, idx_b)
    if not boundary:
        return TopologyRepairResult(
            strategy="illegal_adjacency",
            debug={"reason": "no shared boundary"},
        )

    sub_v, sub_f, global_ids, l2g = extract_boundary_region(
        vertices,
        faces,
        fpi,
        idx_a,
        idx_b,
        ring_rings=ring_rings,
        include_core=include_core,
    )
    if len(sub_f) < 4:
        return TopologyRepairResult(strategy="illegal_adjacency", debug={"reason": "region too small"})
    if len(sub_f) > _MAX_REPAIR_REGION_FACES:
        LOGGER.warning(
            "illegal_adjacency %s↔%s: region %d faces > cap %d, skip",
            part_a,
            part_b,
            len(sub_f),
            _MAX_REPAIR_REGION_FACES,
        )
        return TopologyRepairResult(
            strategy="illegal_adjacency",
            debug={
                "reason": "region too large",
                "region_faces": len(sub_f),
                "cap": _MAX_REPAIR_REGION_FACES,
            },
        )

    seeds_a, seeds_b = boundary_face_seeds(faces, fpi, idx_a, idx_b, local_to_global=l2g)
    spec_labels, spec_dbg = partition_faces_by_fiedler(sub_v, sub_f, seeds_a, seeds_b)
    seam, seam_dbg = find_concave_seam_between_seeds(sub_v, sub_f, seeds_a, seeds_b)

    local_labels = spec_labels.copy()
    if seam:
        local_labels = cut_labels_by_seam(
            sub_f,
            local_labels,
            seam,
            anchor_label=0,
            other_label=1,
        )
    local_labels = enforce_per_label_face_connectivity(sub_f, local_labels)

    overrides: Dict[int, int] = {}
    surgery_faces: List[int] = []
    for li, gfi in enumerate(global_ids):
        gfi_i = int(gfi)
        new_local = int(local_labels[li])
        new_sem = idx_a if new_local == 0 else idx_b
        if int(fpi[gfi_i]) != new_sem:
            overrides[gfi_i] = new_sem
            surgery_faces.append(gfi_i)

    debug: Dict[str, Any] = {
        "part_a": part_a,
        "part_b": part_b,
        "boundary_faces": len(boundary),
        "spectral": spec_dbg,
        "seam": seam_dbg,
        "initial_overrides": len(overrides),
    }

    if refine_boundary and surgery_faces:
        refined, ref_dbg = refine_surgery_boundary(
            vertices,
            faces,
            fpi.tolist(),
            parts,
            surgery_faces,
        )
        overrides.update(refined)
        debug["boundary_refine"] = ref_dbg

    return TopologyRepairResult(
        face_overrides=overrides,
        strategy="illegal_adjacency",
        ok=bool(overrides),
        debug=debug,
    )


def repair_pseudo_bridge(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    semantic_parts: Sequence[str],
    part_a: str,
    part_b: str,
    *,
    ring_rings: int = 2,
    refine_boundary: bool = True,
) -> TopologyRepairResult:
    """
    Error B: cut pseudo-bridge / handle loop (e.g. two legs welded).

    Uses handle-loop detection + relabel + optional boundary refine.
    """
    parts = [str(p).strip() for p in semantic_parts]
    if part_a not in parts or part_b not in parts:
        return TopologyRepairResult(strategy="pseudo_bridge", debug={"reason": "unknown parts"})

    idx_a = parts.index(part_a)
    idx_b = parts.index(part_b)
    fpi = np.asarray(face_semantic_index, dtype=np.int64)

    overrides, dbg = cut_handle_bridge(
        vertices,
        faces,
        fpi,
        idx_a,
        idx_b,
        ring_rings=ring_rings,
    )
    debug: Dict[str, Any] = {"part_a": part_a, "part_b": part_b, **dbg}

    region = expand_face_rings(
        faces,
        semantic_boundary_faces(faces, fpi, idx_a, idx_b),
        rings=ring_rings,
    )
    sub_v, sub_f, _ = extract_boundary_region(
        vertices, faces, fpi, idx_a, idx_b, ring_rings=0, include_core=True
    )
    debug["estimated_genus"] = estimate_genus(sub_v, sub_f) if len(sub_f) else 0.0

    surgery = list(overrides.keys())
    if refine_boundary and surgery:
        refined, ref_dbg = refine_surgery_boundary(
            vertices,
            faces,
            fpi.tolist(),
            parts,
            surgery,
        )
        overrides.update(refined)
        debug["boundary_refine"] = ref_dbg

    return TopologyRepairResult(
        face_overrides=overrides,
        strategy="pseudo_bridge",
        ok=bool(overrides),
        debug=debug,
    )


def repair_topology_violation(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    semantic_parts: Sequence[str],
    violation: Dict[str, Any],
    *,
    topology_constraints: Optional[Sequence[Any]] = None,
    ring_rings: int = 2,
    refine_boundary: bool = True,
) -> TopologyRepairResult:
    """
    Dispatch a topology violation to the appropriate geometric repair tool.

    Supported kinds:
    - illegal_adjacency / forbidden_adjacency
    - pseudo_bridge / handle_loop
    - missing_attachment (tries illegal_adjacency cut toward attach_to parent)
    """
    kind = str(violation.get("kind", "")).strip().lower()
    part_a = str(violation.get("part_a") or violation.get("from_part") or violation.get("part", "")).strip()
    part_b = str(violation.get("part_b") or violation.get("to_part") or "").strip()

    if kind in ("illegal_adjacency", "forbidden_adjacency"):
        if not part_b:
            part_b = str(violation.get("neighbor", "")).strip()
        return repair_illegal_adjacency(
            vertices,
            faces,
            face_semantic_index,
            semantic_parts,
            part_a,
            part_b,
            ring_rings=ring_rings,
            refine_boundary=refine_boundary,
        )

    if kind in ("pseudo_bridge", "handle_loop"):
        if not part_b:
            part_b = str(violation.get("neighbor", "")).strip()
        return repair_pseudo_bridge(
            vertices,
            faces,
            face_semantic_index,
            semantic_parts,
            part_a,
            part_b,
            ring_rings=ring_rings,
            refine_boundary=refine_boundary,
        )

    if kind == "missing_attachment":
        attach = violation.get("attach_to") or []
        if attach:
            parent = str(attach[0]).strip()
            return repair_illegal_adjacency(
                vertices,
                faces,
                face_semantic_index,
                semantic_parts,
                part_a,
                parent,
                ring_rings=ring_rings,
                refine_boundary=refine_boundary,
            )

    return TopologyRepairResult(
        strategy=kind or "unknown",
        debug={"reason": "unsupported violation kind", "violation": violation},
    )


def apply_geometric_repairs(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_index: Sequence[int],
    semantic_parts: Sequence[str],
    violations: Sequence[Dict[str, Any]],
    *,
    topology_constraints: Optional[Sequence[Any]] = None,
    auto_detect_forbidden: bool = True,
    ring_rings: int = 2,
    refine_boundary: bool = False,
    max_repairs: int = 8,
) -> TopologyRepairResult:
    """
    Run geometric repairs for a batch of violations; merge face overrides.
    """
    merged: Dict[int, int] = {}
    strategies: List[str] = []
    debug_all: List[Dict[str, Any]] = []

    seen_pairs: Set[Tuple[str, str, str]] = set()
    vlist: List[Dict[str, Any]] = []

    def _append_violation(item: Dict[str, Any]) -> None:
        kind = str(item.get("kind", "")).strip().lower()
        pa = str(
            item.get("part_a") or item.get("from_part") or item.get("part", "")
        ).strip()
        pb = str(
            item.get("part_b") or item.get("to_part") or item.get("neighbor", "")
        ).strip()
        key = (min(pa, pb), max(pa, pb), kind)
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        vlist.append(item)

    for v in violations:
        _append_violation(v)
    if auto_detect_forbidden and topology_constraints:
        adj_forbidden = infer_forbidden_adjacencies(topology_constraints, semantic_parts)
        fpi = np.asarray(face_semantic_index, dtype=np.int64)
        parts = [str(p).strip() for p in semantic_parts]
        for a, b in adj_forbidden:
            if a not in parts or b not in parts:
                continue
            ia, ib = parts.index(a), parts.index(b)
            if semantic_boundary_faces(faces, fpi, ia, ib):
                _append_violation(
                    {
                        "kind": "illegal_adjacency",
                        "part_a": a,
                        "part_b": b,
                        "detail": f"推断非法邻接: {a} ↔ {b}",
                    }
                )

    for i, v in enumerate(vlist[: max(1, int(max_repairs))]):
        pa = str(v.get("part_a") or v.get("part", "")).strip()
        pb = str(v.get("part_b") or v.get("neighbor", "")).strip()
        LOGGER.info(
            "geo repair %d/%d kind=%s %s↔%s",
            i + 1,
            min(len(vlist), max_repairs),
            v.get("kind", "?"),
            pa,
            pb,
        )
        res = repair_topology_violation(
            vertices,
            faces,
            face_semantic_index,
            semantic_parts,
            v,
            topology_constraints=topology_constraints,
            ring_rings=ring_rings,
            refine_boundary=refine_boundary,
        )
        if res.strategy:
            strategies.append(res.strategy)
        debug_all.append(res.debug)
        merged.update(res.face_overrides)
        LOGGER.info(
            "geo repair %s↔%s done overrides=%d",
            pa,
            pb,
            len(res.face_overrides),
        )

    return TopologyRepairResult(
        face_overrides=merged,
        strategy="+".join(sorted(set(strategies))) if strategies else "none",
        ok=bool(merged),
        debug={"repairs": debug_all, "total_overrides": len(merged)},
    )
