"""Shape-diameter function (SDF) partition."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from geometry.spectral_cluster import _kmeans, compute_vertex_normals

from .common import (
    SubBlockSplit,
    enforce_per_label_face_connectivity,
    extract_submesh,
    face_centroids_and_normals,
)
from .geodesic_fps import split_block_by_geodesic_fps

LOGGER = logging.getLogger("geometry.partition.sdf")


def _compute_face_sdf(
    sub_vertices: np.ndarray,
    sub_faces: np.ndarray,
    *,
    num_rays: int = 16,
    cone_angle_deg: float = 60.0,
) -> np.ndarray:
    import trimesh

    mesh = trimesh.Trimesh(vertices=sub_vertices, faces=sub_faces, process=False)
    centroids = mesh.triangles_center
    fn = mesh.face_normals
    n_faces = len(sub_faces)

    if n_faces == 0:
        return np.zeros(0, dtype=np.float64)

    try:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    except Exception:  # noqa: BLE001
        center = sub_vertices.mean(axis=0)
        return np.linalg.norm(centroids - center, axis=1)

    cone_rad = np.deg2rad(float(cone_angle_deg))
    rng = np.random.default_rng(0)
    sdf = np.zeros(n_faces, dtype=np.float64)

    for fi in range(n_faces):
        origin = centroids[fi] - fn[fi] * 1e-4
        inward = -fn[fi]
        thicknesses: List[float] = []
        directions = [inward]
        if num_rays > 1:
            aux = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(aux, inward))) > 0.9:
                aux = np.array([0.0, 1.0, 0.0])
            u = np.cross(inward, aux)
            u /= max(np.linalg.norm(u), 1e-12)
            v = np.cross(inward, u)
            for _ in range(int(num_rays) - 1):
                theta = rng.uniform(0, 2 * np.pi)
                phi = rng.uniform(0, cone_rad)
                d = (
                    inward * np.cos(phi)
                    + u * np.sin(phi) * np.cos(theta)
                    + v * np.sin(phi) * np.sin(theta)
                )
                d /= max(np.linalg.norm(d), 1e-12)
                directions.append(d)

        for d in directions:
            try:
                loc, _idx_ray, tri_idx = intersector.intersects_location(
                    ray_origins=[origin],
                    ray_directions=[d],
                    multiple_hits=True,
                )
            except Exception:  # noqa: BLE001
                continue
            if len(loc) == 0:
                continue
            valid = [
                float(np.linalg.norm(loc[i] - origin))
                for i, t in enumerate(tri_idx)
                if int(t) != fi and float(np.linalg.norm(loc[i] - origin)) > 1e-5
            ]
            if valid:
                thicknesses.append(min(valid))

        if thicknesses:
            sdf[fi] = float(np.mean(thicknesses))
        else:
            sdf[fi] = float(np.linalg.norm(centroids[fi] - sub_vertices.mean(axis=0)))

    sdf = sdf / max(float(np.percentile(sdf, 95)), 1e-6)
    return sdf


def split_block_by_sdf(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: Sequence[int],
    seed_count: int,
    *,
    num_rays: int = 16,
    cone_angle_deg: float = 60.0,
    vertex_scalars: Optional[np.ndarray] = None,
    scalar_weight: float = 0.35,
) -> SubBlockSplit:
    face_ids_list = list(face_ids)
    seed_count = max(2, int(seed_count))
    if not face_ids_list:
        return SubBlockSplit(
            sub_face_groups=[], sub_labels=np.zeros(0, dtype=np.int64), num_clusters=0
        )

    sub_v, sub_f, _global_v_idx = extract_submesh(vertices, faces, face_ids_list)
    m = len(sub_f)
    if m <= seed_count:
        return SubBlockSplit(
            sub_face_groups=[list(face_ids_list)],
            sub_labels=np.zeros(len(face_ids_list), dtype=np.int64),
            num_clusters=1,
            debug={"reason": "submesh 太小"},
        )

    from .scale import scaled_num_rays

    num_rays = scaled_num_rays(m, int(num_rays))
    sdf_faces = np.arange(m, dtype=np.int64)
    sdf_sample_stride = 1
    max_sdf_faces = 25_000
    if m > max_sdf_faces:
        sdf_sample_stride = max(1, int(np.ceil(m / max_sdf_faces)))
        sdf_faces = np.arange(0, m, sdf_sample_stride, dtype=np.int64)

    sdf_full = np.zeros(m, dtype=np.float64)
    if len(sdf_faces) == m:
        sdf_full = _compute_face_sdf(
            sub_v, sub_f, num_rays=num_rays, cone_angle_deg=float(cone_angle_deg)
        )
    else:
        sdf_sub = _compute_face_sdf(
            sub_v,
            sub_f[sdf_faces],
            num_rays=num_rays,
            cone_angle_deg=float(cone_angle_deg),
        )
        sdf_full[sdf_faces] = sdf_sub
        centroids, _ = face_centroids_and_normals(sub_v, sub_f)
        sample_centroids = centroids[sdf_faces]
        sampled_set = set(sdf_faces.tolist())
        unsampled = [fi for fi in range(m) if fi not in sampled_set]
        if unsampled:
            unsampled_arr = np.asarray(unsampled, dtype=np.int64)
            d = np.linalg.norm(
                sample_centroids[:, None, :] - centroids[unsampled_arr][None, :, :],
                axis=2,
            )
            nearest = np.argmin(d, axis=0)
            sdf_full[unsampled_arr] = sdf_sub[nearest]

    sdf = sdf_full
    centroids, _ = face_centroids_and_normals(sub_v, sub_f)
    center = sub_v.mean(axis=0)
    pos = (centroids - center) / max(
        float(np.linalg.norm(centroids - center, axis=1).max()), 1e-6
    )
    normals = compute_vertex_normals(sub_v, sub_f)
    vn = normals[sub_f].mean(axis=1)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

    features = np.hstack([sdf.reshape(-1, 1), pos, vn * 0.35])

    try:
        face_labels = _kmeans(features.astype(np.float64), seed_count, random_state=0)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("SDF KMeans 失败: %s", exc)
        return split_block_by_geodesic_fps(
            vertices, faces, face_ids, seed_count=seed_count
        )

    from .connectivity import smooth_face_label_boundaries

    face_labels = smooth_face_label_boundaries(sub_f, face_labels, passes=2)
    face_labels = enforce_per_label_face_connectivity(sub_f, face_labels)
    used = sorted({int(v) for v in face_labels})
    groups: List[List[int]] = []
    for lab in used:
        idx = np.flatnonzero(face_labels == lab)
        group = [face_ids_list[i] for i in idx.tolist()]
        if group:
            groups.append(group)

    return SubBlockSplit(
        sub_face_groups=groups,
        sub_labels=face_labels,
        num_clusters=len(groups),
        debug={
            "strategy": "sdf",
            "num_rays": int(num_rays),
            "cone_angle_deg": float(cone_angle_deg),
            "sdf_sample_stride": int(sdf_sample_stride),
            "sdf_faces_sampled": int(len(sdf_faces)),
            "sdf_range": [float(sdf.min()), float(sdf.max())],
            "k_actual": len(groups),
        },
    )
