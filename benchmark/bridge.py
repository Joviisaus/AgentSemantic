"""Transfer per-face mesh labels onto the GT point cloud.

The bundle proxy mesh and the CoSMo3D GT points are coordinate-aligned, so for
each GT point we take the label of the nearest mesh face (by face centroid).
Uses a scipy KD-tree on face centroids (no rtree dependency).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

IGNORE_LABEL = -1


def face_labels_to_point_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Map per-face labels (len == n_faces) to per-point labels (len == n_points).

    A GT point inherits the label of the mesh face whose centroid is closest.
    """
    faces = np.asarray(faces, dtype=np.int64)
    vertices = np.asarray(vertices, dtype=np.float64)
    face_labels = np.asarray(face_labels, dtype=np.int64)
    centroids = vertices[faces].mean(axis=1)  # (F, 3)
    tree = cKDTree(centroids)
    _, idx = tree.query(np.asarray(points, dtype=np.float64), k=1)
    return face_labels[idx]
