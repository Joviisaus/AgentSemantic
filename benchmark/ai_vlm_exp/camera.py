"""
Self-contained pinhole camera that replicates rendering/heatmap_renderer.py's
pyvista camera, so a 2D mask painted on the rendered image can be back-projected
onto the original 3D points consistently.

Camera model (matching heatmap_renderer):
  - up = (0,0,1)
  - camera position = center + dist * [cos(el)cos(az), cos(el)sin(az), sin(el)]
  - looks at center
  - vertical FOV = 30 deg, square window
  - pyvista plotter.camera.zoom(0.9): VTK zoom multiplies the parallel/perspective
    scale; for perspective it scales tan(fov/2) by 1/zoom. zoom<1 -> object smaller.
    So effective tan(fov/2) = tan(fov/2)/zoom.
  - dist = _camera_distance(sphere_r, elev): effective_r/tan(halffov)*margin
"""
from __future__ import annotations

import numpy as np

FOV_DEG = 30.0
MARGIN = 1.12   # tighter framing than the 1.55 default so the object fills the frame
ZOOM = 1.0      # painting-friendly framing; camera stays self-consistent for back-projection


def mesh_center_and_radius(points: np.ndarray):
    center = points.mean(axis=0).astype(np.float64)
    r = np.linalg.norm(points - center, axis=1)
    return center, max(float(r.max()) if r.size else 1e-3, 1e-3)


def camera_distance(sphere_r: float, elev_deg: float,
                    fov_deg: float = FOV_DEG, margin: float = MARGIN) -> float:
    el = np.deg2rad(abs(elev_deg))
    effective_r = sphere_r * (1.0 + 0.35 * np.sin(el))
    half_fov = np.deg2rad(fov_deg) / 2.0
    return effective_r / max(np.tan(half_fov), 1e-6) * margin


def camera_eye(center, dist, elev_deg, azim_deg):
    az = np.deg2rad(azim_deg)
    el = np.deg2rad(elev_deg)
    off = dist * np.array([np.cos(el) * np.cos(az),
                           np.cos(el) * np.sin(az),
                           np.sin(el)], dtype=np.float64)
    return center + off


class Camera:
    def __init__(self, points, elev_deg, azim_deg, width=720, height=720,
                 fov_deg=FOV_DEG, zoom=ZOOM, margin=MARGIN):
        self.elev = float(elev_deg)
        self.azim = float(azim_deg)
        self.width = int(width)
        self.height = int(height)
        self.fov_deg = float(fov_deg)
        self.zoom = float(zoom)
        self.center, self.sphere_r = mesh_center_and_radius(points)
        self.dist = camera_distance(self.sphere_r, elev_deg, fov_deg, margin)
        self.eye = camera_eye(self.center, self.dist, elev_deg, azim_deg)

        up = np.array([0.0, 0.0, 1.0])
        f = (self.center - self.eye)
        f = f / np.linalg.norm(f)              # forward (view dir)
        s = np.cross(f, up)
        if np.linalg.norm(s) < 1e-9:
            up = np.array([0.0, 1.0, 0.0])
            s = np.cross(f, up)
        s = s / np.linalg.norm(s)              # right
        u = np.cross(s, f)                      # true up
        self.R = np.stack([s, u, -f], axis=0)  # world->cam rotation (rows)
        self.forward = f
        # effective half-fov after zoom
        self.tan_half = np.tan(np.deg2rad(fov_deg) / 2.0) / self.zoom

    def project(self, pts: np.ndarray):
        """World points (N,3) -> (u,v) pixel coords float, and depth (cam -z).

        Returns u, v (pixels, origin top-left), depth (positive in front).
        """
        pts = np.asarray(pts, dtype=np.float64)
        cam = (pts - self.eye) @ self.R.T          # (N,3) in camera space
        xc, yc, zc = cam[:, 0], cam[:, 1], cam[:, 2]
        depth = -zc                                # +z_cam is behind (-f)
        eps = 1e-9
        d = np.where(np.abs(depth) < eps, eps, depth)
        # NDC in [-1,1]; vertical fov, square aspect (w==h)
        aspect = self.width / self.height
        ndc_x = (xc / d) / (self.tan_half * aspect)
        ndc_y = (yc / d) / self.tan_half
        u = (ndc_x * 0.5 + 0.5) * self.width
        v = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height  # flip y for image coords
        return u, v, depth

    def to_dict(self):
        return {
            "elev": self.elev, "azim": self.azim,
            "width": self.width, "height": self.height,
            "fov_deg": self.fov_deg, "zoom": self.zoom,
            "center": self.center.tolist(), "eye": self.eye.tolist(),
            "dist": self.dist, "sphere_r": self.sphere_r,
            "tan_half": self.tan_half,
        }
