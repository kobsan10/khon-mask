"""Render the reconstructed mesh from estimated camera poses.

Used for the novel-view comparison: render the mesh from a held-out camera's
exact intrinsics and extrinsics, then compare against the real photograph taken
from that pose.

Rendering is done by CPU ray casting (``open3d.t.geometry.RaycastingScene``)
rather than through the GPU offscreen renderer. Ray casting is headless,
deterministic, and free of the Filament/GL setup that makes the GPU path
unreliable on macOS -- and it returns a depth buffer directly, which is what
provides the silhouette mask for the masked comparison.

Note that ray casting is a pinhole projection and therefore ignores lens
distortion, so the *photograph* is undistorted to match rather than the render
being distorted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap

from .io_utils import get_logger

log = get_logger(__name__)


@dataclass
class RenderResult:
    color: np.ndarray  # HxWx3 uint8, BGR (OpenCV convention)
    depth: np.ndarray  # HxW float32, +inf where nothing was hit
    mask: np.ndarray  # HxW bool, True where the surface was hit


# --------------------------------------------------------------------------
# Camera conversion
# --------------------------------------------------------------------------


def camera_to_opencv(camera: pycolmap.Camera) -> tuple[np.ndarray, np.ndarray]:
    """Convert a COLMAP camera to an OpenCV (K, distCoeffs) pair.

    Covers the models this pipeline actually produces. SIMPLE_RADIAL is the
    default and is what a phone or consumer camera is fitted with.
    """
    params = np.asarray(camera.params, dtype=float)
    model = camera.model_name

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        fx = fy = f
        dist = np.zeros(5)
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        dist = np.zeros(5)
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        fx = fy = f
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0])
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params
        fx = fy = f
        dist = np.array([k1, k2, 0.0, 0.0, 0.0])
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        dist = np.array([k1, k2, p1, p2, 0.0])
    else:
        raise NotImplementedError(
            f"camera model {model!r} is not handled; use SIMPLE_RADIAL, RADIAL, "
            "PINHOLE, SIMPLE_PINHOLE or OPENCV"
        )

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    return K, dist


def extrinsic_matrix(image: pycolmap.Image) -> np.ndarray:
    """4x4 world-to-camera matrix for a registered image."""
    matrix = np.eye(4)
    matrix[:3, :4] = np.asarray(image.cam_from_world().matrix(), dtype=float)
    return matrix


def undistort_photo(photo: np.ndarray, camera: pycolmap.Camera) -> np.ndarray:
    """Undistort a real photograph so it matches the pinhole render."""
    K, dist = camera_to_opencv(camera)
    if not np.any(dist):
        return photo
    return cv2.undistort(photo, K, dist, None, K)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


class MeshRenderer:
    """Ray-casts a mesh from arbitrary camera poses."""

    def __init__(self, mesh, background: tuple[int, int, int] = (255, 255, 255)):
        import open3d as o3d

        self._o3d = o3d
        self.background = background
        self.vertices = np.asarray(mesh.vertices)
        self.triangles = np.asarray(mesh.triangles)
        self.vertex_colors = (
            np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        )
        if self.vertex_colors is None:
            log.warning("mesh has no vertex colours; renders will be shaded grey")
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        self.vertex_normals = np.asarray(mesh.vertex_normals)

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    def render(
        self, camera: pycolmap.Camera, image: pycolmap.Image, scale: float = 1.0
    ) -> RenderResult:
        """Render from a registered image's pose at (optionally scaled) resolution."""
        K, _ = camera_to_opencv(camera)
        width = int(round(camera.width * scale))
        height = int(round(camera.height * scale))
        if scale != 1.0:
            K = K.copy()
            K[:2, :] *= scale

        rays = self.scene.create_rays_pinhole(
            intrinsic_matrix=self._o3d.core.Tensor(K),
            extrinsic_matrix=self._o3d.core.Tensor(extrinsic_matrix(image)),
            width_px=width,
            height_px=height,
        )
        answer = self.scene.cast_rays(rays)

        t_hit = answer["t_hit"].numpy()
        mask = np.isfinite(t_hit)
        depth = np.where(mask, t_hit, np.inf).astype(np.float32)

        color = np.full((height, width, 3), self.background, dtype=np.uint8)
        if mask.any():
            color[mask] = self._shade(answer, mask)

        return RenderResult(color=color, depth=depth, mask=mask)

    def _shade(self, answer: dict, mask: np.ndarray) -> np.ndarray:
        """Interpolate vertex colour across the hit triangle, with light shading."""
        prim_ids = answer["primitive_ids"].numpy()[mask].astype(np.int64)
        uvs = answer["primitive_uvs"].numpy()[mask].astype(np.float64)

        tri = self.triangles[prim_ids]
        u = uvs[:, 0:1]
        v = uvs[:, 1:2]
        w = 1.0 - u - v  # barycentric weight of the first vertex

        if self.vertex_colors is not None:
            rgb = (
                w * self.vertex_colors[tri[:, 0]]
                + u * self.vertex_colors[tri[:, 1]]
                + v * self.vertex_colors[tri[:, 2]]
            )
        else:
            rgb = np.full((tri.shape[0], 3), 0.7)

        # Gentle normal-based shading so geometry stays legible where the
        # texture is flat -- important when judging crown relief by eye.
        normals = (
            w * self.vertex_normals[tri[:, 0]]
            + u * self.vertex_normals[tri[:, 1]]
            + v * self.vertex_normals[tri[:, 2]]
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        view_dir = answer["primitive_normals"].numpy()[mask]
        shade = 0.75 + 0.25 * np.abs((normals * view_dir).sum(axis=1, keepdims=True))
        rgb = np.clip(rgb * shade, 0.0, 1.0)

        bgr = (rgb[:, ::-1] * 255).astype(np.uint8)  # RGB -> BGR for OpenCV
        return bgr


def render_registered_views(
    mesh,
    reconstruction: pycolmap.Reconstruction,
    image_names: list[str] | None,
    images_dir: Path,
    out_dir: Path,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Render the mesh from each requested view and save render/photo pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = MeshRenderer(mesh)
    wanted = set(image_names) if image_names else None
    results: list[dict[str, Any]] = []

    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        if wanted is not None and image.name not in wanted:
            continue
        camera = reconstruction.camera(image.camera_id)

        rendered = renderer.render(camera, image, scale=scale)
        stem = Path(image.name).stem
        render_path = out_dir / f"{stem}_render.png"
        cv2.imwrite(str(render_path), rendered.color)

        photo_path = images_dir / image.name
        photo = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
        undistorted_path = None
        if photo is not None:
            photo = undistort_photo(photo, camera)
            if scale != 1.0:
                photo = cv2.resize(
                    photo,
                    (rendered.color.shape[1], rendered.color.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            undistorted_path = out_dir / f"{stem}_photo.png"
            cv2.imwrite(str(undistorted_path), photo)

        results.append(
            {
                "image": image.name,
                "render": str(render_path),
                "photo": str(undistorted_path) if undistorted_path else None,
                "coverage": float(rendered.mask.mean()),
            }
        )

    log.info("rendered %d view(s) into %s", len(results), out_dir)
    return results
