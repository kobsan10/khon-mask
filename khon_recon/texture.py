"""Colour recovery: project the source photographs onto the reconstructed surface.

This implements what the proposal describes -- "the mask's colour will be
recovered by projecting and blending the source photographs onto the
reconstructed surface" -- rather than the cheaper shortcut of copying colours
from the nearest dense point.

Three things make the blend robust on this particular subject:

*Visibility testing.* A surface point is only coloured from cameras that could
actually see it. Without this, the back of the crown gets coloured by a camera
in front of the mask, which is the classic photogrammetry texturing artifact.
Visibility comes from a rendered depth buffer per camera.

*View-angle weighting.* Observations at a grazing angle cover more surface per
pixel and smear fine ornament, so they are down-weighted and eventually
rejected.

*Median blending.* The proposal predicts gilded and mirrored decoration will be
the main failure mode. A specular highlight is a bright outlier in one or two
views while the remaining views agree, so a median across views rejects it
where a mean would smear it across the surface. This is the cheapest available
defence against the paper's headline risk.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap

from .config import Config
from .io_utils import get_logger, timed
from .render import camera_to_opencv, extrinsic_matrix, undistort_photo

log = get_logger(__name__)


def _sample_bilinear(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear colour lookup at float pixel coordinates."""
    h, w = image.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]
    return (
        image[y0, x0] * (1 - wx) * (1 - wy)
        + image[y0, x1] * wx * (1 - wy)
        + image[y1, x0] * (1 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def gather_multiview_colors(
    points: np.ndarray,
    normals: np.ndarray,
    reconstruction: pycolmap.Reconstruction,
    images_dir: Path,
    scene,
    max_view_angle_deg: float = 70.0,
    blend: str = "median",
    depth_tolerance: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Colour 3D points by blending every camera that can see them.

    Returns ``(colors, n_observations)`` with colours in RGB [0, 1].
    """
    import open3d as o3d

    n_points = points.shape[0]
    samples: list[np.ndarray] = []  # per-camera colour, NaN where not visible
    weights: list[np.ndarray] = []

    image_ids = sorted(reconstruction.reg_image_ids())
    for image_id in image_ids:
        image = reconstruction.image(image_id)
        camera = reconstruction.camera(image.camera_id)

        photo = cv2.imread(str(images_dir / image.name), cv2.IMREAD_COLOR)
        if photo is None:
            continue
        photo = undistort_photo(photo, camera)
        photo_rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        K, _ = camera_to_opencv(camera)
        extrinsic = extrinsic_matrix(image)
        R, t = extrinsic[:3, :3], extrinsic[:3, 3]

        # World -> camera -> image (already undistorted, so a pinhole projection).
        cam_points = points @ R.T + t
        in_front = cam_points[:, 2] > 1e-6
        uv = np.full((n_points, 2), -1.0)
        safe_z = np.where(in_front, cam_points[:, 2], 1.0)
        uv[:, 0] = K[0, 0] * cam_points[:, 0] / safe_z + K[0, 2]
        uv[:, 1] = K[1, 1] * cam_points[:, 1] / safe_z + K[1, 2]

        inside = (
            in_front
            & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width - 1)
            & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height - 1)
        )

        # View angle: reject grazing observations, which smear fine ornament.
        center = np.asarray(image.projection_center(), dtype=float)
        to_camera = center - points
        to_camera /= np.maximum(np.linalg.norm(to_camera, axis=1, keepdims=True), 1e-9)
        cos_angle = (normals * to_camera).sum(axis=1)
        facing = cos_angle > np.cos(np.radians(max_view_angle_deg))

        candidate = inside & facing
        if not candidate.any():
            continue

        # Visibility: is this point the first surface the camera meets?
        idx = np.flatnonzero(candidate)
        origins = np.repeat(center[None, :], idx.size, axis=0)
        directions = points[idx] - origins
        distances = np.linalg.norm(directions, axis=1)
        directions /= np.maximum(distances[:, None], 1e-9)

        rays = o3d.core.Tensor(
            np.hstack([origins, directions]).astype(np.float32), dtype=o3d.core.Dtype.Float32
        )
        hits = scene.cast_rays(rays)["t_hit"].numpy()
        # Occluded if the ray stops meaningfully short of the point itself.
        visible = hits >= distances * (1.0 - depth_tolerance)

        selected = idx[visible]
        if selected.size == 0:
            continue

        colors = np.full((n_points, 3), np.nan)
        colors[selected] = _sample_bilinear(photo_rgb, uv[selected])
        weight = np.zeros(n_points)
        weight[selected] = cos_angle[selected]

        samples.append(colors)
        weights.append(weight)

    if not samples:
        log.warning("no camera saw any surface point; leaving the mesh uncoloured")
        return np.full((n_points, 3), 0.7), np.zeros(n_points, dtype=int)

    stacked = np.stack(samples, axis=0)  # (views, points, 3)
    weight_stack = np.stack(weights, axis=0)  # (views, points)
    counts = np.isfinite(stacked[..., 0]).sum(axis=0)

    # Points no camera saw are all-NaN columns; that is an expected outcome
    # (it is what "unseen_fraction" reports), not a numerical problem.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if blend == "mean":
            colors = np.nanmean(stacked, axis=0)
        elif blend == "weighted":
            w = np.where(np.isfinite(stacked[..., 0]), weight_stack, 0.0)[..., None]
            colors = np.nansum(np.nan_to_num(stacked) * w, axis=0) / np.maximum(
                w.sum(axis=0), 1e-9
            )
        else:  # median -- the default, and the specular-highlight defence
            colors = np.nanmedian(stacked, axis=0)

    colors = np.nan_to_num(colors, nan=0.7)
    unseen = counts == 0
    if unseen.any():
        log.info(
            "%d/%d surface points were never seen by any camera (%.1f%%)",
            int(unseen.sum()), n_points, 100 * unseen.mean(),
        )
    return np.clip(colors, 0.0, 1.0), counts


def texture_mesh_vertices(
    mesh, reconstruction: pycolmap.Reconstruction, images_dir: Path, cfg: Config
) -> tuple[Any, dict[str, Any]]:
    """Colour mesh vertices by multi-view projection and blending."""
    import open3d as o3d

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    points = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    with timed(
        f"projecting {reconstruction.num_reg_images()} photographs onto "
        f"{points.shape[0]} vertices ({cfg.texture.blend} blend)", log
    ):
        colors, counts = gather_multiview_colors(
            points, normals, reconstruction, images_dir, scene,
            max_view_angle_deg=cfg.texture.max_view_angle_deg,
            blend=cfg.texture.blend,
        )

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh, {
        "blend": cfg.texture.blend,
        "max_view_angle_deg": cfg.texture.max_view_angle_deg,
        "mean_views_per_vertex": float(counts.mean()),
        "median_views_per_vertex": float(np.median(counts)),
        "unseen_vertices": int((counts == 0).sum()),
        "unseen_fraction": float((counts == 0).mean()),
    }


def bake_uv_texture(
    mesh, reconstruction: pycolmap.Reconstruction, images_dir: Path, cfg: Config
) -> tuple[Any, dict[str, Any]]:
    """Unwrap with xatlas and bake a UV texture map.

    Produces a proper textured mesh (OBJ + PNG) that opens in MeshLab, Blender
    or a viewer, which is what a "textured 3D mesh" deliverable means. Falls
    back to vertex colours if xatlas is unavailable.
    """
    import open3d as o3d

    try:
        import xatlas
    except ImportError:
        log.warning("xatlas not installed; falling back to vertex colours")
        return texture_mesh_vertices(mesh, reconstruction, images_dir, cfg)

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.uint32)

    with timed("unwrapping UVs with xatlas", log):
        vmapping, indices, uvs = xatlas.parametrize(vertices, triangles)

    # xatlas splits vertices along seams, so rebuild the mesh on its indexing.
    new_vertices = vertices[vmapping]
    unwrapped = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(new_vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(indices.astype(np.int32)),
    )
    unwrapped.compute_vertex_normals()

    size = cfg.texture.texture_size
    texture = np.zeros((size, size, 3), dtype=np.float64)
    filled = np.zeros((size, size), dtype=bool)

    # Rasterise each triangle in texture space and record the 3D position and
    # normal of every texel it covers.
    texel_points: list[np.ndarray] = []
    texel_normals: list[np.ndarray] = []
    texel_coords: list[np.ndarray] = []

    normals = np.asarray(unwrapped.vertex_normals)
    with timed(f"rasterising {indices.shape[0]} triangles into a {size}px atlas", log):
        for tri in indices:
            uv_tri = uvs[tri] * (size - 1)
            pts, nrm, coords = _rasterise_triangle(
                uv_tri, new_vertices[tri], normals[tri], size
            )
            if pts.size:
                texel_points.append(pts)
                texel_normals.append(nrm)
                texel_coords.append(coords)

    if not texel_points:
        log.warning("UV rasterisation produced no texels; falling back to vertex colours")
        return texture_mesh_vertices(mesh, reconstruction, images_dir, cfg)

    points = np.concatenate(texel_points)
    point_normals = np.concatenate(texel_normals)
    coords = np.concatenate(texel_coords)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(unwrapped))

    with timed(f"colouring {points.shape[0]} texels from the photographs", log):
        colors, counts = gather_multiview_colors(
            points, point_normals, reconstruction, images_dir, scene,
            max_view_angle_deg=cfg.texture.max_view_angle_deg,
            blend=cfg.texture.blend,
        )

    texture[coords[:, 1], coords[:, 0]] = colors
    filled[coords[:, 1], coords[:, 0]] = True

    # Dilate outward so bilinear filtering at seams does not sample black.
    texture_u8 = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
    texture_u8 = cv2.inpaint(
        texture_u8, (~filled).astype(np.uint8), 3, cv2.INPAINT_TELEA
    )

    unwrapped.triangle_uvs = o3d.utility.Vector2dVector(
        uvs[indices.reshape(-1)].astype(np.float64)
    )
    unwrapped.textures = [o3d.geometry.Image(texture_u8)]
    unwrapped.triangle_material_ids = o3d.utility.IntVector(
        np.zeros(indices.shape[0], dtype=np.int32)
    )

    return unwrapped, {
        "mode": "uv",
        "texture_size": size,
        "n_texels_filled": int(filled.sum()),
        "atlas_fill_fraction": float(filled.mean()),
        "mean_views_per_texel": float(counts.mean()),
        "unseen_texels": int((counts == 0).sum()),
    }


def _rasterise_triangle(
    uv_tri: np.ndarray, xyz_tri: np.ndarray, normal_tri: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the 3D position, normal and atlas coordinate of each covered texel."""
    min_xy = np.floor(uv_tri.min(axis=0)).astype(int)
    max_xy = np.ceil(uv_tri.max(axis=0)).astype(int)
    min_xy = np.maximum(min_xy, 0)
    max_xy = np.minimum(max_xy, size - 1)
    if np.any(max_xy < min_xy):
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2), dtype=int)

    xs = np.arange(min_xy[0], max_xy[0] + 1)
    ys = np.arange(min_xy[1], max_xy[1] + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    px = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(float)

    # Barycentric coordinates of every candidate texel centre.
    v0 = uv_tri[1] - uv_tri[0]
    v1 = uv_tri[2] - uv_tri[0]
    v2 = px - uv_tri[0]
    denom = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(denom) < 1e-12:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2), dtype=int)

    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / denom
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / denom
    w = 1.0 - u - v
    inside = (u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6)
    if not inside.any():
        return np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 2), dtype=int)

    bary = np.stack([w[inside], u[inside], v[inside]], axis=1)
    points = bary @ xyz_tri
    normals = bary @ normal_tri
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    return points, normals, px[inside].astype(int)


def run_texturing(cfg: Config) -> dict[str, Any]:
    """Texture the reconstructed mesh and write the result."""
    import open3d as o3d

    from .metrics import load_reconstruction

    mesh_path = cfg.mesh_dir / "mesh.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"no mesh at {mesh_path}; run scripts/04_mesh.py first")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    reconstruction = load_reconstruction(cfg.sparse_dir)

    if cfg.texture.mode == "uv":
        mesh, stats = bake_uv_texture(mesh, reconstruction, cfg.images_dir, cfg)
        out_path = cfg.mesh_dir / "mesh_textured.obj"
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_triangle_uvs=True)
    else:
        mesh, stats = texture_mesh_vertices(mesh, reconstruction, cfg.images_dir, cfg)
        out_path = cfg.mesh_dir / "mesh_textured.ply"
        o3d.io.write_triangle_mesh(str(out_path), mesh)

    stats["output"] = str(out_path)
    stats["mode"] = cfg.texture.mode
    log.info("wrote textured mesh to %s", out_path)
    return stats
