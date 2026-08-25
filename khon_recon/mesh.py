"""Surface reconstruction: dense point cloud -> watertight textured mesh.

Poisson surface reconstruction fits a smooth implicit surface to an oriented
point cloud. Two details dominate the result and both get explicit control here:

  *Normal orientation.* Poisson solves for a function whose gradient matches
  the point normals, so normals that flip direction across the surface produce
  an inside-out or blobby mesh. Consistent orientation matters more than the
  reconstruction depth.

  *Density trimming.* Poisson is watertight by construction: it will happily
  invent surface across regions no camera ever saw, producing smooth balloons
  over the recessed eye sockets and open mouth the proposal flags as
  problem areas. Trimming low-density vertices removes that invented geometry
  and turns it back into an honest hole -- which is what the completeness
  metric then measures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .io_utils import get_logger, timed

log = get_logger(__name__)


def load_point_cloud(path: Path):
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"{path} contains no points")
    return pcd


def clean_point_cloud(pcd, cfg: Config) -> tuple[Any, dict[str, Any]]:
    """Remove statistical outliers before meshing.

    MVS produces flyers -- points floating off the surface where photometric
    matching failed, which on this subject means the gilded and mirrored
    regions. Poisson will try to wrap a surface around them.
    """
    before = len(pcd.points)
    with timed(f"removing outliers from {before} points", log):
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=cfg.mesh.outlier_nb_neighbors,
            std_ratio=cfg.mesh.outlier_std_ratio,
        )
    after = len(pcd.points)
    log.info("kept %d/%d points (%.1f%% removed as outliers)",
             after, before, 100 * (before - after) / max(before, 1))
    return pcd, {"points_before": before, "points_after": after}


def estimate_normals(pcd, cfg: Config, camera_centers: np.ndarray | None = None):
    """Estimate and consistently orient point normals.

    Orientation, not estimation, is the hard part. A normal field that is
    globally consistent but points *inward* produces a mesh that looks fine in
    a viewer yet is inside-out: every visibility test then reports the surface
    as self-occluded, and texturing silently colours almost nothing.

    Three sources of orientation, best first:

    1. Normals already on the cloud. COLMAP's ``stereo_fusion`` writes normals
       derived from the depth maps, which are correctly outward-facing by
       construction. Re-estimating would throw that away.
    2. Camera positions. Every reconstructed surface point was seen by some
       camera, so its normal must point towards the cameras that saw it. This
       is exact for object-centric capture and is what fixes an inward field.
    3. Tangent-plane propagation, which is only globally consistent up to an
       arbitrary global sign -- the fallback when no cameras are available.
    """
    import open3d as o3d

    if cfg.mesh.use_existing_normals and pcd.has_normals():
        log.info("using the normals already on the point cloud (COLMAP fusion output)")
    else:
        with timed("estimating normals", log):
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=cfg.mesh.normal_knn)
            )
        with timed("orienting normals consistently", log):
            pcd.orient_normals_consistent_tangent_plane(k=cfg.mesh.normal_orient_knn)

    if camera_centers is not None and len(camera_centers):
        _orient_normals_towards_cameras(pcd, np.asarray(camera_centers, dtype=float))

    return pcd


def _orient_normals_towards_cameras(pcd, camera_centers: np.ndarray) -> None:
    """Flip each normal to face the nearest camera that could have seen it.

    Uses the nearest camera rather than a global centroid because a capture
    orbits the object: the centroid of all camera positions sits near the
    object itself and gives no usable direction.
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    if normals.shape != points.shape:
        return

    # (n_points, n_cameras) distances -> nearest camera per point.
    deltas = points[:, None, :] - camera_centers[None, :, :]
    nearest = np.argmin(np.einsum("pcd,pcd->pc", deltas, deltas), axis=1)
    to_camera = camera_centers[nearest] - points

    flip = np.einsum("pd,pd->p", normals, to_camera) < 0
    if flip.any():
        normals[flip] *= -1.0
        pcd.normals = __import__("open3d").utility.Vector3dVector(normals)
        log.info(
            "flipped %d/%d normals (%.1f%%) to face the observing cameras",
            int(flip.sum()), flip.size, 100 * flip.mean(),
        )


def poisson_mesh(pcd, cfg: Config) -> tuple[Any, dict[str, Any]]:
    """Run Poisson reconstruction and trim invented low-density geometry."""
    import open3d as o3d

    with timed(f"Poisson reconstruction (depth={cfg.mesh.poisson_depth})", log):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=cfg.mesh.poisson_depth, scale=cfg.mesh.poisson_scale, linear_fit=False
        )
    densities = np.asarray(densities)
    stats: dict[str, Any] = {
        "poisson_depth": cfg.mesh.poisson_depth,
        "vertices_raw": len(mesh.vertices),
        "triangles_raw": len(mesh.triangles),
    }

    quantile = cfg.mesh.density_trim_quantile
    if quantile > 0 and densities.size:
        threshold = float(np.quantile(densities, quantile))
        keep = densities >= threshold
        mesh.remove_vertices_by_mask(~keep)
        log.info(
            "trimmed %d low-density vertices (below the %.0f%% quantile) -- "
            "these are surfaces Poisson invented where no camera looked",
            int((~keep).sum()), 100 * quantile,
        )
        stats["density_threshold"] = threshold
        stats["vertices_trimmed"] = int((~keep).sum())

    return mesh, stats


def clean_mesh(mesh, cfg: Config) -> tuple[Any, dict[str, Any]]:
    """Standard mesh hygiene, then optional decimation."""
    before_v, before_t = len(mesh.vertices), len(mesh.triangles)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if cfg.mesh.keep_largest_component and len(mesh.triangles):
        # Keep the mask, drop the debris: turntable edges, leftover background
        # fragments, isolated flyer blobs.
        labels, counts, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        counts = np.asarray(counts)
        if counts.size > 1:
            largest = int(np.argmax(counts))
            mesh.remove_triangles_by_mask(labels != largest)
            mesh.remove_unreferenced_vertices()
            log.info(
                "kept the largest of %d connected components (%d of %d triangles)",
                counts.size, counts[largest], counts.sum(),
            )

    if cfg.mesh.target_triangles > 0 and len(mesh.triangles) > cfg.mesh.target_triangles:
        with timed(f"decimating to ~{cfg.mesh.target_triangles} triangles", log):
            mesh = mesh.simplify_quadric_decimation(cfg.mesh.target_triangles)

    mesh.compute_vertex_normals()
    return mesh, {
        "vertices_before_clean": before_v,
        "triangles_before_clean": before_t,
        "vertices_final": len(mesh.vertices),
        "triangles_final": len(mesh.triangles),
    }


def transfer_vertex_colors(mesh, pcd) -> Any:
    """Colour mesh vertices from the nearest dense point.

    The baseline colour recovery: no UV atlas, but enough for the renders used
    in the novel-view comparison, and it never fails.
    """
    import open3d as o3d

    if not pcd.has_colors():
        log.warning("dense cloud has no colours; mesh will be untextured")
        return mesh

    tree = o3d.geometry.KDTreeFlann(pcd)
    cloud_colors = np.asarray(pcd.colors)
    vertices = np.asarray(mesh.vertices)
    colors = np.zeros_like(vertices)

    with timed(f"transferring colour to {len(vertices)} vertices", log):
        for i, vertex in enumerate(vertices):
            count, idx, _ = tree.search_knn_vector_3d(vertex, 1)
            if count:
                colors[i] = cloud_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def ensure_outward_normals(mesh, camera_centers: np.ndarray) -> float:
    """Verify the finished mesh faces outward, and flip it if it does not.

    Poisson inherits orientation from the input normal field, so this should
    already hold -- but an inside-out mesh is silent and poisons every
    downstream visibility test, so it is checked rather than assumed. The test
    is majority-vote: most of a visible surface should face the cameras.
    """
    import open3d as o3d

    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    if vertices.size == 0:
        return float("nan")

    deltas = vertices[:, None, :] - np.asarray(camera_centers)[None, :, :]
    nearest = np.argmin(np.einsum("pcd,pcd->pc", deltas, deltas), axis=1)
    to_camera = np.asarray(camera_centers)[nearest] - vertices
    outward = float((np.einsum("pd,pd->p", normals, to_camera) > 0).mean())

    if outward < 0.5:
        log.warning(
            "mesh normals point inward (only %.1f%% face the cameras); flipping "
            "triangle winding. Left uncorrected this makes every surface point "
            "test as self-occluded and texturing colours nothing.",
            100 * outward,
        )
        mesh.triangles = o3d.utility.Vector3iVector(
            np.asarray(mesh.triangles)[:, ::-1]
        )
        mesh.compute_vertex_normals()
        outward = 1.0 - outward

    log.info("%.1f%% of mesh normals face the observing cameras", 100 * outward)
    return outward


def run_meshing(cfg: Config, input_ply: Path | None = None) -> dict[str, Any]:
    """Full point-cloud-to-mesh stage."""
    import open3d as o3d

    from .metrics import load_reconstruction, mesh_completeness, point_cloud_density
    from .sfm import camera_centers as get_camera_centers

    source = Path(input_ply) if input_ply else (cfg.dense_dir / "fused.ply")
    if not source.exists():
        raise FileNotFoundError(
            f"no dense point cloud at {source}. Run the Colab dense stage and "
            "import it with scripts/03_dense_import.py, or pass --input to mesh "
            "a different cloud."
        )

    cfg.make_run_dirs()
    pcd = load_point_cloud(source)
    log.info("loaded %d points from %s", len(pcd.points), source.name)

    stats: dict[str, Any] = {"input_ply": str(source)}
    stats["density"] = point_cloud_density(pcd, cfg.eval.density_sample)

    pcd, clean_stats = clean_point_cloud(pcd, cfg)
    stats.update(clean_stats)

    # Camera positions orient the normal field; without them Poisson can
    # produce a geometrically plausible but inside-out surface.
    camera_centers = None
    try:
        camera_centers = get_camera_centers(load_reconstruction(cfg.sparse_dir))
        log.info("orienting normals using %d reconstructed cameras", len(camera_centers))
    except FileNotFoundError:
        log.warning(
            "no sparse model found; falling back to tangent-plane normal "
            "orientation, whose global sign is arbitrary"
        )

    pcd = estimate_normals(pcd, cfg, camera_centers)
    mesh, poisson_stats = poisson_mesh(pcd, cfg)
    stats.update(poisson_stats)

    mesh, mesh_clean_stats = clean_mesh(mesh, cfg)
    stats.update(mesh_clean_stats)

    if camera_centers is not None and len(camera_centers):
        stats["outward_normal_fraction"] = ensure_outward_normals(mesh, camera_centers)

    mesh = transfer_vertex_colors(mesh, pcd)

    out_path = cfg.mesh_dir / "mesh.ply"
    o3d.io.write_triangle_mesh(str(out_path), mesh)
    stats["mesh_ply"] = str(out_path)

    stats["completeness"] = mesh_completeness(mesh)
    log.info(
        "mesh: %d vertices, %d triangles, %d hole(s), watertight=%s",
        stats["completeness"]["n_vertices"],
        stats["completeness"]["n_triangles"],
        stats["completeness"]["n_holes"],
        stats["completeness"]["is_watertight"],
    )
    return stats
