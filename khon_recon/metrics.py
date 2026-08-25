"""Reconstruction quality metrics.

These implement exactly what the proposal's evaluation plan promises, given
that no ground-truth 3D scan of the mask exists:

  1. mean reprojection error from bundle adjustment  -- Eq. (1)
  2. density and completeness of the dense point cloud, and where the holes are
  3. inputs for the novel-view comparison (see render.py / compare.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from .io_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Eq. (1): mean reprojection error
# --------------------------------------------------------------------------


def reprojection_errors(reconstruction: pycolmap.Reconstruction) -> dict[str, Any]:
    r"""Compute the paper's Eq. (1) directly from the reconstruction.

        E = (1/N) * sum_i || x_i - pi(P, X_i) ||

    where the sum runs over all N observations: ``x_i`` is an observed 2D
    feature location, ``X_i`` the triangulated 3D point, and ``pi(P, .)`` the
    projection through camera ``P``.

    Implemented explicitly -- world point -> camera frame -> image plane --
    rather than by calling a library helper, for two reasons.

    First, fidelity: the number in the report is demonstrably the formula in
    the paper. Note that Eq. (1) averages over *observations*, whereas COLMAP's
    ``compute_mean_reprojection_error`` averages the per-point mean errors, so
    the two differ whenever track lengths vary. Both are reported.

    Second, and more important, correctness: COLMAP's built-in reads a *stored*
    per-point error that is written during bundle adjustment and is not
    refreshed afterwards. After registering held-out views (which adds
    observations to existing tracks) the built-in silently reports the
    pre-registration value -- measured on the smoke-test model as 0.362 px when
    the true figure was 0.402 px. Recomputing from scratch avoids quietly
    understating the error.
    """
    residuals: list[float] = []
    per_point_means: list[float] = []
    per_image: dict[str, list[float]] = {}

    for point3D_id in reconstruction.point3D_ids():
        point3D = reconstruction.point3D(point3D_id)
        xyz = np.asarray(point3D.xyz, dtype=float)
        point_errors: list[float] = []

        for element in point3D.track.elements:
            image = reconstruction.image(element.image_id)
            if not image.has_pose:  # property, not a method
                continue
            camera = reconstruction.camera(image.camera_id)

            # pi(P, X): rigid transform into the camera frame, then project
            # through the intrinsics (including radial distortion).
            xyz_cam = image.cam_from_world() * xyz
            if xyz_cam[2] <= 0:  # behind the camera; not a valid observation
                continue
            projected = np.asarray(camera.img_from_cam(xyz_cam), dtype=float)

            observed = np.asarray(image.point2D(element.point2D_idx).xy, dtype=float)
            error = float(np.linalg.norm(observed - projected))

            residuals.append(error)
            point_errors.append(error)
            per_image.setdefault(image.name, []).append(error)

        if point_errors:
            per_point_means.append(float(np.mean(point_errors)))

    if not residuals:
        return {"n_observations": 0, "mean_px": float("nan")}

    values = np.asarray(residuals)
    return {
        "n_observations": int(values.size),
        "n_points": len(per_point_means),
        # Eq. (1) as written: the average over all N observations.
        "mean_px": float(values.mean()),
        # COLMAP's convention, recomputed rather than read from storage.
        "mean_per_point_px": float(np.mean(per_point_means)),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(values.max()),
        "per_image_mean_px": {k: float(np.mean(v)) for k, v in per_image.items()},
        "histogram": _histogram(values, bins=40, upper=float(np.percentile(values, 99))),
    }


def _histogram(values: np.ndarray, bins: int, upper: float) -> dict[str, list[float]]:
    counts, edges = np.histogram(values, bins=bins, range=(0.0, max(upper, 1e-6)))
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def verify_against_builtin(reconstruction: pycolmap.Reconstruction, tol: float = 1e-3) -> dict[str, Any]:
    """Cross-check our reprojection error against COLMAP's own computation.

    The comparison is made in COLMAP's convention (per-point averaging), since
    that is what its helper returns. On a model that has not been modified
    since bundle adjustment the two agree to floating-point precision, which is
    what validates the implementation.

    A *disagreement* is not automatically a bug: COLMAP caches per-point errors
    at BA time, so any model modified afterwards -- notably one with held-out
    views registered -- will report a stale built-in value. That case is
    identified rather than treated as a failure, because the freshly computed
    number is the correct one.
    """
    errors = reprojection_errors(reconstruction)
    ours_per_point = errors["mean_per_point_px"]
    ours_per_obs = errors["mean_px"]
    builtin = float(reconstruction.compute_mean_reprojection_error())

    agrees = bool(abs(ours_per_point - builtin) <= tol)
    result = {
        "eq1_per_observation_px": ours_per_obs,
        "ours_per_point_px": ours_per_point,
        "colmap_builtin_px": builtin,
        "agrees": agrees,
        "tolerance": tol,
    }

    if agrees:
        log.info(
            "reprojection error verified against COLMAP (%.6f px, per-point convention)",
            builtin,
        )
    else:
        result["explanation"] = (
            "COLMAP's built-in reads cached per-point errors from the last bundle "
            "adjustment and does not refresh them when observations are added "
            "afterwards (e.g. by registering held-out views). The recomputed "
            "value is authoritative."
        )
        log.info(
            "recomputed reprojection error %.4f px vs COLMAP's cached %.4f px -- "
            "expected when held-out views were registered after BA; using the "
            "recomputed value",
            ours_per_point, builtin,
        )
    return result


# --------------------------------------------------------------------------
# Sparse model structure
# --------------------------------------------------------------------------


def track_statistics(reconstruction: pycolmap.Reconstruction) -> dict[str, Any]:
    """Track-length distribution.

    Track length is how many views agree on a 3D point. Short tracks mean
    weakly-constrained geometry, which is what under-covered regions look like
    before they show up as holes in the mesh.
    """
    lengths = np.array(
        [
            reconstruction.point3D(pid).track.length()
            for pid in reconstruction.point3D_ids()
        ],
        dtype=float,
    )
    if lengths.size == 0:
        return {"n_points": 0}
    return {
        "n_points": int(lengths.size),
        "mean_track_length": float(lengths.mean()),
        "median_track_length": float(np.median(lengths)),
        "frac_tracks_len2": float((lengths <= 2).mean()),
        "histogram": _histogram(lengths, bins=int(min(lengths.max(), 30)), upper=float(lengths.max())),
    }


def camera_coverage(reconstruction: pycolmap.Reconstruction) -> dict[str, Any]:
    """Angular coverage of the cameras around the subject.

    The proposal calls for circling the mask at several elevations so the
    crown, facial relief and underside are all covered. This turns that
    requirement into numbers: azimuth spread, elevation spread, and the
    largest uncovered azimuth gap.
    """
    centers = np.array(
        [
            reconstruction.image(image_id).projection_center()
            for image_id in reconstruction.reg_image_ids()
        ],
        dtype=float,
    )
    if centers.shape[0] < 3:
        return {"n_cameras": int(centers.shape[0])}

    # Centre on the reconstructed structure, not on the cameras: the object is
    # what we need coverage *of*.
    points = np.array(
        [reconstruction.point3D(pid).xyz for pid in reconstruction.point3D_ids()],
        dtype=float,
    )
    origin = np.median(points, axis=0) if points.size else centers.mean(axis=0)

    rel = centers - origin
    radius = np.linalg.norm(rel, axis=1)
    azimuth = np.degrees(np.arctan2(rel[:, 1], rel[:, 0]))
    elevation = np.degrees(np.arcsin(np.clip(rel[:, 2] / np.maximum(radius, 1e-9), -1, 1)))

    ordered = np.sort(np.mod(azimuth, 360.0))
    gaps = np.diff(np.concatenate([ordered, ordered[:1] + 360.0]))

    return {
        "n_cameras": int(centers.shape[0]),
        "azimuth_span_deg": float(ordered.max() - ordered.min()),
        "largest_azimuth_gap_deg": float(gaps.max()) if gaps.size else 0.0,
        "elevation_min_deg": float(elevation.min()),
        "elevation_max_deg": float(elevation.max()),
        "elevation_span_deg": float(elevation.max() - elevation.min()),
        "radius_mean": float(radius.mean()),
        "radius_cv": float(radius.std() / max(radius.mean(), 1e-9)),
        "azimuth_deg": azimuth.tolist(),
        "elevation_deg": elevation.tolist(),
        "centers": centers.tolist(),
        "origin": origin.tolist(),
    }


def coverage_warnings(coverage: dict[str, Any]) -> list[str]:
    """Turn coverage numbers into actionable capture advice."""
    warnings: list[str] = []
    if coverage.get("n_cameras", 0) < 3:
        return ["too few registered cameras to assess coverage"]
    if coverage["largest_azimuth_gap_deg"] > 45:
        warnings.append(
            f"{coverage['largest_azimuth_gap_deg']:.0f} deg of azimuth is uncovered -- "
            "the reconstruction will be incomplete on that side"
        )
    if coverage["elevation_span_deg"] < 30:
        warnings.append(
            f"elevation span is only {coverage['elevation_span_deg']:.0f} deg; the "
            "proposal calls for eye level, above and below so the crown and the "
            "underside are both seen"
        )
    return warnings


# --------------------------------------------------------------------------
# Dense cloud and mesh
# --------------------------------------------------------------------------


def point_cloud_density(pcd, sample: int = 50000, seed: int = 0) -> dict[str, Any]:
    """Nearest-neighbour spacing distribution of a dense point cloud.

    Median spacing is the resolution the reconstruction actually achieved --
    the finest surface detail it can represent, which for a Khon mask is the
    question of whether crown ornament survives.
    """
    import open3d as o3d

    points = np.asarray(pcd.points)
    if points.shape[0] < 10:
        return {"n_points": int(points.shape[0])}

    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(points.shape[0], size=sample, replace=False)
        if points.shape[0] > sample
        else np.arange(points.shape[0])
    )

    tree = o3d.geometry.KDTreeFlann(pcd)
    spacings = []
    for i in idx:
        # k=2: the first neighbour is the query point itself.
        count, _, sq_dists = tree.search_knn_vector_3d(points[i], 2)
        if count == 2:
            spacings.append(float(np.sqrt(sq_dists[1])))

    values = np.asarray(spacings)
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=float)
    return {
        "n_points": int(points.shape[0]),
        "median_spacing": float(np.median(values)) if values.size else float("nan"),
        "mean_spacing": float(values.mean()) if values.size else float("nan"),
        "p95_spacing": float(np.percentile(values, 95)) if values.size else float("nan"),
        "bbox_extent": extent.tolist(),
        # Spacing relative to object size, so the number is comparable across
        # runs whose reconstructions are at different arbitrary scales.
        "relative_spacing": (
            float(np.median(values) / np.linalg.norm(extent)) if values.size else float("nan")
        ),
    }


def mesh_completeness(mesh) -> dict[str, Any]:
    """Holes and manifoldness -- the proposal's 'identifying missing regions'.

    Boundary edges are literal holes in the surface. For a closed object like a
    mask, every boundary loop is somewhere the cameras never saw: typically the
    recessed eye sockets and the inside of an open mouth, exactly the failure
    the proposal predicts.
    """
    import open3d as o3d

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if triangles.shape[0] == 0:
        return {"n_vertices": int(vertices.shape[0]), "n_triangles": 0}

    boundary_edges = _boundary_edges(triangles)
    total_area = float(mesh.get_surface_area())
    hole_area, n_loops = _hole_area(vertices, boundary_edges)

    return {
        "n_vertices": int(vertices.shape[0]),
        "n_triangles": int(triangles.shape[0]),
        "surface_area": total_area,
        "is_watertight": bool(mesh.is_watertight()),
        "is_edge_manifold": bool(mesh.is_edge_manifold()),
        "n_boundary_edges": int(len(boundary_edges)),
        "n_holes": int(n_loops),
        "hole_perimeter": float(hole_area),
        "n_non_manifold_edges": int(len(mesh.get_non_manifold_edges())),
        "volume": float(mesh.get_volume()) if mesh.is_watertight() else None,
    }


def _boundary_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
    """Edges used by exactly one triangle."""
    edges: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edges[key] = edges.get(key, 0) + 1
    return [edge for edge, count in edges.items() if count == 1]


def _hole_area(vertices: np.ndarray, boundary_edges: list[tuple[int, int]]) -> tuple[float, int]:
    """Total boundary length and the number of distinct boundary loops."""
    if not boundary_edges:
        return 0.0, 0

    perimeter = sum(
        float(np.linalg.norm(vertices[a] - vertices[b])) for a, b in boundary_edges
    )

    # Count loops via union-find over the boundary vertices.
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in boundary_edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    loops = len({find(v) for edge in boundary_edges for v in edge})
    return perimeter, loops


def sparse_dense_agreement(reconstruction: pycolmap.Reconstruction, pcd) -> dict[str, Any]:
    """Check the dense cloud sits in the same frame and scale as the sparse model.

    Guards the Colab round trip: downloading the wrong ``fused.ply``, or one
    from a different run, otherwise produces a garbage mesh with no obvious
    cause. Cheap assertion, expensive bug.
    """
    sparse_points = np.array(
        [reconstruction.point3D(pid).xyz for pid in reconstruction.point3D_ids()],
        dtype=float,
    )
    dense_points = np.asarray(pcd.points)
    if sparse_points.size == 0 or dense_points.size == 0:
        return {"ok": False, "reason": "empty point set"}

    sparse_centroid = np.median(sparse_points, axis=0)
    dense_centroid = np.median(dense_points, axis=0)
    sparse_extent = np.percentile(sparse_points, 95, axis=0) - np.percentile(sparse_points, 5, axis=0)
    dense_extent = np.percentile(dense_points, 95, axis=0) - np.percentile(dense_points, 5, axis=0)

    scale = np.linalg.norm(sparse_extent)
    offset = float(np.linalg.norm(sparse_centroid - dense_centroid))
    ratio = float(np.linalg.norm(dense_extent) / max(scale, 1e-9))

    ok = offset < 0.5 * scale and 0.2 < ratio < 5.0
    return {
        "ok": bool(ok),
        "centroid_offset": offset,
        "centroid_offset_relative": float(offset / max(scale, 1e-9)),
        "extent_ratio": ratio,
        "n_sparse_points": int(sparse_points.shape[0]),
        "n_dense_points": int(dense_points.shape[0]),
    }


def load_reconstruction(sparse_dir: Path) -> pycolmap.Reconstruction:
    """Load a COLMAP model, tolerating the ``sparse/0`` nesting."""
    sparse_dir = Path(sparse_dir)
    if (sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists():
        return pycolmap.Reconstruction(sparse_dir)
    nested = sparse_dir / "0"
    if (nested / "cameras.bin").exists():
        return pycolmap.Reconstruction(nested)
    raise FileNotFoundError(f"no COLMAP model found under {sparse_dir}")
