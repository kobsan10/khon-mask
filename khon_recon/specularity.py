"""Specularity study: does gilding actually break the reconstruction?

The proposal makes a specific prediction:

    "We expect the greatest difficulty in regions with specular gilding or
    mirrored inlay, where multi-view stereo's assumption of consistent surface
    appearance across views breaks down."

That is stated as an expectation. This module turns it into a measurement, by
testing whether the parts of the image that are specular are also the parts
where the reconstruction is thin.

Method
------
1. Detect specular pixels per image: bright and unsaturated in HSV. A gilded
   highlight blows out towards white, losing saturation, while the underlying
   gold paint stays saturated.
2. Project the reconstructed surface into that image and measure local
   reconstruction density per pixel block.
3. Correlate the two across all blocks and all images.

A negative correlation is evidence for the paper's claim. Reporting the
correlation (with its sign and strength) is honest either way -- if gilding
turns out not to hurt, that is also a finding, and a more interesting one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap

from .config import Config
from .io_utils import get_logger, list_images, timed
from .render import camera_to_opencv, extrinsic_matrix, undistort_photo

log = get_logger(__name__)


def detect_specular(image_bgr: np.ndarray, v_min: int, s_max: int) -> np.ndarray:
    """Boolean mask of specular (bright, desaturated) pixels."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    return (value >= v_min) & (saturation <= s_max)


def specular_fraction_per_image(
    images_dir: Path, cfg: Config, masks_dir: Path | None = None
) -> dict[str, float]:
    """How much of each photograph is blown out by specular highlights."""
    fractions: dict[str, float] = {}
    for path in list_images(images_dir):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        specular = detect_specular(image, cfg.eval.specular_v_min, cfg.eval.specular_s_max)
        if masks_dir is not None:
            mask_file = masks_dir / f"{path.name}.png"
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Restrict to the object: a bright backdrop is not gilding.
                foreground = mask > 127
                fractions[path.name] = (
                    float(specular[foreground].mean()) if foreground.any() else 0.0
                )
                continue
        fractions[path.name] = float(specular.mean())
    return fractions


def correlate_specularity_with_density(
    reconstruction: pycolmap.Reconstruction,
    points: np.ndarray,
    images_dir: Path,
    cfg: Config,
    block: int = 32,
    masks_dir: Path | None = None,
) -> dict[str, Any]:
    """Correlate per-block specularity against reconstructed point density.

    Works on image blocks rather than pixels because reconstruction density is
    only meaningful over a neighbourhood, and because it keeps the correlation
    from being dominated by pixel-level noise.
    """
    spec_values: list[float] = []
    density_values: list[float] = []
    per_image: list[dict[str, Any]] = []

    for image_id in sorted(reconstruction.reg_image_ids()):
        image = reconstruction.image(image_id)
        camera = reconstruction.camera(image.camera_id)

        photo = cv2.imread(str(images_dir / image.name), cv2.IMREAD_COLOR)
        if photo is None:
            continue
        photo = undistort_photo(photo, camera)
        specular = detect_specular(photo, cfg.eval.specular_v_min, cfg.eval.specular_s_max)

        foreground = None
        if masks_dir is not None:
            mask = cv2.imread(str(masks_dir / f"{image.name}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != specular.shape:
                    mask = cv2.resize(mask, specular.shape[::-1], interpolation=cv2.INTER_NEAREST)
                foreground = mask > 127

        # Project the reconstructed surface into this view.
        K, _ = camera_to_opencv(camera)
        extrinsic = extrinsic_matrix(image)
        cam_points = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        in_front = cam_points[:, 2] > 1e-6
        if not in_front.any():
            continue
        u = K[0, 0] * cam_points[in_front, 0] / cam_points[in_front, 2] + K[0, 2]
        v = K[1, 1] * cam_points[in_front, 1] / cam_points[in_front, 2] + K[1, 2]

        h, w = specular.shape
        keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        density = np.zeros(((h + block - 1) // block, (w + block - 1) // block), dtype=float)
        np.add.at(
            density,
            (v[keep].astype(int) // block, u[keep].astype(int) // block),
            1.0,
        )

        # Block-average the specular mask onto the same grid.
        spec_blocks = _block_mean(specular.astype(float), block)
        fg_blocks = _block_mean(foreground.astype(float), block) if foreground is not None else None

        valid = np.ones_like(density, dtype=bool)
        if fg_blocks is not None:
            # Only blocks that are mostly object: background blocks have no
            # surface to reconstruct and would create a spurious correlation.
            valid &= fg_blocks > 0.5
        else:
            valid &= density > 0

        if not valid.any():
            continue

        spec_values.extend(spec_blocks[valid].tolist())
        density_values.extend(density[valid].tolist())
        per_image.append(
            {
                "image": image.name,
                "specular_fraction": float(spec_blocks[valid].mean()),
                "mean_density": float(density[valid].mean()),
            }
        )

    if len(spec_values) < 10:
        return {"n_blocks": len(spec_values), "correlation": float("nan")}

    spec = np.asarray(spec_values)
    dens = np.asarray(density_values)

    result: dict[str, Any] = {
        "n_blocks": int(spec.size),
        "n_images": len(per_image),
        "block_px": block,
        "mean_specular_fraction": float(spec.mean()),
        "per_image": per_image,
    }

    if spec.std() > 1e-9 and dens.std() > 1e-9:
        result["correlation"] = float(np.corrcoef(spec, dens)[0, 1])
        # Compare the most specular blocks against the least, which is easier
        # to state in a paper than a correlation coefficient.
        high = spec >= np.quantile(spec, 0.9)
        low = spec <= np.quantile(spec, 0.5)
        if high.any() and low.any():
            result["density_specular_blocks"] = float(dens[high].mean())
            result["density_matte_blocks"] = float(dens[low].mean())
            result["density_ratio"] = float(
                dens[high].mean() / max(dens[low].mean(), 1e-9)
            )
    else:
        result["correlation"] = float("nan")
        result["note"] = "no variation in specularity or density to correlate"

    return result


def _block_mean(array: np.ndarray, block: int) -> np.ndarray:
    """Mean-pool a 2D array into ``block`` x ``block`` cells, padding the edges."""
    h, w = array.shape
    ph = (block - h % block) % block
    pw = (block - w % block) % block
    if ph or pw:
        array = np.pad(array, ((0, ph), (0, pw)), mode="edge")
    hh, ww = array.shape
    return array.reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))


def run_specularity_study(cfg: Config) -> dict[str, Any]:
    """Full study: per-image specularity plus the density correlation."""
    import open3d as o3d

    from .metrics import load_reconstruction

    reconstruction = load_reconstruction(cfg.sparse_dir)
    masks_dir = cfg.masks_dir if (cfg.mask.enabled and cfg.masks_dir.is_dir()) else None

    # Prefer the dense cloud: it is what MVS actually recovered, and MVS is
    # what the proposal predicts will fail on gilding.
    dense_ply = cfg.dense_dir / "fused.ply"
    if dense_ply.exists():
        points = np.asarray(o3d.io.read_point_cloud(str(dense_ply)).points)
        source = "dense"
    else:
        points = np.array(
            [reconstruction.point3D(pid).xyz for pid in reconstruction.point3D_ids()],
            dtype=float,
        )
        source = "sparse"
        log.info("no dense cloud yet; running the study on the sparse points")

    stats: dict[str, Any] = {"point_source": source, "n_points": int(points.shape[0])}
    stats["per_image_specular_fraction"] = specular_fraction_per_image(
        cfg.images_dir, cfg, masks_dir
    )

    with timed("correlating specularity with reconstruction density", log):
        stats["correlation_study"] = correlate_specularity_with_density(
            reconstruction, points, cfg.images_dir, cfg, masks_dir=masks_dir
        )

    correlation = stats["correlation_study"].get("correlation", float("nan"))
    if np.isfinite(correlation):
        ratio = stats["correlation_study"].get("density_ratio")
        log.info(
            "specularity vs reconstruction density: r = %+.3f%s",
            correlation,
            f" (specular blocks recover {ratio:.2f}x the points of matte blocks)"
            if ratio else "",
        )
        if correlation < -0.1:
            log.info(
                "negative correlation supports the proposal's prediction that "
                "gilded regions reconstruct poorly"
            )
    return stats
