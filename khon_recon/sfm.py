"""Sparse reconstruction: feature extraction, matching, incremental mapping.

Wraps pycolmap so every option that affects a reported number is set in code
and recorded in the run manifest, rather than living in a shell command that
nobody can reconstruct three weeks later.

Two things here go beyond a plain COLMAP invocation:

1. **Hold-out split.** Every k-th image is withheld from mapping and
   registered afterwards with the existing poses held fixed. Those images are
   never used to build the geometry, so rendering the mesh from their poses
   and comparing against the real photograph is a genuine novel-view test
   rather than a re-rendering of the training data.

2. **Bundle-adjustment ablation support.** ``minimal`` refinement settings plus
   a re-runnable global BA make it possible to measure what bundle adjustment
   actually buys, which is one of the two ablations the proposal promises.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

from .config import Config
from .io_utils import get_logger, list_images, timed

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Option construction
# --------------------------------------------------------------------------


def _extraction_options(cfg: Config) -> pycolmap.FeatureExtractionOptions:
    options = pycolmap.FeatureExtractionOptions()
    # No CUDA on Apple Silicon: SIFT runs on CPU. Slower, identical results.
    options.use_gpu = cfg.sfm.use_gpu
    options.sift.max_num_features = cfg.sfm.max_num_features
    # Helps on the dark, low-contrast recesses of a mask (eye sockets, the
    # inside of an open mouth) where plain SIFT finds little.
    options.sift.domain_size_pooling = True
    options.sift.estimate_affine_shape = True
    return options


def _reader_options(cfg: Config) -> pycolmap.ImageReaderOptions:
    options = pycolmap.ImageReaderOptions()
    options.camera_model = cfg.sfm.camera_model
    if cfg.mask.enabled and cfg.masks_dir.is_dir():
        # Confines features to the object. Without this a turntable capture
        # reconstructs the background instead of the mask.
        options.mask_path = str(cfg.masks_dir)
        log.info("using foreground masks from %s", cfg.masks_dir)
    return options


def _pipeline_options(cfg: Config, image_names: list[str] | None) -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = cfg.sfm.min_num_matches
    options.ba_global_max_refinements = cfg.sfm.ba_global_max_refinements
    options.ba_local_max_refinements = cfg.sfm.ba_local_max_refinements
    # A single mask is a single object: multiple disconnected models almost
    # always means the capture broke, and silently keeping the largest one
    # hides that. Ask for one model and report if it fragments.
    options.multiple_models = False
    options.extract_colors = True
    options.random_seed = 0  # reproducible runs; ablations must be comparable
    if image_names:
        options.image_names = image_names
    return options


# --------------------------------------------------------------------------
# Hold-out split
# --------------------------------------------------------------------------


def split_holdout(names: list[str], every: int) -> tuple[list[str], list[str]]:
    """Split image names into (train, holdout), taking every k-th as holdout.

    Index 0 is never held out so the mapper always has a well-connected start.
    """
    if every <= 1:
        return list(names), []
    train = [n for i, n in enumerate(names) if i == 0 or i % every != 0]
    holdout = [n for i, n in enumerate(names) if i != 0 and i % every == 0]
    return train, holdout


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def run_sfm(cfg: Config, overwrite: bool = False) -> dict[str, Any]:
    """Run extraction, matching and mapping; return summary statistics."""
    images_dir = cfg.images_dir
    if not images_dir.is_dir():
        raise FileNotFoundError(f"no image directory at {images_dir}")
    all_names = [p.name for p in list_images(images_dir)]
    image_names = all_names[:: max(cfg.sfm.subsample_every, 1)]
    if cfg.sfm.subsample_every > 1:
        log.info(
            "reduced-overlap ablation: using every %d%s image (%d of %d)",
            cfg.sfm.subsample_every,
            {2: "nd", 3: "rd"}.get(cfg.sfm.subsample_every, "th"),
            len(image_names), len(all_names),
        )
    if len(image_names) < 3:
        raise ValueError(f"need at least 3 images, found {len(image_names)}")

    cfg.make_run_dirs()
    database_path = cfg.run_dir / "database.db"
    if overwrite and database_path.exists():
        database_path.unlink()

    stats: dict[str, Any] = {"n_input_images": len(image_names)}

    # ---- features ----
    if database_path.exists():
        log.info("reusing existing database %s", database_path)
    else:
        with timed(f"extracting SIFT features from {len(image_names)} images", log) as t:
            pycolmap.extract_features(
                database_path=database_path,
                image_path=images_dir,
                # Restricting here (rather than filtering later) keeps the
                # database itself free of the dropped images, so matching cost
                # falls with the ablation instead of staying flat.
                image_names=image_names if cfg.sfm.subsample_every > 1 else [],
                camera_mode=(
                    pycolmap.CameraMode.SINGLE
                    if cfg.sfm.single_camera
                    else pycolmap.CameraMode.AUTO
                ),
                reader_options=_reader_options(cfg),
                extraction_options=_extraction_options(cfg),
                device=pycolmap.Device.cpu if not cfg.sfm.use_gpu else pycolmap.Device.auto,
            )
        stats["feature_extraction_seconds"] = t["seconds"]

        # ---- matching ----
        with timed(f"matching features ({cfg.sfm.matcher})", log) as t:
            device = pycolmap.Device.cpu if not cfg.sfm.use_gpu else pycolmap.Device.auto
            if cfg.sfm.matcher == "sequential":
                pycolmap.match_sequential(database_path=database_path, device=device)
            else:
                # Exhaustive is affordable at 60-100 images and more reliable
                # than sequential for a closed orbit, where the last frame
                # must match the first.
                pycolmap.match_exhaustive(database_path=database_path, device=device)
        stats["matching_seconds"] = t["seconds"]

    # ---- hold-out split ----
    train_names, holdout_names = split_holdout(image_names, cfg.sfm.holdout_every)
    stats["n_train_images"] = len(train_names)
    stats["n_holdout_images"] = len(holdout_names)
    if holdout_names:
        log.info(
            "holding out %d of %d images from mapping as a novel-view test set",
            len(holdout_names), len(image_names),
        )
        (cfg.run_dir / "holdout_images.txt").write_text("\n".join(holdout_names))

    # ---- mapping ----
    train_dir = cfg.sparse_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    with timed("incremental mapping (SfM + bundle adjustment)", log) as t:
        reconstructions = pycolmap.incremental_mapping(
            database_path=database_path,
            image_path=images_dir,
            output_path=train_dir,
            options=_pipeline_options(cfg, train_names if holdout_names else None),
        )
    stats["mapping_seconds"] = t["seconds"]

    if not reconstructions:
        raise RuntimeError(
            "COLMAP registered no images. Usual causes: too little overlap "
            "between photos, a textureless or specular subject, or masks that "
            "removed nearly everything. Run scripts/00_prepare_images.py --qc-only "
            "to inspect the capture."
        )
    stats["n_models"] = len(reconstructions)
    if len(reconstructions) > 1:
        log.warning(
            "mapping produced %d disconnected models -- the capture probably "
            "has a gap; using the largest",
            len(reconstructions),
        )

    best_id = max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
    reconstruction = reconstructions[best_id]

    # ---- register held-out views with the geometry held fixed ----
    final_dir = cfg.sparse_dir / "0"
    if holdout_names:
        registered = _register_holdout(
            cfg, database_path, train_dir / str(best_id), final_dir
        )
        stats["n_holdout_registered"] = registered
        reconstruction = pycolmap.Reconstruction(final_dir)
    else:
        final_dir.mkdir(parents=True, exist_ok=True)
        reconstruction.write(final_dir)

    stats.update(reconstruction_stats(reconstruction))
    stats["sparse_dir"] = str(final_dir)
    log.info(
        "registered %d/%d images, %d 3D points, mean reprojection error %.3f px",
        stats["n_registered_images"], len(image_names),
        stats["n_points3D"], stats["mean_reprojection_error_px"],
    )
    reconstruction.export_PLY(str(cfg.sparse_dir / "sparse_points.ply"))
    return stats


def _register_holdout(
    cfg: Config, database_path: Path, input_dir: Path, output_dir: Path
) -> int:
    """Add the held-out images to the model without letting them shape it.

    Uses ``colmap image_registrator`` with ``fix_existing_frames``: the
    held-out cameras get poses, but the 3D structure and the training poses are
    untouched. That is what keeps them a fair test set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    before = pycolmap.Reconstruction(input_dir).num_reg_images()

    command = [
        "colmap", "image_registrator",
        "--database_path", str(database_path),
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
        "--Mapper.fix_existing_frames", "1",
    ]
    with timed("registering held-out views (poses only)", log):
        result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        log.warning(
            "image_registrator failed (%s); continuing without held-out views.\n%s",
            result.returncode, result.stderr[-800:],
        )
        # Fall back to the training model so the pipeline still completes.
        for item in input_dir.iterdir():
            shutil.copy2(item, output_dir / item.name)
        return 0

    after = pycolmap.Reconstruction(output_dir).num_reg_images()
    gained = after - before
    log.info("registered %d held-out view(s)", gained)
    return gained


def refine_with_bundle_adjustment(reconstruction: pycolmap.Reconstruction) -> dict[str, float]:
    """Run a full global bundle adjustment on an existing model.

    Used by the BA ablation: take a minimally-refined reconstruction, measure
    Eq. (1), refine, measure again. Same model, same observations, so the
    difference is attributable to bundle adjustment alone.
    """
    before = reconstruction.compute_mean_reprojection_error()
    options = pycolmap.BundleAdjustmentOptions()
    with timed("global bundle adjustment", log):
        pycolmap.bundle_adjustment(reconstruction, options)
    after = reconstruction.compute_mean_reprojection_error()
    return {
        "mean_reprojection_error_before_px": float(before),
        "mean_reprojection_error_after_px": float(after),
        "improvement_px": float(before - after),
    }


def reconstruction_stats(reconstruction: pycolmap.Reconstruction) -> dict[str, Any]:
    """Summary statistics for a sparse model."""
    return {
        "n_registered_images": int(reconstruction.num_reg_images()),
        "n_points3D": int(reconstruction.num_points3D()),
        "n_observations": int(reconstruction.compute_num_observations()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
        "mean_observations_per_image": float(
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "mean_reprojection_error_px": float(
            reconstruction.compute_mean_reprojection_error()
        ),
    }


def camera_centers(reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    """World-space camera positions of every registered image."""
    centers = [
        reconstruction.image(image_id).projection_center()
        for image_id in reconstruction.reg_image_ids()
    ]
    return np.asarray(centers, dtype=float)
