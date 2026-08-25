"""Ablation studies.

The proposal commits to two comparisons:

    "we plan to compare the full pipeline against a reduced-overlap capture
     (fewer photos, less angular coverage) [...] and against a version without
     bundle adjustment refinement"

Both are implemented here, plus a masking ablation that evidences the
turntable failure mode.

An honest note on the second one, which the report should repeat: COLMAP cannot
run with bundle adjustment *removed* -- incremental SfM triangulates against a
pose graph that BA maintains, and without any refinement the reconstruction
collapses rather than merely degrading. What is actually compared is therefore
*minimal versus full refinement*. To isolate BA's contribution exactly, a
second measurement takes one minimally-refined model and runs a full global
bundle adjustment over it, so the only thing that changes is the refinement.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import Config
from .io_utils import get_logger, write_json

log = get_logger(__name__)


# name -> dotted config overrides
ABLATIONS: dict[str, dict[str, Any]] = {
    "full": {},
    "overlap_half": {"sfm.subsample_every": 2},
    "overlap_third": {"sfm.subsample_every": 3},
    "minimal_ba": {
        "sfm.ba_global_max_refinements": 1,
        "sfm.ba_local_max_refinements": 1,
    },
    "no_mask": {"mask.enabled": False},
}

ABLATION_DESCRIPTIONS = {
    "full": "all images, full bundle adjustment, masks on",
    "overlap_half": "every 2nd image -- half the angular coverage",
    "overlap_third": "every 3rd image -- a third of the angular coverage",
    "minimal_ba": "minimal BA refinement instead of full",
    "no_mask": "no foreground masks (tests the turntable failure mode)",
}


def _apply(cfg: Config, overrides: dict[str, Any], run_id: str) -> Config:
    """Clone a config, apply dotted overrides, and point it at a new run dir."""
    new = copy.deepcopy(cfg)
    new.paths = replace(new.paths, run_id=run_id)
    for dotted, value in overrides.items():
        section, _, field = dotted.partition(".")
        target = getattr(new, section)
        if not hasattr(target, field):
            raise ValueError(f"unknown ablation override: {dotted}")
        setattr(target, field, value)
    return new


def run_ablation(
    base_cfg: Config, name: str, overrides: dict[str, Any], overwrite: bool = False
) -> dict[str, Any]:
    """Run one ablation variant through SfM and collect its metrics."""
    from .metrics import (
        camera_coverage,
        load_reconstruction,
        reprojection_errors,
        track_statistics,
    )
    from .sfm import run_sfm

    run_id = f"{base_cfg.paths.run_id}_{name}"
    cfg = _apply(base_cfg, overrides, run_id)
    cfg.make_run_dirs()
    cfg.save()

    log.info("=" * 64)
    log.info("ablation %-14s : %s", name, ABLATION_DESCRIPTIONS.get(name, ""))
    log.info("=" * 64)

    result: dict[str, Any] = {
        "run": name,
        "description": ABLATION_DESCRIPTIONS.get(name, ""),
        "run_id": run_id,
        "overrides": overrides,
    }

    try:
        sfm_stats = run_sfm(cfg, overwrite=overwrite)
    except Exception as exc:  # a variant may legitimately fail to reconstruct
        log.error("ablation %s failed: %s", name, exc)
        result["failed"] = True
        result["error"] = str(exc)
        return result

    reconstruction = load_reconstruction(cfg.sparse_dir)
    errors = reprojection_errors(reconstruction)
    coverage = camera_coverage(reconstruction)
    tracks = track_statistics(reconstruction)

    result.update(
        {
            "input_images": sfm_stats["n_input_images"],
            "registered_images": sfm_stats["n_registered_images"],
            "registration_rate": (
                sfm_stats["n_registered_images"] / max(sfm_stats["n_input_images"], 1)
            ),
            "points3D": sfm_stats["n_points3D"],
            "observations": sfm_stats["n_observations"],
            "mean_reprojection_error_px": errors["mean_px"],
            "median_reprojection_error_px": errors["median_px"],
            "mean_track_length": tracks.get("mean_track_length", float("nan")),
            "largest_azimuth_gap_deg": coverage.get("largest_azimuth_gap_deg", float("nan")),
            "mapping_seconds": sfm_stats.get("mapping_seconds", float("nan")),
        }
    )
    write_json(cfg.run_dir / "ablation_result.json", result)
    return result


def bundle_adjustment_isolation(cfg: Config, run_id: str | None = None) -> dict[str, Any]:
    """Measure bundle adjustment's effect while changing nothing else.

    Takes the minimally-refined model and runs a full global BA over it. Same
    images, same matches, same tracks -- so the change in Eq. (1) is
    attributable to refinement alone, which the minimal-vs-full comparison
    cannot claim on its own (there, the reconstructions differ).
    """
    from .metrics import load_reconstruction, reprojection_errors
    from .sfm import refine_with_bundle_adjustment

    run_id = run_id or f"{cfg.paths.run_id}_minimal_ba"
    sparse_dir = cfg.data_root / "runs" / run_id / "sparse"
    if not sparse_dir.exists():
        raise FileNotFoundError(
            f"no minimal-BA model at {sparse_dir}; run that ablation first"
        )

    reconstruction = load_reconstruction(sparse_dir)
    before = reprojection_errors(reconstruction)
    summary = refine_with_bundle_adjustment(reconstruction)
    after = reprojection_errors(reconstruction)

    result = {
        "run": "ba_isolation",
        "description": "one model, measured before and after a full global BA",
        "eq1_before_px": before["mean_px"],
        "eq1_after_px": after["mean_px"],
        "eq1_improvement_px": before["mean_px"] - after["mean_px"],
        "eq1_improvement_percent": (
            100 * (before["mean_px"] - after["mean_px"]) / max(before["mean_px"], 1e-9)
        ),
        **summary,
    }
    log.info(
        "bundle adjustment on a fixed model: Eq. (1) %.4f -> %.4f px (%.1f%% better)",
        result["eq1_before_px"], result["eq1_after_px"],
        result["eq1_improvement_percent"],
    )

    # Keep the refined model so the difference is inspectable later.
    refined_dir = cfg.data_root / "runs" / run_id / "sparse_refined"
    refined_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(refined_dir)
    return result


def run_all(
    cfg: Config, names: list[str] | None = None, overwrite: bool = False
) -> list[dict[str, Any]]:
    """Run the requested ablations and return one row per variant."""
    selected = names or list(ABLATIONS)
    unknown = set(selected) - set(ABLATIONS)
    if unknown:
        raise ValueError(f"unknown ablation(s) {sorted(unknown)}; choose from {list(ABLATIONS)}")

    rows = [run_ablation(cfg, name, ABLATIONS[name], overwrite) for name in selected]

    # Sanity check the headline claim rather than assuming it.
    by_name = {r["run"]: r for r in rows if not r.get("failed")}
    if "full" in by_name and "overlap_third" in by_name:
        full_points = by_name["full"]["points3D"]
        third_points = by_name["overlap_third"]["points3D"]
        if third_points > full_points:
            log.warning(
                "the reduced-overlap run reconstructed MORE points (%d) than the "
                "full run (%d). That inverts the expected result -- check the "
                "capture rather than reporting it at face value.",
                third_points, full_points,
            )
    return rows
