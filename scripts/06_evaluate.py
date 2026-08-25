#!/usr/bin/env python
"""Stage 6 -- evaluate the reconstruction.

Produces every number the proposal's evaluation plan promises:

  * mean reprojection error, Eq. (1), verified against COLMAP's own computation
  * point-cloud density and mesh completeness (where the holes are)
  * novel-view PSNR/SSIM against real photographs, on held-out views
  * the specularity study, testing the paper's gilding prediction

    python scripts/06_evaluate.py -s paths.run_id=mask01
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import open3d as o3d

from khon_recon.cli import base_parser, resolve
from khon_recon.compare import evaluate_views
from khon_recon.io_utils import get_logger, update_manifest, write_json
from khon_recon.metrics import (
    camera_coverage,
    coverage_warnings,
    load_reconstruction,
    mesh_completeness,
    point_cloud_density,
    reprojection_errors,
    track_statistics,
    verify_against_builtin,
)
from khon_recon.report import build_report
from khon_recon.sfm import reconstruction_stats
from khon_recon.specularity import run_specularity_study

log = get_logger("evaluate")


def main() -> int:
    parser = base_parser("Evaluate the reconstruction.")
    parser.add_argument("--skip-views", action="store_true",
                        help="skip the novel-view render comparison (the slow part)")
    parser.add_argument("--skip-specularity", action="store_true")
    parser.add_argument("--render-scale", type=float, default=0.5,
                        help="render resolution as a fraction of the source photos")
    args = parser.parse_args()
    cfg = resolve(args)

    reconstruction = load_reconstruction(cfg.sparse_dir)
    evaluation: dict = {"run_id": cfg.paths.run_id, "subject": cfg.paths.subject}

    # ---- sparse model, Eq. (1) ----
    evaluation["sfm"] = reconstruction_stats(reconstruction)
    evaluation["reprojection"] = reprojection_errors(reconstruction)
    evaluation["reprojection_verification"] = verify_against_builtin(reconstruction)
    evaluation["tracks"] = track_statistics(reconstruction)

    log.info(
        "Eq. (1) mean reprojection error: %.4f px over %d observations",
        evaluation["reprojection"]["mean_px"],
        evaluation["reprojection"]["n_observations"],
    )

    # ---- capture coverage ----
    coverage = camera_coverage(reconstruction)
    evaluation["coverage"] = coverage
    for warning in coverage_warnings(coverage):
        log.warning("coverage: %s", warning)

    # ---- dense cloud ----
    dense_ply = cfg.dense_dir / "fused.ply"
    if dense_ply.exists():
        pcd = o3d.io.read_point_cloud(str(dense_ply))
        evaluation["density"] = point_cloud_density(pcd, cfg.eval.density_sample)
        log.info(
            "dense cloud: %d points, median spacing %.5g",
            evaluation["density"]["n_points"],
            evaluation["density"]["median_spacing"],
        )
    else:
        log.warning("no dense cloud at %s; density metrics skipped", dense_ply)

    # ---- mesh ----
    mesh_path = cfg.mesh_dir / "mesh_textured.ply"
    if not mesh_path.exists():
        mesh_path = cfg.mesh_dir / "mesh.ply"
    mesh = None
    if mesh_path.exists():
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        evaluation["completeness"] = mesh_completeness(mesh)
        log.info(
            "mesh completeness: %d holes, %d boundary edges, watertight=%s",
            evaluation["completeness"]["n_holes"],
            evaluation["completeness"]["n_boundary_edges"],
            evaluation["completeness"]["is_watertight"],
        )
    else:
        log.warning("no mesh found; completeness and view metrics skipped")

    # ---- novel-view comparison ----
    holdout_file = cfg.run_dir / "holdout_images.txt"
    holdout = holdout_file.read_text().split() if holdout_file.exists() else []
    if mesh is not None and not args.skip_views:
        evaluation["views"] = evaluate_views(
            mesh, reconstruction, cfg.images_dir, cfg.eval_dir / "views",
            holdout, scale=args.render_scale, use_mask=cfg.eval.masked_compare,
        )
        held = evaluation["views"].get("holdout", {})
        if held:
            log.info(
                "held-out novel views: PSNR %.2f dB, SSIM %.3f over %d view(s)",
                held["psnr_mean"], held["ssim_mean"], held["n"],
            )
        elif not holdout:
            log.info("no held-out views (sfm.holdout_every=0); view scores are on training views")

    # ---- specularity study ----
    if not args.skip_specularity:
        evaluation["specularity"] = run_specularity_study(cfg)

    write_json(cfg.run_dir / "evaluation.json", evaluation)
    update_manifest(cfg.run_dir, "evaluate", {
        "mean_reprojection_error_px": evaluation["reprojection"]["mean_px"],
        "n_registered_images": evaluation["sfm"]["n_registered_images"],
    })

    build_report(cfg.run_dir, cfg.figures_dir)
    log.info("stage 6 complete -- evaluation.json and figures/ are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
