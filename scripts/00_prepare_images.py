#!/usr/bin/env python
"""Stage 0 -- ingest images, run capture QC, and generate foreground masks.

Examples
--------
Smoke test on COLMAP's sample data (no capture needed):
    python scripts/00_prepare_images.py --fetch-sample south-building \
        --sample-limit 25 -s paths.subject=sample -s mask.enabled=false

Rehearse the real shoot with a phone video:
    python scripts/00_prepare_images.py --video ~/orbit.mov -s paths.subject=dryrun

The actual mask photographs:
    python scripts/00_prepare_images.py --input ~/khon_photos -s paths.subject=mask01
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khon_recon.capture_qc import run_qc
from khon_recon.cli import base_parser, resolve
from khon_recon.io_utils import get_logger, update_manifest, write_json
from khon_recon.masking import generate_masks, write_mask_previews
from khon_recon.prepare import fetch_sample, frames_from_video, ingest_images

log = get_logger("prepare")


def main() -> int:
    parser = base_parser("Ingest images, QC the capture, and build masks.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="directory of photographs")
    source.add_argument("--video", type=Path, help="video to sample into frames")
    source.add_argument(
        "--fetch-sample", nargs="?", const="south-building",
        help="download a COLMAP sample dataset instead of using your own images",
    )
    parser.add_argument(
        "--sample-limit", type=int, default=25,
        help="keep only this many sample images (0 = all)",
    )
    parser.add_argument("--qc-only", action="store_true", help="re-run QC on already-ingested images")
    parser.add_argument("--skip-qc", action="store_true")
    args = parser.parse_args()
    cfg = resolve(args)

    log.info("subject=%s  run=%s", cfg.paths.subject, cfg.paths.run_id)
    payload: dict = {}

    # ---- ingest ----
    if not args.qc_only:
        if args.video:
            frames_dir = cfg.raw_dir / "frames"
            frames_from_video(args.video, frames_dir, cfg.prepare.video_fps, args.overwrite)
            src = frames_dir
        elif args.fetch_sample:
            src = fetch_sample(args.fetch_sample, cfg.data_root / "samples", args.sample_limit)
        elif args.input:
            src = args.input
        else:
            parser.error("give one of --input, --video or --fetch-sample (or --qc-only)")
        payload["ingest"] = ingest_images(src, cfg, args.overwrite)

    # ---- capture QC ----
    if not args.skip_qc:
        report = run_qc(cfg.images_dir, cfg.qc)
        print("\n" + report.summary() + "\n")
        write_json(cfg.run_dir / "capture_qc.json", report.to_dict())
        payload["qc"] = {
            "n_images": report.n_images,
            "n_blurry": len(report.blurry_images),
            "exposure_drift": report.exposure_drift,
            "needs_masking": report.needs_masking,
            "warnings": report.warnings,
        }
        if report.needs_masking and not cfg.mask.enabled:
            log.warning(
                "QC detected a static background but mask.enabled is false. "
                "SfM will most likely reconstruct the background, not the mask."
            )

    # ---- masks ----
    if cfg.mask.enabled and not args.qc_only:
        payload["masks"] = generate_masks(
            cfg.images_dir, cfg.masks_dir, cfg.mask, args.overwrite
        )
        previews = write_mask_previews(
            cfg.images_dir, cfg.masks_dir, cfg.figures_dir / "mask_previews"
        )
        log.info("wrote %d mask preview(s) for eyeballing", len(previews))

    cfg.save()
    update_manifest(cfg.run_dir, "prepare", payload)
    log.info("stage 0 complete -- next: python scripts/01_sfm.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
