#!/usr/bin/env python
"""Stage 1 -- sparse reconstruction: features, matching, incremental SfM.

    python scripts/01_sfm.py -s paths.subject=mask01 -s paths.run_id=mask01

Writes camera poses and a sparse point cloud to
``data/runs/<run_id>/sparse/0`` plus a stats JSON carrying the mean
reprojection error of Eq. (1).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khon_recon.cli import base_parser, resolve
from khon_recon.io_utils import get_logger, update_manifest, write_json
from khon_recon.sfm import run_sfm

log = get_logger("sfm")


def main() -> int:
    parser = base_parser("Run structure from motion with COLMAP.")
    args = parser.parse_args()
    cfg = resolve(args)

    stats = run_sfm(cfg, overwrite=args.overwrite)
    write_json(cfg.run_dir / "sfm_stats.json", stats)
    cfg.save()
    update_manifest(cfg.run_dir, "sfm", stats)

    registered = stats["n_registered_images"]
    total = stats["n_input_images"]
    if registered < 0.8 * total:
        log.warning(
            "only %d/%d images registered. Check capture_qc.json for overlap gaps, "
            "and the mask previews if masking is on.",
            registered, total,
        )
    log.info("stage 1 complete -- next: python scripts/02_dense_export.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
