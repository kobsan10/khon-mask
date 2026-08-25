#!/usr/bin/env python
"""Stage 2 -- package the SfM result for the Colab dense stage.

Dense stereo needs CUDA, which Apple Silicon does not have, so this stage only
builds the hand-off bundle and prints what to do with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khon_recon.cli import base_parser, resolve
from khon_recon.dense import cuda_available, dense_instructions, export_bundle
from khon_recon.io_utils import get_logger, update_manifest

log = get_logger("dense_export")


def main() -> int:
    parser = base_parser("Package images + sparse model for Colab densification.")
    parser.add_argument(
        "--no-masks", action="store_true", help="exclude masks from the bundle"
    )
    args = parser.parse_args()
    cfg = resolve(args)

    if cuda_available():
        log.info(
            "this COLMAP build has CUDA -- you can densify locally instead of "
            "using Colab (see the notebook for the exact commands)"
        )

    stats = export_bundle(cfg, include_masks=not args.no_masks)
    update_manifest(cfg.run_dir, "dense_export", stats)
    print(dense_instructions(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
