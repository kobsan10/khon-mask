#!/usr/bin/env python
"""Stage 3 -- import fused.ply back from Colab and verify it.

    python scripts/03_dense_import.py ~/Downloads/fused.ply -s paths.run_id=mask01
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khon_recon.cli import base_parser, resolve
from khon_recon.dense import import_fused
from khon_recon.io_utils import get_logger, update_manifest

log = get_logger("dense_import")


def main() -> int:
    parser = base_parser("Import the fused dense point cloud from Colab.")
    parser.add_argument("ply", type=Path, help="path to the downloaded fused.ply")
    args = parser.parse_args()
    cfg = resolve(args)

    stats = import_fused(cfg, args.ply)
    update_manifest(cfg.run_dir, "dense_import", stats)

    if not stats["agreement"]["ok"]:
        log.error("alignment check FAILED -- do not trust the mesh built from this cloud")
        return 1

    log.info(
        "imported %d dense points -- next: python scripts/04_mesh.py", stats["n_points"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
