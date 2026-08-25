#!/usr/bin/env python
"""Stage 4 -- Poisson surface reconstruction from the dense point cloud.

    python scripts/04_mesh.py -s paths.run_id=mask01

``--input`` meshes an arbitrary point cloud, which is useful for exercising
this stage on the sparse cloud before any dense result exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khon_recon.cli import base_parser, resolve
from khon_recon.io_utils import get_logger, update_manifest, write_json
from khon_recon.mesh import run_meshing

log = get_logger("mesh")


def main() -> int:
    parser = base_parser("Poisson surface reconstruction.")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="point cloud to mesh (default: <run>/dense/fused.ply)",
    )
    args = parser.parse_args()
    cfg = resolve(args)

    stats = run_meshing(cfg, args.input)
    write_json(cfg.run_dir / "mesh_stats.json", stats)
    update_manifest(cfg.run_dir, "mesh", stats)

    holes = stats["completeness"]["n_holes"]
    if holes:
        log.info(
            "%d hole(s) remain -- expected in recessed regions (eye sockets, "
            "open mouth) that the cameras could not see into",
            holes,
        )
    log.info("stage 4 complete -- next: python scripts/05_texture.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
