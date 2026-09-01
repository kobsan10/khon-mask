#!/usr/bin/env python
"""Stage 5 -- recover colour by projecting the photographs onto the mesh.

    python scripts/05_texture.py -s paths.run_id=mask01                # vertex colours
    python scripts/05_texture.py -s paths.run_id=mask01 -s texture.mode=uv   # UV atlas
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khon_recon.cli import base_parser, resolve
from khon_recon.io_utils import get_logger, update_manifest, write_json
from khon_recon.mesh import export_upright
from khon_recon.texture import run_texturing
from khon_recon.previews import write_stage_preview

log = get_logger("texture")


def main() -> int:
    parser = base_parser("Texture the reconstructed mesh from the source photographs.")
    args = parser.parse_args()
    cfg = resolve(args)

    stats = run_texturing(cfg)
    write_json(cfg.run_dir / "texture_stats.json", stats)
    update_manifest(cfg.run_dir, "texture", stats)

    unseen = stats.get("unseen_fraction")
    if unseen and unseen > 0.05:
        log.warning(
            "%.1f%% of the surface was never seen by any camera -- those regions "
            "have no real colour. Add viewpoints covering them.",
            100 * unseen,
        )
    # A viewer-frame copy alongside the canonical mesh. COLMAP's world frame has
    # +Y pointing down, so every mesh viewer opens the model upside down and
    # facing away -- which reads as a failed reconstruction. The canonical mesh
    # must stay in the reconstruction frame for stage 6 to render from the
    # estimated poses, hence a copy rather than a rewrite.
    source = Path(stats.get("output", cfg.mesh_dir / "mesh_textured.ply"))
    if source.suffix == ".ply":
        try:
            upright = export_upright(
                cfg, source, source.with_name(f"{source.stem}_upright.ply")
            )
            update_manifest(cfg.run_dir, "texture_upright", upright)
        except Exception as exc:  # noqa: BLE001 -- a viewing copy must not fail the stage
            log.warning("could not write the upright viewing copy: %s", exc)

    write_stage_preview(cfg, "texture")
    log.info("stage 5 complete -- next: python scripts/06_evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
