#!/usr/bin/env python
"""Stage 7 -- run the ablation studies the proposal promises.

    python scripts/07_ablations.py -s paths.subject=mask01 -s paths.run_id=mask01
    python scripts/07_ablations.py --only full overlap_third

Each variant is a full SfM re-run into its own run directory, so the results
are comparable and independently inspectable. Variants differ only by config.

Dense-dependent metrics are not included here: each variant would need its own
Colab densification run. The SfM-level metrics are where the reduced-overlap
and bundle-adjustment effects are most directly visible anyway.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khon_recon.ablations import ABLATIONS, bundle_adjustment_isolation, run_all
from khon_recon.cli import base_parser, resolve
from khon_recon.io_utils import get_logger, update_manifest, write_json
from khon_recon.report import figure_ablation, latex_table, write_csv

log = get_logger("ablations")


def main() -> int:
    parser = base_parser("Run the reduced-overlap and bundle-adjustment ablations.")
    parser.add_argument(
        "--only", nargs="+", choices=list(ABLATIONS), default=None,
        help="run only these variants",
    )
    parser.add_argument(
        "--skip-ba-isolation", action="store_true",
        help="skip the fixed-model bundle adjustment measurement",
    )
    args = parser.parse_args()
    cfg = resolve(args)

    rows = run_all(cfg, args.only, overwrite=args.overwrite)

    if not args.skip_ba_isolation:
        names = args.only or list(ABLATIONS)
        if "minimal_ba" in names:
            try:
                rows.append(bundle_adjustment_isolation(cfg))
            except Exception as exc:
                log.warning("bundle adjustment isolation skipped: %s", exc)

    write_json(cfg.run_dir / "ablations.json", rows)

    tables_dir = cfg.run_dir / "tables"
    write_csv(rows, tables_dir / "ablations.csv")
    comparable = [r for r in rows if not r.get("failed") and "registered_images" in r]
    (tables_dir / "ablations.tex").write_text(
        latex_table(
            comparable,
            [
                ("run", "Variant"),
                ("registered_images", "Reg. images"),
                ("points3D", "3D points"),
                ("mean_reprojection_error_px", "Eq. (1) error (px)"),
                ("mean_track_length", "Track length"),
            ],
            caption=(
                "Ablation study. Reduced-overlap variants use every 2nd and 3rd "
                "image; the minimal-BA variant limits bundle adjustment "
                "refinement rather than removing it."
            ),
            label="ablations",
        )
    )
    figure_ablation(comparable, cfg.figures_dir)

    print("\n" + "=" * 78)
    print(f"{'variant':<16}{'reg. img':>10}{'3D points':>12}{'Eq.(1) px':>12}{'track len':>11}")
    print("-" * 78)
    for row in comparable:
        print(
            f"{row['run']:<16}{row['registered_images']:>10}{row['points3D']:>12}"
            f"{row['mean_reprojection_error_px']:>12.4f}{row['mean_track_length']:>11.2f}"
        )
    print("=" * 78 + "\n")

    update_manifest(cfg.run_dir, "ablations", {"n_variants": len(rows)})
    log.info("stage 7 complete -- tables/ablations.tex is ready for the paper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
