#!/usr/bin/env python
"""Stage 8 -- regenerate every figure and table for the paper.

    python scripts/08_report.py -s paths.run_id=mask01
    python scripts/08_report.py --all-runs

Reads only the JSON already written by earlier stages, so it is cheap to re-run
and always reflects what is on disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from khon_recon.cli import base_parser, resolve
from khon_recon.io_utils import get_logger, read_json, write_json
from khon_recon.report import _summary_row, build_report, latex_table, write_csv

log = get_logger("report")


def main() -> int:
    parser = base_parser("Regenerate the paper's figures and tables.")
    parser.add_argument(
        "--all-runs", action="store_true",
        help="also build a cross-run summary table over every run directory",
    )
    args = parser.parse_args()
    cfg = resolve(args)

    produced = build_report(cfg.run_dir, cfg.figures_dir)
    for path in produced["figures"]:
        log.info("figure: %s", path)
    for path in produced["tables"]:
        log.info("table : %s", path)

    if args.all_runs:
        runs_root = cfg.data_root / "runs"
        rows = []
        for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
            evaluation_path = run_dir / "evaluation.json"
            if evaluation_path.exists():
                rows.append(_summary_row(run_dir.name, read_json(evaluation_path)))
        if rows:
            tables_dir = cfg.run_dir / "tables"
            write_csv(rows, tables_dir / "all_runs.csv")
            (tables_dir / "all_runs.tex").write_text(
                latex_table(
                    rows,
                    [
                        ("run", "Run"),
                        ("registered_images", "Reg. images"),
                        ("mean_reprojection_error_px", "Eq. (1) error (px)"),
                        ("dense_points", "Dense points"),
                        ("holdout_psnr", "Held-out PSNR (dB)"),
                    ],
                    caption="Reconstruction quality across all runs.",
                    label="allruns",
                )
            )
            write_json(cfg.run_dir / "all_runs.json", rows)
            log.info("cross-run summary over %d run(s)", len(rows))

    log.info("stage 8 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
