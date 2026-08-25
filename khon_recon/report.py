"""Figures and tables for the paper.

Everything here regenerates from the run directories, so a number in the report
always traces back to a run that produced it. Figures are written at IEEE
single-column width as both PDF (for LaTeX) and PNG (for quick viewing).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import get_logger, read_json

log = get_logger(__name__)

# IEEE two-column format: a single column is ~3.5 inches wide.
COLUMN_WIDTH = 3.5
DOUBLE_WIDTH = 7.16


def _plt():
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 200,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path)
        paths.append(path)
    fig.clf()
    return paths


# --------------------------------------------------------------------------
# Individual figures
# --------------------------------------------------------------------------


def figure_reprojection_histogram(errors: dict[str, Any], out_dir: Path) -> list[Path]:
    """Distribution of per-observation reprojection error (Eq. 1)."""
    plt = _plt()
    hist = errors.get("histogram")
    if not hist:
        return []

    counts = np.asarray(hist["counts"], dtype=float)
    edges = np.asarray(hist["edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.2))
    ax.bar(centers, counts, width=np.diff(edges), color="#3b6ea5", edgecolor="none")
    ax.axvline(errors["mean_px"], color="#c1442e", lw=1.2,
               label=f"mean {errors['mean_px']:.3f} px")
    ax.axvline(errors["median_px"], color="#e08b3c", lw=1.2, ls="--",
               label=f"median {errors['median_px']:.3f} px")
    ax.set_xlabel("reprojection error (px)")
    ax.set_ylabel("observations")
    ax.legend(frameon=False)
    return _save(fig, out_dir, "reprojection_error")


def figure_camera_coverage(coverage: dict[str, Any], out_dir: Path) -> list[Path]:
    """Where the cameras were, in azimuth and elevation around the subject."""
    plt = _plt()
    if "azimuth_deg" not in coverage:
        return []

    azimuth = np.asarray(coverage["azimuth_deg"])
    elevation = np.asarray(coverage["elevation_deg"])

    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_WIDTH, 2.4),
        gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.45},
    )

    ax = axes[0]
    ax.remove()
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.scatter(np.radians(azimuth), np.ones_like(azimuth), s=14, c="#3b6ea5")
    ax.set_yticklabels([])
    ax.set_title("azimuth coverage")

    ax2 = axes[1]
    ax2.scatter(azimuth, elevation, s=14, c="#3b6ea5")
    ax2.set_xlabel("azimuth (deg)")
    ax2.set_ylabel("elevation (deg)")
    ax2.set_title(
        f"largest gap {coverage['largest_azimuth_gap_deg']:.0f} deg, "
        f"elevation span {coverage['elevation_span_deg']:.0f} deg"
    )
    return _save(fig, out_dir, "camera_coverage")


def figure_track_lengths(tracks: dict[str, Any], out_dir: Path) -> list[Path]:
    """How many views agree on each 3D point."""
    plt = _plt()
    hist = tracks.get("histogram")
    if not hist:
        return []
    counts = np.asarray(hist["counts"], dtype=float)
    edges = np.asarray(hist["edges"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.0))
    ax.bar(centers, counts, width=np.diff(edges), color="#4c8055", edgecolor="none")
    ax.set_xlabel("track length (views per 3D point)")
    ax.set_ylabel("points")
    ax.set_title(f"mean {tracks['mean_track_length']:.2f} views")
    return _save(fig, out_dir, "track_lengths")


def figure_specularity(study: dict[str, Any], out_dir: Path) -> list[Path]:
    """Specularity against reconstruction density -- the paper's key prediction."""
    plt = _plt()
    per_image = study.get("correlation_study", {}).get("per_image")
    if not per_image:
        return []

    spec = np.array([r["specular_fraction"] for r in per_image])
    dens = np.array([r["mean_density"] for r in per_image])

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.4))
    ax.scatter(spec * 100, dens, s=16, c="#8a4b9c", alpha=0.75)
    if spec.size > 2 and spec.std() > 1e-9:
        slope, intercept = np.polyfit(spec * 100, dens, 1)
        xs = np.linspace((spec * 100).min(), (spec * 100).max(), 50)
        ax.plot(xs, slope * xs + intercept, color="#c1442e", lw=1.2)
    correlation = study["correlation_study"].get("correlation", float("nan"))
    ax.set_xlabel("specular area of the mask (%)")
    ax.set_ylabel("reconstructed points per block")
    ax.set_title(f"r = {correlation:+.3f}")
    return _save(fig, out_dir, "specularity_vs_density")


def figure_ablation(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Bar comparison across ablation runs."""
    plt = _plt()
    if not rows:
        return []

    labels = [r["run"] for r in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_WIDTH, 2.3))
    specs = [
        ("registered_images", "registered images", "#3b6ea5"),
        ("mean_reprojection_error_px", "mean reproj. error (px)", "#c1442e"),
        ("holdout_psnr", "held-out PSNR (dB)", "#4c8055"),
    ]
    for ax, (key, title, color) in zip(axes, specs):
        values = [r.get(key, np.nan) for r in rows]
        ax.bar(x, values, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
    return _save(fig, out_dir, "ablation_comparison")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write rows to CSV, unioning the keys across rows."""
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def latex_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], caption: str,
                label: str) -> str:
    """Render an IEEE-style booktabs table, ready to paste into the paper."""
    header = " & ".join(title for _, title in columns) + r" \\"
    body_lines = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append("--" if not np.isfinite(value) else f"{value:.3f}")
            else:
                cells.append(str(value))
        body_lines.append(" & ".join(cells) + r" \\")

    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{" + caption + "}",
            r"\label{tab:" + label + "}",
            r"\centering",
            r"\begin{tabular}{l" + "r" * (len(columns) - 1) + "}",
            r"\hline",
            header,
            r"\hline",
            *body_lines,
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def pipeline_stage_table() -> str:
    """The pipeline-stages table already drafted in the proposal."""
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{Reconstruction pipeline stages and tools}",
            r"\label{tab:pipeline}",
            r"\centering",
            r"\begin{tabular}{llll}",
            r"\hline",
            r"Stage & Method & Output & Tool \\",
            r"\hline",
            r"Sparse & Feature matching, incremental SfM, & Camera poses + & COLMAP \\",
            r"       & bundle adjustment & sparse point cloud & \\",
            r"Dense  & Multi-view stereo & Dense point cloud & COLMAP (CUDA) \\",
            r"Surface & Poisson surface reconstruction & Textured 3D mesh & Open3D \\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def build_report(run_dir: Path, figures_dir: Path | None = None) -> dict[str, Any]:
    """Regenerate every figure and table available for one run."""
    run_dir = Path(run_dir)
    figures_dir = figures_dir or (run_dir / "figures")
    produced: dict[str, Any] = {"figures": [], "tables": []}

    def load(name: str) -> dict[str, Any] | None:
        path = run_dir / name
        return read_json(path) if path.exists() else None

    evaluation = load("evaluation.json")
    if evaluation:
        if "reprojection" in evaluation:
            produced["figures"] += [
                str(p) for p in figure_reprojection_histogram(
                    evaluation["reprojection"], figures_dir
                )
            ]
        if "coverage" in evaluation:
            produced["figures"] += [
                str(p) for p in figure_camera_coverage(evaluation["coverage"], figures_dir)
            ]
        if "tracks" in evaluation:
            produced["figures"] += [
                str(p) for p in figure_track_lengths(evaluation["tracks"], figures_dir)
            ]
        if "specularity" in evaluation:
            produced["figures"] += [
                str(p) for p in figure_specularity(evaluation["specularity"], figures_dir)
            ]

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "pipeline_stages.tex").write_text(pipeline_stage_table())
    produced["tables"].append(str(tables_dir / "pipeline_stages.tex"))

    if evaluation:
        summary_rows = [_summary_row(run_dir.name, evaluation)]
        write_csv(summary_rows, tables_dir / "summary.csv")
        (tables_dir / "summary.tex").write_text(
            latex_table(
                summary_rows,
                [
                    ("run", "Run"),
                    ("registered_images", "Reg. images"),
                    ("mean_reprojection_error_px", "Eq. (1) error (px)"),
                    ("dense_points", "Dense points"),
                    ("holdout_psnr", "Held-out PSNR (dB)"),
                    ("holdout_ssim", "Held-out SSIM"),
                ],
                caption="Reconstruction quality summary.",
                label="summary",
            )
        )
        produced["tables"] += [
            str(tables_dir / "summary.csv"), str(tables_dir / "summary.tex")
        ]

    log.info(
        "wrote %d figure file(s) and %d table(s) into %s",
        len(produced["figures"]), len(produced["tables"]), run_dir,
    )
    return produced


def _summary_row(run_name: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    reprojection = evaluation.get("reprojection", {})
    views = evaluation.get("views", {})
    holdout = views.get("holdout", {}) if isinstance(views, dict) else {}
    return {
        "run": run_name,
        "registered_images": evaluation.get("sfm", {}).get("n_registered_images", ""),
        "mean_reprojection_error_px": reprojection.get("mean_px", float("nan")),
        "dense_points": evaluation.get("density", {}).get("n_points", ""),
        "n_holes": evaluation.get("completeness", {}).get("n_holes", ""),
        "holdout_psnr": holdout.get("psnr_mean", float("nan")),
        "holdout_ssim": holdout.get("ssim_mean", float("nan")),
    }
