#!/usr/bin/env python3
"""Compare density posteriors with fixed and uncertain atmospheric flux."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from plot_density import (
    BASE_DIR,
    EDGES_KM,
    R_EARTH_KM,
    SAVE_DPI,
    apply_style,
    load_prem_profile,
    prem_mean_densities,
    style_density_axis,
)

BASELINE_COLOR = "#888888"
NUISANCE_COLOR = "#ff5515"
LAYER_NAMES = [
    "Inner core",
    "Outer core",
    "Lower mantle",
    "Middle mantle",
    "Upper mantle",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare five-layer density posteriors."
    )
    parser.add_argument(
        "--nlive",
        type=int,
        choices=[5000, 8000],
        required=True,
        help="Select the run with five flux nuisance parameters.",
    )
    parser.add_argument(
        "--run-dir",
        default="result/tomography_mnest_5unc",
        help="Nuisance run directory, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--baseline-dir",
        default="result/tomography_mnest",
        help="Fixed-flux run directory, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic density axis.",
    )
    return parser.parse_args()


def resolve_path(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def load_density_samples(
    filename: Path, n_dim: int, mean_rho: np.ndarray
) -> np.ndarray:
    data = np.loadtxt(filename, ndmin=2)
    if data.shape[0] < 3 or data.shape[1] != n_dim + 1 or not np.isfinite(data).all():
        raise RuntimeError(
            f"Expected at least 3 finite rows with {n_dim + 1} columns: {filename}"
        )

    # The first five columns are density scale factors in both cases.
    # All rows are retained, marginalizing over nuisance parameters.
    return data[:, :5] * mean_rho[None, :]


def summarize_density(rho: np.ndarray) -> np.ndarray:
    return np.percentile(rho, [16.0, 50.0, 84.0], axis=0)


def write_summary(filename: Path, results):
    filename.parent.mkdir(parents=True, exist_ok=True)

    with filename.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "case",
                "layer",
                "inner_radius_km",
                "outer_radius_km",
                "median_g_cm3",
                "lower_16_g_cm3",
                "upper_84_g_cm3",
                "minus_error_g_cm3",
                "plus_error_g_cm3",
            ]
        )

        for label, summary in results:
            lower, median, upper = summary

            print(f"\n{label}: density [g/cm^3]")
            print(
                f"{'Layer':<15} {'Median':>10} {'-error':>10} "
                f"{'+error':>10} {'Lower':>10} {'Upper':>10}"
            )

            for i, name in enumerate(LAYER_NAMES):
                minus = median[i] - lower[i]
                plus = upper[i] - median[i]

                print(
                    f"{name:<15} {median[i]:10.6f} {minus:10.6f} "
                    f"{plus:10.6f} {lower[i]:10.6f} {upper[i]:10.6f}"
                )
                writer.writerow(
                    [
                        label,
                        name,
                        EDGES_KM[i],
                        EDGES_KM[i + 1],
                        median[i],
                        lower[i],
                        upper[i],
                        minus,
                        plus,
                    ]
                )


def draw_density_intervals(ax, summary, color, offset, label):
    lower, median, upper = summary
    widths = np.diff(EDGES_KM)
    centers = 0.5 * (EDGES_KM[:-1] + EDGES_KM[1:])

    # Rectangles span the physical layer; only the markers are offset.
    for i in range(5):
        ax.add_patch(
            Rectangle(
                (EDGES_KM[i], lower[i]),
                widths[i],
                upper[i] - lower[i],
                facecolor=color,
                edgecolor=color,
                linewidth=0.9,
                alpha=0.18,
                zorder=2,
            )
        )

    ax.errorbar(
        centers + offset * widths,
        median,
        yerr=np.vstack((median - lower, upper - median)),
        fmt="o",
        color=color,
        markersize=5.8,
        elinewidth=1.5,
        capsize=4,
        capthick=1.3,
        label=label,
        zorder=6,
    )


def plot_comparison(
    prem_r,
    prem_rho,
    baseline,
    nuisance,
    nlive,
    log_y,
    filename,
):
    if log_y and (
        np.any(prem_rho <= 0) or np.any(baseline[0] <= 0) or np.any(nuisance[0] <= 0)
    ):
        raise RuntimeError("A logarithmic axis requires positive densities.")

    fig, ax = plt.subplots(figsize=(8.9, 5.65), dpi=120)

    ax.plot(
        prem_r,
        prem_rho,
        color="#1f2937",
        linestyle="--",
        linewidth=1.3,
        label="PREM truth",
        zorder=5,
    )

    draw_density_intervals(
        ax,
        baseline,
        BASELINE_COLOR,
        -0.065,
        "Fixed flux",
    )
    draw_density_intervals(
        ax,
        nuisance,
        NUISANCE_COLOR,
        +0.065,
        f"5 flux nuisances ($N_{{live}}={nlive}$)",
    )

    ymax = max(
        float(np.max(prem_rho)),
        float(np.max(baseline[2])),
        float(np.max(nuisance[2])),
    )

    if log_y:
        ymin = min(
            float(np.min(prem_rho)),
            float(np.min(baseline[0])),
            float(np.min(nuisance[0])),
        )
        ax.set_yscale("log")
        ax.set_ylim(ymin / 1.15, ymax * 1.35)
    else:
        ax.set_ylim(0.0, ymax * 1.15)

    ax.set_xlim(0.0, R_EARTH_KM)
    ax.set_xlabel("Radius from Earth's center (km)")
    ax.set_ylabel(r"Density (g cm$^{-3}$)")
    ax.set_title(
        "5-layer constant-density tomography, 1 year\n"
        "Posterior medians and central 68% intervals",
        pad=12,
    )
    style_density_axis(ax)
    ax.legend(loc="upper right", frameon=True, framealpha=0.90)

    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = resolve_path(args.run_dir)
    baseline_dir = resolve_path(args.baseline_dir)

    baseline_file = baseline_dir / "raw_output" / "post_equal_weights.dat"
    nuisance_file = run_dir / f"raw_output_{args.nlive}" / "post_equal_weights.dat"
    prem_file = BASE_DIR / "data/PREM/EARTH_MODEL_PREM.dat"

    out_dir = run_dir / "plots"
    suffix = "_log" if args.log_y else ""
    plot_file = out_dir / f"density_comparison_{args.nlive}{suffix}.png"
    summary_file = out_dir / f"density_comparison_{args.nlive}.csv"

    prem_r, prem_rho = load_prem_profile(prem_file)
    mean_rho = prem_mean_densities(prem_r, prem_rho, EDGES_KM)

    baseline_rho = load_density_samples(baseline_file, 5, mean_rho)
    nuisance_rho = load_density_samples(nuisance_file, 10, mean_rho)

    baseline = summarize_density(baseline_rho)
    nuisance = summarize_density(nuisance_rho)

    apply_style()
    plot_comparison(
        prem_r,
        prem_rho,
        baseline,
        nuisance,
        args.nlive,
        args.log_y,
        plot_file,
    )

    write_summary(
        summary_file,
        [
            ("fixed_flux", baseline),
            (f"5_flux_nuisances_nlive_{args.nlive}", nuisance),
        ],
    )

    print(f"\n[input] baseline: {baseline_file}")
    print(f"[input] nuisance: {nuisance_file}")
    print(f"[plot] output   : {plot_file}")
    print(f"[summary] output: {summary_file}")


if __name__ == "__main__":
    main()
