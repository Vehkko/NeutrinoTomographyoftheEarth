#!/usr/bin/env python3
"""Plot the five-layer density posterior with flux nuisances marginalized."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

BASE_DIR = Path(__file__).resolve().parents[2]
R_EARTH_KM = 6371.0
EDGES_KM = np.array([0.0, 1221.0, 3480.0, 4811.0, 5700.0, 6371.0])

SAVE_DPI = 300
COLOR = "#355C7D"
PREM_COLOR = "#1f2937"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot the five-layer density posterior."
    )
    parser.add_argument(
        "--run-dir",
        default="result/tomography_mnest_5unc",
        help="Run directory, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--nlive",
        type=int,
        choices=[5000, 8000],
        required=True,
        help="Read raw_output_5000 or raw_output_8000.",
    )
    return parser.parse_args()


def resolve_run_dir(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def load_posterior(filename: Path) -> np.ndarray:
    # MultiNest equal-weight output:
    # q1,...,q5,z1,...,z5,loglike
    data = np.loadtxt(filename, ndmin=2)
    if data.shape[0] < 3 or data.shape[1] != 11 or not np.isfinite(data).all():
        raise RuntimeError(
            "Expected at least 3 finite rows with 11 columns "
            f"(q1,...,q5,z1,...,z5,loglike): {filename}"
        )

    # Keeping every posterior row marginalizes over the flux parameters.
    return data[:, :5]


def load_prem_profile(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(filename)
    radius = data[:, 0]
    density = data[:, 1]

    if np.max(radius) <= 2.0:
        radius = radius * R_EARTH_KM

    # Stable ordering preserves the two sides of PREM discontinuities.
    order = np.argsort(radius, kind="stable")
    return radius[order], density[order]


def prem_mean_densities(
    radius_km: np.ndarray,
    density_gcm3: np.ndarray,
    edges_km: np.ndarray,
) -> np.ndarray:
    """Volume-weighted layer means of the piecewise-linear PREM table."""
    radius = radius_km / R_EARTH_KM
    edges = edges_km / R_EARTH_KM
    integrals = np.zeros(len(edges) - 1)

    for i in range(len(radius) - 1):
        ra, rb = radius[i], radius[i + 1]
        if rb <= ra:
            continue

        slope = (density_gcm3[i + 1] - density_gcm3[i]) / (rb - ra)
        intercept = density_gcm3[i] - slope * ra

        for layer in range(len(integrals)):
            lo = max(ra, edges[layer])
            hi = min(rb, edges[layer + 1])
            if hi <= lo:
                continue

            integrals[layer] += (
                slope * (hi**4 - lo**4) / 4.0 + intercept * (hi**3 - lo**3) / 3.0
            )

    volume_weights = (edges[1:] ** 3 - edges[:-1] ** 3) / 3.0
    return integrals / volume_weights


def apply_style():
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "figure.dpi": 120,
            "savefig.dpi": SAVE_DPI,
            "font.size": 10.5,
            "axes.labelsize": 11.2,
            "axes.titlesize": 13.0,
            "legend.fontsize": 9.4,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "mathtext.fontset": "dejavuserif",
        }
    )


def add_layer_background(ax, edges: np.ndarray):
    colors = ["#7f1d1d", "#c2410c", "#ea580c", "#f59e0b", "#fbbf24"]
    for i, color in enumerate(colors):
        ax.axvspan(
            edges[i],
            edges[i + 1],
            color=color,
            alpha=0.04,
            lw=0,
            zorder=0,
        )

    for radius in edges[1:-1]:
        ax.axvline(radius, color="#6b2f1a", lw=0.8, alpha=0.50, zorder=1)


def style_density_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.20)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.11)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.45)
    ax.spines["bottom"].set_alpha(0.45)


def plot_density_profile(
    prem_r: np.ndarray,
    prem_rho: np.ndarray,
    q: np.ndarray,
    mean_rho: np.ndarray,
    outname: Path,
):
    rho = q * mean_rho[None, :]
    rho16, rho50, rho84 = np.percentile(rho, [16.0, 50.0, 84.0], axis=0)

    fig, ax = plt.subplots(figsize=(8.9, 5.65), dpi=120)
    add_layer_background(ax, EDGES_KM)

    ax.plot(
        prem_r,
        prem_rho,
        color=PREM_COLOR,
        lw=1.15,
        alpha=0.95,
        zorder=8,
    )

    for i in range(5):
        left, right = EDGES_KM[i], EDGES_KM[i + 1]
        width = right - left
        x0 = left + 0.08 * width
        x1 = right - 0.08 * width

        ax.add_patch(
            Rectangle(
                (x0, rho16[i]),
                x1 - x0,
                rho84[i] - rho16[i],
                facecolor=COLOR,
                edgecolor=COLOR,
                linewidth=0.9,
                alpha=0.22,
                zorder=3,
            )
        )
        ax.scatter(
            0.5 * (left + right),
            rho50[i],
            s=42,
            color=COLOR,
            edgecolor="white",
            linewidth=0.85,
            zorder=14,
        )

    ax.set_xlabel("Radius from Earth's center (km)")
    ax.set_ylabel(r"Density (g cm$^{-3}$)")
    ax.set_xlim(0.0, R_EARTH_KM)
    ax.set_ylim(
        0.0,
        1.08 * max(float(np.max(prem_rho)), float(np.max(rho84))),
    )
    style_density_axis(ax)
    ax.set_title(
        "Density profile with PREM Asimov truth\n"
        "5-layer constant-density, 1 year; flux nuisances marginalized",
        pad=12,
    )

    handles = [
        Line2D([0], [0], color=PREM_COLOR, lw=1.15, label="PREM truth"),
        Patch(
            facecolor=COLOR,
            edgecolor=COLOR,
            alpha=0.22,
            label="Central 68% interval",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="o",
            markersize=6.2,
            markerfacecolor=COLOR,
            markeredgecolor="white",
            label="Posterior median",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.90)

    fig.tight_layout()
    outname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outname, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def write_density_summary(q: np.ndarray, mean_rho: np.ndarray, filename: Path):
    """Summarize marginal layer densities in g/cm^3."""
    layer_names = [
        "Inner core",
        "Outer core",
        "Lower mantle",
        "Middle mantle",
        "Upper mantle",
    ]

    rho = q * mean_rho[None, :]
    lower, median, upper = np.percentile(rho, [16.0, 50.0, 84.0], axis=0)

    filename.parent.mkdir(parents=True, exist_ok=True)

    print("\nDensity posterior [g/cm^3]")
    print("Center: median; interval: central 68% (16th–84th percentiles)")
    print(
        f"{'Layer':<15} {'Median':>10} {'-error':>10} "
        f"{'+error':>10} {'Lower':>10} {'Upper':>10}"
    )

    with filename.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "layer",
                "inner_radius_km",
                "outer_radius_km",
                "prem_mean_density_g_cm3",
                "median_g_cm3",
                "lower_16_g_cm3",
                "upper_84_g_cm3",
                "minus_error_g_cm3",
                "plus_error_g_cm3",
            ]
        )

        for i, name in enumerate(layer_names):
            minus = median[i] - lower[i]
            plus = upper[i] - median[i]

            print(
                f"{name:<15} {median[i]:10.6f} {minus:10.6f} "
                f"{plus:10.6f} {lower[i]:10.6f} {upper[i]:10.6f}"
            )

            writer.writerow(
                [
                    name,
                    EDGES_KM[i],
                    EDGES_KM[i + 1],
                    mean_rho[i],
                    median[i],
                    lower[i],
                    upper[i],
                    minus,
                    plus,
                ]
            )

    print(f"\n[summary] output: {filename}")


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)

    posterior_file = run_dir / f"raw_output_{args.nlive}" / "post_equal_weights.dat"
    prem_file = BASE_DIR / "data/PREM/EARTH_MODEL_PREM.dat"

    out_dir = run_dir / "plots"
    output_file = out_dir / f"density_profile_{args.nlive}.png"
    summary_file = out_dir / f"density_summary_{args.nlive}.csv"

    q = load_posterior(posterior_file)
    prem_r, prem_rho = load_prem_profile(prem_file)
    mean_rho = prem_mean_densities(prem_r, prem_rho, EDGES_KM)

    apply_style()
    plot_density_profile(prem_r, prem_rho, q, mean_rho, output_file)
    write_density_summary(q, mean_rho, summary_file)

    print(f"[plot] nlive  : {args.nlive}")
    print(f"[plot] input  : {posterior_file}")
    print(f"[plot] samples: {len(q)}")
    print(f"[plot] output : {output_file}")


if __name__ == "__main__":
    main()
