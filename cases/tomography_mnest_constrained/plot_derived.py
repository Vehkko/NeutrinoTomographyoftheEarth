#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

BASE_DIR = Path(__file__).resolve().parents[2]
R_EARTH_KM = 6371.0
SAVE_DPI = 300
Q_MIN = 0.0
Q_MAX = 5.0
COLOR = "#355C7D"
PREM_COLOR = "#1f2937"
REF_COLOR = "#9f1239"
USE_TEX = False

plt.rc("text", usetex=USE_TEX)
plt.rc("font", family="serif")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot derived quantities for the constrained five-layer run."
    )
    parser.add_argument(
        "--run-dir",
        default="result/tomography_mnest_constrained",
        help="Run directory, relative to the project root or absolute.",
    )
    return parser.parse_args()


def resolve_run_dir(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def load_run_info(run_dir: Path) -> dict[str, str]:
    filename = run_dir / "run_info.dat"
    if not filename.is_file():
        raise FileNotFoundError(f"run_info.dat not found: {filename}")

    info: dict[str, str] = {}
    for raw in filename.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"invalid run_info.dat line: {line}")
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()

    for key in ("model", "exposure_years", "mass_kg", "inertia_kg_m2"):
        if key not in info:
            raise RuntimeError(f"run_info.dat is missing '{key}': {filename}")

    return info


def load_posterior(filename: Path, n_dim: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[0] < 3 or data.shape[1] != n_dim + 1 or not np.isfinite(data).all():
        raise RuntimeError(
            f"posterior must contain at least 3 finite rows of q1,q2,q3,q4,q5,loglike: {filename}"
        )
    return data[:, :n_dim], data[:, n_dim]


def load_prem_profile(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(filename)
    x = arr[:, 0]
    rho = arr[:, 1]
    r_km = x * R_EARTH_KM if np.nanmax(x) <= 2.0 else x
    order = np.argsort(r_km, kind="stable")
    return r_km[order], rho[order]


def prem_piecewise_integrals(
    r_km: np.ndarray, rho_gcm3: np.ndarray, edges_km: np.ndarray
):
    n = len(edges_km) - 1
    mass = np.zeros(n)
    inertia = np.zeros(n)

    r_m = r_km * 1000.0
    rho_kgm3 = rho_gcm3 * 1000.0
    edges_m = edges_km * 1000.0

    for i in range(len(r_m) - 1):
        ra, rb = r_m[i], r_m[i + 1]
        if rb <= ra:
            continue

        slope = (rho_kgm3[i + 1] - rho_kgm3[i]) / (rb - ra)
        intercept = rho_kgm3[i] - slope * ra

        for layer in range(n):
            lo = max(ra, edges_m[layer])
            hi = min(rb, edges_m[layer + 1])
            if hi <= lo:
                continue

            mass_int = slope * (hi**4 - lo**4) / 4.0 + intercept * (hi**3 - lo**3) / 3.0
            inertia_int = (
                slope * (hi**6 - lo**6) / 6.0 + intercept * (hi**5 - lo**5) / 5.0
            )
            mass[layer] += 4.0 * np.pi * mass_int
            inertia[layer] += 8.0 * np.pi / 3.0 * inertia_int

    edges_m = edges_km * 1000.0
    volumes = (4.0 / 3.0) * np.pi * (edges_m[1:] ** 3 - edges_m[:-1] ** 3)
    mean_rho = (mass / volumes) / 1000.0
    return mass, inertia, volumes, mean_rho


def apply_style():
    plt.rcParams.update(
        {
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


def profile_display_name(profile: str) -> str:
    return "constant-density" if profile == "constant" else "PREM-scaled"


def title_suffix(layers: int, profile: str, exposure: int) -> str:
    year = "year" if exposure == 1 else "years"
    return f"{layers}-layer {profile_display_name(profile)}, {exposure:g} {year}"


def derived_quantities(
    q: np.ndarray,
    profile: str,
    prem_mass: np.ndarray,
    prem_inertia: np.ndarray,
    volumes: np.ndarray,
    mean_rho: np.ndarray,
    edges: np.ndarray,
    core_end: int,
):
    masses = q * (mean_rho * 1000.0 * volumes)[None, :]

    if profile == "prem_scaled":
        inertias = q * prem_inertia[None, :]
    else:
        edges_m = edges * 1000.0
        inertia_coeff = (8.0 * np.pi / 15.0) * (edges_m[1:] ** 5 - edges_m[:-1] ** 5)
        inertias = q * (mean_rho * 1000.0 * inertia_coeff)[None, :]

    total_mass = masses.sum(axis=1)
    core_mass = masses[:, :core_end].sum(axis=1)
    mantle_mass = masses[:, core_end:].sum(axis=1)
    total_inertia = inertias.sum(axis=1)

    core_volume = volumes[:core_end].sum()
    mantle_volume = volumes[core_end:].sum()
    core_rho = (core_mass / core_volume) / 1000.0
    mantle_rho = (mantle_mass / mantle_volume) / 1000.0

    return {
        "M_earth_1e24kg": total_mass / 1.0e24,
        "M_core_1e24kg": core_mass / 1.0e24,
        "I_earth_1e37": total_inertia / 1.0e37,
        "delta_rho_gcm3": core_rho - mantle_rho,
    }


def prem_reference(
    prem_mass: np.ndarray, prem_inertia: np.ndarray, volumes: np.ndarray, core_end: int
):
    core_mass = prem_mass[:core_end].sum()
    mantle_mass = prem_mass[core_end:].sum()
    core_rho = (core_mass / volumes[:core_end].sum()) / 1000.0
    mantle_rho = (mantle_mass / volumes[core_end:].sum()) / 1000.0
    return {
        "M_earth_1e24kg": prem_mass.sum() / 1.0e24,
        "M_core_1e24kg": core_mass / 1.0e24,
        "I_earth_1e37": prem_inertia.sum() / 1.0e37,
        "delta_rho_gcm3": core_rho - mantle_rho,
    }


def summarize_1d(x: np.ndarray) -> dict[str, float]:
    q16, q50, q84 = np.percentile(x, [16.0, 50.0, 84.0])
    q2p5, q97p5 = np.percentile(x, [2.5, 97.5])
    return {
        "mean": float(np.mean(x)),
        "median": float(q50),
        "std": 0.0 if np.ptp(x) == 0 else float(np.std(x, ddof=1)),
        "q16": float(q16),
        "q84": float(q84),
        "q2p5": float(q2p5),
        "q97p5": float(q97p5),
    }


def write_derived_summary(
    filename: Path, derived: dict[str, np.ndarray], reference: dict[str, float]
):
    with filename.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "quantity",
                "mean",
                "median",
                "std",
                "q16",
                "q84",
                "q2p5",
                "q97p5",
                "prem_reference",
            ]
        )
        for key, values in derived.items():
            s = summarize_1d(values)
            writer.writerow(
                [
                    key,
                    s["mean"],
                    s["median"],
                    s["std"],
                    s["q16"],
                    s["q84"],
                    s["q2p5"],
                    s["q97p5"],
                    reference[key],
                ]
            )


def kde_curve(x: np.ndarray, ref: float):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    xmin = min(float(np.min(x)), ref)
    xmax = max(float(np.max(x)), ref)
    dx = xmax - xmin
    if dx <= 0.0:
        dx = max(abs(xmin), 1.0)
    grid = np.linspace(xmin - 0.16 * dx, xmax + 0.16 * dx, 900)

    if len(x) < 3 or np.std(x) <= 0.0:
        width = 0.05 * dx
        y = np.exp(-0.5 * ((grid - np.mean(x)) / width) ** 2)
    else:
        kde = gaussian_kde(x, bw_method="scott")
        kde.set_bandwidth(bw_method=kde.factor * 1.25)
        y = kde(grid)

    if np.max(y) > 0.0:
        y = y / np.max(y)
    return grid, y


PANEL_CONFIGS = [
    ("M_earth_1e24kg", "(a) Total Earth mass", r"$M_\oplus\ [10^{24}\,\mathrm{kg}]$"),
    ("M_core_1e24kg", "(b) Core mass", r"$M_{\rm core}\ [10^{24}\,\mathrm{kg}]$"),
    (
        "I_earth_1e37",
        "(c) Moment of inertia",
        r"$I_\oplus\ [10^{37}\,\mathrm{kg\,m^2}]$",
    ),
    (
        "delta_rho_gcm3",
        "(d) Core--mantle density contrast",
        r"$\bar{\rho}_{\rm core}-\bar{\rho}_{\rm mantle}\ [\mathrm{g\,cm^{-3}}]$",
    ),
]


def plot_derived(
    derived: dict[str, np.ndarray],
    reference: dict[str, float],
    title: str,
    outname: Path,
    fixed: dict[str, float],
):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.3), dpi=120)

    for ax, (key, panel_title, xlabel) in zip(axes.ravel(), PANEL_CONFIGS):
        x = derived[key]
        if key in fixed:
            value = fixed[key]
            # M and I are fixed by elimination. Do not KDE their roundoff noise.
            ax.axvline(value, color=COLOR, lw=1.75)
            ax.axvline(reference[key], color=REF_COLOR, linestyle=(0, (4, 4)), lw=1.15)
            width = max(abs(value - reference[key]) * 1.6, abs(value) * 0.002)
            ax.set_xlim(
                min(value, reference[key]) - width, max(value, reference[key]) + width
            )
            ax.text(
                0.5,
                0.80,
                f"Hard constraint: {value:.8g}",
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
            )
            ax.set_yticks([])
        else:
            grid, y = kde_curve(x, reference[key])
            q16, q50, q84 = np.percentile(x, [16.0, 50.0, 84.0])
            ax.fill_between(grid, 0.0, y, color=COLOR, alpha=0.16, linewidth=0.0)
            ax.plot(grid, y, color=COLOR, lw=1.75)
            ax.axvspan(q16, q84, color=COLOR, alpha=0.08, lw=0)
            ax.axvline(q50, color=COLOR, lw=1.0, alpha=0.8)
            ax.axvline(reference[key], color=REF_COLOR, linestyle=(0, (4, 4)), lw=1.15)

        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("" if key in fixed else "Normalized density")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=COLOR, lw=1.75, label="Posterior KDE / fixed value"),
        Patch(facecolor=COLOR, alpha=0.08, label="Central 68% interval"),
        Line2D(
            [0],
            [0],
            color=REF_COLOR,
            lw=1.15,
            linestyle=(0, (4, 4)),
            label="PREM reference",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(f"Derived posteriors: {title}", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    outname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outname, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    info = load_run_info(run_dir)
    if info["model"] != "5layer_constant":
        raise RuntimeError("Expected model=5layer_constant in run_info.dat")
    layers, profile, exposure, core_end = (
        5,
        "constant",
        float(info["exposure_years"]),
        2,
    )
    edges = np.array([0.0] + [float(info[f"outer_radius_km_{i}"]) for i in range(1, 6)])
    mean_rho = np.array(
        [float(info[f"prem_mean_density_g_cm3_{i}"]) for i in range(1, 6)]
    )
    if not np.all(np.diff(edges) > 0) or not np.all(mean_rho > 0):
        raise RuntimeError("Invalid layer radii or reference densities in run_info.dat")
    layer_names = [
        "Inner core",
        "Outer core",
        "Lower mantle",
        "Middle mantle",
        "Upper mantle",
    ]
    param_labels = [
        r"q_{\rm IC}",
        r"q_{\rm OC}",
        r"q_{\rm LM}",
        r"q_{\rm MM}",
        r"q_{\rm UM}",
    ]
    global Q_MIN, Q_MAX
    Q_MIN, Q_MAX = float(info["q_min"]), float(info["q_max"])

    posterior_file = run_dir / "raw_output" / "post_equal_weights.dat"
    prem_file = BASE_DIR / "data/PREM/EARTH_MODEL_PREM.dat"
    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not posterior_file.is_file():
        raise FileNotFoundError(f"posterior file not found: {posterior_file}")
    if not prem_file.is_file():
        raise FileNotFoundError(f"PREM file not found: {prem_file}")

    apply_style()

    q, loglike = load_posterior(posterior_file, layers)
    prem_r, prem_rho = load_prem_profile(prem_file)
    prem_mass, prem_inertia, volumes, _ = prem_piecewise_integrals(
        prem_r, prem_rho, edges
    )
    rho = q * mean_rho[None, :]
    title = title_suffix(layers, profile, exposure) + "; hard M + I + EHE"

    if (
        np.any(q < Q_MIN)
        or np.any(q > Q_MAX)
        or np.any(rho[:, -1] <= 0)
        or np.any(np.diff(rho, axis=1) >= 0)
    ):
        raise RuntimeError("Posterior violates q support or EHE; check the input run")
    derived = derived_quantities(
        q, profile, prem_mass, prem_inertia, volumes, mean_rho, edges, core_end
    )
    reference = prem_reference(prem_mass, prem_inertia, volumes, core_end)
    fixed = {
        "M_earth_1e24kg": float(info["mass_kg"]) / 1e24,
        "I_earth_1e37": float(info["inertia_kg_m2"]) / 1e37,
    }
    for key, value in fixed.items():
        residual = float(np.max(np.abs(derived[key] / value - 1.0)))
        print(f"[check] {key}: max relative constraint residual = {residual:.3e}")
        if residual > 1e-7:
            raise RuntimeError(f"Posterior violates the hard constraint for {key}")
        derived[key] = np.full(len(q), value)
    write_derived_summary(out_dir / "derived_summary.csv", derived, reference)
    plot_derived(derived, reference, title, out_dir / "derived_posteriors.png", fixed)
    print(f"[plot] {title}")
    print(f"[plot] input : {posterior_file}")
    print(f"[plot] output: {out_dir}")


if __name__ == "__main__":
    main()
