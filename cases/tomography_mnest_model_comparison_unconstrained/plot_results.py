#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import gaussian_kde

BASE_DIR = Path(__file__).resolve().parents[2]
R_EARTH_KM = 6371.0
SAVE_DPI = 300
Q_MIN = 0.0
Q_MAX = 4.0
COLOR = "#355C7D"
PREM_COLOR = "#1f2937"
REF_COLOR = "#9f1239"
USE_TEX = False

plt.rc("text", usetex=USE_TEX)
plt.rc("font", family="serif")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot one unconstrained MultiNest model-comparison run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory, relative to the project root or absolute.",
    )
    return parser.parse_args()


def resolve_run_dir(text: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def load_run_info(run_dir: Path) -> dict[str, str]:
    filename = run_dir / "run.info"
    if not filename.is_file():
        raise FileNotFoundError(f"run.info not found: {filename}")

    info: dict[str, str] = {}
    for raw in filename.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"invalid run.info line: {line}")
        key, value = line.split("=", 1)
        info[key.strip()] = value.strip()

    for key in ("layers", "profile", "exposure_years"):
        if key not in info:
            raise RuntimeError(f"run.info is missing '{key}': {filename}")

    return info


def model_config(info: dict[str, str]):
    layers = int(info["layers"])
    profile = info["profile"]
    exposure = int(info["exposure_years"])

    if layers == 5:
        edges = np.array([0.0, 1221.0, 3480.0, 4811.0, 5700.0, 6371.0])
        names = [
            "Inner core",
            "Outer core",
            "Lower mantle",
            "Middle mantle",
            "Upper mantle",
        ]
        labels = [
            r"q_{\rm IC}",
            r"q_{\rm OC}",
            r"q_{\rm LM}",
            r"q_{\rm MM}",
            r"q_{\rm UM}",
        ]
        core_end = 2
    elif layers == 3:
        edges = np.array([0.0, 3480.0, 5700.0, 6371.0])
        names = ["Core", "Lower mantle", "Upper mantle"]
        labels = [r"q_{\rm C}", r"q_{\rm LM}", r"q_{\rm UM}"]
        core_end = 1
    else:
        raise RuntimeError(f"unsupported layer count: {layers}")

    if profile not in ("constant", "prem_scaled"):
        raise RuntimeError(f"unsupported profile: {profile}")

    return layers, profile, exposure, edges, names, labels, core_end


def load_posterior(filename: Path, n_dim: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < n_dim + 1:
        raise RuntimeError(f"posterior file has too few columns: {filename}")
    return data[:, :n_dim], data[:, n_dim]


def load_prem_profile(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(filename)
    x = arr[:, 0]
    rho = arr[:, 1]
    r_km = x * R_EARTH_KM if np.nanmax(x) <= 2.0 else x
    order = np.argsort(r_km)
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
    return "layered constant-density" if profile == "constant" else "PREM-scaled"


def title_suffix(layers: int, profile: str, exposure: int) -> str:
    year = "year" if exposure == 1 else "years"
    return f"{layers}-layer {profile_display_name(profile)}, {exposure} {year}"


def posterior_stats(
    q: np.ndarray, rho: np.ndarray, loglike: np.ndarray
) -> dict[str, np.ndarray | int]:
    q16, q50, q84 = np.percentile(q, [16.0, 50.0, 84.0], axis=0)
    q2p5, q97p5 = np.percentile(q, [2.5, 97.5], axis=0)
    rho16, rho50, rho84 = np.percentile(rho, [16.0, 50.0, 84.0], axis=0)
    rho2p5, rho97p5 = np.percentile(rho, [2.5, 97.5], axis=0)

    return {
        "best_idx": int(np.argmax(loglike)),
        "q_mean": q.mean(axis=0),
        "q_std": q.std(axis=0, ddof=1),
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "q2p5": q2p5,
        "q97p5": q97p5,
        "rho_mean": rho.mean(axis=0),
        "rho_std": rho.std(axis=0, ddof=1),
        "rho16": rho16,
        "rho50": rho50,
        "rho84": rho84,
        "rho2p5": rho2p5,
        "rho97p5": rho97p5,
    }


def write_posterior_summary(
    filename: Path, q: np.ndarray, rho: np.ndarray, stats: dict, names: list[str]
):
    filename.parent.mkdir(parents=True, exist_ok=True)
    best = int(stats["best_idx"])

    with filename.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "parameter",
                "layer",
                "best_q",
                "mean_q",
                "median_q",
                "std_q",
                "q16",
                "q84",
                "q2p5",
                "q97p5",
                "best_rho_g_cm3",
                "mean_rho_g_cm3",
                "median_rho_g_cm3",
                "std_rho_g_cm3",
                "rho16",
                "rho84",
                "rho2p5",
                "rho97p5",
            ]
        )
        for i, name in enumerate(names):
            writer.writerow(
                [
                    f"q{i + 1}",
                    name,
                    q[best, i],
                    stats["q_mean"][i],
                    stats["q50"][i],
                    stats["q_std"][i],
                    stats["q16"][i],
                    stats["q84"][i],
                    stats["q2p5"][i],
                    stats["q97p5"][i],
                    rho[best, i],
                    stats["rho_mean"][i],
                    stats["rho50"][i],
                    stats["rho_std"][i],
                    stats["rho16"][i],
                    stats["rho84"][i],
                    stats["rho2p5"][i],
                    stats["rho97p5"][i],
                ]
            )


def plot_corner(q: np.ndarray, labels: list[str], title: str, outname: Path):
    names = [f"q{i + 1}" for i in range(q.shape[1])]
    samples = MCSamples(
        samples=q,
        names=names,
        labels=labels,
        ranges={name: (Q_MIN, Q_MAX) for name in names},
        settings={"smooth_scale_1D": 0.30, "smooth_scale_2D": 0.30},
    )

    g = plots.get_subplot_plotter()
    g.settings.num_plot_contours = 2
    g.settings.lab_fontsize = 15
    g.settings.axes_fontsize = 11
    g.settings.alpha_filled_add = 0.34
    g.settings.linewidth = 1.2
    g.triangle_plot(
        [samples],
        filled=True,
        contour_colors=[COLOR],
        line_args=[{"lw": 1.45, "color": COLOR, "alpha": 0.92}],
        contour_lws=[1.15],
    )
    g.fig.suptitle(f"Posterior density factors: {title}", fontsize=13.5, y=0.995)
    g.fig.subplots_adjust(top=0.95)
    outname.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(outname, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(g.fig)


def add_layer_background(ax, edges: np.ndarray):
    colors = ["#7f1d1d", "#c2410c", "#ea580c", "#f59e0b", "#fbbf24"]
    for i in range(len(edges) - 1):
        ax.axvspan(
            edges[i],
            edges[i + 1],
            color=colors[min(i, len(colors) - 1)],
            alpha=0.04,
            lw=0,
            zorder=0,
        )
    for x in edges[1:-1]:
        ax.axvline(x, color="#6b2f1a", lw=0.8, alpha=0.50, zorder=1)


def style_density_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.20)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, alpha=0.11)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.45)
    ax.spines["bottom"].set_alpha(0.45)


def layer_indices(r_km: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges[1:], r_km, side="left")
    return np.clip(idx, 0, len(edges) - 2)


def plot_density_profile(
    prem_r: np.ndarray,
    prem_rho: np.ndarray,
    q: np.ndarray,
    mean_rho: np.ndarray,
    profile: str,
    edges: np.ndarray,
    title: str,
    outname: Path,
):
    q16, q50, q84 = np.percentile(q, [16.0, 50.0, 84.0], axis=0)

    fig, ax = plt.subplots(figsize=(8.9, 5.65), dpi=120)
    add_layer_background(ax, edges)
    ax.plot(
        prem_r,
        prem_rho,
        color=PREM_COLOR,
        lw=1.15,
        alpha=0.95,
        label="PREM truth",
        zorder=8,
    )

    if profile == "constant":
        rho16 = q16 * mean_rho
        rho50 = q50 * mean_rho
        rho84 = q84 * mean_rho
        for i in range(len(mean_rho)):
            left, right = edges[i], edges[i + 1]
            width = right - left
            x0, x1 = left + 0.08 * width, right - 0.08 * width
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
    else:
        idx = layer_indices(prem_r, edges)
        lo = prem_rho * q16[idx]
        med = prem_rho * q50[idx]
        hi = prem_rho * q84[idx]
        ax.fill_between(prem_r, lo, hi, color=COLOR, alpha=0.22, linewidth=0, zorder=3)
        ax.plot(prem_r, med, color=COLOR, lw=1.45, alpha=0.95, zorder=10)

    ax.set_xlabel("Radius from Earth's center (km)")
    ax.set_ylabel(r"Density (g cm$^{-3}$)")
    ax.set_xlim(0.0, R_EARTH_KM)
    ax.set_ylim(0.0, max(float(np.nanmax(prem_rho) * 1.08), ax.get_ylim()[1]))
    style_density_axis(ax)
    ax.set_title(f"Density profile with PREM Asimov truth\n{title}", pad=12)

    handles = [Line2D([0], [0], color=PREM_COLOR, lw=1.15, label="PREM truth")]
    if profile == "constant":
        handles += [
            Patch(facecolor=COLOR, edgecolor=COLOR, alpha=0.22, label="68% interval"),
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
    else:
        handles += [
            Patch(facecolor=COLOR, edgecolor=COLOR, alpha=0.22, label="68% interval"),
            Line2D([0], [0], color=COLOR, lw=1.45, label="Posterior median"),
        ]
    ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.90)
    fig.tight_layout()
    outname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outname, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


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
    masses = q * prem_mass[None, :]

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
        "std": float(np.std(x, ddof=1)),
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
):
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.3), dpi=120)

    for ax, (key, panel_title, xlabel) in zip(axes.ravel(), PANEL_CONFIGS):
        x = derived[key]
        grid, y = kde_curve(x, reference[key])
        q16, q50, q84 = np.percentile(x, [16.0, 50.0, 84.0])

        ax.fill_between(grid, 0.0, y, color=COLOR, alpha=0.16, linewidth=0.0)
        ax.plot(grid, y, color=COLOR, lw=1.75)
        ax.axvspan(q16, q84, color=COLOR, alpha=0.08, lw=0)
        ax.axvline(q50, color=COLOR, lw=1.0, alpha=0.8)
        ax.axvline(
            reference[key], color=REF_COLOR, linestyle=(0, (4, 4)), lw=1.15, alpha=0.9
        )

        ax.set_title(panel_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized density")
        ax.set_ylim(bottom=0.0)
        ax.grid(True, which="major", linestyle="--", linewidth=0.55, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=COLOR, lw=1.75, label="Posterior KDE"),
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
    layers, profile, exposure, edges, layer_names, param_labels, core_end = (
        model_config(info)
    )

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
    prem_mass, prem_inertia, volumes, mean_rho = prem_piecewise_integrals(
        prem_r, prem_rho, edges
    )
    rho = q * mean_rho[None, :]
    title = title_suffix(layers, profile, exposure)

    stats = posterior_stats(q, rho, loglike)
    write_posterior_summary(
        out_dir / "posterior_summary.csv", q, rho, stats, layer_names
    )
    plot_corner(q, param_labels, title, out_dir / "corner_q.png")
    plot_density_profile(
        prem_r,
        prem_rho,
        q,
        mean_rho,
        profile,
        edges,
        title,
        out_dir / "density_profile.png",
    )

    derived = derived_quantities(
        q, profile, prem_mass, prem_inertia, volumes, mean_rho, edges, core_end
    )
    reference = prem_reference(prem_mass, prem_inertia, volumes, core_end)
    write_derived_summary(out_dir / "derived_summary.csv", derived, reference)
    plot_derived(derived, reference, title, out_dir / "derived_posteriors.png")

    print(f"[plot] {title}")
    print(f"[plot] input : {posterior_file}")
    print(f"[plot] output: {out_dir}")


if __name__ == "__main__":
    main()
