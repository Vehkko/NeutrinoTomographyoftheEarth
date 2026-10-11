#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

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
        description="Plot posterior corner and density profile for the constrained five-layer run."
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

    for i, name in enumerate(layer_names):
        print(
            f"q{i + 1} ({name}): median={stats['q50'][i]:.6g}, "
            f"68%=[{stats['q16'][i]:.6g}, {stats['q84'][i]:.6g}]"
        )

    print(f"[plot] {title}")
    print(f"[plot] input : {posterior_file}")
    print(f"[plot] output: {out_dir}")


if __name__ == "__main__":
    main()
