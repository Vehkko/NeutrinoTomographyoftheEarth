#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# ============================================================
# 用户配置区
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

N_DIM = 5
R_EARTH_KM = 6371.0
SAVE_DPI = 300

POSTERIOR_FILE = BASE_DIR / "result/tomography_mnest/raw_output/post_equal_weights.dat"

OUT_DIR = BASE_DIR / "result/tomography_mnest"

CORNER_OUT = OUT_DIR / "corner_q.png"
PROFILE_OUT = OUT_DIR / "density_profile.png"
SUMMARY_OUT = OUT_DIR / "posterior_summary.csv"

PREM_FILE = BASE_DIR / "data/PREM/EARTH_MODEL_PREM.dat"

LAYER_EDGES_KM = np.array([0.0, 1221.0, 3480.0, 4811.0, 5700.0, 6371.0])

LAYER_NAMES = [
    "Inner core",
    "Outer core",
    "Lower mantle",
    "Middle mantle",
    "Upper mantle",
]

PARAM_NAMES = [f"q{i + 1}" for i in range(N_DIM)]

PARAM_LABELS = [
    r"q_{\rm IC}",
    r"q_{\rm OC}",
    r"q_{\rm LM}",
    r"q_{\rm MM}",
    r"q_{\rm UM}",
]

Q_MIN = 0.0
Q_MAX = 4.0

COLOR = "#355C7D"

USE_TEX = False

plt.rc("text", usetex=USE_TEX)
plt.rc("font", family="serif")

# ---------- corner 图 ----------
CORNER_SMOOTH_SCALE_1D = 0.30
CORNER_SMOOTH_SCALE_2D = 0.30
CORNER_FILL_ALPHA = 0.34

# ---------- 密度图 ----------
BAR_ALPHA = 0.22
BAR_EDGE_ALPHA = 0.46

PREM_LINEWIDTH = 1.15
MEDIAN_MARKER_SIZE = 34
BOUNDARY_LINEWIDTH = 0.8

LAYER_BAR_PAD_FRAC = 0.08


# ============================================================
# 读取 posterior
# ============================================================


def load_posterior(
    filename: Path,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(filename)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[1] < N_DIM + 1:
        raise RuntimeError(f"posterior file has too few columns: {filename}")

    q = data[:, :N_DIM]
    loglike = data[:, N_DIM]

    return q, loglike


# ============================================================
# PREM
# ============================================================


def load_prem_profile(
    prem_file: Path,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(prem_file)

    x = arr[:, 0]
    rho = arr[:, 1]

    if np.nanmax(x) <= 2.0:
        r_km = x * R_EARTH_KM
    else:
        r_km = x

    order = np.argsort(r_km)

    return r_km[order], rho[order]


def integrate_prem_rho_x2(
    r_km: np.ndarray,
    rho: np.ndarray,
    r0_km: float,
    r1_km: float,
) -> float:
    x = r_km / R_EARTH_KM

    a = r0_km / R_EARTH_KM
    b = r1_km / R_EARTH_KM

    total = 0.0

    for i in range(len(x) - 1):
        x0 = x[i]
        x1 = x[i + 1]

        u = max(a, x0)
        v = min(b, x1)

        if v <= u:
            continue

        slope = (rho[i + 1] - rho[i]) / (x1 - x0)
        intercept = rho[i] - slope * x0

        total += intercept * (v**3 - u**3) / 3.0
        total += slope * (v**4 - u**4) / 4.0

    return total


def mean_prem_layer_densities(
    r_km: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    means = np.zeros(N_DIM, dtype=float)

    for i in range(N_DIM):
        a = LAYER_EDGES_KM[i] / R_EARTH_KM
        b = LAYER_EDGES_KM[i + 1] / R_EARTH_KM

        volume_weight = (b**3 - a**3) / 3.0

        means[i] = (
            integrate_prem_rho_x2(
                r_km,
                rho,
                LAYER_EDGES_KM[i],
                LAYER_EDGES_KM[i + 1],
            )
            / volume_weight
        )

    return means


# ============================================================
# posterior summary
# ============================================================


def summarize(
    q: np.ndarray,
    rho: np.ndarray,
    loglike: np.ndarray,
) -> dict:
    q16, q50, q84 = np.percentile(
        q,
        [16.0, 50.0, 84.0],
        axis=0,
    )

    q2p5, q97p5 = np.percentile(
        q,
        [2.5, 97.5],
        axis=0,
    )

    rho16, rho50, rho84 = np.percentile(
        rho,
        [16.0, 50.0, 84.0],
        axis=0,
    )

    rho2p5, rho97p5 = np.percentile(
        rho,
        [2.5, 97.5],
        axis=0,
    )

    q_mean = q.mean(axis=0)
    q_std = q.std(axis=0, ddof=1)

    rho_mean = rho.mean(axis=0)
    rho_std = rho.std(axis=0, ddof=1)

    best_idx = int(np.argmax(loglike))

    print("===== five-layer constant-density posterior =====")

    for i in range(N_DIM):
        print(
            f"{PARAM_NAMES[i]} ({LAYER_NAMES[i]}): "
            f"best={q[best_idx, i]:.6f}, "
            f"median={q50[i]:.6f}, "
            f"mean={q_mean[i]:.6f}, "
            f"std={q_std[i]:.6f}, "
            f"68%=[{q16[i]:.6f}, {q84[i]:.6f}], "
            f"95%=[{q2p5[i]:.6f}, {q97p5[i]:.6f}], "
            f"rho_median={rho50[i]:.6f} g/cm^3"
        )

    print(f"best loglike = {loglike[best_idx]:.10g}")

    print()

    return {
        "best_idx": best_idx,
        "q_mean": q_mean,
        "q_std": q_std,
        "q16": q16,
        "q50": q50,
        "q84": q84,
        "q2p5": q2p5,
        "q97p5": q97p5,
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "rho16": rho16,
        "rho50": rho50,
        "rho84": rho84,
        "rho2p5": rho2p5,
        "rho97p5": rho97p5,
    }


def write_summary(
    filename: Path,
    q: np.ndarray,
    rho: np.ndarray,
    stats: dict,
):
    best_idx = stats["best_idx"]

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with filename.open(
        "w",
        newline="",
    ) as file:
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

        for i in range(N_DIM):
            writer.writerow(
                [
                    PARAM_NAMES[i],
                    LAYER_NAMES[i],
                    q[best_idx, i],
                    stats["q_mean"][i],
                    stats["q50"][i],
                    stats["q_std"][i],
                    stats["q16"][i],
                    stats["q84"][i],
                    stats["q2p5"][i],
                    stats["q97p5"][i],
                    rho[best_idx, i],
                    stats["rho_mean"][i],
                    stats["rho50"][i],
                    stats["rho_std"][i],
                    stats["rho16"][i],
                    stats["rho84"][i],
                    stats["rho2p5"][i],
                    stats["rho97p5"][i],
                ]
            )

    print(f"[+] posterior summary saved to: {filename}")


# ============================================================
# 风格
# ============================================================


def apply_global_style():
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


def add_layer_background(ax):
    colors = [
        "#7f1d1d",
        "#c2410c",
        "#ea580c",
        "#f59e0b",
        "#fbbf24",
    ]

    alphas = [
        0.050,
        0.046,
        0.042,
        0.038,
        0.034,
    ]

    for i in range(N_DIM):
        ax.axvspan(
            LAYER_EDGES_KM[i],
            LAYER_EDGES_KM[i + 1],
            color=colors[i],
            alpha=alphas[i],
            lw=0,
            zorder=0,
        )

    for x in LAYER_EDGES_KM[1:-1]:
        ax.axvline(
            x,
            color="#6b2f1a",
            lw=BOUNDARY_LINEWIDTH,
            alpha=0.50,
            zorder=1,
        )


def add_layer_labels(ax):
    ymin, ymax = ax.get_ylim()

    y = ymax - 0.055 * (ymax - ymin)

    x_offsets = np.array([0.0, 0.0, 0.0, -115.0, 120.0])

    for i, name in enumerate(LAYER_NAMES):
        x = 0.5 * (LAYER_EDGES_KM[i] + LAYER_EDGES_KM[i + 1]) + x_offsets[i]

        ax.text(
            x,
            y,
            name,
            ha="center",
            va="top",
            fontsize=8.6,
            alpha=0.86,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="white",
                edgecolor="none",
                alpha=0.60,
            ),
            zorder=30,
        )


def style_density_axes(ax):
    ax.set_facecolor("white")

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.55,
        alpha=0.20,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.35,
        alpha=0.11,
    )

    ax.minorticks_on()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_alpha(0.45)
    ax.spines["bottom"].set_alpha(0.45)

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=10,
        length=5,
        width=0.8,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        length=3,
        width=0.6,
    )


# ============================================================
# Corner
# ============================================================


def plot_corner(
    q: np.ndarray,
    outname: Path,
):
    samples = MCSamples(
        samples=q,
        names=PARAM_NAMES,
        labels=PARAM_LABELS,
        label="1 year",
        ranges={
            PARAM_NAMES[i]: (
                Q_MIN,
                Q_MAX,
            )
            for i in range(N_DIM)
        },
        settings={
            "smooth_scale_1D": CORNER_SMOOTH_SCALE_1D,
            "smooth_scale_2D": CORNER_SMOOTH_SCALE_2D,
        },
    )

    g = plots.get_subplot_plotter()

    g.settings.figure_legend_loc = "upper right"
    g.settings.num_plot_contours = 2
    g.settings.lab_fontsize = 15
    g.settings.legend_fontsize = 12
    g.settings.axes_fontsize = 11
    g.settings.alpha_filled_add = CORNER_FILL_ALPHA
    g.settings.linewidth = 1.2

    g.triangle_plot(
        [samples],
        filled=True,
        contour_colors=[COLOR],
        legend_labels=["1 year"],
        line_args=[
            {
                "lw": 1.45,
                "color": COLOR,
                "alpha": 0.92,
            }
        ],
        contour_lws=[1.15],
    )

    g.fig.suptitle(
        "Posterior of five-layer constant-density factors",
        fontsize=13.5,
        y=0.995,
    )

    g.fig.subplots_adjust(top=0.95)

    outname.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    g.fig.savefig(
        outname,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close(g.fig)

    print(f"[+] corner plot saved to: {outname}")


# ============================================================
# 密度图
# ============================================================


def add_density_bar(
    ax,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    color: str,
    alpha: float,
    zorder: int,
):
    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor=color,
        edgecolor=color,
        linewidth=0.9,
        alpha=alpha,
        zorder=zorder,
    )

    ax.add_patch(rect)


def plot_density_profile(
    prem_r_km: np.ndarray,
    prem_rho: np.ndarray,
    rho: np.ndarray,
    outname: Path,
):
    q16, q50, q84 = np.percentile(
        rho,
        [16.0, 50.0, 84.0],
        axis=0,
    )

    fig, ax = plt.subplots(
        figsize=(8.9, 5.65),
        dpi=120,
    )

    add_layer_background(ax)

    # ------------------------------------------------------------
    # PREM reference
    # ------------------------------------------------------------

    ax.plot(
        prem_r_km,
        prem_rho,
        color="#1f2937",
        lw=PREM_LINEWIDTH,
        alpha=0.95,
        label="PREM",
        zorder=8,
    )

    # ------------------------------------------------------------
    # posterior 区间
    # ------------------------------------------------------------

    for i in range(N_DIM):
        left = LAYER_EDGES_KM[i]
        right = LAYER_EDGES_KM[i + 1]

        width = right - left

        x0 = left + LAYER_BAR_PAD_FRAC * width

        x1 = right - LAYER_BAR_PAD_FRAC * width

        add_density_bar(
            ax=ax,
            x0=x0,
            x1=x1,
            y0=q16[i],
            y1=q84[i],
            color=COLOR,
            alpha=BAR_ALPHA,
            zorder=3,
        )

        xc = 0.5 * (left + right)

        ax.scatter(
            xc,
            q50[i],
            s=MEDIAN_MARKER_SIZE * 1.25,
            color=COLOR,
            edgecolor="white",
            linewidth=0.85,
            alpha=0.98,
            zorder=14,
        )

    # ------------------------------------------------------------
    # 坐标与范围
    # ------------------------------------------------------------

    ax.set_xlabel(
        "Radius from Earth's center (km)",
        fontsize=11.5,
    )

    ax.set_ylabel(
        r"Density (g cm$^{-3}$)",
        fontsize=11.5,
    )

    ax.set_xlim(
        0.0,
        R_EARTH_KM,
    )

    ymax = (
        max(
            np.nanmax(prem_rho),
            np.max(q84),
        )
        * 1.08
    )

    ax.set_ylim(
        0.0,
        ymax,
    )

    style_density_axes(ax)
    add_layer_labels(ax)

    ax.set_title(
        "Five-layer constant-density fit with PREM Asimov truth",
        fontsize=13,
        pad=12,
    )

    # ------------------------------------------------------------
    # 图例
    # ------------------------------------------------------------

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#1f2937",
            lw=PREM_LINEWIDTH,
            label="PREM truth",
        ),
        Patch(
            facecolor=COLOR,
            edgecolor=COLOR,
            alpha=BAR_ALPHA,
            label="68% interval",
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

    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.83),
        fontsize=8.8,
        frameon=True,
        fancybox=True,
        framealpha=0.90,
        borderpad=0.65,
        labelspacing=0.48,
        handlelength=1.8,
    )

    leg.get_frame().set_facecolor("white")

    leg.get_frame().set_edgecolor((0, 0, 0, 0.14))

    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    outname.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        outname,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[+] density profile saved to: {outname}")


# ============================================================
# 主程序
# ============================================================


def main():
    apply_global_style()

    if not POSTERIOR_FILE.exists():
        raise FileNotFoundError(f"posterior file not found: {POSTERIOR_FILE}")

    if not PREM_FILE.exists():
        raise FileNotFoundError(f"PREM file not found: {PREM_FILE}")

    q, loglike = load_posterior(POSTERIOR_FILE)

    prem_r_km, prem_rho = load_prem_profile(PREM_FILE)

    mean_rho = mean_prem_layer_densities(
        prem_r_km,
        prem_rho,
    )

    rho = q * mean_rho[None, :]

    print("===== PREM volume-weighted layer densities =====")

    for i in range(N_DIM):
        print(f"{LAYER_NAMES[i]}: {mean_rho[i]:.8f} g/cm^3")

    print()

    stats = summarize(
        q,
        rho,
        loglike,
    )

    write_summary(
        SUMMARY_OUT,
        q,
        rho,
        stats,
    )

    plot_corner(
        q,
        CORNER_OUT,
    )

    plot_density_profile(
        prem_r_km,
        prem_rho,
        rho,
        PROFILE_OUT,
    )


if __name__ == "__main__":
    main()
