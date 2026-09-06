#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# ============================================================
# 用户配置区
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

N_DIM = 5
R_EARTH_KM = 6371.0
SAVE_DPI = 300

POSTERIOR_FILE = BASE_DIR / "result/tomography_mnest/raw_output/post_equal_weights.dat"

OUT_DIR = BASE_DIR / "result/tomography_mnest"

FIGURE_OUT = OUT_DIR / "derived_posteriors.png"

SUMMARY_OUT = OUT_DIR / "derived_summary.csv"

PREM_FILE = BASE_DIR / "data/PREM/EARTH_MODEL_PREM.dat"

LAYER_EDGES_KM = np.array(
    [
        0.0,
        1221.0,
        3480.0,
        4811.0,
        5700.0,
        6371.0,
    ]
)

LAYER_NAMES = [
    "Inner core",
    "Outer core",
    "Lower mantle",
    "Middle mantle",
    "Upper mantle",
]

CORE_END_LAYER = 2

COLOR = "#355C7D"

KDE_BW_SCALE = 1.25
KDE_NGRID = 900
KDE_PAD_FRAC = 0.16

USE_TEX = False
FONT_FAMILY = "serif"

plt.rc(
    "text",
    usetex=USE_TEX,
)

plt.rc(
    "font",
    family=FONT_FAMILY,
)


# ============================================================
# 读取 posterior
# ============================================================


def load_posterior(
    post_file: Path,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(post_file)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[1] < N_DIM + 1:
        raise RuntimeError(f"posterior file has too few columns: {post_file}")

    q = data[
        :,
        :N_DIM,
    ]

    loglike = data[
        :,
        N_DIM,
    ]

    return q, loglike


# ============================================================
# PREM
# ============================================================


def load_prem_profile(
    prem_file: Path,
) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(prem_file)

    x = arr[:, 0]
    rho_gcm3 = arr[:, 1]

    if np.nanmax(x) <= 2.0:
        r_km = x * R_EARTH_KM
    else:
        r_km = x

    order = np.argsort(r_km)

    return (
        r_km[order],
        rho_gcm3[order],
    )


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

        u = max(
            a,
            x0,
        )

        v = min(
            b,
            x1,
        )

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
    means = np.zeros(
        N_DIM,
        dtype=float,
    )

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
# 五层常数密度模型的 derived quantities
# ============================================================


def layer_volumes_m3(
    edges_km: np.ndarray,
) -> np.ndarray:
    edges_m = edges_km * 1000.0

    r0 = edges_m[:-1]
    r1 = edges_m[1:]

    return (4.0 / 3.0) * np.pi * (r1**3 - r0**3)


def layer_inertia_coeff_m5(
    edges_km: np.ndarray,
) -> np.ndarray:
    edges_m = edges_km * 1000.0

    r0 = edges_m[:-1]
    r1 = edges_m[1:]

    return (8.0 * np.pi / 15.0) * (r1**5 - r0**5)


def compute_layered_quantities(
    rho_gcm3: np.ndarray,
    volumes_m3: np.ndarray,
    inertia_coeff_m5: np.ndarray,
) -> dict[str, np.ndarray]:
    rho_kgm3 = rho_gcm3 * 1000.0

    masses = rho_kgm3 * volumes_m3[None, :]

    inertias = rho_kgm3 * inertia_coeff_m5[None, :]

    m_total = masses.sum(axis=1)

    m_core = masses[
        :,
        :CORE_END_LAYER,
    ].sum(axis=1)

    m_mantle = masses[
        :,
        CORE_END_LAYER:,
    ].sum(axis=1)

    i_total = inertias.sum(axis=1)

    v_core = volumes_m3[:CORE_END_LAYER].sum()

    v_mantle = volumes_m3[CORE_END_LAYER:].sum()

    rho_core_avg = (m_core / v_core) / 1000.0

    rho_mantle_avg = (m_mantle / v_mantle) / 1000.0

    delta_rho = rho_core_avg - rho_mantle_avg

    return {
        "M_earth_1e24kg": (m_total / 1.0e24),
        "M_core_1e24kg": (m_core / 1.0e24),
        "I_earth_1e37": (i_total / 1.0e37),
        "delta_rho_gcm3": (delta_rho),
    }


# ============================================================
# 直接由 PREM 曲线计算 reference
# ============================================================


def compute_prem_direct_reference(
    r_km: np.ndarray,
    rho_gcm3: np.ndarray,
    layer_edges_km: np.ndarray,
) -> dict[str, float]:
    n_layer = len(layer_edges_km) - 1

    mass_layer = np.zeros(
        n_layer,
        dtype=float,
    )

    inertia_layer = np.zeros(
        n_layer,
        dtype=float,
    )

    r_m = r_km * 1000.0

    rho_kgm3 = rho_gcm3 * 1000.0

    edges_m = layer_edges_km * 1000.0

    for i in range(len(r_m) - 1):
        ra = r_m[i]
        rb = r_m[i + 1]

        if rb <= ra:
            continue

        rho_a = rho_kgm3[i]
        rho_b = rho_kgm3[i + 1]

        slope = (rho_b - rho_a) / (rb - ra)

        intercept = rho_a - slope * ra

        for layer in range(n_layer):
            lo = max(
                ra,
                edges_m[layer],
            )

            hi = min(
                rb,
                edges_m[layer + 1],
            )

            if hi <= lo:
                continue

            mass_integral = (
                slope * (hi**4 - lo**4) / 4.0 + intercept * (hi**3 - lo**3) / 3.0
            )

            mass_layer[layer] += 4.0 * np.pi * mass_integral

            inertia_integral = (
                slope * (hi**6 - lo**6) / 6.0 + intercept * (hi**5 - lo**5) / 5.0
            )

            inertia_layer[layer] += 8.0 * np.pi / 3.0 * inertia_integral

    volumes = layer_volumes_m3(layer_edges_km)

    m_total = mass_layer.sum()

    m_core = mass_layer[:CORE_END_LAYER].sum()

    m_mantle = mass_layer[CORE_END_LAYER:].sum()

    i_total = inertia_layer.sum()

    v_core = volumes[:CORE_END_LAYER].sum()

    v_mantle = volumes[CORE_END_LAYER:].sum()

    rho_core_avg = (m_core / v_core) / 1000.0

    rho_mantle_avg = (m_mantle / v_mantle) / 1000.0

    delta_rho = rho_core_avg - rho_mantle_avg

    return {
        "M_earth_1e24kg": float(m_total / 1.0e24),
        "M_core_1e24kg": float(m_core / 1.0e24),
        "I_earth_1e37": float(i_total / 1.0e37),
        "delta_rho_gcm3": float(delta_rho),
    }


# ============================================================
# Summary
# ============================================================


def summarize_1d(
    x: np.ndarray,
) -> dict[str, float]:
    q16, q50, q84 = np.percentile(
        x,
        [16, 50, 84],
    )

    q2p5, q97p5 = np.percentile(
        x,
        [2.5, 97.5],
    )

    return {
        "mean": float(np.mean(x)),
        "median": float(q50),
        "std": float(
            np.std(
                x,
                ddof=1,
            )
        ),
        "q16": float(q16),
        "q84": float(q84),
        "q2p5": float(q2p5),
        "q97p5": float(q97p5),
    }


# ============================================================
# KDE
# ============================================================


def kde_curve(
    x: np.ndarray,
    extra_points: list[float] | None = None,
    ngrid: int = KDE_NGRID,
    pad_frac: float = KDE_PAD_FRAC,
    bw_scale: float = KDE_BW_SCALE,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        x,
        dtype=float,
    )

    x = x[np.isfinite(x)]

    if extra_points is None:
        extra_points = []

    range_values = list(x) + [p for p in extra_points if np.isfinite(p)]

    xmin = float(np.min(range_values))

    xmax = float(np.max(range_values))

    dx = xmax - xmin

    if dx <= 0.0:
        dx = max(
            abs(xmin),
            1.0,
        )

    grid = np.linspace(
        xmin - pad_frac * dx,
        xmax + pad_frac * dx,
        ngrid,
    )

    if np.std(x) <= 0.0 or len(x) < 3:
        width = 0.05 * dx if dx > 0.0 else 1.0

        y = np.exp(-0.5 * ((grid - np.mean(x)) / width) ** 2)

        y /= np.max(y)

        return grid, y

    kde = gaussian_kde(
        x,
        bw_method="scott",
    )

    kde.set_bandwidth(bw_method=(kde.factor * bw_scale))

    y = kde(grid)

    if np.max(y) > 0.0:
        y = y / np.max(y)

    return grid, y


# ============================================================
# 绘图风格
# ============================================================


def apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": SAVE_DPI,
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "axes.linewidth": 0.85,
            "lines.linewidth": 1.7,
            "mathtext.fontset": "dejavuserif",
        }
    )


def style_one_axis(ax):
    ax.set_facecolor("#fbfaf8")

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.55,
        alpha=0.22,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.35,
        alpha=0.12,
    )

    ax.minorticks_on()

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_alpha(0.50)

    ax.spines["bottom"].set_alpha(0.50)

    ax.tick_params(
        axis="both",
        which="major",
        length=5,
        width=0.8,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        length=3,
        width=0.6,
    )


PANEL_CONFIGS = [
    {
        "key": "M_earth_1e24kg",
        "title": "(a) Total Earth mass",
        "xlabel": (r"$M_\oplus" r"\ [10^{24}\,\mathrm{kg}]$"),
        "show_zero": False,
    },
    {
        "key": "M_core_1e24kg",
        "title": "(b) Core mass",
        "xlabel": (r"$M_{\rm core}" r"\ [10^{24}\,\mathrm{kg}]$"),
        "show_zero": False,
    },
    {
        "key": "I_earth_1e37",
        "title": "(c) Moment of inertia",
        "xlabel": (r"$I_\oplus" r"\ [10^{37}\,\mathrm{kg\,m^2}]$"),
        "show_zero": False,
    },
    {
        "key": "delta_rho_gcm3",
        "title": (r"(d) Core--mantle density contrast"),
        "xlabel": (
            r"$\bar{\rho}_{\rm core}"
            r"-\bar{\rho}_{\rm mantle}"
            r"\ [\mathrm{g\,cm^{-3}}]$"
        ),
        "show_zero": True,
    },
]


# ============================================================
# 绘图
# ============================================================


def plot_one_panel(
    ax,
    x: np.ndarray,
    color: str,
    title: str,
    xlabel: str,
    direct_ref_value: float | None,
    show_zero: bool = False,
):
    extra_points = []

    if direct_ref_value is not None and np.isfinite(direct_ref_value):
        extra_points.append(direct_ref_value)

    if show_zero:
        extra_points.append(0.0)

    grid, y = kde_curve(
        x,
        extra_points=extra_points,
    )

    ax.fill_between(
        grid,
        0.0,
        y,
        color=color,
        alpha=0.16,
        linewidth=0.0,
    )

    ax.plot(
        grid,
        y,
        color=color,
        lw=1.75,
    )

    if direct_ref_value is not None and np.isfinite(direct_ref_value):
        ax.axvline(
            direct_ref_value,
            color="#9f1239",
            linestyle=(0, (4, 4)),
            lw=1.15,
            alpha=0.9,
        )

    if show_zero:
        p_less_0 = float(np.mean(x < 0.0))

        ax.axvline(
            0.0,
            color="#111827",
            linestyle="-",
            lw=1.05,
            alpha=0.78,
        )

        mask = grid < 0.0

        if np.any(mask):
            ax.fill_between(
                grid[mask],
                0.0,
                y[mask],
                color="#ef4444",
                alpha=0.22,
                linewidth=0.0,
            )

        if p_less_0 < 1.0e-4:
            p_text = r"$P(<0)<10^{-4}$"
        else:
            p_text = rf"$P(<0)=" rf"{100.0 * p_less_0:.2f}" rf"\%$"

        ax.text(
            0.97,
            0.86,
            p_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="#dddddd",
                linewidth=0.5,
                alpha=0.82,
            ),
        )

        ax.text(
            0.0,
            0.04,
            r"$0$",
            transform=(ax.get_xaxis_transform()),
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#111827",
            alpha=0.85,
        )

    stats = summarize_1d(x)

    ax.text(
        0.035,
        0.84,
        (
            rf"median = "
            rf"{stats['median']:.3g}" + "\n" + rf"68\% = "
            rf"[{stats['q16']:.3g}, "
            rf"{stats['q84']:.3g}]"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.72,
        ),
    )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold",
        pad=7,
    )

    ax.set_xlabel(xlabel)

    ax.set_ylim(
        0.0,
        1.08,
    )

    ax.set_yticks([])

    style_one_axis(ax)


def plot_derived_posteriors(
    derived: dict[str, np.ndarray],
    direct_ref: dict[str, float],
    outfig: Path,
):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.4, 7.2),
    )

    axes = axes.ravel()

    for ax, cfg in zip(
        axes,
        PANEL_CONFIGS,
    ):
        key = cfg["key"]

        plot_one_panel(
            ax=ax,
            x=derived[key],
            color=COLOR,
            title=cfg["title"],
            xlabel=cfg["xlabel"],
            direct_ref_value=(direct_ref[key]),
            show_zero=(cfg["show_zero"]),
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=COLOR,
            lw=1.75,
            label="Posterior KDE",
        ),
        Line2D(
            [0],
            [0],
            color="#9f1239",
            lw=1.15,
            linestyle=(0, (4, 4)),
            label="PREM reference",
        ),
        Line2D(
            [0],
            [0],
            color="#111827",
            lw=1.05,
            label=r"$0$ line in panel (d)",
        ),
    ]

    fig.suptitle(
        ("Derived 1D posteriors from five-layer constant-density fit"),
        fontsize=14.2,
        y=0.975,
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.925,
        ),
        handlelength=2.8,
        columnspacing=1.8,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        bottom=0.075,
        top=0.835,
        wspace=0.18,
        hspace=0.34,
    )

    outfig.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        outfig,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[+] saved figure to: {outfig}")


def write_summary(
    summary_csv: Path,
    derived: dict[str, np.ndarray],
    direct_ref: dict[str, float],
):
    summary_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_csv.open(
        "w",
        newline="",
    ) as file:
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
                "PREM_reference",
                "P_less_than_0",
            ]
        )

        for cfg in PANEL_CONFIGS:
            key = cfg["key"]

            values = derived[key]

            stats = summarize_1d(values)

            if key == "delta_rho_gcm3":
                probability = float(np.mean(values < 0.0))
            else:
                probability = np.nan

            writer.writerow(
                [
                    key,
                    stats["mean"],
                    stats["median"],
                    stats["std"],
                    stats["q16"],
                    stats["q84"],
                    stats["q2p5"],
                    stats["q97p5"],
                    direct_ref[key],
                    probability,
                ]
            )

    print(f"[+] saved summary to: {summary_csv}")


# ============================================================
# 主程序
# ============================================================


def main():
    apply_style()

    if not POSTERIOR_FILE.exists():
        raise FileNotFoundError(f"posterior file not found: {POSTERIOR_FILE}")

    if not PREM_FILE.exists():
        raise FileNotFoundError(f"PREM file not found: {PREM_FILE}")

    q, loglike = load_posterior(POSTERIOR_FILE)

    r_km, prem_rho = load_prem_profile(PREM_FILE)

    mean_rho = mean_prem_layer_densities(
        r_km,
        prem_rho,
    )

    rho_gcm3 = q * mean_rho[None, :]

    volumes_m3 = layer_volumes_m3(LAYER_EDGES_KM)

    inertia_coeff_m5 = layer_inertia_coeff_m5(LAYER_EDGES_KM)

    derived = compute_layered_quantities(
        rho_gcm3=rho_gcm3,
        volumes_m3=volumes_m3,
        inertia_coeff_m5=inertia_coeff_m5,
    )

    direct_ref = compute_prem_direct_reference(
        r_km=r_km,
        rho_gcm3=prem_rho,
        layer_edges_km=LAYER_EDGES_KM,
    )

    print("===== PREM direct references from curve integration =====")

    for key, value in direct_ref.items():
        print(f"{key}: {value:.10g}")

    print()

    best_idx = int(np.argmax(loglike))

    print("===== best-fit five-layer model =====")

    for i, name in enumerate(LAYER_NAMES):
        print(
            f"q{i + 1} "
            f"({name}): "
            f"best={q[best_idx, i]:.6g}, "
            f"rho={rho_gcm3[best_idx, i]:.6g} "
            f"g/cm^3"
        )

    print(f"best loglike = {loglike[best_idx]:.10g}")

    print()

    print("===== derived posterior summary =====")

    for key, values in derived.items():
        stats = summarize_1d(values)

        print(
            f"{key}: "
            f"mean={stats['mean']:.6g}, "
            f"median={stats['median']:.6g}, "
            f"std={stats['std']:.6g}, "
            f"68%=[{stats['q16']:.6g}, "
            f"{stats['q84']:.6g}], "
            f"PREM={direct_ref[key]:.6g}"
        )

    p_neg = float(np.mean(derived["delta_rho_gcm3"] < 0.0))

    print(f"P(delta_rho < 0) = {p_neg:.6e}")

    print()

    write_summary(
        SUMMARY_OUT,
        derived,
        direct_ref,
    )

    plot_derived_posteriors(
        derived,
        direct_ref,
        FIGURE_OUT,
    )


if __name__ == "__main__":
    main()
